from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MarketReaderConfig:
    model_dim: int = 256
    layers: int = 4
    vocab_size: int = 4096
    output_dim: int = 64
    sequence_length: int = 512
    optimizer: str = "AdamW"
    learning_rate: float = 3e-4
    batch_size: int = 16
    epochs: int = 1

    def validate(self) -> None:
        if self.vocab_size <= 0 or self.output_dim <= 0:
            raise ValueError("vocab_size and output_dim must be positive")
        if self.layers < 1:
            raise ValueError("layers must be >= 1")
        if self.sequence_length < 8:
            raise ValueError("sequence_length must be >= 8")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be >= 1")


def build_marketreader_training_plan(
    config: MarketReaderConfig,
    corpus_path: Path,
    tokenizer_output_path: Path,
    model_output_path: Path,
) -> dict[str, object]:
    config.validate()
    return {
        **asdict(config),
        "corpus_path": str(corpus_path),
        "tokenizer_output_path": str(tokenizer_output_path),
        "model_output_path": str(model_output_path),
        "objective": "masked_language_modeling_plus_finance_embedding_projection",
    }


def _extract_text(payload: Any) -> str | None:
    if isinstance(payload, str):
        text = payload.strip()
        return text if text else None

    if isinstance(payload, dict):
        for key in ("text", "content", "body"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        title = payload.get("title")
        summary = payload.get("summary")
        if isinstance(title, str) and isinstance(summary, str):
            composite = f"{title.strip()} {summary.strip()}".strip()
            return composite if composite else None
        for key in ("headline", "title", "summary"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def load_corpus_texts(corpus_path: Path, max_documents: int | None = None) -> list[str]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    texts: list[str] = []
    for line_number, raw_line in enumerate(corpus_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue

        payload: Any
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            payload = raw_line

        extracted = _extract_text(payload)
        if extracted:
            texts.append(extracted)
            if max_documents is not None and len(texts) >= max_documents:
                break
        elif isinstance(payload, dict):
            raise ValueError(f"No text-like field found in corpus JSONL line {line_number}")

    if not texts:
        raise ValueError("Corpus did not contain any usable text documents")
    return texts


def train_bpe_tokenizer(texts: list[str], output_path: Path, vocab_size: int):
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers  # type: ignore
    except ImportError as error:
        raise RuntimeError("tokenizers is required. Install with: pip install -e '.[train]'") from error

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def _encode_sequences(tokenizer, texts: list[str], sequence_length: int) -> tuple[list[list[int]], list[list[int]], int]:
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        pad_id = 0

    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for text in texts:
        token_ids = tokenizer.encode(text).ids
        if len(token_ids) < 2:
            continue
        token_ids = token_ids[: sequence_length + 1]
        x = token_ids[:-1]
        y = token_ids[1:]
        if len(x) < sequence_length:
            pad = [pad_id] * (sequence_length - len(x))
            x = x + pad
            y = y + pad
        inputs.append(x)
        targets.append(y)

    if not inputs:
        raise ValueError("No tokenizable sequences produced from corpus")
    return inputs, targets, pad_id


def run_marketreader_training(
    config: MarketReaderConfig,
    corpus_path: Path,
    tokenizer_output_path: Path,
    model_output_path: Path,
    metrics_output_path: Path | None = None,
    max_documents: int | None = None,
    device: str | None = None,
) -> dict[str, object]:
    config.validate()
    texts = load_corpus_texts(corpus_path, max_documents=max_documents)
    tokenizer = train_bpe_tokenizer(texts, tokenizer_output_path, config.vocab_size)
    x_rows, y_rows, pad_id = _encode_sequences(tokenizer, texts, config.sequence_length)

    try:
        import torch  # type: ignore
        from torch import nn  # type: ignore
        from torch.utils.data import DataLoader, TensorDataset  # type: ignore
    except ImportError as error:
        raise RuntimeError("torch is required. Install with: pip install -e '.[train]'") from error

    train_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    class TinyMarketReader(nn.Module):
        def __init__(self, vocab_size: int, model_dim: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_id)
            self.gru = nn.GRU(model_dim, model_dim, batch_first=True)
            self.head = nn.Linear(model_dim, vocab_size)

        def forward(self, token_ids):
            embedded = self.embedding(token_ids)
            out, _ = self.gru(embedded)
            return self.head(out)

    model = TinyMarketReader(config.vocab_size, config.model_dim).to(train_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    x_tensor = torch.tensor(x_rows, dtype=torch.long)
    y_tensor = torch.tensor(y_rows, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    loss_history: list[float] = []
    model.train()
    for _epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(train_device)
            batch_y = batch_y.to(train_device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits.reshape(-1, config.vocab_size), batch_y.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            epoch_steps += 1
        loss_history.append(epoch_loss / max(1, epoch_steps))

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "tokenizer_path": str(tokenizer_output_path),
            "pad_id": pad_id,
        },
        model_output_path,
    )

    metrics = {
        "documents_used": len(texts),
        "sequences": len(x_rows),
        "loss_history": loss_history,
        "device": train_device,
        "tokenizer_output_path": str(tokenizer_output_path),
        "model_output_path": str(model_output_path),
    }
    target_metrics_path = metrics_output_path or model_output_path.with_suffix(".metrics.json")
    target_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    target_metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    metrics["metrics_output_path"] = str(target_metrics_path)
    return metrics
