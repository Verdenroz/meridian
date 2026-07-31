"""Distill the Pi-native MarketReader encoder from the harrier-oss + FinBERT teachers.

Spec §7.10. The on-device 3.5M MarketReader must reproduce, in-process on a Pi 5,
what two heavy teachers produce offline:

  • Deep head (Block 8, 64-dim)  ← harrier-oss-0.6b (1024-dim) reduced by PCA(64)
  • Headline head (sentiment)    ← FinBERT signed score  P(pos) − P(neg)

Because the trained student *replaces* both teachers at inference, the distillation
target only needs a fixed, internally-consistent 64-dim reduction — so this script
is fully self-contained: it calls the same teacher endpoints the collector uses
(harrier-oss `/v1/embeddings`, FinBERT `/predict`), fits its own PCA(64) over the
harrier embeddings, and caches `{text, deep[64], sentiment}` records. The distill
phase then trains MarketReader against those records.

Modes (``--mode``):
  build-targets  text corpus (*.txt) → teacher targets JSONL   [needs GPU teachers]
  distill        targets JSONL + BPE tokenizer → MarketReader checkpoint
  smoke          synthetic end-to-end self-test of the distill loop (no network)

Pipeline (once the eGPU + teachers are up):
  make text-corpus   # dump FNSPID/EDGAR text the collector already gathered
  make tokenizer     # python -m train.train_tokenizer  → bpe_4096.json
  make marketreader  # build-targets (cached) then distill
Then re-anchor features to the student (Rust forward port, A5) before retraining.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from market_data.marketreader import MarketReader

log = logging.getLogger("train_marketreader")

DEEP_DIM = 64
HEADLINE_DIM = 4


# ── Corpus dump (dump-corpus) ────────────────────────────────────────────────


def dump_corpus(args: argparse.Namespace) -> None:
    """Dump the text the collector already gathered (FNSPID headline caches +
    EDGAR filing text) into newline-delimited *.txt for the tokenizer + targets.

    FNSPID cache: ``data/.fnspid_cache/{SYMBOL}.jsonl`` of ``{"ts","text"}``.
    EDGAR text:   ``data/edgar_text/**`` plain-text documents.
    """
    out_dir = Path(args.corpus_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n = 0
    with (out_dir / "corpus.txt").open("w", encoding="utf-8") as out:
        fnspid = Path(args.fnspid_dir)
        if fnspid.is_dir():
            for jl in sorted(fnspid.glob("*.jsonl")):
                for line in jl.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        text = str(json.loads(line).get("text", "")).strip()
                    except json.JSONDecodeError:
                        continue
                    if text and text not in seen:
                        seen.add(text)
                        out.write(text + "\n")
                        n += 1
        edgar = Path(args.edgar_dir)
        if edgar.is_dir():
            for doc in sorted(edgar.rglob("*")):
                if not doc.is_file():
                    continue
                for line in doc.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if line and line not in seen:
                        seen.add(line)
                        out.write(line + "\n")
                        n += 1
    if n == 0:
        raise ValueError(
            f"no text found under {args.fnspid_dir} or {args.edgar_dir} "
            "— run `make collect` (or `make fnspid`) first"
        )
    log.info("dumped %d unique snippets → %s", n, out_dir / "corpus.txt")


# ── Teacher-target generation (build-targets) ────────────────────────────────


def _embed_harrier(texts: list[str], endpoint: str, model: str) -> np.ndarray:
    """POST texts to the OpenAI-compatible harrier-oss embeddings endpoint."""
    import urllib.request

    body = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        endpoint, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read())
    # OpenAI shape: {"data": [{"embedding": [...]}, ...]} (order preserved)
    return np.asarray([d["embedding"] for d in payload["data"]], dtype=np.float32)


def _finbert_signed(text: str, endpoint: str) -> float:
    """POST one text to the FinBERT sidecar → signed P(pos) − P(neg) in [-1, 1]."""
    import urllib.request

    body = json.dumps({"inputs": text}).encode()
    req = urllib.request.Request(
        endpoint, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        scores = json.loads(resp.read())
    if scores and isinstance(scores[0], list):  # nested [[{label,score}]]
        scores = scores[0]
    pos = neg = 0.0
    for s in scores:
        label = str(s.get("label", "")).lower()
        if label == "positive":
            pos = float(s["score"])
        elif label == "negative":
            neg = float(s["score"])
    return float(np.clip(pos - neg, -1.0, 1.0))


def _fit_pca(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA(k): returns (mean[D], components[k, D]) via SVD on centered data."""
    mean = x.mean(axis=0)
    xc = x - mean
    # economy SVD; rows of Vt are principal directions
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return mean, vt[:k]


def build_targets(args: argparse.Namespace) -> None:
    """Read *.txt corpus, query teachers, fit PCA(64), write targets JSONL."""
    corpus_dir = Path(args.corpus_dir)
    texts = [
        line.strip()
        for p in sorted(corpus_dir.rglob("*.txt"))
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if not texts:
        raise ValueError(f"no text under {corpus_dir}")
    log.info("corpus: %d snippets", len(texts))

    # Embed in batches → harrier 1024-dim.
    embs: list[np.ndarray] = []
    bs = args.embed_batch
    for i in range(0, len(texts), bs):
        embs.append(_embed_harrier(texts[i : i + bs], args.embed_endpoint, args.embed_model))
        log.info("embedded %d / %d", min(i + bs, len(texts)), len(texts))
    emb = np.concatenate(embs, axis=0)

    mean, comps = _fit_pca(emb, DEEP_DIM)
    deep = (emb - mean) @ comps.T  # (N, 64)
    deep /= np.linalg.norm(deep, axis=1, keepdims=True).clip(min=1e-8)

    out = Path(args.targets)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Persist the PCA basis alongside the targets for provenance / re-anchoring.
    np.savez(out.with_suffix(".pca.npz"), mean=mean, components=comps)
    with out.open("w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            sentiment = _finbert_signed(text, args.sentiment_endpoint)
            f.write(json.dumps({"text": text, "deep": deep[i].tolist(), "sentiment": sentiment}) + "\n")
            if (i + 1) % 200 == 0:
                log.info("scored %d / %d", i + 1, len(texts))
    log.info("wrote targets → %s  (+ %s)", out, out.with_suffix(".pca.npz"))


# ── Distillation ─────────────────────────────────────────────────────────────


@dataclass
class DistillConfig:
    steps: int = 5_000
    batch_size: int = 64
    lr: float = 3e-4
    min_lr: float = 1e-5
    val_frac: float = 0.1
    lambda_deep: float = 1.0
    lambda_sent: float = 0.5
    max_deep_len: int = MarketReader.MAX_DEEP_LEN
    log_every: int = 100
    seed: int = 0


def _student_sentiment(headline_logits: torch.Tensor) -> torch.Tensor:
    """Map the 4-dim headline head → signed sentiment in [-1, 1]: tanh(pos − neg)."""
    return torch.tanh(headline_logits[:, 0] - headline_logits[:, 1])


def distill_loss(
    model: MarketReader,
    ids: torch.Tensor,
    pad_mask: torch.Tensor,
    deep_t: torch.Tensor,
    sent_t: torch.Tensor,
    cfg: DistillConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    deep_p = F.normalize(model.encode_deep(ids, pad_mask), dim=-1)
    deep_t = F.normalize(deep_t, dim=-1)
    deep_loss = (1.0 - (deep_p * deep_t).sum(-1)).mean() + F.mse_loss(deep_p, deep_t)

    sent_p = _student_sentiment(model.encode_headline(ids[:, : MarketReader.MAX_HEADLINE_LEN]))
    sent_loss = F.mse_loss(sent_p, sent_t)

    total = cfg.lambda_deep * deep_loss + cfg.lambda_sent * sent_loss
    return total, {
        "loss": float(total.detach()),
        "deep": float(deep_loss.detach()),
        "sent": float(sent_loss.detach()),
    }


def _load_targets(path: Path) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    texts, deep, sent = [], [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            deep.append(r["deep"])
            sent.append(r["sentiment"])
    return texts, torch.tensor(deep, dtype=torch.float32), torch.tensor(sent, dtype=torch.float32)


def distill(args: argparse.Namespace) -> None:
    cfg = DistillConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts, deep, sent = _load_targets(Path(args.targets))
    log.info("targets: %d records", len(texts))

    model = MarketReader(deep_dim=DEEP_DIM, headline_dim=HEADLINE_DIM).to(device)
    model.load_tokenizer(args.tokenizer)
    # Pre-tokenize once (deep length) — headline reuses the prefix.
    ids, pad = model.tokenize(texts, cfg.max_deep_len)
    n = len(texts)
    n_val = max(1, int(n * cfg.val_frac))
    perm = torch.randperm(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps, eta_min=cfg.min_lr)

    best_val = float("inf")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for step in range(cfg.steps):
        model.train()
        b = train_idx[torch.randint(len(train_idx), (cfg.batch_size,))]
        loss, parts = distill_loss(
            model, ids[b].to(device), pad[b].to(device),
            deep[b].to(device), sent[b].to(device), cfg,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            val = _eval(model, ids, pad, deep, sent, val_idx, cfg, device)
            log.info("step %d  train=%.4f deep=%.4f sent=%.4f  val_cos=%.4f val_sent=%.4f",
                     step, parts["loss"], parts["deep"], parts["sent"],
                     val["deep_cos"], val["sent_mae"])
            if val["combined"] < best_val:
                best_val = val["combined"]
                _save(model, out, val)
    log.info("done — best checkpoint → %s", out)


@torch.no_grad()
def _eval(model, ids, pad, deep, sent, idx, cfg, device) -> dict[str, float]:
    model.eval()
    deep_p = F.normalize(model.encode_deep(ids[idx].to(device), pad[idx].to(device)), dim=-1)
    deep_t = F.normalize(deep[idx].to(device), dim=-1)
    cos = float((deep_p * deep_t).sum(-1).mean())
    sent_p = _student_sentiment(model.encode_headline(ids[idx, : MarketReader.MAX_HEADLINE_LEN].to(device)))
    mae = float((sent_p - sent[idx].to(device)).abs().mean())
    return {"deep_cos": cos, "sent_mae": mae, "combined": (1.0 - cos) + mae}


def _save(model: MarketReader, path: Path, metrics: dict[str, float]) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "format_version": 2,
            "architecture_version": "v4.0",
            "kind": "marketreader",
            "deep_dim": DEEP_DIM,
            "headline_dim": HEADLINE_DIM,
            "val": metrics,
        },
        path,
    )


# ── Synthetic smoke test (no network, no corpus, CPU) ────────────────────────


def smoke(_args: argparse.Namespace | None = None) -> None:
    """End-to-end exercise of the distill loop on random data: assert the loss is
    finite and decreases. Validates model + loss + optimizer wiring without the
    tokenizer, teachers, or GPU."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    vocab, dim, n, seq = 256, 64, 96, 48
    model = MarketReader(vocab_size=vocab, dim=dim, deep_dim=DEEP_DIM, headline_dim=HEADLINE_DIM).to(device)

    ids = torch.randint(1, vocab, (n, seq))
    pad = torch.zeros(n, seq, dtype=torch.bool)
    # A learnable-but-deterministic target so the loss can actually descend.
    deep_t = F.normalize(torch.randn(n, DEEP_DIM), dim=-1)
    sent_t = torch.tanh(torch.randn(n))
    cfg = DistillConfig(batch_size=32)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = last = None
    for step in range(60):
        b = torch.randint(n, (cfg.batch_size,))
        loss, parts = distill_loss(model, ids[b], pad[b], deep_t[b], sent_t[b], cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), "non-finite loss"
        if step == 0:
            first = parts["loss"]
        last = parts["loss"]
    assert last < first, f"loss did not decrease ({first:.4f} → {last:.4f})"
    n_params = sum(p.numel() for p in MarketReader().parameters())
    log.info("SMOKE OK  loss %.4f → %.4f   full-model params=%.2fM", first, last, n_params / 1e6)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Distill the Pi-native MarketReader encoder")
    p.add_argument(
        "--mode",
        choices=["dump-corpus", "build-targets", "distill", "smoke"],
        default="smoke",
    )
    # dump-corpus
    p.add_argument("--fnspid-dir", default="data/.fnspid_cache")
    p.add_argument("--edgar-dir", default="data/edgar_text")
    # build-targets
    p.add_argument("--corpus-dir", default="data/text_corpus")
    p.add_argument("--targets", default="data/text_corpus/targets.jsonl")
    p.add_argument("--embed-endpoint", default="http://127.0.0.1:12434/v1/embeddings")
    p.add_argument("--embed-model", default="harrier-oss:0.6b")
    p.add_argument("--embed-batch", type=int, default=32)
    p.add_argument("--sentiment-endpoint", default="http://127.0.0.1:8088/predict")
    # distill
    p.add_argument("--tokenizer", default="models/tokenizer/bpe_4096.json")
    p.add_argument("--out", default="checkpoints/marketreader/marketreader_best.pt")
    p.add_argument("--steps", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    {
        "dump-corpus": dump_corpus,
        "build-targets": build_targets,
        "distill": distill,
        "smoke": smoke,
    }[args.mode](args)


if __name__ == "__main__":
    main()
