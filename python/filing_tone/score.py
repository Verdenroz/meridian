"""Filing-text FinBERT scoring into the standing sentiment cache.

Cache contract is exactly the tone study's: sha256(model\x00text) ->
little-endian f32 of signed P(pos) − P(neg), under
`data/sentiment_cache/<h[:2]>/<h>`. Text preprocessing is frozen in the
study registration; the key is the processed text, so scores are
deterministic and shared across studies.

Scoring needs torch + transformers (not project deps — run via
`uv run --with torch --with transformers python -m filing_tone score`).
Reads happen with stdlib only.
"""

from __future__ import annotations

import hashlib
import html
import re
import struct
from pathlib import Path

SENTIMENT_CACHE = Path("data/sentiment_cache")
SENTIMENT_MODEL = "ProsusAI/finbert"
MAX_CHARS = 2000
_WS = re.compile(r"\s+")


def processed_text(path: Path) -> str:
    """The frozen preprocessing: unescape entities, collapse whitespace,
    anchor at the first "Item " heading, first 2000 chars."""
    raw = path.read_text(errors="ignore")
    text = _WS.sub(" ", html.unescape(raw)).strip()
    anchor = text.find("Item ")
    if anchor > 0:
        text = text[anchor:]
    return text.strip()[:MAX_CHARS]


def _cache_path(text: str) -> Path:
    h = hashlib.sha256(f"{SENTIMENT_MODEL}\x00{text}".encode()).hexdigest()
    return SENTIMENT_CACHE / h[:2] / h


def read_cached_tone(text: str) -> float | None:
    if not text:
        return None
    try:
        return struct.unpack("<f", _cache_path(text).read_bytes())[0]
    except (OSError, struct.error):
        return None


def write_tone(text: str, tone: float) -> None:
    path = _cache_path(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<f", tone))


def score_missing(texts: list[str], batch_size: int = 64) -> int:
    """Score uncached texts on the best available device; returns count
    written. Imports torch lazily — see module docstring."""
    todo = [t for t in dict.fromkeys(texts) if t and read_cached_tone(t) is None]
    if not todo:
        return 0
    import torch  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
    model.to(device).eval()
    labels = {i: label for i, label in model.config.id2label.items()}
    written = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        enc = tokenizer(
            batch, truncation=True, padding=True, max_length=512, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        for text, p in zip(batch, probs):
            by_label = {labels[j].lower(): float(p[j]) for j in range(len(p))}
            write_tone(text, by_label["positive"] - by_label["negative"])
            written += 1
        if (i // batch_size) % 20 == 0:
            print(f"  scored {min(i + batch_size, len(todo))}/{len(todo)}", flush=True)
    return written
