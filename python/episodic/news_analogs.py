"""News-day analogs: which historical days *read* like this one?

Explanation-only, like everything episodic: a per-UTC-date centroid of
headline embeddings (legacy harrier-oss:0.6b cache over the FNSPID
corpus) is matched by cosine against the query day. Query days inside
corpus coverage use their stored centroid; later days (the live era)
embed the feeds archive's titles through the local endpoint — the same
model the cache was built with (verified cosine 0.9995 on cached text),
and newly embedded titles are written back into the cache.

The centroid store is derived and rebuildable (`build-news` CLI step);
analogs never touch allocation.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from tone_study.study import FNSPID_CACHE

EMBED_CACHE = Path("data/embed_cache")
EMBED_ENDPOINT = "http://127.0.0.1:12434/v1/embeddings"
EMBED_MODEL = "harrier-oss:0.6b"
EMBED_DIM = 1024
CENTROID_STORE = Path("data/episodic/news_centroids.npz")
FEEDS_DIR = Path("archive/feeds")
MIN_HEADLINES = 30


def _cached_embedding(text: str) -> np.ndarray | None:
    h = hashlib.sha256(text.encode()).hexdigest()
    path = EMBED_CACHE / h[:2] / h
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return np.frombuffer(raw, dtype="<f4") if len(raw) == 4 * EMBED_DIM else None


def _write_embedding(text: str, vec: np.ndarray) -> None:
    h = hashlib.sha256(text.encode()).hexdigest()
    path = EMBED_CACHE / h[:2] / h
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(vec.astype("<f4").tobytes())


def _endpoint_embed(texts: list[str], batch: int = 64) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch):
        req = urllib.request.Request(
            EMBED_ENDPOINT,
            data=json.dumps(
                {"model": EMBED_MODEL, "input": texts[i : i + batch]}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            rows = json.load(resp)["data"]
        out.extend(np.array(r["embedding"], dtype=np.float32) for r in rows)
    return out


def _unit(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else None


def build_centroids() -> dict:
    """Scan the FNSPID corpus once; store per-day unit centroids."""
    sums: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(EMBED_DIM))
    counts: dict[str, int] = defaultdict(int)
    for path in sorted(FNSPID_CACHE.glob("*.jsonl")):
        with path.open() as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    ts, text = row["ts"], row["text"]
                except (ValueError, KeyError):
                    continue
                emb = _cached_embedding(text)
                if emb is None:
                    continue
                unit = _unit(emb)
                if unit is None:
                    continue
                date = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
                sums[date] += unit
                counts[date] += 1
    dates = sorted(d for d, c in counts.items() if c >= MIN_HEADLINES)
    matrix = np.zeros((len(dates), EMBED_DIM), dtype=np.float16)
    for i, d in enumerate(dates):
        unit = _unit(sums[d])
        assert unit is not None
        matrix[i] = unit.astype(np.float16)
    CENTROID_STORE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CENTROID_STORE, dates=np.array(dates), centroids=matrix
    )
    return {"n_days": len(dates), "span": (dates[0], dates[-1]) if dates else None}


def _live_titles(date: str) -> list[str]:
    path = FEEDS_DIR / f"{date}.jsonl"
    titles: list[str] = []
    if not path.is_file():
        return titles
    with path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
                title = row["payload"]["title"]
            except (ValueError, KeyError, TypeError):
                continue
            if title:
                titles.append(title)
    return list(dict.fromkeys(titles))


def day_centroid(date: str) -> np.ndarray | None:
    """Stored centroid if covered; else embed the day's archived titles
    (cache-first, endpoint for the rest, written back to the cache)."""
    if CENTROID_STORE.is_file():
        doc = np.load(CENTROID_STORE, allow_pickle=False)
        idx = np.flatnonzero(doc["dates"] == date)
        if len(idx):
            return doc["centroids"][idx[0]].astype(np.float32)
    titles = _live_titles(date)
    if len(titles) < MIN_HEADLINES:
        return None
    units: list[np.ndarray] = []
    missing: list[str] = []
    for t in titles:
        emb = _cached_embedding(t)
        if emb is None:
            missing.append(t)
        else:
            unit = _unit(emb)
            if unit is not None:
                units.append(unit)
    if missing:
        for text, vec in zip(missing, _endpoint_embed(missing)):
            _write_embedding(text, vec)
            unit = _unit(vec)
            if unit is not None:
                units.append(unit)
    if len(units) < MIN_HEADLINES:
        return None
    return _unit(np.sum(units, axis=0))


def nearest_news_days(date: str, k: int = 5) -> list[tuple[str, float]] | None:
    """Top-k historical days by news-centroid cosine (query day excluded)."""
    query = day_centroid(date)
    if query is None or not CENTROID_STORE.is_file():
        return None
    doc = np.load(CENTROID_STORE, allow_pickle=False)
    dates = [str(d) for d in doc["dates"]]
    matrix = doc["centroids"].astype(np.float32)
    sims = matrix @ query
    order = np.argsort(-sims)
    out = [
        (dates[i], float(sims[i])) for i in order if dates[i] != date
    ][:k]
    return out
