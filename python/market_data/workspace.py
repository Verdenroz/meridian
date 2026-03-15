"""Data workspace management.

Workspace directory layout (meridian data init):
  <root>/
    data/
      raw/          — raw JSONL rows from DataBus or manual collection
      processed/    — regime feature files, normalized statistics
      mmaps/
        train/      — memory-mapped f32 shards (*.f32bin)
        validation/
        test/
    artifacts/
      tokenizer.json   — BPE tokenizer for MarketReader
      hmm.pkl          — fitted 4-state HMM
    checkpoints/       — model checkpoints (backbone_pretrain.pt, etc.)
    reports/
      ingestion/    — per-asset ingestion manifests
      backtests/    — backtest result JSON files
    logs/           — paper trading SQLite database
"""

from __future__ import annotations

# TODO: implement workspace initialization and path resolution
