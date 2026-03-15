"""meridian CLI entrypoint.

Commands (Section 16 of meridian-spec.md):

  check:
    meridian check spec                                   — validate architecture hyperparameters
    meridian check features --asset-class <cls>           — validate feature schema for asset class
    meridian check split --timestamp <ts>                 — validate train/val/test split assignment
    meridian check env [--allow-non-h100]                 — validate H100 training environment

  data:
    meridian data init --root <path>                      — initialize workspace directory layout
    meridian data plan --assets <syms> --output <path>    — generate data pipeline manifest
    meridian data ingest --input <jsonl> --asset <sym> --output-root <path>
                                                          — ingest JSONL rows → memory-mapped shards
    meridian data collect --asset <sym> --start <date> --end <date>
                                                          — collect historical data via Rust DataBus

  train:
    meridian train plan --stage <stage>                   — print training plan for a phase
    meridian train marketreader [--plan] --corpus <jsonl> [--epochs N] [--max-documents N]
                                                          — train MarketReader text encoder
    meridian train hmm [--plan] --input <jsonl> --features <list>
                                                          — pre-fit 4-state HMM on regime features

  smoke:
    meridian smoke core --batch-size N --seq-len N        — smoke test model core
    meridian smoke backbone --batch-size N --seq-len N    — smoke test RA-GDN backbone
    meridian smoke cascade --batch-size N --seq-len N     — smoke test temporal cascade
    meridian smoke fusion --n-assets N --n-ctx N --batch-size N
                                                          — smoke test cross-asset fusion
    meridian smoke train --steps N [options]              — smoke test training loop
    meridian smoke eval --shards-root <path> --splits <list> [options]
                                                          — smoke test evaluation on shards
"""

from __future__ import annotations

# TODO: implement CLI (Section 16 of meridian-spec.md)


def main() -> None:
    # TODO: implement using argparse or click
    raise NotImplementedError
