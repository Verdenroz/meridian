"""Train the MarketReader BPE tokenizer on a financial text corpus.

Spec §7.10: vocab_size=4096, trained on headlines, article summaries, and
earnings transcripts collected by the DataBus.

Usage:
    python -m python.train.train_tokenizer \\
        --corpus-dir data/text_corpus \\
        --output     models/tokenizer/bpe_4096.json

Corpus directory layout (any .txt file is accepted):
    data/text_corpus/
        headlines/    *.txt   (one headline per line)
        summaries/    *.txt   (one article summary per line)
        transcripts/  *.txt   (full earnings call transcripts)

The tokenizer uses byte-level BPE with a financial-domain pre-tokenisation
pattern that keeps ticker symbols, currency strings, and percentages intact.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)

VOCAB_SIZE = 4_096

# Keep ticker symbols (1-5 caps), currency + numbers, and percentage strings
# together as single pre-tokens so BPE doesn't split them.
_PRETOKENIZE_PATTERN = (
    r"[A-Z]{1,5}"                   # ticker symbols
    r"|[$€£¥]\d[\d,\.]*"            # currency amounts
    r"|\d[\d,\.]*%"                  # percentages
    r"|\d[\d,\.]*"                   # plain numbers
    r"|[a-zA-Z]+(?:'[a-z]+)?"       # words (with contractions)
    r"|[^\s]"                        # any other non-whitespace
)


def _iter_lines(corpus_dir: Path):
    """Yield every non-empty line from every .txt file under corpus_dir."""
    for path in sorted(corpus_dir.rglob("*.txt")):
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def train(corpus_dir: Path, output_path: Path) -> None:
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
    except ImportError as e:
        raise ImportError(
            "huggingface tokenizers not installed — run: pip install tokenizers"
        ) from e

    log.info("Scanning corpus: %s", corpus_dir)
    lines = list(_iter_lines(corpus_dir))
    if not lines:
        raise ValueError(f"No text found in {corpus_dir}")
    log.info("Corpus: %d lines", len(lines))

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    tokenizer.normalizer = Sequence([NFD(), StripAccents()])

    from tokenizers.pre_tokenizers import Split
    tokenizer.pre_tokenizer = Split(
        pattern=_PRETOKENIZE_PATTERN,
        behavior="isolated",
    )

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(lines, trainer=trainer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    log.info("Saved tokenizer → %s  (vocab=%d)", output_path, tokenizer.get_vocab_size())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train MarketReader BPE tokenizer")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/text_corpus"),
        help="Root directory of .txt corpus files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/tokenizer/bpe_4096.json"),
        help="Output path for the tokenizer JSON",
    )
    args = parser.parse_args()

    train(args.corpus_dir, args.output)


if __name__ == "__main__":
    main()
