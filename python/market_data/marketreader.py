"""MarketReader text encoder — Section 7.10 of meridian-spec.md.

Three-tier processing pipeline (Section 6.5):

  Tier 1 — Headline-only (every 5 min, 31 RSS sources):
    Input:  title + source name
    Output: 4-dim sentiment vector (positive, negative, uncertainty, finance-relevance)
    Model:  lightweight BPE tokenizer + single-layer transformer

  Tier 2 — Summary-level (every 15 min, high-impact articles):
    Input:  FeedEntry.summary (1–3 paragraphs)
    Output: 32-dim topic + sentiment vector
    Model:  same tokenizer + deeper transformer head

  Tier 3 — Deep-level (on arrival, queued):
    Input:  full article text / earnings transcript
    Output: 64-dim embedding
    Model:  full MarketReader encoder

  Per-asset text aggregation:
    text_embedding_i = Σ_j (attention_weight_j × decay_j × embedding_j)
    decay_j = exp(-minutes_since_j / (τ × 60))
    τ = 4h (headlines), 12h (summaries), 48h (transcripts)

Training:
  - BPE tokenizer: 16K vocab, trained on financial news corpus
  - Self-supervised: masked language modeling on financial text
  - Fine-tuned: contrastive loss on (article, subsequent_return) pairs

~3.5M parameters, INT8 quantized, runs async on Core 3 of Pi 5.
"""

from __future__ import annotations

# TODO: implement MarketReader (Section 7.10 of meridian-spec.md)
