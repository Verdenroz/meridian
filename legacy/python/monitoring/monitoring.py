"""Adaptation monitoring dashboard — Section 13.3 of meridian-spec.md.

Tracks health of all three adaptation tiers and model internals during paper trading.

Tier 1 (Surprise-gated dynamics):
  - Plot: surprise signal over time with regime labels
  - Alert: if surprise EMA trends upward for 5+ consecutive days

Tier 2 (Normalization adaptation):
  - Plot: RMSNorm parameter drift from checkpoint
  - Alert: if drift > 20%

Tier 3 (EWC + LoRA consolidation):
  - Plot: nightly consolidation loss trajectory
  - Plot: Fisher diagonal magnitude distribution
  - Alert: if consolidation loss fails to decrease

State health:
  - Plot: GDN hidden state norms per layer over time
  - Plot: per-asset state norms
  - Alert: monotonically increasing state norms (drift — Risk #1)

Comparison to frozen model:
  - Run second Meridian instance with all adaptation disabled
  - Track: Sharpe_adaptive − Sharpe_frozen (target: +0.1 after 3 months)
"""

from __future__ import annotations

# TODO: implement monitoring dashboard (Section 13.3 of meridian-spec.md)
