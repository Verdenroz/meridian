"""4-state Hidden Markov Model for regime pre-fitting — Section 7.2.

The HMM provides prior regime posteriors that seed the neural regime detector.
It is fitted once on historical data and frozen during model training.

Regimes (4 states):
  0: Low-vol bull    — trending up, VIX < 15, ADX > 25
  1: High-vol bull   — trending up, VIX 15–25, elevated correlation
  2: Low-vol bear    — trending down, VIX < 20, low breadth
  3: Crisis          — VIX > 30, high correlation, large drawdowns

Input features (4-dim per timestep):
  vix_level, adx14, corr_dispersion, drawdown

Output:
  - Transition matrix A (4×4)
  - Emission parameters per state (mean + covariance)
  - Initial state distribution π

Usage:
  meridian train hmm --input data/processed/regime_features.jsonl \
                     --features vix_level,adx14,corr_dispersion,drawdown
  → writes artifacts/hmm.pkl
"""

from __future__ import annotations

# TODO: implement HMM training pipeline (Section 7.2 of meridian-spec.md)
