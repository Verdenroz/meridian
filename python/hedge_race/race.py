"""Hedge-veto race: is TLT a safe asset during rate turbulence?
Pre-registered before any result was computed.

The spine's defensive sleeve admits TLT while its trailing 60-session
correlation with SPY is negative. 2022 is the failure mode: when stress
originates in *rates*, stocks and bonds fall together, and a 60-session
correlation flips sign only after the damage — the rule is right about
the state but late by construction. Hypothesis: an explicitly causal
rate-turbulence veto removes TLT from the sleeve faster than the
correlation check can, improving the held credit-spine candidate's book
in bond-led stress while changing nothing anywhere else (the veto only
bars a sleeve asset; equity sizing and the regime model are untouched).

Veto signal, fixed a priori: causal trailing z (window 252, min 120) of
the log EWMA vol of daily 10Y yield changes (the rate_race feature
series, shared construction) >= 2 activates the veto for 21 sessions —
burst mechanics inherited frozen from the brake family; 21 is the
codebase's standing monthly horizon (tuner refit cadence, episodic
separation), a single a-priori choice, not tuned. TLT only: rate vol is
bond-specific toxicity, GLD's sleeve rule is untouched.

Challenger: the held 5f credit spine with the TLT veto. References
(non-gating): buy-hold, the untouched null, the held 5f credit spine
without the veto — so the veto's contribution is attributable.

Gate: the challenger clears the charter's amended bar (§1.3) against the
untouched vol-target null — outright after-cost Sharpe win, or held
route (b): non-inferiority 0.05 + max drawdown <= 2/3 of the null's +
beat of the matched-vol placebo. Same engine, full walk-forward span,
same panel as the credit race (SPY/TLT/GLD). Changing any constant after
seeing results invalidates the race.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from credit_race.race import credit_spread_series
from harness.engine import Strategy
from harness.gates import amended_bar
from harness.panel import build_panel
from harness.racing import race_contenders
from harness.signals import brake_series, causal_zscores
from harness.strategies import BuyHold, VolTarget
from rate_race.race import rate_vol_feature
from spine.features import build_features
from spine.strategy import TunedSpineStrategy

__all__ = ["rate_veto_series", "run_race"]

GATE = {
    "burst_z": 2.0,
    "veto_days": 21,
    "z_window": 252,
    "z_min_obs": 120,
    "noninferiority_margin": 0.05,
    "maxdd_ratio": 2 / 3,
}
RACE_SYMBOLS = ("SPY", "TLT", "GLD")
NULL = "vol-target SPY 15% (null)"
CHALLENGER = "spine 5f-credit + TLT veto"


def rate_veto_series(store_dir: Path, dates: list[str]) -> np.ndarray:
    """Veto-active series from the causal z of the rate-vol level."""
    z = causal_zscores(
        rate_vol_feature(store_dir, dates),
        int(GATE["z_window"]),
        int(GATE["z_min_obs"]),
    )
    return brake_series(z, GATE["burst_z"], int(GATE["veto_days"]))


def run_race(store_dir: Path) -> dict:
    panel = build_panel(store_dir, RACE_SYMBOLS)
    features5 = np.column_stack(
        [build_features(panel, store_dir), credit_spread_series(store_dir, panel.dates)]
    )
    veto = rate_veto_series(store_dir, panel.dates)
    first = panel.first_index_with_data(("SPY",))

    contenders: dict[str, Callable[[], Strategy]] = {
        "buy-hold SPY": lambda: BuyHold("SPY"),
        NULL: lambda: VolTarget("SPY"),
        "spine 5f-credit (held ref)": lambda: TunedSpineStrategy(panel, features5),
        CHALLENGER: lambda: TunedSpineStrategy(
            panel, features5, safe_veto={"TLT": veto}
        ),
    }
    runs = race_contenders(panel, contenders, first)
    verdict = amended_bar(runs[CHALLENGER], runs[NULL], panel, first, GATE)
    return {
        "gate": dict(GATE),
        "veto_active_days": int(veto.sum()),
        "n_days": panel.n_days,
        "oos_span": (
            runs[NULL]["result"]["dates"][0],
            runs[NULL]["result"]["dates"][-1],
        ),
        "metrics": {label: r["metrics"] for label, r in runs.items()},
        "verdict": {"challenger": verdict, "race_pass": bool(verdict["passes"])},
    }
