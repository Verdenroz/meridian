"""Shadow-equivalent decision replay for fidelity (C2) and reference (C3).

Contender feature series are causal by construction (asserted by the race
test suites), so replaying against the full panel with views capped at t
reproduces what the shadow job computed on a panel that ended at t — up
to upstream data revisions, which the fidelity budget deliberately
charges. Expect the full evaluation to take tens of minutes: the spine
tuner races six configs through the engine at every refit.
"""

from __future__ import annotations

from pathlib import Path

from credit_race.race import RACE_SYMBOLS
from harness.panel import Panel, build_panel
from harness.view import HistoryView
from shadow.contenders import build_contenders, extract_meta, round_weights
from spine.strategy import TunedSpineStrategy

from behavior_match.criteria import CRITERIA


def build_race_panel(store_dir: Path) -> Panel:
    return build_panel(store_dir, RACE_SYMBOLS)


def trading_days_between(panel: Panel, start: str, end: str) -> list[str]:
    return [d for d in panel.dates if start <= d <= end]


def _decision_row(name: str, strategy: object, view: HistoryView) -> dict:
    weights = strategy.target_weights(view) or {}  # type: ignore[attr-defined]
    weights = round_weights(weights)
    return {
        "decision_date": view.today,
        "strategy": name,
        "weights": weights,
        "cash_weight": round(1.0 - sum(weights.values()), 6),
        "meta": extract_meta(name, strategy, view),
    }


def retro_weights(
    panel: Panel, store_dir: Path, decision_dates: list[str]
) -> dict[tuple[str, str], dict[str, float]]:
    """Re-run the exact shadow procedure (fresh fit per date) and return
    weights keyed by (decision_date, strategy)."""
    contenders = build_contenders(panel, store_dir)
    index = {d: i for i, d in enumerate(panel.dates)}
    out: dict[tuple[str, str], dict[str, float]] = {}
    for date in decision_dates:
        if date not in index:
            raise ValueError(f"logged decision date {date} not in panel calendar")
        view = HistoryView(panel, index[date])
        for name, strategy in contenders.items():
            strategy.fit(view)  # type: ignore[attr-defined]
            out[(date, name)] = _decision_row(name, strategy, view)["weights"]
    return out


def reference_rows(panel: Panel, store_dir: Path) -> dict[str, list[dict]]:
    """Daily shadow-semantics replay over the pre-shadow panel. Fresh band
    state every day; the spine's full tuner refit runs on the frozen stride."""
    contenders = build_contenders(panel, store_dir)
    start = panel.first_index_with_data(("SPY",)) + int(
        CRITERIA["reference_warmup_sessions"]
    )
    end_date = str(CRITERIA["reference_end"])
    end = max(i for i, d in enumerate(panel.dates) if d <= end_date)
    stride = int(CRITERIA["reference_fit_stride"])
    out: dict[str, list[dict]] = {}
    for name, strategy in contenders.items():
        rows: list[dict] = []
        for step, t in enumerate(range(start, end + 1)):
            view = HistoryView(panel, t)
            if isinstance(strategy, TunedSpineStrategy):
                if step % stride == 0:
                    strategy.fit(view)
                else:
                    # Daily band reset mirrors the shadow job's fresh-fit-per-day
                    # semantics; the filter itself advances deterministically.
                    assert strategy._inner is not None
                    strategy._inner._last_target = None
            else:
                strategy.fit(view)  # type: ignore[attr-defined]
            rows.append(_decision_row(name, strategy, view))
        out[name] = rows
    return out
