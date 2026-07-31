"""Point-in-time view of a panel: structural lookahead prevention.

Strategies never touch the panel directly — the engine hands them a view whose
accessors cannot reach past the current day, so leakage is a type error rather
than a discipline.
"""

from __future__ import annotations

from harness.panel import Panel


class HistoryView:
    """Read-only window over `panel` ending at index `t` (inclusive). Decisions
    happen after the close of day t and take effect on day t+1's returns."""

    def __init__(self, panel: Panel, t: int) -> None:
        self._panel = panel
        self._t = t

    @property
    def today(self) -> str:
        return self._panel.dates[self._t]

    @property
    def t(self) -> int:
        return self._t

    def returns_window(self, symbol: str, n: int) -> list[float]:
        """Last ≤n defined daily returns up to and including day t."""
        rets = self._panel.returns[symbol]
        out: list[float] = []
        for i in range(self._t, 0, -1):
            r = rets[i]
            if r is not None:
                out.append(r)
                if len(out) == n:
                    break
        out.reverse()
        return out

    def adj_close(self, symbol: str) -> float | None:
        return self._panel.adj_close[symbol][self._t]

    def cash_rate_daily(self) -> float:
        return self._panel.cash_returns[self._t]
