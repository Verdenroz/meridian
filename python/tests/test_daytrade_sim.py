from __future__ import annotations

from datetime import date, datetime, time

import pytest
from bars_archive.fetch import ET
from daytrade import GATE
from daytrade.signal import Trade, bar_time, find_trade, vwap_path
from daytrade.sim import fill, open_to_close, random_control

SESSION = date(2026, 9, 30)
SIGMA = 0.0002


def _session(prices: list[float], volume: int = 1000) -> list[dict]:
    open_ts = int(datetime.combine(SESSION, time(9, 30), ET).timestamp())
    return [
        {"ts": open_ts + 60 * i, "o": p, "h": p, "l": p, "c": p, "v": volume}
        for i, p in enumerate(prices)
    ]


def _dip_at(minute: int, depth: float = 0.02) -> list[float]:
    """Flat at 100, a dip at `minute`, recovered two minutes later."""
    prices = [100.0] * 390
    prices[minute] = prices[minute + 1] = 100.0 * (1 - depth)
    return prices


class TestSignal:
    def test_entry_fills_on_the_bar_after_the_decision(self) -> None:
        spy = _session(_dip_at(120))
        trade = find_trade(spy, SIGMA, GATE)
        assert trade is not None and trade.symbol == "SPY"
        assert trade.entry_ts == spy[121]["ts"]

    def test_the_entry_never_reads_a_later_bar(self) -> None:
        spy = _session(_dip_at(120))
        altered = spy[:122] + _session([150.0] * 390)[122:]
        original, rerun = find_trade(spy, SIGMA, GATE), find_trade(altered, SIGMA, GATE)
        assert original is not None and rerun is not None
        assert original.entry_ts == rerun.entry_ts

    def test_an_upward_stretch_buys_the_inverse(self) -> None:
        prices = [100.0] * 390
        prices[120] = 102.0
        trade = find_trade(_session(prices), SIGMA, GATE)
        assert trade is not None and trade.symbol == "SH"

    def test_no_entry_before_the_window_opens(self) -> None:
        assert find_trade(_session(_dip_at(10)), SIGMA, GATE) is None

    def test_an_unreverted_position_is_flat_by_the_close(self) -> None:
        prices = [100.0] * 120 + [98.0] * 270
        trade = find_trade(_session(prices), SIGMA, GATE)
        assert trade is not None
        assert bar_time(trade.exit_ts) == time.fromisoformat(GATE["flat_by"])

    def test_only_one_trade_per_session(self) -> None:
        prices = _dip_at(120)
        prices[200] = prices[201] = 98.0
        trade = find_trade(_session(prices), SIGMA, GATE)
        assert trade is not None and trade.entry_ts == _session(prices)[121]["ts"]

    def test_zero_volume_opening_bars_fall_back_to_the_close(self) -> None:
        bars = _session([100.0, 101.0], volume=0)
        assert vwap_path(bars) == [100.0, 101.0]


class TestSim:
    def test_fill_pays_costs_on_both_legs(self) -> None:
        spy = _session([100.0] * 390)
        row = fill(spy, "SPY", spy[0]["ts"], spy[10]["ts"], GATE)
        assert row["pnl"] < 0 and row["qty"] == 99
        assert row["pnl_stress"] == pytest.approx(row["qty"] * (row["exit_px"] - row["entry_px"]) - 2.0)

    def test_random_control_is_reproducible_and_keeps_the_side(self) -> None:
        spy = _session(_dip_at(120))
        trade = find_trade(spy, SIGMA, GATE)
        assert trade is not None
        a = random_control(trade, spy, SESSION, GATE)
        assert a == random_control(trade, spy, SESSION, GATE)
        assert a.symbol == trade.symbol
        assert a != random_control(trade, spy, date(2026, 10, 1), GATE)

    def test_random_control_never_holds_past_the_flat_time(self) -> None:
        spy = _session([100.0] * 390)
        long_hold = Trade("SPY", spy[31]["ts"], spy[385]["ts"])
        control = random_control(long_hold, spy, SESSION, GATE)
        assert bar_time(control.exit_ts) <= time.fromisoformat(GATE["flat_by"])

    def test_open_to_close_spans_the_session(self) -> None:
        spy = _session([100.0] * 389 + [101.0])
        row = open_to_close(spy, GATE)
        assert row["pnl"] > 0
