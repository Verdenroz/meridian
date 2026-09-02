"""Invariant tests for the paper broker: cash-account rules, fills,
settlement, idempotent submission, and the target-weight planner."""

from datetime import date
from pathlib import Path

import pytest
from execution.broker.paper import PaperBroker, settlement_date
from execution.broker.paper_state import PaperState, PaperStore
from execution.cycle import run_cycle
from execution.orders import Account, OrderRequest, Position, Quote
from execution.quotes import quote_from_payload
from execution.targets import live_targets, plan_orders
from shadow.log import append_decision

NOW = "2026-09-02T13:40:00+00:00"
TODAY = date(2026, 9, 2)


def _quote(symbol: str, last: float, spread: float = 0.01) -> Quote:
    return Quote(symbol, last - spread, last + spread, last, "REGULAR", NOW)


QUOTES = {"SPLG": _quote("SPLG", 80.0), "SGOV": _quote("SGOV", 100.4)}


def _broker(tmp_path: Path, cash: float = 10_000.0) -> PaperBroker:
    store = PaperStore(tmp_path / "paper")
    store.save(PaperState.initial(cash))
    return PaperBroker(store)


class TestPaperBroker:
    def test_marketable_buy_fills_at_ask(self, tmp_path: Path) -> None:
        broker = _broker(tmp_path)
        order = broker.submit(OrderRequest("k:SPLG:buy", "SPLG", "buy", 10, 80.5), NOW)
        assert order.status == "open"
        (fill,) = broker.process(QUOTES, NOW, TODAY)
        assert fill.fill_price == pytest.approx(80.01)
        assert broker.state.cash == pytest.approx(10_000 - 800.1)
        held = broker.state.position("SPLG")
        assert (held.qty, held.avg_cost) == (10, pytest.approx(80.01))

    def test_non_marketable_order_stays_open_then_cancels(self, tmp_path: Path) -> None:
        broker = _broker(tmp_path)
        order = broker.submit(OrderRequest("k:SPLG:buy", "SPLG", "buy", 10, 79.0), NOW)
        assert broker.process(QUOTES, NOW, TODAY) == ()
        cancelled = broker.cancel(order.id, NOW, "day")
        assert cancelled.status == "cancelled"
        assert broker.open_orders() == ()
        assert broker.state.cash == 10_000

    def test_buy_beyond_cash_is_rejected(self, tmp_path: Path) -> None:
        broker = _broker(tmp_path, cash=1_000.0)
        order = broker.submit(OrderRequest("k:SPLG:buy", "SPLG", "buy", 13, 80.5), NOW)
        assert order.status == "rejected"
        assert order.reason == "insufficient cash"

    def test_open_buys_count_against_cash(self, tmp_path: Path) -> None:
        broker = _broker(tmp_path, cash=1_000.0)
        broker.submit(OrderRequest("a", "SPLG", "buy", 6, 80.5), NOW)
        second = broker.submit(OrderRequest("b", "SPLG", "buy", 7, 80.5), NOW)
        assert second.status == "rejected"

    def test_sell_beyond_position_is_rejected(self, tmp_path: Path) -> None:
        broker = _broker(tmp_path)
        order = broker.submit(OrderRequest("k:SPLG:sell", "SPLG", "sell", 1, 79.0), NOW)
        assert order.status == "rejected"
        assert order.reason == "insufficient shares"

    def test_sale_proceeds_settle_next_weekday(self, tmp_path: Path) -> None:
        broker = _broker(tmp_path)
        broker.submit(OrderRequest("buy", "SPLG", "buy", 10, 80.5), NOW)
        broker.process(QUOTES, NOW, TODAY)
        broker.submit(OrderRequest("sell", "SPLG", "sell", 10, 79.5), NOW)
        broker.process(QUOTES, NOW, date(2026, 9, 4))
        account = broker.account(QUOTES)
        assert account.settled_cash == pytest.approx(account.cash - 799.9)
        broker.settle(date(2026, 9, 4))
        assert broker.state.unsettled
        broker.settle(date(2026, 9, 7))
        assert broker.state.unsettled == ()
        assert broker.account(QUOTES).settled_cash == pytest.approx(account.cash)

    def test_settlement_skips_weekends(self) -> None:
        assert settlement_date(date(2026, 9, 4)) == date(2026, 9, 7)
        assert settlement_date(date(2026, 9, 2)) == date(2026, 9, 3)

    def test_resubmitting_a_client_id_returns_the_same_order(self, tmp_path: Path) -> None:
        broker = _broker(tmp_path)
        first = broker.submit(OrderRequest("k:SPLG:buy", "SPLG", "buy", 10, 80.5), NOW)
        broker.process(QUOTES, NOW, TODAY)
        again = PaperBroker(broker.store).submit(
            OrderRequest("k:SPLG:buy", "SPLG", "buy", 10, 80.5), NOW
        )
        assert again.id == first.id
        assert again.status == "filled"
        assert broker.store.load().position("SPLG").qty == 10

    def test_fractional_qty_is_rejected(self, tmp_path: Path) -> None:
        broker = _broker(tmp_path)
        order = broker.submit(OrderRequest("k", "SPLG", "buy", 0, 80.5), NOW)
        assert order.status == "rejected"

    def test_state_round_trips_through_disk(self, tmp_path: Path) -> None:
        broker = _broker(tmp_path)
        broker.submit(OrderRequest("k", "SPLG", "buy", 10, 80.5), NOW)
        broker.process(QUOTES, NOW, TODAY)
        assert broker.store.load() == broker.state
        kinds = [r["kind"] for r in broker.store.journal_rows()]
        assert kinds == ["submit", "fill"]


class TestPlanner:
    def test_live_targets_map_research_symbols(self) -> None:
        assert live_targets({"SPY": 0.6}, 0.4) == {"SPLG": 0.6, "SGOV": 0.4}
        assert live_targets({"SPY": 0.0}, 1.0) == {"SGOV": 1.0}

    def test_sells_precede_buys_and_use_the_collar(self) -> None:
        account = Account(200.0, 200.0, 10_240.0, (Position("SGOV", 100, 100.0),))
        orders = plan_orders(account, {"SPLG": 0.7, "SGOV": 0.3}, QUOTES, "k")
        assert [o.side for o in orders] == ["sell", "buy"]
        sell, buy = orders
        assert (sell.symbol, sell.qty, sell.limit) == ("SGOV", 70, 99.89)
        assert buy.symbol == "SPLG" and buy.limit == 80.41
        assert buy.qty == int(10_240 * 0.7 // 80.41)

    def test_holdings_inside_the_rounding_zone_are_left_alone(self) -> None:
        account = Account(0.0, 0.0, 8_000.0, (Position("SPLG", 100, 80.0),))
        assert plan_orders(account, {"SPLG": 1.0, "SGOV": 0.0}, QUOTES, "k") == ()

    def test_unlisted_holdings_are_sold_down(self) -> None:
        account = Account(0.0, 0.0, 8_000.0, (Position("SPLG", 100, 80.0),))
        (sell,) = plan_orders(account, {"SGOV": 1.0}, QUOTES, "k")[:1]
        assert (sell.symbol, sell.qty) == ("SPLG", 100)


class TestQuotes:
    def test_touch_is_synthesized_from_cost_table_when_absent(self) -> None:
        q = quote_from_payload(
            {"symbol": "SPLG", "regularMarketPrice": 80.0, "marketState": "POST"}, NOW
        )
        assert q.bid == pytest.approx(80 * (1 - 1.5e-4))
        assert q.ask == pytest.approx(80 * (1 + 1.5e-4))

    def test_feed_bid_ask_win_when_present(self) -> None:
        q = quote_from_payload(
            {"symbol": "SPLG", "regularMarketPrice": 80.0, "bid": 79.9, "ask": 80.1}, NOW
        )
        assert (q.bid, q.ask, q.market_state) == (79.9, 80.1, "UNKNOWN")


class TestCycle:
    def _log(self, tmp_path: Path, weight: float) -> Path:
        path = tmp_path / "decisions.jsonl"
        append_decision(
            path,
            {
                "decision_date": "2026-09-01",
                "strategy": "null_voltarget",
                "weights": {"SPY": weight},
                "cash_weight": 1.0 - weight,
            },
        )
        return path

    def test_cycle_trades_once_and_then_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("execution.cycle.fetch_quotes", lambda *_: QUOTES)
        store = PaperStore(tmp_path / "paper")
        store.save(PaperState.initial(10_000.0))
        log = self._log(tmp_path, 0.6)
        first = run_cycle(store, log, "http://x", today=TODAY)
        assert first["status"] == "done"
        assert first["positions"] == {"SGOV": 39, "SPLG": 74}
        assert all(o["status"] == "filled" for o in first["orders"])
        second = run_cycle(store, log, "http://x", today=TODAY)
        assert second["status"] == "skipped"
        assert store.load().position("SPLG").qty == 74

    def test_closed_market_holds_without_orders(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        closed = {s: Quote(s, q.bid, q.ask, q.last, "PRE", NOW) for s, q in QUOTES.items()}
        monkeypatch.setattr("execution.cycle.fetch_quotes", lambda *_: closed)
        store = PaperStore(tmp_path / "paper")
        store.save(PaperState.initial(10_000.0))
        report = run_cycle(store, self._log(tmp_path, 1.0), "http://x", today=TODAY)
        assert report["status"] == "market_closed"
        assert store.load().orders == ()
        assert [r["kind"] for r in store.journal_rows()] == []
