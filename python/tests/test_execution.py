"""Invariant tests for the paper broker: cash-account rules, fills,
settlement, idempotent submission, and the target-weight planner."""

import http.server
import random
import threading
from datetime import date, datetime
from pathlib import Path

import pytest
from execution.broker.paper import PaperBroker, settlement_date
from execution.broker.paper_state import PaperState, PaperStore
from execution.cycle import run_cycle
from execution.guardrails import LIMITS, Rejection, validate
from execution.notify import alert
from execution.orders import Account, OrderRequest, Position, Quote
from execution.quotes import quote_from_payload
from execution.targets import live_targets, plan_orders
from execution.watchdog import last_session, paper_findings, shadow_findings
from shadow.log import append_decision

NOW = "2026-09-02T13:40:00+00:00"
TODAY = date(2026, 9, 2)


def _quote(symbol: str, last: float, spread: float = 0.01) -> Quote:
    return Quote(symbol, last - spread, last + spread, last, "REGULAR", NOW)


QUOTES = {"SPYM": _quote("SPYM", 80.0), "SGOV": _quote("SGOV", 100.4)}


def _broker(tmp_path: Path, cash: float = 10_000.0) -> PaperBroker:
    store = PaperStore(tmp_path / "paper")
    store.save(PaperState.initial(cash))
    return PaperBroker(store)


class TestPaperBroker:
    def test_marketable_buy_fills_at_ask(self, tmp_path: Path) -> None:
        broker = _broker(tmp_path)
        order = broker.submit(OrderRequest("k:SPYM:buy", "SPYM", "buy", 10, 80.5), NOW)
        assert order.status == "open"
        (fill,) = broker.process(QUOTES, NOW, TODAY)
        assert fill.fill_price == pytest.approx(80.01)
        assert broker.state.cash == pytest.approx(10_000 - 800.1 - 0.35)
        held = broker.state.position("SPYM")
        assert (held.qty, held.avg_cost) == (10, pytest.approx(80.01))

    def test_non_marketable_order_stays_open_then_cancels(self, tmp_path: Path) -> None:
        broker = _broker(tmp_path)
        order = broker.submit(OrderRequest("k:SPYM:buy", "SPYM", "buy", 10, 79.0), NOW)
        assert broker.process(QUOTES, NOW, TODAY) == ()
        cancelled = broker.cancel(order.id, NOW, "day")
        assert cancelled.status == "cancelled"
        assert broker.open_orders() == ()
        assert broker.state.cash == 10_000

    def test_buy_beyond_cash_is_rejected(self, tmp_path: Path) -> None:
        broker = _broker(tmp_path, cash=1_000.0)
        order = broker.submit(OrderRequest("k:SPYM:buy", "SPYM", "buy", 13, 80.5), NOW)
        assert order.status == "rejected"
        assert order.reason == "insufficient cash"

    def test_open_buys_count_against_cash(self, tmp_path: Path) -> None:
        broker = _broker(tmp_path, cash=1_000.0)
        broker.submit(OrderRequest("a", "SPYM", "buy", 6, 80.5), NOW)
        second = broker.submit(OrderRequest("b", "SPYM", "buy", 7, 80.5), NOW)
        assert second.status == "rejected"

    def test_sell_beyond_position_is_rejected(self, tmp_path: Path) -> None:
        broker = _broker(tmp_path)
        order = broker.submit(OrderRequest("k:SPYM:sell", "SPYM", "sell", 1, 79.0), NOW)
        assert order.status == "rejected"
        assert order.reason == "insufficient shares"

    def test_sale_proceeds_settle_next_weekday(self, tmp_path: Path) -> None:
        broker = _broker(tmp_path)
        broker.submit(OrderRequest("buy", "SPYM", "buy", 10, 80.5), NOW)
        broker.process(QUOTES, NOW, TODAY)
        broker.submit(OrderRequest("sell", "SPYM", "sell", 10, 79.5), NOW)
        broker.process(QUOTES, NOW, date(2026, 9, 4))
        account = broker.account(QUOTES)
        assert account.settled_cash == pytest.approx(account.cash - (799.9 - 0.35))
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
        first = broker.submit(OrderRequest("k:SPYM:buy", "SPYM", "buy", 10, 80.5), NOW)
        broker.process(QUOTES, NOW, TODAY)
        again = PaperBroker(broker.store).submit(
            OrderRequest("k:SPYM:buy", "SPYM", "buy", 10, 80.5), NOW
        )
        assert again.id == first.id
        assert again.status == "filled"
        assert broker.store.load().position("SPYM").qty == 10

    def test_fractional_qty_is_rejected(self, tmp_path: Path) -> None:
        broker = _broker(tmp_path)
        order = broker.submit(OrderRequest("k", "SPYM", "buy", 0, 80.5), NOW)
        assert order.status == "rejected"

    def test_state_round_trips_through_disk(self, tmp_path: Path) -> None:
        broker = _broker(tmp_path)
        broker.submit(OrderRequest("k", "SPYM", "buy", 10, 80.5), NOW)
        broker.process(QUOTES, NOW, TODAY)
        assert broker.store.load() == broker.state
        kinds = [r["kind"] for r in broker.store.journal_rows()]
        assert kinds == ["submit", "fill"]


class TestPlanner:
    def test_live_targets_map_research_symbols(self) -> None:
        assert live_targets({"SPY": 0.6}, 0.4) == {"SPYM": 0.6, "SGOV": 0.4}
        assert live_targets({"SPY": 0.0}, 1.0) == {"SGOV": 1.0}

    def test_sells_precede_buys_and_use_the_collar(self) -> None:
        account = Account(200.0, 200.0, 10_240.0, (Position("SGOV", 100, 100.0),))
        orders = plan_orders(account, {"SPYM": 0.7, "SGOV": 0.3}, QUOTES, "k")
        assert [o.side for o in orders] == ["sell", "buy"]
        sell, buy = orders
        assert (sell.symbol, sell.qty, sell.limit) == ("SGOV", 70, 99.89)
        assert buy.symbol == "SPYM" and buy.limit == 80.41
        assert buy.qty == int(0.25 * 10_240 // 80.41)

    def test_entry_from_cash_ramps_at_the_risk_cap(self) -> None:
        account = Account(10_000.0, 10_000.0, 10_000.0, ())
        (buy,) = plan_orders(account, {"SPYM": 1.0, "SGOV": 0.0}, QUOTES, "k")
        assert buy.qty * buy.limit <= 2_500
        assert buy.qty == 31

    def test_cash_leg_buys_are_not_ramped(self) -> None:
        account = Account(10_000.0, 10_000.0, 10_000.0, ())
        (buy,) = plan_orders(account, {"SGOV": 1.0}, QUOTES, "k")
        assert buy.qty * buy.limit > 9_000

    def test_holdings_inside_the_rounding_zone_are_left_alone(self) -> None:
        account = Account(0.0, 0.0, 8_000.0, (Position("SPYM", 100, 80.0),))
        assert plan_orders(account, {"SPYM": 1.0, "SGOV": 0.0}, QUOTES, "k") == ()

    def test_unlisted_holdings_are_sold_down(self) -> None:
        account = Account(0.0, 0.0, 8_000.0, (Position("SPYM", 100, 80.0),))
        (sell,) = plan_orders(account, {"SGOV": 1.0}, QUOTES, "k")[:1]
        assert (sell.symbol, sell.qty) == ("SPYM", 100)


class TestQuotes:
    def test_touch_is_synthesized_from_cost_table_when_absent(self) -> None:
        q = quote_from_payload(
            {"symbol": "SPYM", "regularMarketPrice": 80.0, "marketState": "POST"}, NOW
        )
        assert q.bid == pytest.approx(80 * (1 - 1.5e-4))
        assert q.ask == pytest.approx(80 * (1 + 1.5e-4))

    def test_feed_bid_ask_win_when_present(self) -> None:
        q = quote_from_payload(
            {"symbol": "SPYM", "regularMarketPrice": 80.0, "bid": 79.9, "ask": 80.1}, NOW
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
        assert first["positions"] == {"SGOV": 39, "SPYM": 31}
        assert all(o["status"] == "filled" for o in first["orders"])
        assert first["drift"] == pytest.approx(0.6 - 31 * 80.0 / first["equity"], abs=1e-4)
        second = run_cycle(store, log, "http://x", today=TODAY)
        assert second["status"] == "skipped"
        assert store.load().position("SPYM").qty == 31

    def test_halt_flag_holds_and_journals(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("execution.cycle.fetch_quotes", lambda *_: QUOTES)
        store = PaperStore(tmp_path / "paper")
        store.save(PaperState.initial(10_000.0))
        halt = tmp_path / "HALT"
        halt.touch()
        report = run_cycle(store, self._log(tmp_path, 1.0), "http://x", today=TODAY, halt_path=halt)
        assert report["status"] == "halted"
        assert [r["kind"] for r in store.journal_rows()] == ["halt"]
        assert store.load().orders == ()

    def test_guardrail_rejection_places_nothing_and_journals(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("execution.cycle.fetch_quotes", lambda *_: QUOTES)
        hostile = (OrderRequest("k:SPYM:buy", "SPYM", "buy", 120, 80.41),)
        monkeypatch.setattr("execution.cycle.plan_orders", lambda *_: hostile)
        store = PaperStore(tmp_path / "paper")
        store.save(PaperState.initial(10_000.0))
        report = run_cycle(store, self._log(tmp_path, 1.0), "http://x", today=TODAY)
        assert report["status"] == "rejected"
        assert "exceeds 0.25" in report["reason"]
        assert store.load().orders == ()
        assert [r["kind"] for r in store.journal_rows()] == ["rejection"]

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


class TestGuardrails:
    account = Account(2_000.0, 2_000.0, 10_000.0, (Position("SPYM", 100, 80.0),))

    def _ok(self, *requests: OrderRequest) -> tuple[OrderRequest, ...]:
        result = validate(self.account, requests, QUOTES)
        assert not isinstance(result, Rejection), result
        return result

    def _reject(self, *requests: OrderRequest, account: Account | None = None) -> str:
        result = validate(account or self.account, requests, QUOTES)
        assert isinstance(result, Rejection)
        return result.reason

    def test_valid_batch_passes_unchanged(self) -> None:
        batch = (
            OrderRequest("a", "SPYM", "sell", 20, 79.6),
            OrderRequest("b", "SGOV", "buy", 15, 100.91),
        )
        assert self._ok(*batch) == batch

    def test_unlisted_symbol(self) -> None:
        assert "not an allowed" in self._reject(OrderRequest("a", "SPY", "buy", 1, 600.0))

    def test_collar_breach(self) -> None:
        assert "collar" in self._reject(OrderRequest("a", "SPYM", "buy", 1, 81.0))

    def test_sell_over_holding_and_shorts(self) -> None:
        assert "exceeds holding" in self._reject(OrderRequest("a", "SPYM", "sell", 101, 79.6))
        assert "exceeds holding" in self._reject(OrderRequest("a", "SGOV", "sell", 1, 100.0))

    def test_unsettled_funded_position_cannot_be_sold(self) -> None:
        account = Account(0.0, -500.0, 10_000.0, (Position("SPYM", 100, 80.0, True),))
        assert "unsettled" in self._reject(OrderRequest("a", "SPYM", "sell", 1, 79.6), account=account)

    def test_risk_adding_cap_per_order_and_per_batch(self) -> None:
        assert "exceeds 0.25" in self._reject(OrderRequest("a", "SPYM", "buy", 32, 80.41))
        assert "exceeds 0.25" in self._reject(
            OrderRequest("a", "SPYM", "buy", 20, 80.41), OrderRequest("b", "SPYM", "buy", 20, 80.41)
        )

    def test_risk_adding_disabled_mode(self) -> None:
        result = validate(self.account, (OrderRequest("a", "SPYM", "buy", 1, 80.41),), QUOTES, allow_risk_adding=False)
        assert isinstance(result, Rejection) and "disabled" in result.reason
        sells = (OrderRequest("a", "SPYM", "sell", 50, 79.6),)
        assert validate(self.account, sells, QUOTES, allow_risk_adding=False) == sells

    def test_batch_cannot_overdraw_cash(self) -> None:
        assert "cash" in self._reject(OrderRequest("a", "SGOV", "buy", 20, 100.91))

    def test_too_many_orders(self) -> None:
        batch = tuple(OrderRequest(str(i), "SGOV", "buy", 1, 100.91) for i in range(11))
        assert "exceeds 10" in self._reject(*batch)

    def test_hostile_batches_never_produce_a_violating_approval(self) -> None:
        rng = random.Random(20260901)
        symbols = ["SPYM", "SGOV", "SPY", "TQQQ"]
        approvals = 0
        for _ in range(3000):
            held = tuple(
                Position(s, rng.randint(0, 120), 50.0, rng.random() < 0.1)
                for s in ("SPYM", "SGOV")
                if rng.random() < 0.8
            )
            cash = rng.uniform(0, 5_000)
            equity = cash + sum(p.qty * QUOTES[p.symbol].last for p in held)
            account = Account(cash, cash, equity, held)
            quotes = {**QUOTES, "SPY": _quote("SPY", 600.0), "TQQQ": _quote("TQQQ", 60.0)}
            batch = tuple(
                OrderRequest(
                    str(i),
                    rng.choice(symbols),
                    rng.choice(["buy", "sell"]),
                    rng.choice([-3, 0, 1, 5, 30, 200]),
                    round(rng.uniform(40, 700), 2),
                )
                for i in range(rng.randint(0, 12))
            )
            result = validate(account, batch, quotes)
            if isinstance(result, Rejection):
                continue
            approvals += 1
            _assert_invariants(account, result, quotes)
        assert approvals > 0


def _assert_invariants(account: Account, batch: tuple[OrderRequest, ...], quotes: dict[str, Quote]) -> None:
    held = {p.symbol: p.qty for p in account.positions}
    cash = account.cash
    risk_add = 0.0
    for r in batch:
        assert r.symbol in LIMITS["allowed_symbols"]
        assert r.qty > 0
        ref = quotes[r.symbol].ask if r.side == "buy" else quotes[r.symbol].bid
        assert abs(r.limit - ref) / ref <= LIMITS["max_collar"]
        if r.side == "buy":
            held[r.symbol] = held.get(r.symbol, 0) + r.qty
            cash -= r.qty * r.limit
            if r.symbol != LIMITS["cash_symbol"]:
                risk_add += r.qty * r.limit
        else:
            held[r.symbol] = held.get(r.symbol, 0) - r.qty
            cash += r.qty * r.limit
        assert held[r.symbol] >= 0
    assert cash >= 0
    assert risk_add <= LIMITS["max_risk_add"] * account.equity + 1e-9
    gross = sum(q * quotes[s].last for s, q in held.items())
    assert gross <= account.equity * (LIMITS["max_gross"] + LIMITS["gross_tolerance"]) + 1e-9


class TestNotify:
    def test_unset_webhook_is_a_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MERIDIAN_ALERT_WEBHOOK", raising=False)
        assert alert("t", {"a": 1}) is False

    def test_posts_row_with_title_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        received: list[tuple[str, bytes]] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                received.append((self.headers["X-Title"], body))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *_: object) -> None:
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        monkeypatch.setenv("MERIDIAN_ALERT_WEBHOOK", f"http://127.0.0.1:{server.server_port}/")
        assert alert("hello", {"kind": "cycle"}) is True
        thread.join()
        assert received == [("hello", b'{"kind": "cycle"}')]

    def test_failed_delivery_is_reported_not_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MERIDIAN_ALERT_WEBHOOK", "http://127.0.0.1:9/")
        assert alert("t", {}) is False


class TestWatchdog:
    def test_last_session_skips_the_in_progress_bar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def bar(day: str) -> dict:
            ts = int(datetime.fromisoformat(f"{day}T14:30:00+00:00").timestamp())
            return {"timestamp": ts, "close": 100.0}

        candles = {"candles": [bar("2026-09-02"), bar("2026-09-03"), bar("2026-09-04")]}
        monkeypatch.setattr("execution.watchdog.fetch_chart", lambda *_, **__: candles)
        assert last_session("http://x", today="2026-09-04") == "2026-09-03"
        assert last_session("http://x", today="2026-09-05") == "2026-09-04"
        assert last_session("http://x", today="2026-09-07") == "2026-09-04"

    def test_shadow_missing_last_session_is_a_finding(self, tmp_path: Path) -> None:
        log = tmp_path / "decisions.jsonl"
        append_decision(log, {"decision_date": "2026-08-31", "strategy": "null_voltarget"})
        assert shadow_findings(log, "2026-08-31", ("null_voltarget",)) == []
        findings = shadow_findings(log, "2026-09-01", ("null_voltarget", "other"))
        assert len(findings) == 2 and "2026-09-01" in findings[0]

    def test_paper_findings_follow_the_journal(self, tmp_path: Path) -> None:
        store = PaperStore(tmp_path / "paper")
        assert paper_findings(store, "2026-09-01") == []
        store.save(PaperState.initial(1.0))
        assert paper_findings(store, "2026-09-01") == ["paper: account open but no cycle ever logged"]
        store.journal({"kind": "cycle", "decision_date": "2026-08-31"})
        assert "not acted on" in paper_findings(store, "2026-09-01")[0]
        store.journal({"kind": "rejection", "decision_date": "2026-09-01", "reason": "collar"})
        assert paper_findings(store, "2026-09-01") == ["paper: rejection on 2026-09-01: collar"]
        store.journal({"kind": "cycle", "decision_date": "2026-09-01"})
        assert paper_findings(store, "2026-09-01") == []


class TestCommissionModel:
    def test_floor_binds_on_a_few_shares(self) -> None:
        from execution.commissions import tiered_commission

        assert tiered_commission(3, 90.0) == 0.35
        assert tiered_commission(200, 90.0) == pytest.approx(0.70)

    def test_capped_at_one_percent_of_value(self) -> None:
        from execution.commissions import tiered_commission

        assert tiered_commission(1, 1.0) == 0.01


class TestBook:
    def test_book_cash_follows_its_own_fills_only(self) -> None:
        from execution.book import book_account

        rows: list[dict] = [
            {"kind": "fill", "order": {"id": "a", "side": "buy", "qty": 10, "fill_price": 80.0, "commission": 0.35}},
            {"kind": "fill", "order": {"id": "a", "side": "buy", "qty": 10, "fill_price": 80.0, "commission": 0.35}},
            {"kind": "fill", "order": {"id": "b", "side": "sell", "qty": 2, "fill_price": 81.0, "commission": 0.35}},
            {"kind": "cycle", "orders": []},
        ]
        account = Account(1_000_000.0, 1_000_000.0, 1_000_640.0, (Position("SPYM", 8, 80.0),))
        book = book_account(account, rows, 10_000.0)
        assert book.cash == pytest.approx(10_000 - 800.35 + 161.65)
        assert book.equity == pytest.approx(book.cash + 640.0)
        assert book.positions == account.positions

    def test_book_never_exceeds_the_account(self) -> None:
        from execution.book import book_account

        account = Account(5_000.0, 4_000.0, 5_000.0, ())
        book = book_account(account, [], 10_000.0)
        assert (book.cash, book.settled_cash, book.equity) == (5_000.0, 4_000.0, 5_000.0)
        assert book_account(account, [], None) is account
