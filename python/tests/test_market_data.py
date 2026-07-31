"""Invariant tests for market_data: adjClose cross-check and store behavior.

These guard the places where bugs silently poison data: the total-return
reconstruction, the loud-failure paths, and snapshot integrity.
"""

import pytest
from market_data.crosscheck import (
    CrossCheckError,
    dividends_by_date,
    reconstruct_adj_close,
    verify_adj_close,
)
from market_data.macro import normalize_treasury
from market_data.panel import load_total_return_panel
from market_data.series import normalize_candles
from market_data.store import list_snapshots, read_snapshot, write_snapshot

# 2024-01-02 .. 2024-01-05 as market-open timestamps (14:30 UTC)
TS = [1704205800, 1704292200, 1704378600, 1704465000]


def make_rows(closes, adj=None):
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    return [
        {
            "date": d,
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "adj_close": (adj[i] if adj else c),
            "volume": 100,
        }
        for i, (d, c) in enumerate(zip(dates, closes))
    ]


class TestReconstruction:
    def test_no_dividends_adj_equals_close(self):
        rows = make_rows([100.0, 101.0, 102.0, 103.0])
        assert reconstruct_adj_close(rows, {}) == [100.0, 101.0, 102.0, 103.0]

    def test_single_dividend_scales_earlier_closes(self):
        rows = make_rows([100.0, 101.0, 102.0, 103.0])
        # $1 dividend ex 2024-01-04: days before scale by (1 - 1/101)
        recon = reconstruct_adj_close(rows, {"2024-01-04": 1.0})
        factor = 1.0 - 1.0 / 101.0
        assert recon[0] == pytest.approx(100.0 * factor)
        assert recon[1] == pytest.approx(101.0 * factor)
        assert recon[2] == 102.0
        assert recon[3] == 103.0

    def test_verify_passes_on_consistent_data(self):
        closes = [100.0, 101.0, 102.0, 103.0]
        factor = 1.0 - 2.0 / 101.0
        adj = [closes[0] * factor, closes[1] * factor, closes[2], closes[3]]
        rows = make_rows(closes, adj)
        dividends = [{"timestamp": TS[2], "amount": 2.0}]
        report = verify_adj_close("TEST", rows, dividends)
        assert report["adjustment_events"] == 1
        assert report["mismatches"] == []

    def test_verify_fails_when_all_events_mismatch(self):
        rows = make_rows([100.0, 101.0, 102.0, 103.0])
        # Upstream claims adj == close but a real dividend exists → the only
        # adjustment event is inconsistent → systematic failure.
        dividends = [{"timestamp": TS[2], "amount": 2.0}]
        with pytest.raises(CrossCheckError, match="inconsistent"):
            verify_adj_close("TEST", rows, dividends)

    def test_verify_fails_loudly_on_missing_adj_close(self):
        rows = make_rows([100.0, 101.0, 102.0, 103.0])
        rows[1]["adj_close"] = None
        with pytest.raises(CrossCheckError, match="missing"):
            verify_adj_close("TEST", rows, [])

    def test_implausible_adjustment_step_is_corruption(self):
        closes = [100.0, 101.0, 102.0, 103.0]
        # Upstream back-adjusts early days by 50% with no matching dividend.
        adj = [closes[0] * 0.5, closes[1] * 0.5, closes[2], closes[3]]
        with pytest.raises(CrossCheckError, match="corruption"):
            verify_adj_close("TEST", make_rows(closes, adj), [])

    def test_isolated_endpoint_gap_is_tolerated_and_recorded(self):
        # 30 dividends correctly applied upstream; endpoint lost one of them
        # (the KO 2001-09-10 scenario). Verification passes, gap is recorded.
        n = 62
        base_ts = 1704119400  # 2024-01-01 14:30 UTC, daily steps
        stamps = [base_ts + i * 86400 for i in range(n)]
        closes = [100.0 + i for i in range(n)]
        div_days = {i: 0.5 for i in range(2, n, 2)}
        factor = 1.0
        factors = [1.0] * n
        for i in range(n - 1, 0, -1):
            if i in div_days:
                factor *= 1.0 - div_days[i] / closes[i - 1]
            factors[i - 1] = factor
        from market_data.series import utc_date

        rows = [
            {
                "date": utc_date(stamps[i]),
                "close": closes[i],
                "adj_close": closes[i] * factors[i],
            }
            for i in range(n)
        ]
        dropped_day = 10
        dividends = [
            {"timestamp": stamps[i], "amount": amt}
            for i, amt in div_days.items()
            if i != dropped_day
        ]
        report = verify_adj_close("TEST", rows, dividends)
        assert len(report["mismatches"]) == 1
        assert report["mismatches"][0]["date"] == utc_date(stamps[dropped_day])
        assert report["mismatches"][0]["endpoint_dividend"] == 0.0

    def test_dividend_exceeding_prev_close_is_corrupt(self):
        rows = make_rows([100.0, 101.0, 102.0, 103.0])
        dividends = [{"timestamp": TS[2], "amount": 500.0}]
        with pytest.raises(CrossCheckError):
            verify_adj_close("TEST", rows, dividends)

    def test_same_day_dividends_sum(self):
        divs = [
            {"timestamp": TS[2], "amount": 1.0},
            {"timestamp": TS[2] + 60, "amount": 0.5},
        ]
        assert dividends_by_date(divs) == {"2024-01-04": 1.5}

    def test_exact_duplicate_rows_are_data_bugs_not_payouts(self):
        # Upstream duplication observed on MSFT 2026-02-19: same ts, same amount.
        divs = [
            {"timestamp": TS[2], "amount": 0.91},
            {"timestamp": TS[2], "amount": 0.91},
        ]
        assert dividends_by_date(divs) == {"2024-01-04": 0.91}


class TestNormalization:
    def test_drops_null_close_and_counts_missing_adj(self):
        candles = [
            {"timestamp": TS[0], "close": 100.0, "adjClose": 99.0},
            {"timestamp": TS[1], "close": None, "adjClose": None},
            {"timestamp": TS[2], "close": 102.0, "adjClose": None},
        ]
        rows, issues = normalize_candles(candles)
        assert [r["date"] for r in rows] == ["2024-01-02", "2024-01-04"]
        assert issues["dropped_no_close"] == 1
        assert issues["missing_adj_close"] == 1
        assert rows[1]["adj_close"] is None

    def test_duplicate_dates_collapse_to_one_row(self):
        candles = [
            {"timestamp": TS[0], "close": 100.0, "adjClose": 100.0},
            {"timestamp": TS[0] + 3600, "close": 101.0, "adjClose": 101.0},
        ]
        rows, issues = normalize_candles(candles)
        assert len(rows) == 1
        assert rows[0]["close"] == 101.0
        assert issues["duplicate_dates"] == 1


class TestStore:
    def test_roundtrip_and_fetched_at(self, tmp_path):
        write_snapshot(tmp_path, "charts", "SPY", {"rows": [{"date": "2024-01-02"}]})
        doc = read_snapshot(tmp_path, "charts", "SPY")
        assert doc is not None
        assert doc["data"]["rows"][0]["date"] == "2024-01-02"
        assert "fetched_at" in doc

    def test_no_tmp_file_left_behind(self, tmp_path):
        write_snapshot(tmp_path, "charts", "SPY", {"rows": []})
        assert list((tmp_path / "charts").glob("*.tmp")) == []

    def test_overwrite_replaces_cleanly(self, tmp_path):
        write_snapshot(tmp_path, "charts", "SPY", {"rows": [1]})
        write_snapshot(tmp_path, "charts", "SPY", {"rows": [1, 2]})
        doc = read_snapshot(tmp_path, "charts", "SPY")
        assert doc is not None and doc["data"]["rows"] == [1, 2]

    def test_list_reports_unreadable_snapshot(self, tmp_path):
        (tmp_path / "charts").mkdir(parents=True)
        (tmp_path / "charts" / "BAD.json").write_text("{not json")
        write_snapshot(tmp_path, "charts", "SPY", {"rows": []})
        snaps = {s["name"]: s for s in list_snapshots(tmp_path)}
        assert "error" in snaps["BAD"]
        assert "fetched_at" in snaps["SPY"]


class TestPanel:
    def test_union_alignment_with_different_inceptions(self, tmp_path):
        write_snapshot(
            tmp_path,
            "charts",
            "AAA",
            {"rows": [
                {"date": "2024-01-02", "adj_close": 10.0},
                {"date": "2024-01-03", "adj_close": 11.0},
            ]},
        )
        write_snapshot(
            tmp_path,
            "charts",
            "BBB",
            {"rows": [{"date": "2024-01-03", "adj_close": 20.0}]},
        )
        panel = load_total_return_panel(tmp_path, ("AAA", "BBB"))
        assert panel["dates"] == ["2024-01-02", "2024-01-03"]
        assert panel["adj_close"]["AAA"] == [10.0, 11.0]
        assert panel["adj_close"]["BBB"] == [None, 20.0]

    def test_missing_symbol_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_total_return_panel(tmp_path, ("NOPE",))


class TestMacro:
    def test_treasury_dates_converted_and_sorted(self):
        rows = normalize_treasury(
            [
                {"date": "07/30/2026", "y10": 4.52},
                {"date": "01/02/1990", "y10": 7.94},
            ]
        )
        assert [r["date"] for r in rows] == ["1990-01-02", "2026-07-30"]
        assert rows[1]["y10"] == 4.52
