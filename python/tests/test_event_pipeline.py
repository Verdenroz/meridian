"""Invariant tests for the 8-K pipeline: extraction, versioning, store, scorer."""

import json

import pytest
from event_pipeline.filings import document_url, html_to_text
from event_pipeline.scorer import PROMPT, PROMPT_VERSION, EventRiskScore, EventScorer
from event_pipeline.store import append_score, load_scored_keys, load_scores


class TestExtraction:
    def test_strips_tags_scripts_and_collapses_whitespace(self):
        html = (
            "<html><head><title>ignore</title><script>var x=1;</script></head>"
            "<body><div>Item 5.02</div>\n\n<p>Departure of  Directors</p></body></html>"
        )
        assert html_to_text(html) == "Item 5.02 Departure of Directors"

    def test_document_url_strips_accession_dashes(self):
        url = document_url("320193", "0000320193-26-000018", "aapl-20260730.htm")
        assert url == (
            "https://www.sec.gov/Archives/edgar/data/320193/"
            "000032019326000018/aapl-20260730.htm"
        )


class TestVersioning:
    def test_prompt_version_tracks_prompt_text(self):
        import hashlib

        assert PROMPT_VERSION == hashlib.sha256(PROMPT.encode()).hexdigest()[:12]

    def test_scorer_version_binds_model_and_prompt(self):
        scorer = EventScorer(model="some-model", client=object())
        assert scorer.scorer_version == f"some-model@{PROMPT_VERSION}"


class TestStore:
    def test_append_and_reload_keys(self, tmp_path):
        path = tmp_path / "scores.jsonl"
        append_score(path, "AAPL", "acc-1", "2026-07-30", "m@v1", {"severity": 3})
        append_score(path, "AAPL", "acc-1", "2026-07-30", "m@v2", {"severity": 4})
        keys = load_scored_keys(path)
        assert keys == {("acc-1", "m@v1"), ("acc-1", "m@v2")}

    def test_load_scores_filters_by_version(self, tmp_path):
        path = tmp_path / "scores.jsonl"
        append_score(path, "AAPL", "acc-1", "2026-07-30", "m@v1", {"severity": 3})
        append_score(path, "MSFT", "acc-2", "2026-07-29", "m@v2", {"severity": 7})
        v2 = load_scores(path, "m@v2")
        assert len(v2) == 1 and v2[0]["symbol"] == "MSFT"
        assert len(load_scores(path)) == 2

    def test_rows_carry_metadata_and_score_fields(self, tmp_path):
        path = tmp_path / "scores.jsonl"
        row = append_score(
            path, "KO", "acc-9", "2026-01-02", "m@v1",
            {"severity": 5, "scope": "asset", "direction": "risk_up", "rationale": "x"},
        )
        assert row["scored_at"]
        stored = json.loads(path.read_text())
        assert stored["direction"] == "risk_up"
        assert stored["filing_date"] == "2026-01-02"


class FakeResponse:
    def __init__(self, parsed):
        self.parsed = parsed


class FakeClient:
    def __init__(self, parsed):
        self._parsed = parsed
        self.calls = []

    def generate(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return FakeResponse(self._parsed)


class TestScorer:
    def test_score_returns_parsed_model_and_formats_prompt(self):
        expected = EventRiskScore(
            severity=6, scope="asset", direction="risk_up", rationale="CFO resigned"
        )
        client = FakeClient(expected)
        scorer = EventScorer(client=client)
        result = scorer.score("AAPL", "2026-07-30", "CFO resigned effective today")
        assert result == expected
        call = client.calls[0]
        assert "SYMBOL: AAPL" in call["messages"]
        assert "CFO resigned effective today" in call["messages"]
        assert call["response_model"] is EventRiskScore
        assert call["temperature"] == 0

    def test_unparsed_response_raises(self):
        scorer = EventScorer(client=FakeClient(parsed=None))
        with pytest.raises(TypeError, match="unparsed"):
            scorer.score("AAPL", "2026-07-30", "text")

    def test_schema_bounds(self):
        with pytest.raises(ValueError):
            EventRiskScore(severity=11, scope="asset", direction="neutral", rationale="")
        with pytest.raises(ValueError):
            EventRiskScore(severity=3, scope="galaxy", direction="neutral", rationale="")
