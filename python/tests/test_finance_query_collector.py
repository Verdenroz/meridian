import sys
import unittest
from datetime import date

from meridian_train.finance_query_collector import (
    FinanceQueryCollectorConfig,
    build_collect_command,
    collect_feature_rows,
    default_rust_collector_command_template,
)


class FinanceQueryCollectorTest(unittest.TestCase):
    def test_default_command_template_targets_rust_collector(self) -> None:
        template = default_rust_collector_command_template()
        self.assertEqual(template[:3], ("cargo", "run", "--quiet"))
        self.assertIn("--symbol", template)
        self.assertIn("--start", template)
        self.assertIn("--end", template)

    def test_build_collect_command_substitutes_placeholders(self) -> None:
        command = build_collect_command(
            symbol="AAPL",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            command_template=("tool", "--symbol", "{symbol}", "--start", "{start}", "--end", "{end}"),
        )
        self.assertEqual(
            command,
            ["tool", "--symbol", "AAPL", "--start", "2024-01-01", "--end", "2024-12-31"],
        )

    def test_collect_feature_rows_parses_jsonl(self) -> None:
        script = (
            "import json; "
            "row={'timestamp':'2024-06-30T10:00:00','asset_class':'equity',"
            "'price_volume':[0.0]*10,'technical':[0.0]*20,'options':[0.0]*5,"
            "'fundamental':[0.0]*10,'text':[0.0]*64,'asset_calendar':[0.0]*3}; "
            "print(json.dumps(row))"
        )
        config = FinanceQueryCollectorConfig(
            command_template=(sys.executable, "-c", script),
            timeout_seconds=30,
        )
        rows = collect_feature_rows(
            symbol="AAPL",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            config=config,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["timestamp"], "2024-06-30T10:00:00")

    def test_collect_feature_rows_fails_when_executable_missing(self) -> None:
        config = FinanceQueryCollectorConfig(
            command_template=("missing_finance_query_binary_12345",),
            timeout_seconds=5,
        )
        with self.assertRaises(RuntimeError):
            collect_feature_rows(
                symbol="AAPL",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
                config=config,
            )


if __name__ == "__main__":
    unittest.main()
