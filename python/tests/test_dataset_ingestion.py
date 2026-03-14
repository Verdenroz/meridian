import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from meridian_train import cli
from meridian_train.dataset_ingestion import ingest_feature_jsonl


def _sample_row(timestamp: str) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "asset_class": "equity",
        "price_volume": [0.0] * 10,
        "technical": [0.1] * 20,
        "options": [0.2] * 5,
        "fundamental": [0.3] * 10,
        "text": [0.4] * 64,
        "asset_calendar": [1.0, 0.5, 0.0],
    }


class DatasetIngestionTest(unittest.TestCase):
    def test_ingest_jsonl_materializes_split_shards(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "features.jsonl"
            rows = [
                _sample_row("2024-06-30T10:00:00"),
                _sample_row("2024-07-15T10:00:00"),
                _sample_row("2025-02-15T10:00:00"),
            ]
            input_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            manifest = ingest_feature_jsonl(
                input_path=input_path,
                asset_symbol="AAPL",
                output_root=root,
            )
            self.assertEqual(manifest["feature_dim"], 112)
            self.assertEqual(manifest["split_counts"]["train"], 1)
            self.assertEqual(manifest["split_counts"]["validation"], 1)
            self.assertEqual(manifest["split_counts"]["test"], 1)
            self.assertTrue((root / "data/mmaps/train/AAPL.f32bin").exists())

    def test_ingest_jsonl_accepts_unix_timestamp(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "features_unix.jsonl"
            row = _sample_row("2024-06-30T10:00:00")
            row["timestamp"] = 1719741600
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest = ingest_feature_jsonl(
                input_path=input_path,
                asset_symbol="AAPL",
                output_root=root,
            )
            self.assertEqual(manifest["split_counts"]["train"], 1)

    def test_ingest_jsonl_rejects_bad_dimensions(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "bad_features.jsonl"
            row = _sample_row("2024-06-30T10:00:00")
            row["technical"] = [0.1] * 19
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                ingest_feature_jsonl(
                    input_path=input_path,
                    asset_symbol="AAPL",
                    output_root=root,
                )

    def test_cli_ingest_jsonl_command(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "features.jsonl"
            input_path.write_text(
                json.dumps(_sample_row("2024-06-30T10:00:00")) + "\n",
                encoding="utf-8",
            )
            output_capture = io.StringIO()
            with redirect_stdout(output_capture):
                exit_code = cli.main(
                    argv=[
                        "meridian",
                        "ingest-jsonl",
                        "--input",
                        str(input_path),
                        "--asset",
                        "AAPL",
                        "--output-root",
                        str(root),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertIn("manifest", output_capture.getvalue())

    def test_cli_collect_ingest_command(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mocked_rows = [_sample_row("2024-06-30T10:00:00")]
            output_capture = io.StringIO()
            with patch("meridian_train.cli.collect_feature_rows", return_value=mocked_rows):
                with redirect_stdout(output_capture):
                    exit_code = cli.main(
                        argv=[
                            "meridian",
                            "collect-ingest",
                            "--asset",
                            "AAPL",
                            "--start",
                            "2024-01-01",
                        "--end",
                        "2024-12-31",
                        "--output-root",
                        str(root),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertTrue((root / "data/mmaps/train/AAPL.f32bin").exists())


if __name__ == "__main__":
    unittest.main()
