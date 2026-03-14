import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from meridian_train import cli
from meridian_train.marketreader import load_corpus_texts


class MarketReaderTrainingTest(unittest.TestCase):
    def test_load_corpus_texts_supports_json_fields_and_raw_text(self) -> None:
        with TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "news.jsonl"
            rows = [
                {"title": "Fed decision", "summary": "Rates unchanged this meeting"},
                {"text": "Tech earnings surprise to upside"},
                "Plain text line with macro signal",
            ]
            corpus_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            texts = load_corpus_texts(corpus_path)
            self.assertEqual(len(texts), 3)
            self.assertIn("Fed decision", texts[0])

    def test_cli_marketreader_train_wiring(self) -> None:
        with TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "news.jsonl"
            corpus_path.write_text(json.dumps({"text": "sample"}) + "\n", encoding="utf-8")
            output_capture = io.StringIO()
            with patch("meridian_train.cli.run_marketreader_training") as mocked:
                mocked.return_value = {
                    "documents_used": 1,
                    "model_output_path": str(Path(temp_dir) / "model.pt"),
                    "tokenizer_output_path": str(Path(temp_dir) / "tokenizer.json"),
                }
                with redirect_stdout(output_capture):
                    exit_code = cli.main(
                        argv=[
                            "meridian",
                            "marketreader-train",
                            "--corpus",
                            str(corpus_path),
                            "--model-output",
                            str(Path(temp_dir) / "model.pt"),
                            "--tokenizer-output",
                            str(Path(temp_dir) / "tokenizer.json"),
                            "--epochs",
                            "1",
                        ]
                    )
            self.assertEqual(exit_code, 0)
            mocked.assert_called_once()
            self.assertIn("documents_used", output_capture.getvalue())


if __name__ == "__main__":
    unittest.main()

