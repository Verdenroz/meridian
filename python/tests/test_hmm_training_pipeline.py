import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cli
from market_data.regime_hmm import (
    RegimeHMMConfig,
    load_regime_feature_rows,
    run_hmm_training_pipeline,
)


class DummyHMM:
    def __init__(self) -> None:
        self.transmat_ = [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ]
        self.startprob_ = [0.25, 0.25, 0.25, 0.25]
        self.means_ = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
        self.covars_ = [[1.0, 1.0, 1.0] for _ in range(4)]

    def predict(self, observations):
        return [index % 4 for index, _ in enumerate(observations)]


class HMMTrainingPipelineTest(unittest.TestCase):
    def test_load_regime_feature_rows_from_json_objects(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "features.jsonl"
            rows = [
                {"vix_level": 20.0, "adx14": 12.0, "corr_dispersion": 0.2},
                {"vix_level": 25.0, "adx14": 18.0, "corr_dispersion": 0.4},
            ]
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            loaded = load_regime_feature_rows(
                input_path=input_path,
                feature_names=["vix_level", "adx14", "corr_dispersion"],
            )
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0], [20.0, 12.0, 0.2])

    def test_run_hmm_pipeline_writes_model_and_meta(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "features.jsonl"
            rows = [
                {"vix_level": 20.0, "adx14": 12.0, "corr_dispersion": 0.2},
                {"vix_level": 25.0, "adx14": 18.0, "corr_dispersion": 0.4},
                {"vix_level": 30.0, "adx14": 26.0, "corr_dispersion": 0.6},
                {"vix_level": 35.0, "adx14": 30.0, "corr_dispersion": 0.7},
            ]
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            model_output = root / "checkpoints" / "regime_hmm.pkl"
            with patch("market_data.regime_hmm.fit_regime_hmm", return_value=DummyHMM()):
                summary = run_hmm_training_pipeline(
                    config=RegimeHMMConfig(),
                    input_path=input_path,
                    feature_names=["vix_level", "adx14", "corr_dispersion"],
                    model_output_path=model_output,
                )
            self.assertTrue(model_output.exists())
            self.assertTrue(Path(summary["meta_output_path"]).exists())
            self.assertEqual(summary["rows_used"], 4)

    def test_cli_hmm_train_wiring(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "features.jsonl"
            input_path.write_text(
                json.dumps({"vix_level": 20.0, "adx14": 12.0, "corr_dispersion": 0.2}) + "\n",
                encoding="utf-8",
            )
            captured = io.StringIO()
            with patch("cli.run_hmm_training_pipeline") as mocked:
                mocked.return_value = {"rows_used": 1, "meta_output_path": str(Path(temp_dir) / "meta.json")}
                with redirect_stdout(captured):
                    exit_code = cli.main(
                        argv=[
                            "meridian",
                            "hmm-train",
                            "--input",
                            str(input_path),
                            "--features",
                            "vix_level,adx14,corr_dispersion",
                            "--model-output",
                            str(Path(temp_dir) / "model.pkl"),
                        ]
                    )
            self.assertEqual(exit_code, 0)
            mocked.assert_called_once()
            self.assertIn("rows_used", captured.getvalue())


if __name__ == "__main__":
    unittest.main()

