import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from meridian_train import cli
from meridian_train.phase1_training import (
    Phase1TrainingConfig,
    build_phase1_training_spec,
    run_phase1_training_smoke,
)


class Phase1TrainingTest(unittest.TestCase):
    def test_phase1_training_spec_contains_expected_fields(self) -> None:
        config = Phase1TrainingConfig()
        spec = build_phase1_training_spec(
            config=config,
            checkpoint_output_path=Path("checkpoints/phase1_backbone.pt"),
            metrics_output_path=Path("reports/phase1_training_metrics.json"),
        )
        self.assertEqual(spec["steps"], config.steps)
        self.assertEqual(spec["batch_size"], config.batch_size)
        self.assertEqual(spec["seq_len"], config.seq_len)
        self.assertEqual(spec["objective"], "self_supervised_backbone_scaffold")

    def test_phase1_train_smoke_raises_when_torch_missing(self) -> None:
        with patch("meridian_train.phase1_training.TORCH_AVAILABLE", False):
            with self.assertRaises(RuntimeError):
                run_phase1_training_smoke(
                    config=Phase1TrainingConfig(),
                    checkpoint_output_path=Path("checkpoints/phase1_backbone.pt"),
                )

    def test_cli_phase1_train_smoke_wiring(self) -> None:
        captured = io.StringIO()
        with patch("meridian_train.cli.run_phase1_training_smoke") as mocked:
            mocked.return_value = {
                "steps": 2,
                "device": "cpu",
                "avg_total_loss": 0.123,
                "checkpoint_output_path": "checkpoints/phase1_backbone.pt",
                "metrics_output_path": "reports/phase1_training_metrics.json",
            }
            with redirect_stdout(captured):
                exit_code = cli.main(
                    argv=[
                        "meridian",
                        "phase1-train-smoke",
                        "--steps",
                        "2",
                        "--batch-size",
                        "2",
                        "--seq-len",
                        "8",
                    ]
                )
        self.assertEqual(exit_code, 0)
        mocked.assert_called_once()
        self.assertIn("avg_total_loss", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
