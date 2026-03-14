import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from meridian_train import cli
from meridian_train.phase1_core import (
    run_phase1_backbone_smoke_check,
    run_phase1_smoke_check,
)


class Phase1CoreTest(unittest.TestCase):
    def test_phase1_smoke_raises_when_torch_missing(self) -> None:
        with patch("meridian_train.phase1_core.TORCH_AVAILABLE", False):
            with self.assertRaises(RuntimeError):
                run_phase1_smoke_check()

    def test_phase1_backbone_smoke_raises_when_torch_missing(self) -> None:
        with patch("meridian_train.phase1_core.TORCH_AVAILABLE", False):
            with self.assertRaises(RuntimeError):
                run_phase1_backbone_smoke_check()

    def test_cli_phase1_smoke_wiring(self) -> None:
        captured = io.StringIO()
        with patch("meridian_train.cli.run_phase1_smoke_check") as mocked:
            mocked.return_value = {
                "block_output_shape": [2, 16, 768],
                "state_shape": [2, 4, 64, 192],
                "alpha_min": 0.1,
                "alpha_max": 0.9,
                "beta_min": 0.1,
                "beta_max": 0.9,
            }
            with redirect_stdout(captured):
                exit_code = cli.main(
                    argv=["meridian", "phase1-smoke", "--batch-size", "2", "--seq-len", "16"]
                )
        self.assertEqual(exit_code, 0)
        mocked.assert_called_once()
        self.assertIn("block_output_shape", captured.getvalue())

    def test_cli_phase1_backbone_smoke_wiring(self) -> None:
        captured = io.StringIO()
        with patch("meridian_train.cli.run_phase1_backbone_smoke_check") as mocked:
            mocked.return_value = {
                "backbone_output_shape": [2, 16, 768],
                "num_blocks": 8,
                "num_moe_insertions": 3,
                "moe_usage": {"2": [0.5, 0.5, 0.0, 0.0]},
            }
            with redirect_stdout(captured):
                exit_code = cli.main(
                    argv=[
                        "meridian",
                        "phase1-backbone-smoke",
                        "--batch-size",
                        "2",
                        "--seq-len",
                        "16",
                    ]
                )
        self.assertEqual(exit_code, 0)
        mocked.assert_called_once()
        self.assertIn("backbone_output_shape", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
