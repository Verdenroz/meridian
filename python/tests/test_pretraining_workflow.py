import io
import json
import struct
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cli
from core.self_supervised_pretraining import (
    BackboneTrainingConfig,
    TORCH_AVAILABLE,
    build_backbone_pretraining_spec,
    build_multi_shard_sequence_windows,
    build_shard_sequence_windows,
    discover_split_shards,
    load_f32_feature_shard_rows,
    resolve_objective_weights,
    run_backbone_pretraining_smoke,
    run_split_shard_evaluation,
    validate_checkpoint_compatibility,
)


class PretrainingWorkflowTest(unittest.TestCase):
    def test_training_spec_contains_expected_fields(self) -> None:
        config = BackboneTrainingConfig()
        spec = build_backbone_pretraining_spec(
            config=config,
            checkpoint_output_path=Path("checkpoints/backbone_pretrain.pt"),
            metrics_output_path=Path("reports/backbone_training_metrics.json"),
        )
        self.assertEqual(spec["steps"], config.steps)
        self.assertEqual(spec["batch_size"], config.batch_size)
        self.assertEqual(spec["seq_len"], config.seq_len)
        self.assertEqual(spec["objective"], "self_supervised_backbone_scaffold")
        self.assertEqual(spec["return_prediction_weight"], 0.5)
        self.assertEqual(spec["volatility_prediction_weight"], 0.3)
        self.assertEqual(spec["correlation_prediction_weight"], 0.2)

    def test_training_smoke_raises_when_torch_missing(self) -> None:
        with patch("core.self_supervised_pretraining.TORCH_AVAILABLE", False):
            with self.assertRaises(RuntimeError):
                run_backbone_pretraining_smoke(
                    config=BackboneTrainingConfig(),
                    checkpoint_output_path=Path("checkpoints/backbone_pretrain.pt"),
                )

    def test_prediction_objective_weights_ramp_linearly(self) -> None:
        config = BackboneTrainingConfig(
            return_prediction_weight=0.5,
            volatility_prediction_weight=0.3,
            correlation_prediction_weight=0.2,
            prediction_ramp_steps=4,
        )
        step_0 = resolve_objective_weights(config, step_index=0)
        step_1 = resolve_objective_weights(config, step_index=1)
        step_3 = resolve_objective_weights(config, step_index=3)
        step_8 = resolve_objective_weights(config, step_index=8)

        self.assertAlmostEqual(step_0["return_prediction_weight"], 0.125)
        self.assertAlmostEqual(step_1["volatility_prediction_weight"], 0.15)
        self.assertAlmostEqual(step_3["correlation_prediction_weight"], 0.2)
        self.assertAlmostEqual(step_8["return_prediction_weight"], 0.5)
        self.assertEqual(step_0["reconstruction_weight"], config.reconstruction_weight)

    def test_load_f32_feature_shard_rows_and_windowing(self) -> None:
        row_struct = struct.Struct("<112f")
        with TemporaryDirectory() as temp_dir:
            shard_path = Path(temp_dir) / "AAPL.f32bin"
            rows = [[float(i + j) for j in range(112)] for i in range(8)]
            with shard_path.open("wb") as handle:
                for row in rows:
                    handle.write(row_struct.pack(*row))

            loaded_rows = load_f32_feature_shard_rows(shard_path=shard_path)
            self.assertEqual(len(loaded_rows), 8)
            self.assertEqual(len(loaded_rows[0]), 112)

            windows = build_shard_sequence_windows(loaded_rows, seq_len=4, max_sequences=2)
            self.assertEqual(len(windows), 2)
            self.assertEqual(len(windows[0]), 4)
            self.assertEqual(len(windows[0][0]), 112)

    def test_load_f32_feature_shard_rows_rejects_bad_size(self) -> None:
        with TemporaryDirectory() as temp_dir:
            shard_path = Path(temp_dir) / "bad.f32bin"
            shard_path.write_bytes(b"\x00\x01\x02")
            with self.assertRaises(ValueError):
                load_f32_feature_shard_rows(shard_path=shard_path)

    def test_build_multi_shard_sequence_windows_combines_shards(self) -> None:
        row_struct = struct.Struct("<112f")
        with TemporaryDirectory() as temp_dir:
            shard_a = Path(temp_dir) / "AAPL.f32bin"
            shard_b = Path(temp_dir) / "SPY.f32bin"
            rows_a = [[float(i + j) for j in range(112)] for i in range(8)]
            rows_b = [[float(100 + i + j) for j in range(112)] for i in range(8)]
            with shard_a.open("wb") as handle:
                for row in rows_a:
                    handle.write(row_struct.pack(*row))
            with shard_b.open("wb") as handle:
                for row in rows_b:
                    handle.write(row_struct.pack(*row))

            windows = build_multi_shard_sequence_windows(
                shard_paths=[shard_a, shard_b],
                seq_len=4,
                max_sequences_per_shard=1,
            )
            self.assertEqual(len(windows), 2)
            self.assertEqual(len(windows[0]), 4)
            self.assertEqual(len(windows[0][0]), 112)

    def test_discover_split_shards_groups_by_split(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "mmaps"
            (root / "train").mkdir(parents=True, exist_ok=True)
            (root / "validation").mkdir(parents=True, exist_ok=True)
            (root / "test").mkdir(parents=True, exist_ok=True)
            (root / "train" / "AAPL.f32bin").write_bytes(b"")
            (root / "validation" / "SPY.f32bin").write_bytes(b"")

            discovered = discover_split_shards(
                shards_root=root,
                splits=["train", "validation", "test"],
            )
            self.assertEqual(len(discovered["train"]), 1)
            self.assertEqual(len(discovered["validation"]), 1)
            self.assertEqual(len(discovered["test"]), 0)

    def test_validate_checkpoint_compatibility_detects_core_mismatch(self) -> None:
        payload = {
            "config": {
                "core": {
                    "model_dim": 512,
                    "regime_input_dim": 91,
                    "regime_dim": 64,
                    "n_regimes": 4,
                    "n_heads": 4,
                    "d_head": 64,
                    "d_v": 192,
                    "conv_kernel": 4,
                }
            },
            "backbone_state_dict": {},
            "reconstruction_head_state_dict": {},
            "return_head_state_dict": {},
            "volatility_head_state_dict": {},
            "correlation_head_state_dict": {},
        }
        report = validate_checkpoint_compatibility(
            payload,
            current_config=BackboneTrainingConfig(),
            requires_input_projection=False,
        )
        self.assertFalse(report["compatible"])
        self.assertTrue(any("core.model_dim" in reason for reason in report["errors"]))

    def test_validate_checkpoint_compatibility_warns_about_missing_projection(self) -> None:
        payload = {
            "config": {
                "core": {
                    "model_dim": 768,
                    "regime_input_dim": 91,
                    "regime_dim": 64,
                    "n_regimes": 4,
                    "n_heads": 4,
                    "d_head": 64,
                    "d_v": 192,
                    "conv_kernel": 4,
                }
            },
            "backbone_state_dict": {},
            "reconstruction_head_state_dict": {},
            "return_head_state_dict": {},
            "volatility_head_state_dict": {},
            "correlation_head_state_dict": {},
        }
        report = validate_checkpoint_compatibility(
            payload,
            current_config=BackboneTrainingConfig(),
            requires_input_projection=True,
        )
        self.assertTrue(report["compatible"])
        self.assertTrue(any("input projection" in warning for warning in report["warnings"]))

    def test_validate_checkpoint_compatibility_warns_when_format_version_missing(self) -> None:
        payload = {
            "config": {
                "core": {
                    "model_dim": 768,
                    "regime_input_dim": 91,
                    "regime_dim": 64,
                    "n_regimes": 4,
                    "n_heads": 4,
                    "d_head": 64,
                    "d_v": 192,
                    "conv_kernel": 4,
                }
            },
            "checkpoint_track": "backbone_pretraining",
            "backbone_state_dict": {},
            "reconstruction_head_state_dict": {},
            "return_head_state_dict": {},
            "volatility_head_state_dict": {},
            "correlation_head_state_dict": {},
        }
        report = validate_checkpoint_compatibility(
            payload,
            current_config=BackboneTrainingConfig(),
            requires_input_projection=False,
        )
        self.assertTrue(report["compatible"])
        self.assertTrue(
            any(
                "checkpoint_format_version" in warning
                for warning in report["warnings"]
            )
        )

    def test_validate_checkpoint_compatibility_rejects_newer_format_version(self) -> None:
        payload = {
            "config": {
                "core": {
                    "model_dim": 768,
                    "regime_input_dim": 91,
                    "regime_dim": 64,
                    "n_regimes": 4,
                    "n_heads": 4,
                    "d_head": 64,
                    "d_v": 192,
                    "conv_kernel": 4,
                }
            },
            "checkpoint_format_version": 999,
            "checkpoint_track": "backbone_pretraining",
            "backbone_state_dict": {},
            "reconstruction_head_state_dict": {},
            "return_head_state_dict": {},
            "volatility_head_state_dict": {},
            "correlation_head_state_dict": {},
        }
        report = validate_checkpoint_compatibility(
            payload,
            current_config=BackboneTrainingConfig(),
            requires_input_projection=False,
        )
        self.assertFalse(report["compatible"])
        self.assertTrue(
            any(
                "checkpoint_format_version" in reason
                for reason in report["errors"]
            )
        )

    def test_cli_training_smoke_wiring(self) -> None:
        captured = io.StringIO()
        with patch("cli.run_backbone_pretraining_smoke") as mocked:
            mocked.return_value = {
                "steps": 2,
                "device": "cpu",
                "avg_total_loss": 0.123,
                "checkpoint_output_path": "checkpoints/backbone_pretrain.pt",
                "metrics_output_path": "reports/backbone_training_metrics.json",
            }
            with redirect_stdout(captured):
                exit_code = cli.main(
                    argv=[
                        "meridian",
                        "smoke", "train",
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

    def test_cli_training_smoke_passes_single_input_shard(self) -> None:
        with TemporaryDirectory() as temp_dir:
            shard_path = Path(temp_dir) / "AAPL.f32bin"
            shard_path.write_bytes(b"")
            with patch("cli.run_backbone_pretraining_smoke") as mocked:
                mocked.return_value = {
                    "steps": 1,
                    "device": "cpu",
                    "avg_total_loss": 0.1,
                    "checkpoint_output_path": "checkpoints/backbone_pretrain.pt",
                    "metrics_output_path": "reports/backbone_training_metrics.json",
                }
                with redirect_stdout(io.StringIO()):
                    exit_code = cli.main(
                        argv=[
                            "meridian",
                            "smoke", "train",
                            "--steps",
                            "1",
                            "--input-shard",
                            str(shard_path),
                            "--max-sequences",
                            "8",
                        ]
                    )
            self.assertEqual(exit_code, 0)
            called_kwargs = mocked.call_args.kwargs
            self.assertEqual(called_kwargs["input_shard_path"], shard_path)
            self.assertEqual(called_kwargs["max_sequences"], 8)

    def test_cli_training_smoke_passes_resume_and_eval(self) -> None:
        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "backbone_pretrain.pt"
            checkpoint_path.write_bytes(b"")
            with patch("cli.run_backbone_pretraining_smoke") as mocked:
                mocked.return_value = {
                    "steps": 1,
                    "device": "cpu",
                    "avg_total_loss": 0.1,
                    "checkpoint_output_path": None,
                    "metrics_output_path": "reports/backbone_eval_metrics.json",
                    "eval_only": True,
                }
                with redirect_stdout(io.StringIO()):
                    exit_code = cli.main(
                        argv=[
                            "meridian",
                            "smoke", "train",
                            "--steps",
                            "1",
                            "--resume-checkpoint",
                            str(checkpoint_path),
                            "--eval-only",
                        ]
                    )
            self.assertEqual(exit_code, 0)
            called_kwargs = mocked.call_args.kwargs
            self.assertEqual(called_kwargs["resume_from_checkpoint_path"], checkpoint_path)
            self.assertTrue(called_kwargs["eval_only"])

    def test_cli_training_smoke_passes_prediction_ramp_steps(self) -> None:
        with patch("cli.run_backbone_pretraining_smoke") as mocked:
            mocked.return_value = {
                "steps": 1,
                "device": "cpu",
                "avg_total_loss": 0.1,
                "checkpoint_output_path": "checkpoints/backbone_pretrain.pt",
                "metrics_output_path": "reports/backbone_training_metrics.json",
            }
            with redirect_stdout(io.StringIO()):
                exit_code = cli.main(
                    argv=[
                        "meridian",
                        "smoke", "train",
                        "--steps",
                        "1",
                        "--prediction-ramp-steps",
                        "7",
                    ]
                )
        self.assertEqual(exit_code, 0)
        called_config = mocked.call_args.kwargs["config"]
        self.assertEqual(called_config.prediction_ramp_steps, 7)

    def test_cli_training_smoke_passes_multi_input_shards(self) -> None:
        with TemporaryDirectory() as temp_dir:
            shard_a = Path(temp_dir) / "AAPL.f32bin"
            shard_b = Path(temp_dir) / "SPY.f32bin"
            shard_a.write_bytes(b"")
            shard_b.write_bytes(b"")
            with patch("cli.run_backbone_pretraining_smoke") as mocked:
                mocked.return_value = {
                    "steps": 1,
                    "device": "cpu",
                    "avg_total_loss": 0.1,
                    "checkpoint_output_path": "checkpoints/backbone_pretrain.pt",
                    "metrics_output_path": "reports/backbone_training_metrics.json",
                }
                with redirect_stdout(io.StringIO()):
                    exit_code = cli.main(
                        argv=[
                            "meridian",
                            "smoke", "train",
                            "--steps",
                            "1",
                            "--input-shards",
                            f"{shard_a},{shard_b}",
                        ]
                    )
            self.assertEqual(exit_code, 0)
            called_kwargs = mocked.call_args.kwargs
            self.assertEqual(called_kwargs["input_shard_paths"], [shard_a, shard_b])

    def test_cli_split_shard_eval_wiring(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with patch("cli.run_split_shard_evaluation") as mocked:
                mocked.return_value = {
                    "splits": {
                        "train": {"avg_total_loss": 0.2, "shard_count": 2},
                        "validation": {"avg_total_loss": 0.3, "shard_count": 1},
                    }
                }
                with redirect_stdout(io.StringIO()):
                    exit_code = cli.main(
                        argv=[
                            "meridian",
                            "smoke", "eval",
                            "--shards-root",
                            temp_dir,
                            "--splits",
                            "train,validation",
                            "--steps",
                            "2",
                        ]
                    )
            self.assertEqual(exit_code, 0)
            called_kwargs = mocked.call_args.kwargs
            self.assertEqual(called_kwargs["shards_root"], Path(temp_dir))
            self.assertEqual(called_kwargs["splits"], ["train", "validation"])

    def test_cli_split_shard_eval_passes_per_shard_flag(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with patch("cli.run_split_shard_evaluation") as mocked:
                mocked.return_value = {"splits": {"train": {"shard_count": 1}}}
                with redirect_stdout(io.StringIO()):
                    exit_code = cli.main(
                        argv=[
                            "meridian",
                            "smoke", "eval",
                            "--shards-root",
                            temp_dir,
                            "--splits",
                            "train",
                            "--include-per-shard",
                        ]
                    )
            self.assertEqual(exit_code, 0)
            called_kwargs = mocked.call_args.kwargs
            self.assertTrue(called_kwargs["include_per_shard"])

    def test_cli_split_shard_eval_writes_report_output(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "split_eval_report.json"
            with patch("cli.run_split_shard_evaluation") as mocked:
                mocked.return_value = {
                    "splits": {"train": {"avg_total_loss": 0.2, "shard_count": 1}}
                }
                with redirect_stdout(io.StringIO()):
                    exit_code = cli.main(
                        argv=[
                            "meridian",
                            "smoke", "eval",
                            "--shards-root",
                            temp_dir,
                            "--splits",
                            "train",
                            "--report-output",
                            str(report_path),
                        ]
                    )
            self.assertEqual(exit_code, 0)
            self.assertTrue(report_path.exists())
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("splits", payload)

    def test_cli_split_shard_eval_passes_max_shards_per_split(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with patch("cli.run_split_shard_evaluation") as mocked:
                mocked.return_value = {"splits": {"train": {"shard_count": 1}}}
                with redirect_stdout(io.StringIO()):
                    exit_code = cli.main(
                        argv=[
                            "meridian",
                            "smoke", "eval",
                            "--shards-root",
                            temp_dir,
                            "--splits",
                            "train",
                            "--max-shards-per-split",
                            "3",
                        ]
                    )
            self.assertEqual(exit_code, 0)
            called_kwargs = mocked.call_args.kwargs
            self.assertEqual(called_kwargs["max_shards_per_split"], 3)

    def test_split_shard_eval_applies_max_shards_per_split(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "mmaps"
            train_dir = root / "train"
            validation_dir = root / "validation"
            train_dir.mkdir(parents=True, exist_ok=True)
            validation_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(3):
                (train_dir / f"TRAIN_{idx}.f32bin").write_bytes(b"")
                (validation_dir / f"VAL_{idx}.f32bin").write_bytes(b"")

            with patch("core.self_supervised_pretraining.run_backbone_pretraining_smoke") as mocked:
                mocked.return_value = {
                    "avg_total_loss": 0.1,
                    "avg_reconstruction_loss": 0.01,
                    "avg_regime_smoothness_loss": 0.02,
                    "avg_moe_balance_loss": 0.03,
                    "avg_return_prediction_loss": 0.04,
                    "avg_volatility_prediction_loss": 0.05,
                    "avg_correlation_prediction_loss": 0.06,
                }
                report = run_split_shard_evaluation(
                    config=BackboneTrainingConfig(steps=1, batch_size=1, seq_len=2),
                    shards_root=root,
                    splits=["train", "validation"],
                    checkpoint_output_path=Path(temp_dir) / "eval_placeholder.pt",
                    include_per_shard=True,
                    max_shards_per_split=2,
                )

            self.assertEqual(report["splits"]["train"]["shard_count"], 2)
            self.assertEqual(report["splits"]["validation"]["shard_count"], 2)
            self.assertEqual(len(report["splits"]["train"]["per_shard"]), 2)
            self.assertEqual(len(report["splits"]["validation"]["per_shard"]), 2)

    def test_split_shard_eval_returns_per_shard_diagnostics(self) -> None:
        if not TORCH_AVAILABLE:
            self.skipTest("torch is required for split-shard integration test")

        row_struct = struct.Struct("<112f")
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "mmaps"
            train_dir = root / "train"
            validation_dir = root / "validation"
            train_dir.mkdir(parents=True, exist_ok=True)
            validation_dir.mkdir(parents=True, exist_ok=True)

            train_shard = train_dir / "AAPL.f32bin"
            validation_shard = validation_dir / "SPY.f32bin"
            rows = [[float(i + j) for j in range(112)] for i in range(4)]
            for shard_path in (train_shard, validation_shard):
                with shard_path.open("wb") as handle:
                    for row in rows:
                        handle.write(row_struct.pack(*row))

            checkpoint_output = Path(temp_dir) / "backbone_pretrain.pt"
            config = BackboneTrainingConfig(steps=1, batch_size=1, seq_len=2)
            run_backbone_pretraining_smoke(
                config=config,
                checkpoint_output_path=checkpoint_output,
                input_shard_path=train_shard,
                max_sequences=1,
            )

            report = run_split_shard_evaluation(
                config=BackboneTrainingConfig(steps=1, batch_size=1, seq_len=2),
                shards_root=root,
                splits=["train", "validation"],
                checkpoint_output_path=Path(temp_dir) / "eval_placeholder.pt",
                resume_from_checkpoint_path=checkpoint_output,
                max_sequences=1,
                include_per_shard=True,
            )

            self.assertIn("per_shard", report["splits"]["train"])
            self.assertEqual(len(report["splits"]["train"]["per_shard"]), 1)
            self.assertEqual(
                report["splits"]["train"]["per_shard"][0]["shard_path"],
                str(train_shard),
            )


if __name__ == "__main__":
    unittest.main()
