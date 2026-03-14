import io
import json
import unittest
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cli
from market_data.environment import build_environment_report, validate_training_environment
from market_data.pipeline import PipelineConfig, build_pipeline_plan, write_pipeline_manifest


class _FakeCuda:
    def __init__(self, available: bool, device_name: str) -> None:
        self._available = available
        self._device_name = device_name

    def is_available(self) -> bool:
        return self._available

    def get_device_name(self, _index: int) -> str:
        return self._device_name


class _FakeTorch:
    def __init__(self, version: str, cuda_available: bool, device_name: str) -> None:
        self.__version__ = version
        self.cuda = _FakeCuda(cuda_available, device_name)


class EnvironmentAndPipelineTest(unittest.TestCase):
    def test_environment_report_detects_h100(self) -> None:
        fake_torch = _FakeTorch("2.3.1", True, "NVIDIA H100 80GB HBM3")
        report = build_environment_report(torch_module=fake_torch, fla_available=True)
        self.assertTrue(report["torch_available"])
        self.assertTrue(report["cuda_available"])
        self.assertTrue(report["is_h100"])
        self.assertTrue(report["ready_for_training"])

    def test_environment_validation_fails_without_h100(self) -> None:
        fake_torch = _FakeTorch("2.3.1", True, "NVIDIA A10")
        with self.assertRaises(RuntimeError):
            validate_training_environment(
                require_h100=True, torch_module=fake_torch, fla_available=True
            )

    def test_pipeline_plan_contains_spec_contracts(self) -> None:
        config = PipelineConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2025, 6, 30),
            assets=["AAPL", "SPY", "BTC-USD"],
        )
        plan = build_pipeline_plan(config)
        self.assertEqual(plan["per_asset_feature_dim"], 112)
        self.assertEqual(plan["split_windows"]["train"]["start"], "2020-01-01T00:00:00")
        self.assertGreater(plan["estimated_equity_market_minutes_per_asset"], 400_000)

    def test_pipeline_manifest_writer_creates_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "plan.json"
            config = PipelineConfig(
                start_date=date(2020, 1, 1),
                end_date=date(2025, 6, 30),
                assets=["AAPL", "SPY"],
                output_root=Path(temp_dir),
            )
            write_pipeline_manifest(config, manifest_path)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["asset_count"], 2)
            self.assertIn("data_sources", payload)

    def test_cli_data_plan_command_writes_manifest(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "manifest.json"
            argv = [
                "meridian",
                "data-plan",
                "--assets",
                "AAPL,SPY,QQQ",
                "--output",
                str(output_path),
            ]
            captured = io.StringIO()
            with redirect_stdout(captured):
                exit_code = cli.main(argv=argv)
            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())

    def test_cli_env_check_success_path(self) -> None:
        argv = ["meridian", "env-check", "--allow-non-h100"]
        with patch("cli.build_environment_report") as mock_report:
            with patch("cli.validate_training_environment") as mock_validate:
                mock_report.return_value = {"ready_for_training": True}
                mock_validate.return_value = {"ready_for_training": True}
                captured = io.StringIO()
                with redirect_stdout(captured):
                    exit_code = cli.main(argv=argv)
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
