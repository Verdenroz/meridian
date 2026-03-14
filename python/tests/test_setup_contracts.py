import unittest
from datetime import datetime
from tempfile import TemporaryDirectory
from pathlib import Path

from train.feature_schema import PER_ASSET_FEATURE_DIM, validate_feature_schema
from train.spec import DEFAULT_ARCHITECTURE, TRAINING_STAGES
from train.splits import resolve_split
from market_data.workspace import WORKSPACE_DIRS, initialize_workspace


class SetupContractsTest(unittest.TestCase):
    def test_feature_dimension_is_112(self) -> None:
        validate_feature_schema()
        self.assertEqual(PER_ASSET_FEATURE_DIM, 112)

    def test_architecture_defaults_validate(self) -> None:
        DEFAULT_ARCHITECTURE.validate()
        self.assertEqual(DEFAULT_ARCHITECTURE.model_dim, 768)
        self.assertEqual(DEFAULT_ARCHITECTURE.max_assets, 100)

    def test_training_stage_keys_present(self) -> None:
        self.assertEqual(
            set(TRAINING_STAGES.keys()),
            {"foundation", "optimization", "adaptation", "compression"},
        )

    def test_data_split_resolution(self) -> None:
        self.assertEqual(resolve_split(datetime(2020, 6, 15, 10, 0, 0)), "train")
        self.assertEqual(resolve_split(datetime(2024, 8, 15, 13, 30, 0)), "validation")
        self.assertEqual(resolve_split(datetime(2025, 3, 5, 9, 45, 0)), "test")

    def test_workspace_initialization(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            created = initialize_workspace(root)
            self.assertEqual(len(created), len(WORKSPACE_DIRS))
            for relative_dir in WORKSPACE_DIRS:
                self.assertTrue((root / relative_dir).is_dir())


if __name__ == "__main__":
    unittest.main()
