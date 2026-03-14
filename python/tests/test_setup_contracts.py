import unittest
from datetime import datetime
from tempfile import TemporaryDirectory
from pathlib import Path

from meridian_train.feature_schema import PER_ASSET_FEATURE_DIM, validate_feature_schema
from meridian_train.spec import DEFAULT_ARCHITECTURE, TRAINING_PHASES
from meridian_train.splits import resolve_split
from meridian_train.workspace import WORKSPACE_DIRS, initialize_workspace


class SetupContractsTest(unittest.TestCase):
    def test_feature_dimension_is_112(self) -> None:
        validate_feature_schema()
        self.assertEqual(PER_ASSET_FEATURE_DIM, 112)

    def test_architecture_defaults_validate(self) -> None:
        DEFAULT_ARCHITECTURE.validate()
        self.assertEqual(DEFAULT_ARCHITECTURE.model_dim, 768)
        self.assertEqual(DEFAULT_ARCHITECTURE.max_assets, 100)

    def test_training_phases_present(self) -> None:
        self.assertEqual(set(TRAINING_PHASES.keys()), {"phase1", "phase2", "phase3", "phase4"})

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
