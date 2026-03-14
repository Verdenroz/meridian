import io
import unittest
from pathlib import Path
from contextlib import redirect_stdout

import cli
from market_data.feature_assembly import AssetMinuteInput, FeatureAssembler
from market_data.marketreader import (
    MarketReaderConfig,
    build_marketreader_training_plan,
)
from market_data.regime_hmm import RegimeHMMConfig, build_hmm_training_spec


class FeatureAssemblyAndSpecsTest(unittest.TestCase):
    def test_feature_assembler_handles_missing_options_block(self) -> None:
        assembler = FeatureAssembler()
        asset_input = AssetMinuteInput(
            price_volume=[0.0] * 10,
            technical=[0.1] * 20,
            options=None,
            fundamental=[0.2] * 10,
            text=[0.3] * 64,
            asset_calendar=[1.0, 0.5, 1.0],
            asset_class="equity",
        )
        vector = assembler.assemble_asset(asset_input)
        self.assertEqual(len(vector), 112)
        self.assertEqual(vector[30:35], assembler.options_defaults["equity"])

    def test_feature_assembler_rejects_bad_dimension(self) -> None:
        assembler = FeatureAssembler()
        asset_input = AssetMinuteInput(
            price_volume=[0.0] * 9,
            technical=[0.1] * 20,
            options=[0.0] * 5,
            fundamental=[0.2] * 10,
            text=[0.3] * 64,
            asset_calendar=[1.0, 0.5, 1.0],
            asset_class="equity",
        )
        with self.assertRaises(ValueError):
            assembler.assemble_asset(asset_input)

    def test_marketreader_plan_matches_spec_defaults(self) -> None:
        config = MarketReaderConfig()
        plan = build_marketreader_training_plan(
            config=config,
            corpus_path=Path("data/raw/news_corpus.jsonl"),
            tokenizer_output_path=Path("artifacts/tokenizer.json"),
            model_output_path=Path("checkpoints/marketreader.pt"),
        )
        self.assertEqual(plan["vocab_size"], 4096)
        self.assertEqual(plan["layers"], 4)
        self.assertEqual(plan["output_dim"], 64)

    def test_hmm_spec_uses_four_states(self) -> None:
        spec = build_hmm_training_spec(
            RegimeHMMConfig(),
            feature_names=["vix_level", "adx14", "corr_dispersion", "drawdown"],
        )
        self.assertEqual(spec["n_states"], 4)
        self.assertEqual(spec["covariance_type"], "diag")
        self.assertEqual(spec["feature_count"], 4)

    def test_cli_marketreader_and_hmm_commands(self) -> None:
        captured = io.StringIO()
        with redirect_stdout(captured):
            marketreader_exit = cli.main(argv=["meridian", "marketreader-plan"])
            hmm_exit = cli.main(
                argv=["meridian", "hmm-spec", "--features", "vix_level,adx14,drawdown"]
            )
            vector_exit = cli.main(
                argv=["meridian", "feature-vector-check", "--asset-class", "equity"]
            )
        self.assertEqual(marketreader_exit, 0)
        self.assertEqual(hmm_exit, 0)
        self.assertEqual(vector_exit, 0)


if __name__ == "__main__":
    unittest.main()
