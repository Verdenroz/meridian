import unittest
from unittest.mock import patch

from core.cross_asset_fusion import (
    CrossAssetFusion,
    FusionConfig,
    TORCH_AVAILABLE,
    _NO_SECTOR,
    run_fusion_smoke_check,
)


class FusionConfigTest(unittest.TestCase):
    def test_defaults(self) -> None:
        c = FusionConfig()
        self.assertEqual(c.n_heads * c.d_head, c.model_dim)
        self.assertEqual(c.n_sectors, 11)
        self.assertEqual(c.n_layers, 2)
        self.assertEqual(c.ffn_dim, c.model_dim * 2)

    def test_invalid_head_config_raises(self) -> None:
        if not TORCH_AVAILABLE:
            return
        with self.assertRaises(AssertionError):
            # 7 heads × 96 ≠ 768
            from core.cross_asset_fusion import _FusionLayer
            _FusionLayer(FusionConfig(n_heads=7, d_head=96))


class CrossAssetFusionTest(unittest.TestCase):
    def setUp(self) -> None:
        if not TORCH_AVAILABLE:
            self.skipTest("torch not available")
        import torch
        self.torch = torch
        # Small config for fast tests
        self.config = FusionConfig(model_dim=64, n_heads=4, d_head=16, n_layers=2, ffn_dim=128)
        self.model  = CrossAssetFusion(self.config)

    def test_output_shape_with_context(self) -> None:
        B, N, M = 2, 10, 3
        D = self.config.model_dim
        out = self.model(
            self.torch.randn(B, N, D),
            self.torch.randn(B, M, D),
            self.torch.randint(0, self.config.n_sectors, (B, N)),
        )
        self.assertEqual(list(out.shape), [B, N, D])

    def test_output_shape_no_context(self) -> None:
        B, N = 2, 10
        D = self.config.model_dim
        out = self.model(self.torch.randn(B, N, D), None, None)
        self.assertEqual(list(out.shape), [B, N, D])

    def test_sector_bias_shape(self) -> None:
        n = self.config.n_sectors + 1  # includes no-sector sentinel
        self.assertEqual(list(self.model.sector_bias.shape), [n, n])

    def test_sector_bias_applied_to_asset_block(self) -> None:
        """With a non-zero sector bias, output should differ from zero-bias run."""
        import torch
        B, N, M = 1, 5, 1
        D = self.config.model_dim
        asset  = torch.randn(B, N, D)
        ctx    = torch.randn(B, M, D)
        sids   = torch.zeros(B, N, dtype=torch.long)

        out_before = self.model(asset, ctx, sids).detach().clone()

        # Manually set a large sector bias and re-run
        with torch.no_grad():
            self.model.sector_bias.fill_(5.0)
        out_after = self.model(asset, ctx, sids).detach()

        self.assertFalse(torch.allclose(out_before, out_after))
        # Reset bias
        with torch.no_grad():
            self.model.sector_bias.zero_()

    def test_no_sector_sentinel_is_outside_gics_range(self) -> None:
        self.assertEqual(_NO_SECTOR, 11)
        self.assertNotIn(_NO_SECTOR, range(11))

    def test_context_tokens_stripped_from_output(self) -> None:
        """Output should have exactly N_assets tokens regardless of context size."""
        B, N = 1, 8
        D = self.config.model_dim
        for n_ctx in (0, 1, 5):
            ctx = self.torch.randn(B, n_ctx, D) if n_ctx > 0 else None
            out = self.model(self.torch.randn(B, N, D), ctx, None)
            self.assertEqual(out.shape[1], N, f"n_ctx={n_ctx}")

    def test_gradients_flow_through_sector_bias(self) -> None:
        B, N = 1, 4
        D = self.config.model_dim
        asset = self.torch.randn(B, N, D)
        sids  = self.torch.zeros(B, N, dtype=self.torch.long)
        out   = self.model(asset, None, sids)
        out.sum().backward()
        self.assertIsNotNone(self.model.sector_bias.grad)

    def test_smoke_check_raises_without_torch(self) -> None:
        with patch("core.cross_asset_fusion.TORCH_AVAILABLE", False):
            with self.assertRaises(RuntimeError):
                run_fusion_smoke_check()

    def test_smoke_check_output_keys(self) -> None:
        result = run_fusion_smoke_check(
            n_assets=5, n_ctx=2, batch_size=1, config=self.config
        )
        for key in ("device", "input_asset_shape", "input_context_shape",
                    "output_shape", "n_layers", "n_heads"):
            self.assertIn(key, result)
        self.assertEqual(result["output_shape"][1], 5)  # n_assets preserved

    def test_n_layers_affects_depth(self) -> None:
        for n_layers in (1, 2, 3):
            model = CrossAssetFusion(FusionConfig(
                model_dim=64, n_heads=4, d_head=16, n_layers=n_layers, ffn_dim=128
            ))
            self.assertEqual(len(model.layers), n_layers)

    def test_single_asset_no_crash(self) -> None:
        D = self.config.model_dim
        out = self.model(self.torch.randn(1, 1, D), None, None)
        self.assertEqual(out.shape, (1, 1, D))


if __name__ == "__main__":
    unittest.main()
