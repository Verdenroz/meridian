import unittest
from unittest.mock import patch

from core.temporal_cascade import (
    CascadeConfig,
    TemporalCascade,
    TORCH_AVAILABLE,
    _block_config,
    run_cascade_smoke_check,
)


class BlockConfigTest(unittest.TestCase):
    def test_d_v_constraint_hourly(self) -> None:
        cfg = _block_config(512, 64, 64, 4)
        self.assertEqual(cfg.n_heads * cfg.d_v, cfg.model_dim)

    def test_d_v_constraint_weekly(self) -> None:
        cfg = _block_config(384, 64, 48, 4)
        self.assertEqual(cfg.n_heads * cfg.d_v, cfg.model_dim)


class CascadeConfigTest(unittest.TestCase):
    def test_default_dims(self) -> None:
        c = CascadeConfig()
        self.assertEqual(c.minute_dim, 768)
        self.assertEqual(c.hourly_dim, 512)
        self.assertEqual(c.daily_dim, 512)
        self.assertEqual(c.weekly_dim, 384)

    def test_injection_block_defaults(self) -> None:
        c = CascadeConfig()
        self.assertEqual(c.hourly_injection_block, 1)
        self.assertEqual(c.daily_injection_block, 4)
        self.assertEqual(c.weekly_injection_block, 7)


class TemporalCascadeTest(unittest.TestCase):
    def setUp(self) -> None:
        if not TORCH_AVAILABLE:
            self.skipTest("torch not available")
        import torch
        self.torch = torch
        self.config = CascadeConfig(minute_dim=64, hourly_dim=32, daily_dim=32, weekly_dim=16)
        self.model = TemporalCascade(self.config)

    def test_output_shapes(self) -> None:
        c = self.config
        B, T = 2, 8
        minute_repr  = self.torch.randn(B, T, c.minute_dim)
        regime_emb   = self.torch.randn(B, T, c.regime_dim)
        ctx, fused = self.model(minute_repr, regime_emb)

        self.assertEqual(fused.shape, (B, T, c.minute_dim))
        self.assertIn(c.hourly_injection_block, ctx)
        self.assertIn(c.daily_injection_block, ctx)
        self.assertIn(c.weekly_injection_block, ctx)
        for tensor in ctx.values():
            self.assertEqual(tensor.shape, (B, T, c.minute_dim))

    def test_injection_blocks_match_config(self) -> None:
        c = self.config
        ctx, _ = self.model(
            self.torch.randn(1, 4, c.minute_dim),
            self.torch.randn(1, 4, c.regime_dim),
        )
        self.assertEqual(set(ctx.keys()), {
            c.hourly_injection_block,
            c.daily_injection_block,
            c.weekly_injection_block,
        })

    def test_surprise_ema_changes_gate_output(self) -> None:
        """Non-zero surprise should produce different contexts than no-surprise."""
        c = self.config
        minute = self.torch.randn(1, 4, c.minute_dim)
        regime = self.torch.randn(1, 4, c.regime_dim)
        surprise_low  = self.torch.zeros(1, 4)
        surprise_high = self.torch.ones(1, 4) * 10.0

        ctx_low,  _ = self.model(minute, regime, surprise_ema=surprise_low)
        ctx_high, _ = self.model(minute, regime, surprise_ema=surprise_high)

        # Gate values differ → contexts differ (for any level where gate_w ≠ 0)
        # We just verify the outputs are not identical
        key = c.hourly_injection_block
        self.assertFalse(self.torch.allclose(ctx_low[key], ctx_high[key]))

    def test_no_surprise_runs_without_error(self) -> None:
        c = self.config
        ctx, fused = self.model(
            self.torch.randn(2, 4, c.minute_dim),
            self.torch.randn(2, 4, c.regime_dim),
        )
        self.assertIsNotNone(fused)

    def test_gradients_flow(self) -> None:
        c = self.config
        minute = self.torch.randn(1, 4, c.minute_dim, requires_grad=True)
        regime = self.torch.randn(1, 4, c.regime_dim)
        _, fused = self.model(minute, regime)
        fused.sum().backward()
        self.assertIsNotNone(minute.grad)
        self.assertFalse(self.torch.all(minute.grad == 0))

    def test_smoke_check_raises_without_torch(self) -> None:
        with patch("core.temporal_cascade.TORCH_AVAILABLE", False):
            with self.assertRaises(RuntimeError):
                run_cascade_smoke_check()

    def test_smoke_check_returns_expected_keys(self) -> None:
        result = run_cascade_smoke_check(batch_size=1, seq_len=4, config=self.config)
        for key in ("device", "injection_blocks", "fused_shape",
                    "hourly_ctx_shape", "daily_ctx_shape", "weekly_ctx_shape"):
            self.assertIn(key, result)
        self.assertEqual(result["fused_shape"], [1, 4, self.config.minute_dim])

    def test_custom_injection_blocks(self) -> None:
        cfg = CascadeConfig(
            minute_dim=64, hourly_dim=32, daily_dim=32, weekly_dim=16,
            hourly_injection_block=2,
            daily_injection_block=5,
            weekly_injection_block=6,
        )
        model = TemporalCascade(cfg)
        ctx, _ = model(
            self.torch.randn(1, 4, cfg.minute_dim),
            self.torch.randn(1, 4, cfg.regime_dim),
        )
        self.assertIn(2, ctx)
        self.assertIn(5, ctx)
        self.assertIn(6, ctx)


if __name__ == "__main__":
    unittest.main()
