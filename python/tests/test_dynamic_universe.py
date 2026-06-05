"""Dynamic-universe masking contracts (fixed max-N + active mask).

Run: PYTHONPATH=python python -m pytest python/tests/test_dynamic_universe.py
"""

from __future__ import annotations

import torch

from core.cross_asset_fusion import CrossAssetFusion
from optimization.portfolio_heads import MeridianR, MeridianS
from train.spec import MeridianSpec


def test_portfolio_state_dim_derives_from_n_assets() -> None:
    assert MeridianSpec(n_assets=100).portfolio_state_dim == 201
    assert MeridianSpec(n_assets=128).portfolio_state_dim == 257
    MeridianSpec(n_assets=128).validate()  # property keeps the invariant


def _mask(n: int, live: int) -> torch.Tensor:
    m = torch.zeros(n, dtype=torch.bool)
    m[:live] = True
    return m


def test_heads_zero_padded_and_renormalize() -> None:
    B, N, D = 2, 8, 768
    mask = _mask(N, live=5)
    for Head in (MeridianR, MeridianS):
        head = Head(D).eval()
        w = head(torch.randn(B, N, D), active_mask=mask)
        assert torch.allclose(w.sum(-1), torch.ones(B), atol=1e-5)
        assert (w[:, ~mask].abs() < 1e-12).all()
        assert not torch.isnan(w).any()


def test_fusion_isolates_padded_slots() -> None:
    """Perturbing padded asset tokens must not change live-asset outputs."""
    torch.manual_seed(0)
    B, N, D, k = 2, 8, 768, 8
    mask = _mask(N, live=5)
    f = CrossAssetFusion(D).eval()
    sector_ids = torch.randint(0, 11, (N,))
    at = torch.randn(B, N, D)
    mk, pf, ep = torch.randn(B, D), torch.randn(B, D), torch.randn(B, k, D)
    with torch.no_grad():
        out_a, _ = f(at, mk, pf, ep, sector_ids, active_mask=mask)
        at2 = at.clone()
        at2[:, ~mask] = torch.randn(B, int((~mask).sum()), D) * 99.0
        out_b, _ = f(at2, mk, pf, ep, sector_ids, active_mask=mask)
    assert (out_a[:, mask] - out_b[:, mask]).abs().max() < 1e-5
    assert not torch.isnan(out_a).any()


def test_masking_is_noop_when_mask_none() -> None:
    """Back-compat: no mask → identical to the legacy full-universe path."""
    torch.manual_seed(1)
    B, N, D = 2, 6, 768
    head = MeridianR(D).eval()
    x = torch.randn(B, N, D)
    full = _mask(N, live=N)
    assert torch.allclose(head(x), head(x, active_mask=full), atol=1e-6)
