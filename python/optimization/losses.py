"""Loss functions for portfolio optimization training.

Return head (Meridian-R) loss:
  L = -DSR(w, r) + λ_turn · TurnoverPenalty(w) + λ_vol · VolatilityAux(w, r)

  DSR (Differential Sharpe Ratio):
    Online estimate of d/dη [E[R] / σ[R]] using exponential moving moments.
    Differentiable surrogate for Sharpe ratio (Moody & Saffell 2001).

  TurnoverPenalty:
    λ_turn · Σ_t Σ_i |w_{i,t} - w_{i,t-1}|
    Discourages excessive rebalancing (target: < 10% daily turnover)

  VolatilityAux:
    MSE between predicted portfolio vol and realised vol.
    Auxiliary task that improves risk-awareness of the backbone.

Sharpe head (Meridian-S) loss:
  L = -SharpeRatio(w, r)_{rolling_window}
  Backbone frozen; only Meridian-S head parameters updated.

Hyperparameters:
  λ_turn = 0.1   — turnover penalty weight
  λ_vol  = 0.05  — volatility auxiliary weight
  η      = 0.01  — DSR EMA adaptation rate
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ── DSR ───────────────────────────────────────────────────────────────────────

def portfolio_returns(
    weights: torch.Tensor,   # (B, T, N)
    asset_returns: torch.Tensor,  # (B, T, N) — next-step log-returns
) -> torch.Tensor:           # (B, T)
    """Compute per-step portfolio log-return: R_t = Σ_i w_{i,t} · r_{i,t+1}."""
    return (weights * asset_returns).sum(dim=-1)


def dsr_loss(
    weights: torch.Tensor,       # (B, T, N)
    asset_returns: torch.Tensor, # (B, T, N)
    eta: float = 0.01,
) -> torch.Tensor:               # scalar
    """Differential Sharpe Ratio loss (Moody & Saffell 2001).

    Computes the online gradient of the Sharpe ratio with respect to the
    current portfolio return using exponential moving moment estimates.

    The DSR at time t is:
        DSR_t = [B_{t-1}·ΔA_t - 0.5·A_{t-1}·ΔB_t] / (B_{t-1} - A_{t-1}²)^{3/2}

    where:
        R_t   = portfolio return at t
        A_t   = (1-η)·A_{t-1} + η·R_t   (EMA of returns)
        B_t   = (1-η)·B_{t-1} + η·R_t²  (EMA of squared returns)
        ΔA_t  = R_t - A_{t-1}
        ΔB_t  = R_t² - B_{t-1}

    Returns the *negative* mean DSR (to be minimised).
    """
    R = portfolio_returns(weights, asset_returns)  # (B, T)
    B, T = R.shape

    A = torch.zeros(B, device=R.device)
    Bm = torch.ones(B, device=R.device) * 1e-6  # B_moment, init > 0 to avoid div/0

    dsr_values = []
    for t in range(T):
        Rt = R[:, t]
        dA = Rt - A
        dB = Rt.pow(2) - Bm
        denom = (Bm - A.pow(2)).clamp(min=1e-4).pow(1.5)
        dsr_t = ((Bm * dA - 0.5 * A * dB) / denom).clamp(-100, 100)
        dsr_values.append(dsr_t)
        # Update moments
        A  = (1 - eta) * A  + eta * Rt
        Bm = (1 - eta) * Bm + eta * Rt.pow(2)

    dsr = torch.stack(dsr_values, dim=1)  # (B, T)
    return -dsr.mean()


# ── Turnover penalty ─────────────────────────────────────────────────────────

def turnover_penalty(weights: torch.Tensor) -> torch.Tensor:  # (B, T, N) → scalar
    """L1 turnover: mean absolute weight change per step.

    Target: < 10% of portfolio rebalanced per day
    (10% / 390 minutes ≈ 0.026% per minute across all assets).
    """
    if weights.shape[1] < 2:
        return weights.new_zeros(1).squeeze()
    delta = (weights[:, 1:] - weights[:, :-1]).abs()  # (B, T-1, N)
    return delta.sum(dim=-1).mean()


# ── Volatility auxiliary ──────────────────────────────────────────────────────

def realised_vol(
    asset_returns: torch.Tensor,  # (B, T, N)
    weights: torch.Tensor,        # (B, T, N)
    window: int = 5,
) -> torch.Tensor:                # (B, T-window+1)
    """Rolling realised portfolio volatility (std of portfolio returns)."""
    R = portfolio_returns(weights, asset_returns)  # (B, T)
    if R.shape[1] < window:
        return R.std(dim=1, keepdim=True).expand(-1, max(1, R.shape[1] - window + 1))
    windows = R.unfold(1, window, 1)  # (B, T-window+1, window)
    return windows.std(dim=-1)        # (B, T-window+1)


def volatility_aux_loss(
    pred_vol: torch.Tensor,       # (B, T', horizon) from VolatilityHead
    weights: torch.Tensor,        # (B, T, N)
    asset_returns: torch.Tensor,  # (B, T, N)
    vol_window: int = 5,
) -> torch.Tensor:
    """MSE between predicted portfolio vol and realised portfolio vol."""
    target = realised_vol(asset_returns, weights, vol_window)  # (B, T')
    T_pred = min(pred_vol.shape[1], target.shape[1])
    pred = pred_vol[:, :T_pred, 0]    # use first horizon step
    tgt  = target[:, :T_pred]
    return F.mse_loss(pred, tgt.detach())


# ── Rolling Sharpe ────────────────────────────────────────────────────────────

def rolling_sharpe_loss(
    weights: torch.Tensor,        # (B, T, N)
    asset_returns: torch.Tensor,  # (B, T, N)
    annual_factor: float = 252.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Negative annualised Sharpe ratio over the full sequence window.

    Used for Meridian-S head training (backbone frozen).
    Computes Sharpe over the entire T-step window, which at 200K steps
    of 64-step sequences covers substantial regime variation.

    annualised Sharpe = (μ / σ) × sqrt(annual_factor)
    """
    R = portfolio_returns(weights, asset_returns)  # (B, T)
    mu    = R.mean(dim=1)                           # (B,)
    sigma = R.std(dim=1).clamp(min=eps)             # (B,)
    sharpe = (mu / sigma) * (annual_factor ** 0.5)
    return -sharpe.mean()


# ── Combined return-head loss ─────────────────────────────────────────────────

def return_head_loss(
    weights: torch.Tensor,        # (B, T, N)
    asset_returns: torch.Tensor,  # (B, T, N)
    pred_vol: torch.Tensor | None = None,  # (B, T, horizon) from vol head
    lambda_turn: float = 0.1,
    lambda_vol: float = 0.05,
    dsr_eta: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Combined Meridian-R training loss.

    Turnover and vol_aux penalties are adaptively scaled relative to
    the DSR magnitude so their gradients remain meaningful regardless
    of how large DSR grows during training.

    Returns a dict with individual components and 'total' for logging.
    """
    l_dsr  = dsr_loss(weights, asset_returns, eta=dsr_eta)
    l_turn = turnover_penalty(weights)
    l_vol  = (
        volatility_aux_loss(pred_vol, weights, asset_returns)
        if pred_vol is not None
        else weights.new_zeros(1).squeeze()
    )
    # Adaptive scaling: make λ control *relative* importance to DSR,
    # not absolute magnitude.  stop-gradient ratio so scaling doesn't
    # affect the DSR gradient itself.
    dsr_scale = l_dsr.detach().abs().clamp(min=1.0, max=100.0)
    turn_scale = dsr_scale / l_turn.detach().clamp(min=1e-6)
    vol_scale  = dsr_scale / l_vol.detach().clamp(min=1e-6) if l_vol.abs() > 0 else 1.0

    total = l_dsr + lambda_turn * turn_scale * l_turn + lambda_vol * vol_scale * l_vol
    return {"dsr": l_dsr, "turnover": l_turn, "vol_aux": l_vol, "total": total}
