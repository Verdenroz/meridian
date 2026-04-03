"""MarketReader text encoder — 3-tier financial text processing (Section 7.10)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from tokenizers import Tokenizer  # type: ignore[import]


# ── Shared backbone primitives ───────────────────────────────────────────────

class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class _GDNBlock(nn.Module):
    """Single GDN block used inside MarketReader.

    MarketReader GDN (D=256, n_heads=4, d_head=32, d_v=64).
    Implemented as causal self-attention with a GDN-style recurrent state;
    for training we use the parallel (attention) formulation.
    """

    def __init__(self, dim: int = 256, n_heads: int = 4, d_head: int = 32, d_v: int = 64) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_head
        self.d_v     = d_v

        self.q_proj = nn.Linear(dim, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(dim, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(dim, n_heads * d_v,   bias=False)
        self.o_proj = nn.Linear(n_heads * d_v, dim,   bias=False)

        self.log_alpha = nn.Parameter(torch.full((n_heads,), -1.0))
        self.norm      = _RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) — runs in parallel during training
        residual = x
        x = self.norm(x)
        B, T, _ = x.shape
        H, Dh, Dv = self.n_heads, self.d_head, self.d_v

        q = self.q_proj(x).view(B, T, H, Dh).permute(0, 2, 1, 3)  # (B, H, T, Dh)
        k = self.k_proj(x).view(B, T, H, Dh).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, T, H, Dv).permute(0, 2, 1, 3)

        # Forget-gate decay mask: alpha^|i-j| for causal attention
        alpha = torch.sigmoid(self.log_alpha)  # (H,)
        pos   = torch.arange(T, device=x.device, dtype=x.dtype)
        dist  = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(min=0)  # (T, T) causal dist
        decay = alpha.view(H, 1, 1) ** dist.unsqueeze(0)            # (H, T, T)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        decay = decay.masked_fill(causal_mask.unsqueeze(0), 0.0)

        scale = Dh ** -0.5
        attn = torch.einsum("bhin,bhjn->bhij", q, k) * scale  # (B, H, T, T)
        attn = attn * decay.unsqueeze(0) - 1e9 * causal_mask.view(1, 1, T, T).float()
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bhij,bhjv->bhiv", attn, v)         # (B, H, T, Dv)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, H * Dv)
        return residual + self.o_proj(out)


class _AttentionPool(nn.Module):
    """Learned attention pooling over a token sequence → single vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T) bool — True = padding
        scores = self.score(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, T)
        return torch.einsum("bt,btd->bd", weights, x)


# ── MarketReader ─────────────────────────────────────────────────────────────

class MarketReader(nn.Module):
    """Shared GDN backbone with three tier-specific output heads.

    Shared:
      Token embedding:    Embedding(4096, 256)
      Position embedding: Embedding(512,  256)
      4 × GDNBlock(D=256, n_heads=4, d_head=32, d_v=64)
      AttentionPool → pooled vector (256-dim)

    Output heads:
      Headline:  Linear(256 →  4) — positive, negative, uncertainty, finance-relevance
      Summary:   Linear(256 → 32) — topic + sentiment
      Deep:      Linear(256 → 64) — full embedding

    Total: ~3.5M parameters.

    Max sequence lengths:
      Headline: 32 tokens
      Summary:  128 tokens
      Deep:     512 tokens (sliding-window + max-pool for transcripts > 512)
    """

    MAX_HEADLINE_LEN = 32
    MAX_SUMMARY_LEN  = 128
    MAX_DEEP_LEN     = 512

    # Time-decay τ (hours) for each tier
    HEADLINE_TAU_HOURS = 4.0
    SUMMARY_TAU_HOURS  = 12.0
    DEEP_TAU_HOURS     = 48.0

    def __init__(
        self,
        vocab_size: int = 4_096,
        dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_head: int = 32,
        d_v: int = 64,
        max_seq_len: int = 512,
        headline_dim: int = 4,
        summary_dim: int = 32,
        deep_dim: int = 64,
    ) -> None:
        super().__init__()
        self.dim         = dim
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb   = nn.Embedding(max_seq_len, dim)

        self.blocks = nn.ModuleList([
            _GDNBlock(dim, n_heads, d_head, d_v) for _ in range(n_layers)
        ])
        self.pool = _AttentionPool(dim)
        self.norm = _RMSNorm(dim)

        self.headline_head = nn.Linear(dim, headline_dim)
        self.summary_head  = nn.Linear(dim, summary_dim)
        self.deep_head     = nn.Linear(dim, deep_dim)

        self._tokenizer: Tokenizer | None = None

    # ── Tokenizer ────────────────────────────────────────────────────────────

    def load_tokenizer(self, path: str) -> None:
        from tokenizers import Tokenizer  # type: ignore[import]
        self._tokenizer = Tokenizer.from_file(path)

    def tokenize(self, texts: list[str], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a list of texts and return (token_ids, padding_mask)."""
        if self._tokenizer is None:
            raise RuntimeError("Call load_tokenizer() before encoding text.")
        ids_list = []
        for text in texts:
            enc = self._tokenizer.encode(text)
            ids = enc.ids[:max_len]
            ids_list.append(ids)
        max_actual = max(len(ids) for ids in ids_list)
        padded     = torch.zeros(len(texts), max_actual, dtype=torch.long)
        pad_mask   = torch.ones(len(texts), max_actual, dtype=torch.bool)
        for i, ids in enumerate(ids_list):
            padded[i, : len(ids)]   = torch.tensor(ids)
            pad_mask[i, : len(ids)] = False
        return padded, pad_mask

    # ── Shared encoder ───────────────────────────────────────────────────────

    def _encode(
        self,
        token_ids: torch.Tensor,         # (B, T) int64
        pad_mask: torch.Tensor | None,   # (B, T) bool — True = padding
    ) -> torch.Tensor:                   # (B, dim)
        B, T = token_ids.shape
        pos   = torch.arange(T, device=token_ids.device)
        x     = self.token_emb(token_ids) + self.pos_emb(pos.unsqueeze(0))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.pool(x, pad_mask)

    def _sliding_window_encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode sequences longer than MAX_DEEP_LEN via sliding window + max-pool."""
        T = token_ids.shape[1]
        step = self.MAX_DEEP_LEN // 2
        chunk_embeds = []
        for start in range(0, T, step):
            chunk = token_ids[:, start : start + self.MAX_DEEP_LEN]
            if chunk.shape[1] == 0:
                break
            chunk_embeds.append(self._encode(chunk, None))
        stacked = torch.stack(chunk_embeds, dim=1)  # (B, n_chunks, dim)
        return stacked.max(dim=1).values            # (B, dim)

    # ── Public API ───────────────────────────────────────────────────────────

    def encode_headline(self, token_ids: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """(B, T≤32) → (B, 4) sentiment vector."""
        pooled = self._encode(token_ids, pad_mask)
        return self.headline_head(pooled)

    def encode_summary(self, token_ids: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """(B, T≤128) → (B, 32) topic+sentiment vector."""
        pooled = self._encode(token_ids, pad_mask)
        return self.summary_head(pooled)

    def encode_deep(self, token_ids: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """(B, T≤512, or longer for transcripts) → (B, 64) full embedding."""
        if token_ids.shape[1] > self.MAX_DEEP_LEN:
            pooled = self._sliding_window_encode(token_ids)
        else:
            pooled = self._encode(token_ids, pad_mask)
        return self.deep_head(pooled)

    # ── Per-asset text aggregation (Section 6.5) ─────────────────────────────

    @staticmethod
    def aggregate_embeddings(
        embeddings: torch.Tensor,     # (N, dim) — all text items for this asset
        minutes_since: torch.Tensor,  # (N,) — age of each item in minutes
        tau_hours: float,             # time-decay constant in hours
        attn_proj: nn.Linear,         # learned attention scorer
    ) -> torch.Tensor:                # (dim,)
        """Weighted pooling with time decay and learned attention weights.

        text_emb_i = Σ_j (attn_j × decay_j × emb_j) / Z
        decay_j    = exp(-minutes_since_j / (τ × 60))
        """
        tau_minutes = tau_hours * 60.0
        decay   = torch.exp(-minutes_since / tau_minutes)        # (N,)
        scores  = attn_proj(embeddings).squeeze(-1)              # (N,)
        weights = F.softmax(scores, dim=0) * decay               # (N,)
        weights = weights / (weights.sum() + 1e-8)
        return (weights.unsqueeze(-1) * embeddings).sum(dim=0)   # (dim,)
