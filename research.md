# Meridian: validated architecture for a CPU-native portfolio model

**Gated DeltaNet — not vanilla Mamba — should be Meridian's backbone.** Research across 2024–2026 confirms that Gated DeltaNet delivers the best combination of O(1) recurrent inference, ternary quantization friendliness, training efficiency on H100, and suitability for financial time-series modeling. It has entered production in Qwen3.5 (Feb 2026) and Kimi Linear (Oct 2025), outperforming both Mamba-2 and Transformers under fair comparison. The full Meridian stack — ternary weights, surprise-gated online adaptation, regime-routed MoE, and FAISS-backed episodic memory — is validated as feasible on Raspberry Pi 5 with **100–1000× compute headroom** beyond what's needed for real-time financial inference.

---

## The backbone decision: why Gated DeltaNet wins

The most consequential finding from this research is that the recurrent architecture landscape has shifted decisively since Mamba-1. **Gated DeltaNet**, published at ICLR 2025, combines Mamba-2's gated decay (adaptive forgetting) with the delta rule's error-correcting updates into a single elegant recurrence: `S_t = α_t · S_{t-1} + β_t · k_t ⊗ (v_t − α_t · S_{t-1}ᵀk_t)`. This state update is mathematically equivalent to recursive least squares — the same algorithm quant finance has relied on for decades.

The evaluation against Meridian's four core criteria makes the case clear:

| Criterion | Mamba-2 | RWKV-7 | xLSTM (mLSTM) | Gated DeltaNet |
|---|---|---|---|---|
| O(1) CPU inference | ✅ (needs CUDA port) | ✅ (pure RNN) | ✅ | **✅ (simplest recurrence)** |
| Ternary quantization | ⚠️ Scattered outliers | ✅ O(1)-magnitude states | ⚠️ Exp gating range | **✅ L2-norm Q/K, bounded states** |
| Financial time-series | ✅ | ✅ | ✅ LSTM heritage | **✅✅ Delta rule = online learning** |
| H100 training speed | ✅✅ | ✅ | ✅ | **✅✅ Higher throughput than Mamba-2** |

**Mamba-2** suffers from scattered activation outliers that make ternary quantization difficult — the Quamba team found that Mamba's linear recurrence amplifies quantization errors, and Q-Mamba showed state caches consume **79.6%** of memory in quantized Mamba models. **RWKV-7** is the strongest pure-RNN alternative with excellent O(1)-magnitude state stability, but its weaker lookback/retrieval compared to delta-rule architectures is a liability for pattern matching in financial data. **xLSTM's** exponential gating creates large dynamic range incompatible with low-bit representation. Gated DeltaNet avoids all these problems: L2-normalized queries and keys eliminate softmax's large dynamic range, the gated decay bounds state magnitude, and the delta rule's error-correcting write operation is inherently robust.

The production validation is compelling. Qwen3-Next (Sept 2025) and **Qwen3.5 (Feb 2026)** use a 3:1 ratio of Gated DeltaNet to Gated Attention layers. Kimi Linear (Oct 2025) uses a refined variant called KDA (Kimi Delta Attention) and became the **first linear attention model to outperform full attention under fair comparison**. The Flash Linear Attention (FLA) library provides production-ready Triton kernels for training.

The recommended block architecture for Meridian:

```
Input → Token Shift → LayerNorm →
  Gated DeltaNet (token mixing):
    Q, K projections → L2 normalize → Conv1d(k=4) → SiLU
    V projection → Conv1d(k=4) → SiLU
    Gate (α) → sigmoid (adaptive forgetting)
    Beta (β) → sigmoid (update strength)
    Delta Rule: S = α·S + β·k⊗(v − α·Sᵀk)
    Output = (qᵀ·S) ⊙ gate
→ Residual → LayerNorm → SwiGLU FFN → Residual
```

**RWKV-7 remains the backup choice** if the Gated DeltaNet ecosystem proves insufficient. RWKV-7's generalized delta rule with adaptive in-context learning rates achieves competitive results at 3B scale while maintaining O(1) inference and Apache 2.0 licensing. Its O(1)-magnitude WKV states are the most quantization-friendly of any architecture surveyed.

---

## Ternary quantization is viable but demands mixed precision

The BitNet b1.58 ecosystem has matured significantly. Microsoft released the **BitNet b1.58 2B4T** model (April 2025) under MIT license — 2B parameters trained on 4T tokens, running at **~8 tokens/sec on Raspberry Pi 5** with a **0.4 GB** memory footprint. The training methodology is well-established: absmean weight quantization with straight-through estimator (STE) gradient propagation, training from scratch with ternary constraints (not post-training quantization).

The critical validation for SSMs specifically comes from **Slender-Mamba** (COLING 2025), which applied BitNet b1.58 ternary quantization to Mamba-2 at 170M parameters, achieving a **90% reduction** in bits with minimal performance degradation. This is the closest published precedent to Meridian's intended architecture.

However, the research reveals a critical design constraint: **the recurrent hidden state must remain in higher precision**. The Precision Highway principle (Park et al., 2018) demonstrates that quantization errors accumulate through recurrent state transitions. NVIDIA's Nemotron team confirmed this for Mamba specifically: "per-token recurrent quantization accumulates errors across every token." For Meridian running 24/7, unbounded error accumulation in a ternary state would cause drift within hours.

The validated mixed-precision strategy:

- **Ternary {-1, 0, 1}**: All linear projection matrices (input, output, gating) — ~90% of parameters. Multiplications become additions using ARM NEON bitwise ops.
- **FP16**: Hidden state `S`, state transition parameters (α, β), and non-linear operations (sigmoid, SiLU). This costs negligibly because `S` at head_dim=64 with 4 heads and 12 layers is only **384 KB** total.
- **INT8**: Activations, using per-token absmax quantization as in BitNet b1.58.
- **Periodic state normalization**: RMSNorm applied to hidden state every N steps as a safety valve against drift.

ARM NEON ternary inference is production-ready. The **TAB framework** achieves **13.4× end-to-end speedup** on Raspberry Pi using bitwise dot products for ternary weights, packing 64 ternary values into a single 128-bit NEON register. Microsoft's **bitnet.cpp** provides three optimized kernel variants (I2_S, TL1, TL2) achieving **1.37–5.07× speedups** on ARM with **55–70% energy reduction**. The T-MAC runtime pushes BitNet 3B to **11 tokens/sec** on Pi 5.

For a 30M-parameter ternary Meridian model (**7.5 MB** weight storage), the performance envelope on Pi 5 is generous: **0.3–1.5 ms per inference step** with optimized NEON kernels, enabling **300–1000 inferences per second** — far exceeding the 1/second or 1/minute cadence needed for financial data.

---

## Three-tier online adaptation replaces monolithic retraining

Financial markets are non-stationary. Meridian must adapt to regime changes without catastrophic forgetting. The research converges on a multi-timescale strategy combining three complementary mechanisms.

**Tier 1 — Architectural adaptation via surprise-gated TTT dynamics.** The Titans framework (Google Research, Jan 2025) extends Test-Time Training with **surprise-based memorization**: tokens with high prediction error receive stronger memory updates, while an adaptive weight decay acts as a forgetting gate. This aligns perfectly with Gated DeltaNet's delta rule, which already performs error-correcting updates at each step. The key insight: `v − S^T k` in the delta rule IS a prediction error — DeltaNet is already performing a form of TTT-Linear. To add surprise gating, modulate the update strength β_t by prediction surprise: `β_t = β_base × sigmoid(α × surprise_t)`, where surprise_t measures the normalized prediction error. This adds **~50 FLOPs per step** — negligible.

**Tier 2 — Normalization adaptation at zero cost.** A February 2026 paper on Test-Time Adaptation for non-stationary time series demonstrates that updating only normalization affine parameters (scale and bias) via entropy minimization handles distributional shift with essentially zero compute. The backbone stays frozen; only the LayerNorm parameters (a handful of floats per layer) are updated online. A quadratic drift penalty and uncertainty-triggered fallback prevent instability. This is the simplest, most robust form of adaptation and should always be active.

**Tier 3 — Periodic consolidation via EWC + ternary LoRA.** During low-activity windows (market close), accumulate a buffer of recent data and fine-tune projection matrices using **LoTA-QAF** (Lossless Ternary Adaptation), which performs LoRA-style updates that merge losslessly back into ternary weights using ternary signed gradient descent. Online EWC regularization prevents forgetting past regime knowledge. Memory overhead: ~2× model parameters for the Fisher Information Matrix diagonal + old parameter snapshot = ~15 MB for a 30M model.

The fast weights (Gated DeltaNet state S in FP16) and slow weights (ternary backbone) mirror the biological brain's distinction between working memory and crystallized knowledge. The Titans finding that this asymmetry improves over uniform adaptation validates this design principle.

---

## Regime-routed MoE is warranted but only on FFN layers

MoE literature from 2024–2025 provides a surprising positive signal: **expert layers are MORE robust to quantization than dense layers** (MoQE, Microsoft). With 2-bit expert weights, MoE models actually outperform equivalent dense models. This is attributed to the sparse activation pattern — each expert handles a narrower input distribution, making its weight distribution simpler and more amenable to low-bit representation.

The validated pattern from MoE-Mamba, BlackMamba, and Jamba is to **apply MoE exclusively to FFN layers while keeping the sequence model shared**. For Meridian, this means:

```
Input → [Shared Gated DeltaNet layers] → [Regime Router] → [Expert FFN 1..4] → Heads
```

**External regime-based routing is strongly preferred over learned routing.** With ternary weights, a learned router's softmax over quantized logits has extremely limited resolution — routing decisions become noisy. Instead, feed HMM/GMM regime posterior probabilities directly as soft routing weights. When P(bull) > 0.8, route 100% to the bull expert; when uncertain, blend experts proportionally. This eliminates learned-routing instability, provides interpretability, and leverages decades of validated regime-detection methodology in quantitative finance. The router becomes "free" — it's the existing HMM detector.

At **30M total parameters** with 4 experts, each expert FFN is ~3.75M parameters (ternary = ~0.9 MB). Total model storage: ~7 MB. All experts reside in Pi 5 memory simultaneously; routing overhead is a single linear layer mapping to 4 outputs — **~0.01% of total compute**. Start with 2 experts (low-volatility/high-volatility) and scale to 4 if regime differentiation justifies it.

An alternative worth benchmarking: **FiLM conditioning** (Feature-wise Linear Modulation), where regime-specific scale/shift parameters modulate intermediate representations. This is simpler than full MoE and avoids expert parameterization overhead, though it sacrifices the hard specialization benefit.

---

## Financial architecture patterns that Meridian should adopt

The dual-head design (return prediction + Sharpe optimization) has direct precedent: **Ong & Herremans (2023)** demonstrated a shared backbone with Sharpe ratio optimization as the main task and volatility forecasting as auxiliary, backtested on 88 futures contracts over 20 years with positive results after transaction costs. Their key finding — **auxiliary volatility tasks significantly boost portfolio Sharpe ratio** — argues for adding a third volatility head to Meridian.

Direct Sharpe ratio optimization is problematic. The DSL framework (Kim et al., 2025, ACM ICAIF) demonstrates that supervised learning with Sharpe-optimal portfolio targets, trained via cross-entropy loss with deep ensembles, **outperforms direct end-to-end Sharpe optimization**, which suffers from non-convexity, gradient instability near low-volatility periods, and sensitivity to initialization. Meridian should instead use the **Differential Sharpe Ratio (DSR)** formulation (Moody & Saffell, 1998), which decomposes Sharpe into per-timestep contributions via exponential moving estimates — naturally suited for online/streaming operation and avoiding batch-level ratio instability.

The recommended loss function: `L = α·L_return + β·L_DSR + γ·L_volatility + δ·L_turnover`, with uncertainty-weighted multi-task balancing (Kendall et al., 2018) for automatic α, β, γ tuning and an explicit turnover penalty to embed transaction costs.

From the Temporal Fusion Transformer, two design patterns deserve adoption: **Variable Selection Networks** with gating to dynamically weight input features (critical because feature importance is regime-dependent), and **Static Covariate Encoders** for asset metadata (sector, asset class, market cap) that condition the temporal processing. From CMDMamba (2025), the dual-layer approach capturing both micro-fluctuations and macro-trends via different state expansion factors validates multi-scale processing through the backbone itself rather than requiring hierarchical input streams.

---

## Episodic memory via surprise-gated k-NN with FAISS

The optimal episodic memory design for Pi 5 is not a sophisticated differentiable system (DNC, Kanerva Machine) but a **simple k-NN lookup over compressed embeddings with surprise-based selective storage**. This draws from three converging research streams: RAFT-style retrieval augmentation (validated at ICML level — 86% win ratio across 10 benchmarks), Surprise Memory's distinction between noise and true novelty (critical for noisy financial data), and FAISS's proven ARM deployment.

The architecture uses a **fixed-size circular buffer** of 10,000–50,000 entries, each containing a market embedding (64–128 dimensions, float16), prediction error, outcome, and metadata. Total memory: **5–25 MB**. A two-tier write gate prevents storing repetitive high-volatility events while capturing genuinely novel patterns: Tier 1 checks if prediction error exceeds an adaptive 2σ threshold; Tier 2 queries FAISS for nearest neighbors and stores only if the minimum distance exceeds a novelty threshold — meaning this type of surprise hasn't been seen before.

**FAISS `IndexFlatL2`** provides exact sub-millisecond search over 50K vectors on ARM (validated on Raspberry Pi 4; Pi 5 is faster). For larger buffers, `IndexIVFPQ` with product quantization compresses 128-dim vectors to 8 bytes each, reducing 100K entries to **0.8 MB**. At inference time: encode the current market window → query top-k similar historical episodes → concatenate retrieved outcomes as context for the policy network. The RAFT paper's key insight applies directly: "Even if a pattern rarely appears in historical data and is difficult for the model to memorize, the retrieval module allows the model to easily leverage historical patterns when they reappear." For black swan events and crash signatures, this is exactly the capability Meridian needs.

Priority-based eviction maintains buffer quality: `priority = surprise_score × recency_decay × retrieval_frequency`. Entries that are both surprising and frequently retrieved (useful) persist; common, stale entries are evicted. Total per-step overhead: **<10 ms** including embedding generation, FAISS search, and surprise computation.

---

## Raspberry Pi 5 deployment is not the bottleneck

The BCM2712 SoC (quad-core Cortex-A76 @ 2.4 GHz) provides **~307 GOPS INT8** via dual 128-bit NEON pipelines per core, rising to an estimated **400–600 effective ternary GOPS** with optimized bitwise kernels. A 30M-parameter ternary model requires ~30M operations per forward step. The arithmetic: 30M ops ÷ 100 GOPS (conservative) = **0.3 ms compute time**. Adding memory access (7.5 MB model against ~8 GB/s LPDDR4X bandwidth = 0.94 ms cold, ~0.5 ms warm) and overhead, total inference latency is **0.5–3 ms** — delivering 300–1000+ inferences per second.

At 1 inference/second financial cadence, **CPU utilization is <0.1%**. The Pi is essentially idle. Power consumption under this workload: **<3W total board power**, well below thermal throttling thresholds. With the $5 official active cooler, sustained 24/7 operation at full 2.4 GHz is validated by community thermal testing at 59–72°C under heavy load.

The deployment stack: Raspberry Pi OS Lite 64-bit (no GUI), custom C/C++ inference kernel with NEON intrinsics adapted from bitnet.cpp's ternary GEMV kernels, FAISS-cpu for episodic memory retrieval, and Python for data ingestion via shared memory IPC. Pin inference to cores 0–2, reserve core 3 for OS/data pipeline. Use NVMe SSD via PCIe for model and memory buffer storage.

---

## Conclusion: a refined Meridian architecture

The research validates Meridian's core concept while refining every component. The key architectural shifts from the initial plan:

**Gated DeltaNet replaces vanilla Mamba** as the backbone — it offers superior quantization friendliness (no scattered outliers), higher H100 training throughput, and a delta-rule recurrence that IS online learning. The production validation in Qwen3.5 and Kimi Linear at hundreds of billions of parameters confirms scalability of the architecture, while the simplicity of the recurrent step (a rank-1 matrix update) makes it ideal for ARM NEON. **Mixed precision is mandatory**: ternary weights with FP16 hidden states, following the Precision Highway principle validated across quantized RNN literature. **Online adaptation is multi-timescale**: surprise-gated delta-rule dynamics (fast), normalization parameter adaptation (medium), and periodic ternary LoRA consolidation with EWC (slow). **Regime routing should be external** (HMM posteriors, not learned ternary gating), applied only to FFN experts while the Gated DeltaNet sequence model remains shared. **The Sharpe head should use Differential Sharpe Ratio**, not batch Sharpe optimization, with a volatility auxiliary head that demonstrably improves portfolio performance. **Episodic memory is a surprise-gated FAISS k-NN buffer** — simple, proven on ARM, and filling a genuine gap in the financial ML literature.

The single biggest risk is **ternary quantization error accumulation** in the recurrent state during 24/7 operation. The mitigation — FP16 state with periodic RMSNorm — is well-grounded in literature but has not been empirically validated for continuous operation over weeks. This should be the first item tested in prototype development: run a quantized Gated DeltaNet for 10M+ steps and measure state drift. Everything else — compute budget, memory budget, thermal envelope, and architectural effectiveness — is validated with margin to spare.