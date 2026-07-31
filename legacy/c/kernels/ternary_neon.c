/*
 * ternary_neon.c — ARM NEON ternary GEMV kernel
 * Section 9 (Quantization Strategy) of meridian-spec.md
 *
 * Compiled as a shared library and called from python/quantization/ternary_kernels.py:
 *   gcc -O3 -march=armv8-a+simd -shared -fPIC -o libternary.so c/kernels/ternary_neon.c
 *
 * =============================================================================
 * TERNARY WEIGHT FORMAT
 * =============================================================================
 *
 * Each weight w ∈ {-1, 0, +1} is packed as 2 bits:
 *   00 → -1
 *   01 →  0
 *   10 → +1
 *
 * A weight matrix W of shape (M, K) is stored as:
 *   - w_packed: uint8_t[(M * K + 3) / 4]   — 4 weights per byte
 *   - scale:    float16_t[M]                — one absmean scale per output row
 *
 * Quantization (done once at export, in Python):
 *   scale_i = mean(|W[i, :]|)
 *   w_ternary[i, j] = round(W[i, j] / scale_i).clamp(-1, 1)
 *
 * =============================================================================
 * FUNCTIONS TO IMPLEMENT
 * =============================================================================
 *
 * void ternary_gemv(
 *     const uint8_t* w_packed,   // ternary weight matrix (M × K), 2-bit packed
 *     const float16_t* x,        // input vector (K,), FP16
 *     const float16_t* scales,   // per-row scale factors (M,), FP16
 *     float16_t* out,            // output vector (M,), FP16
 *     int M, int K
 * );
 *
 * Computes: out[i] = scales[i] * Σ_j w_ternary[i,j] * x[j]
 *
 * ARM NEON strategy:
 *   1. Load 16 packed bytes (= 64 weights) with vld1q_u8
 *   2. Extract ternary values with bit masking (& 0x03) and shift
 *   3. Map 00→-1, 01→0, 10→+1 using lookup or conditional
 *   4. Accumulate dot product with vmull_s8 + vpaddlq_s16 + vpaddlq_s32
 *   5. Convert accumulator to FP16, multiply by scale, store
 *
 * Target performance: ~4 ms for full RA-GDN backbone (100 assets × 8 blocks)
 * on Raspberry Pi 5 Cortex-A76 @ 2.4 GHz.
 *
 * =============================================================================
 * ADDITIONAL FUNCTIONS
 * =============================================================================
 *
 * void pack_ternary(
 *     const float* w_fp32,       // full-precision weight matrix (M × K)
 *     uint8_t* w_packed,         // output: 2-bit packed (M × K + 3) / 4 bytes
 *     float16_t* scales,         // output: per-row scales (M,)
 *     int M, int K
 * );
 *
 * int ternary_gemv_latency_us(int M, int K);
 *   // Benchmark helper: returns measured latency in microseconds.
 *   // Used by `meridian check env` to validate Pi 5 performance.
 */

/* TODO: implement ternary GEMV kernel (Section 9 of meridian-spec.md) */
