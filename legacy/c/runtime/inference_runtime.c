/*
 * inference_runtime.c — Raspberry Pi 5 inference runtime glue
 * Section 14 (Inference Runtime on Pi 5) of meridian-spec.md
 *
 * This module sits between the Python inference pipeline
 * (python/runtime/inference_pipeline.py) and the low-level NEON kernel
 * (c/kernels/ternary_neon.c). It is compiled into a shared library:
 *   gcc -O3 -march=armv8-a+simd -shared -fPIC \
 *       -o libruntime.so c/runtime/inference_runtime.c c/kernels/ternary_neon.c
 *
 * =============================================================================
 * RESPONSIBILITIES
 * =============================================================================
 *
 * 1. MODEL STATE MANAGEMENT
 *    Owns all persistent recurrent state in a single contiguous allocation:
 *      - GDN hidden states: 100 assets × 8 layers × D² (FP16)   ~75.5 MB
 *      - Temporal cascade states: 3 levels × 100 × D²            ~16.4 MB
 *      - Conv1d caches                                            ~ 3.9 MB
 *    State is updated in-place each forward pass (O(1) per asset per layer).
 *    Persisted to disk on SIGTERM; restored on startup.
 *
 * 2. FORWARD PASS ORCHESTRATION
 *    Called once per minute by the Python pipeline with pre-assembled features:
 *      int meridian_forward(
 *          const float16_t* per_asset_features,  // 100 × 288, FP16
 *          const float16_t* market_features,     // 96, FP16
 *          const float16_t* portfolio_state,     // 201, FP16
 *          const float16_t* calendar_features,   // 20, FP16
 *          const EventToken* events,             // variable length, may be NULL
 *          int n_events,
 *          float16_t* weights_out,               // 100, FP16 — allocation weights
 *          float*     surprise_out,              // scalar aggregate surprise
 *          uint8_t    active_head                // 0=Meridian-R, 1=Meridian-S
 *      );
 *    Returns 0 on success, non-zero on error.
 *
 * 3. MULTI-CORE SCHEDULING (Pi 5 — 4 × Cortex-A76)
 *    - Core 0–2: parallel RA-GDN inference across 100 assets (33-34 assets each)
 *    - Core 3:   data pipeline (async, runs concurrently with inference)
 *    Thread affinity set via pthread_setaffinity_np().
 *
 * 4. MEMORY LAYOUT
 *    All model weights loaded once at startup into a locked mmap() region
 *    (mlock to prevent swapping). Weight layout:
 *      - Ternary weights (2-bit packed): loaded by ternary_neon.c routines
 *      - FP16 weights (heads, norms):    raw float16_t arrays
 *      - INT8 weights (regime, MarketReader): int8_t arrays with per-tensor scale
 *
 * 5. LATENCY BUDGET (per-minute, total ~7 ms for inference cores)
 *    Input projection + VSN:       0.5 ms
 *    Regime detection:             0.1 ms
 *    RA-GDN × 8 blocks × 100:     4.0 ms   ← ternary_gemv() hot path
 *    FMoE × 3 blocks × 100:       1.0 ms
 *    Event token processing:       0.3 ms
 *    Temporal cascade:             0.2 ms
 *    Cross-asset attention (110t): 0.5 ms
 *    FAISS episodic memory:        0.5 ms
 *    Output head:                  0.05 ms
 *
 * =============================================================================
 * EVENT TOKEN STRUCT
 * =============================================================================
 *
 * typedef struct {
 *     uint8_t  event_type;        // 0–17, see EventType enum in spec Section 6.4
 *     uint8_t  asset_id;          // 0–99
 *     float    magnitude;         // event magnitude (e.g., surprise %, dollar value)
 *     uint32_t minutes_since;     // minutes elapsed since event occurred
 * } EventToken;
 *
 * =============================================================================
 * DRIFT MONITORING
 * =============================================================================
 *
 * void meridian_get_state_norms(float* norms_out, int n);
 *   // Returns L2 norm of each asset's GDN hidden state (100 values).
 *   // Called by python/monitoring/monitoring.py for drift detection (Risk #1).
 *   // Alert condition: monotonically increasing norms over 5+ consecutive days.
 */

/* TODO: implement Pi 5 inference runtime (Section 14 of meridian-spec.md) */
