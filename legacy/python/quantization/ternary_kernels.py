"""Python bindings for the ARM NEON ternary inference kernel.

The kernel is implemented in C at c/kernels/ternary_neon.c and compiled
as a shared library for use on Raspberry Pi 5.

Ternary packing format:
  Each weight w ∈ {-1, 0, +1} packed as 2 bits.
  4 weights per byte → 75% memory reduction vs FP16.
  Per-matrix scale factor stored in FP16.

Kernel operations exposed:
  ternary_gemv(W_packed, x_fp16, scale, out_fp16)
    — multiply ternary matrix W by FP16 vector x
    — uses ARM NEON SIMD (vld1q_s8, vmull, vpaddl, vcvt)
    — target: ~4 ms for full 100-asset RA-GDN backbone on Cortex-A76

  pack_ternary(W_fp32) → (W_packed, scale)
    — quantize and pack a float32 weight matrix
    — scale = mean(|W|), W_ternary = round(W / scale).clamp(-1, 1)

Build:
  gcc -O3 -march=armv8-a+simd -shared -fPIC -o libternary.so c/kernels/ternary_neon.c
"""

from __future__ import annotations

# TODO: implement ctypes/cffi bindings to libternary.so
# (Section 9 and c/kernels/ternary_neon.c of meridian-spec.md)
