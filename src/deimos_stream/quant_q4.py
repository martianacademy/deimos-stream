#!/usr/bin/env python3
"""
DeimosStream Universal 4-Bit Affine Group-64 Quantizer & Unpacker (`src/deimos_stream/quant_q4.py`)
===================================================================================================
Guarantees 100% binary leaf compatibility (`[gu_w || gu_s || gu_b || dn_w || dn_s || dn_b]`)
between Apple Silicon MLX (`mx.quantize` / `mx.gather_qmm`) and NVIDIA CUDA / Linux PyTorch:
  - Packing: 8 unsigned 4-bit nibbles `[0..15]` per little-endian `uint32` (`2` nibbles per `uint8`:
    low nibble `w_u8 & 0x0F`, high nibble `(w_u8 >> 4) & 0x0F`).
  - Affine Reconstruction: `w_hat = float16(float32(q) * float32(scale) + float32(bias))`,
    yielding `0.0` exact FP16 parity with `mx.dequantize(..., group_size=64, bits=4)`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


def quantize_fp16_np_to_q4(
    weight_fp16: np.ndarray, group_size: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantizes a 2D `[M, N]` or 3D `[E, M, N]` FP16 NumPy array into MLX-compatible
    4-bit affine `(qw_u32, scales_f16, biases_f16)` without requiring `mlx`.
    """
    orig_shape = weight_fp16.shape
    if orig_shape[-1] % group_size != 0 or group_size % 8 != 0:
        raise ValueError(f"Last dim {orig_shape[-1]} must be divisible by group_size={group_size}")

    w_f32 = np.asarray(weight_fp16, dtype=np.float32)
    grouped = w_f32.reshape(*orig_shape[:-1], orig_shape[-1] // group_size, group_size)

    w_min = np.min(grouped, axis=-1)
    w_max = np.max(grouped, axis=-1)
    scales_f16 = ((w_max - w_min) / 15.0).astype(np.float16)
    biases_f16 = w_min.astype(np.float16)

    s_f32 = scales_f16.astype(np.float32)[..., None]
    b_f32 = biases_f16.astype(np.float32)[..., None]
    safe_s = np.where(np.abs(s_f32) > 1e-12, s_f32, 1.0)
    q = np.where(np.abs(s_f32) > 1e-12, np.round((grouped - b_f32) / safe_s), 0.0)
    q_u8 = np.clip(q, 0.0, 15.0).astype(np.uint8).reshape(*orig_shape)

    lo = q_u8[..., 0::2]
    hi = q_u8[..., 1::2]
    packed_u8 = np.ascontiguousarray(lo | (hi << 4), dtype=np.uint8)
    qw_u32 = packed_u8.view(np.uint32).reshape(*orig_shape[:-1], orig_shape[-1] // 8)
    return qw_u32, scales_f16, biases_f16


def dequantize_q4_np_to_fp16(
    qw_u32: np.ndarray, scales_f16: np.ndarray, biases_f16: np.ndarray, group_size: int = 64
) -> np.ndarray:
    """Dequantizes `(qw_u32, scales_f16, biases_f16)` to FP16 NumPy with exact F32 FMA rounding."""
    prefix = qw_u32.shape[:-1]
    in_dim = qw_u32.shape[-1] * 8
    packed_u8 = np.ascontiguousarray(qw_u32).view(np.uint8).reshape(*prefix, in_dim // 2)
    nibbles = np.empty((*prefix, in_dim), dtype=np.uint8)
    nibbles[..., 0::2] = packed_u8 & 0x0F
    nibbles[..., 1::2] = (packed_u8 >> 4) & 0x0F
    grouped = nibbles.reshape(*prefix, in_dim // group_size, group_size).astype(np.float32)
    deq = grouped * scales_f16.astype(np.float32)[..., None] + biases_f16.astype(np.float32)[..., None]
    return deq.astype(np.float16).reshape(*prefix, in_dim)


def unpack_q4_torch(
    packed_u8: "torch.Tensor",
    scales_f16: "torch.Tensor",
    biases_f16: "torch.Tensor",
    group_size: int = 64,
    out_dtype: "torch.dtype" = None,  # type: ignore[assignment]
) -> "torch.Tensor":
    """
    Fast vectorized GPU/CPU unpacking of `[K, out_dim, in_dim // 2]` `uint8` 4-bit weights.
    Produces `[K, out_dim, in_dim]` in `out_dtype` (defaults to `scales_f16.dtype`).
    """
    assert torch is not None
    target_dtype = out_dtype if out_dtype is not None else scales_f16.dtype
    k_dim, out_dim, half_in = packed_u8.shape
    in_dim = half_in * 2
    n_groups = in_dim // group_size

    nibbles = torch.stack([packed_u8 & 0x0F, (packed_u8 >> 4) & 0x0F], dim=-1).reshape(
        k_dim, out_dim, n_groups, group_size
    )
    deq = nibbles.to(torch.float32) * scales_f16.to(torch.float32).unsqueeze(-1) + biases_f16.to(
        torch.float32
    ).unsqueeze(-1)
    return deq.to(target_dtype).reshape(k_dim, out_dim, in_dim)
