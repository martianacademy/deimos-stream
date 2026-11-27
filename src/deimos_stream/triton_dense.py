#!/usr/bin/env python3
"""
DeimosStream Fused CUDA Triton Dense INT8, Q4 Group-64 & SwiGLU Kernels
(`src/deimos_stream/triton_dense.py`)
=======================================================================
Provides high-bandwidth (`~190-220 GB/s` on Tesla T4 `sm_75`) dense linear and activation
kernels for `--backbone fp16` (`lm_head_int8`) and `--backbone q4` (INT8 `GatedDeltaNet` +
vectorized `uint32` Q4 `SelfAttention` / `shared_expert` / `lm_head`):
  - `triton_int8_gemv` & `quantize_2d_int8_torch`
  - `triton_q4_dense_linear` & `quantize_2d_q4_torch`
  - `triton_swiglu_gated`
"""

from __future__ import annotations

from typing import Optional, Tuple

from deimos_stream.cuda_kernels import cuda_int8_gemv, cuda_q4_dense_linear
from deimos_stream.quant_q4 import unpack_q4_torch

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False


def triton_kernels_available(device: "torch.device") -> bool:
    """Returns True when Triton CUDA fused kernels can run on `device`."""
    return bool(HAS_TORCH and HAS_TRITON and device.type == "cuda")


if HAS_TRITON:

    @triton.jit
    def _int8_gemv_kernel(
        x_ptr, w_ptr, s_ptr, out_ptr, vocab_size, in_dim,
        stride_xn, stride_wv, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0).to(tl.int64)
        pid_m = tl.program_id(1).to(tl.int64)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        row_mask = rows < vocab_size
        k_offs = tl.arange(0, BLOCK_K).to(tl.int64)

        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, in_dim, BLOCK_K):
            cols = k0 + k_offs
            k_mask = cols < in_dim
            x_v = tl.load(x_ptr + pid_n * stride_xn + cols, mask=k_mask, other=0.0).to(tl.float32)
            w_v = tl.load(
                w_ptr + rows[:, None] * stride_wv + cols[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0,
            ).to(tl.float32)
            acc += tl.sum(w_v * x_v[None, :], axis=1)

        sc = tl.load(s_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + pid_n * stride_on + rows, (acc * sc).to(out_ptr.dtype.element_ty), mask=row_mask)

    @triton.jit
    def _q4_dense_gemv_kernel(
        x_ptr, w32_ptr, s_ptr, b_ptr, out_ptr, out_dim, num_steps,
        stride_xn, stride_wr, stride_sr, stride_on,
        WORDS_G: tl.constexpr, WORDS_STEP: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < out_dim
        w_idx = tl.arange(0, WORDS_STEP)
        sub_g = w_idx // WORDS_G
        groups_step: tl.constexpr = WORDS_STEP // WORDS_G
        step_elems: tl.constexpr = WORDS_STEP * 8

        w_row_base = rows.to(tl.int64)[:, None] * stride_wr
        sb_row_base = rows.to(tl.int64) * stride_sr
        x_row_ptr = x_ptr + pid_n.to(tl.int64) * stride_xn

        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for st in range(0, num_steps):
            xg = x_row_ptr + st * step_elems + w_idx * 8
            x0 = tl.load(xg + 0).to(tl.float32)
            x1 = tl.load(xg + 1).to(tl.float32)
            x2 = tl.load(xg + 2).to(tl.float32)
            x3 = tl.load(xg + 3).to(tl.float32)
            x4 = tl.load(xg + 4).to(tl.float32)
            x5 = tl.load(xg + 5).to(tl.float32)
            x6 = tl.load(xg + 6).to(tl.float32)
            x7 = tl.load(xg + 7).to(tl.float32)
            x_wsum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7

            col_off = (st * WORDS_STEP + w_idx)[None, :]
            grp_off = (st * groups_step + sub_g)[None, :]
            w32 = tl.load(w32_ptr + w_row_base + col_off, mask=row_mask[:, None], other=0).to(tl.uint32)
            dw = (
                (w32 & 0xF).to(tl.float32) * x0[None, :]
                + ((w32 >> 4) & 0xF).to(tl.float32) * x1[None, :]
                + ((w32 >> 8) & 0xF).to(tl.float32) * x2[None, :]
                + ((w32 >> 12) & 0xF).to(tl.float32) * x3[None, :]
                + ((w32 >> 16) & 0xF).to(tl.float32) * x4[None, :]
                + ((w32 >> 20) & 0xF).to(tl.float32) * x5[None, :]
                + ((w32 >> 24) & 0xF).to(tl.float32) * x6[None, :]
                + ((w32 >> 28) & 0xF).to(tl.float32) * x7[None, :]
            )
            sc = tl.load(s_ptr + sb_row_base[:, None] + grp_off, mask=row_mask[:, None], other=0.0).to(tl.float32)
            bi = tl.load(b_ptr + sb_row_base[:, None] + grp_off, mask=row_mask[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(dw * sc + x_wsum[None, :] * bi, axis=1)

        tl.store(out_ptr + pid_n.to(tl.int64) * stride_on + rows, acc.to(out_ptr.dtype.element_ty), mask=row_mask)

    @triton.jit
    def _swiglu_gated_kernel(gu_ptr, gate_ptr, out_ptr, inter_dim, stride_gu, BLOCK_I: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_i = tl.program_id(1)
        cols = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        mask = cols < inter_dim
        base = gu_ptr + pid_n * stride_gu
        g = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(base + inter_dim + cols, mask=mask, other=0.0).to(tl.float32)
        act = (g * tl.sigmoid(g)) * u
        if gate_ptr is not None:
            act = act * tl.sigmoid(tl.load(gate_ptr + pid_n).to(tl.float32))
        tl.store(out_ptr + pid_n * inter_dim + cols, act.to(out_ptr.dtype.element_ty), mask=mask)


def triton_swiglu_gated(gu: "torch.Tensor", shared_gate_logit: Optional["torch.Tensor"] = None) -> "torch.Tensor":
    """Fuses `silu(gu[:, :I]) * gu[:, I:] * sigmoid(shared_gate_logit)` into a single Triton launch."""
    N, two_i = gu.shape
    inter = two_i // 2
    if not triton_kernels_available(gu.device):
        act = F.silu(gu[:, :inter]) * gu[:, inter:]
        return act * torch.sigmoid(shared_gate_logit) if shared_gate_logit is not None else act
    out = torch.empty((N, inter), device=gu.device, dtype=gu.dtype)
    block_i = 256
    _swiglu_gated_kernel[(N, triton.cdiv(inter, block_i))](
        gu, shared_gate_logit, out, inter, int(gu.stride(0)), BLOCK_I=block_i, num_warps=2
    )
    return out


def triton_int8_gemv(x_in: "torch.Tensor", w_i8: "torch.Tensor", scale_1d: "torch.Tensor") -> "torch.Tensor":
    """Fused INT8 linear GEMV `[..., D] @ [Out, D].T * scale[Out]` (~190+ GB/s on Tesla T4)."""
    assert HAS_TORCH
    orig_prefix = x_in.shape[:-1]
    in_dim = int(x_in.shape[-1])
    x_c = x_in.contiguous().view(-1, in_dim)
    N, out_dim = int(x_c.shape[0]), int(w_i8.shape[0])
    if N <= 16 and x_c.device.type == "cuda":
        fast = cuda_int8_gemv(x_c, w_i8, scale_1d)
        if fast is not None:
            return fast.view(*orig_prefix, out_dim)
    if N > 8 or not triton_kernels_available(x_c.device):
        sc_2d = scale_1d.view(-1, 1).to(x_c.dtype)
        return torch.cat(
            [x_c @ (w_i8[r0 : min(out_dim, r0 + 16384)].to(x_c.dtype) * sc_2d[r0 : min(out_dim, r0 + 16384)]).T
             for r0 in range(0, out_dim, 16384)],
            dim=-1,
        ).view(*orig_prefix, out_dim)

    sc_c = scale_1d.reshape(-1).contiguous()
    out = torch.empty((N, out_dim), device=x_c.device, dtype=x_c.dtype)
    block_m, block_k = 64, 128
    _int8_gemv_kernel[(N, triton.cdiv(out_dim, block_m))](
        x_c, w_i8, sc_c, out, out_dim, in_dim,
        int(x_c.stride(0)), int(w_i8.stride(0)), int(out.stride(0)),
        BLOCK_M=block_m, BLOCK_K=block_k, num_warps=4,
    )
    return out.view(*orig_prefix, out_dim)


def quantize_2d_int8_torch(weight_2d: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Quantizes `[Out, In]` FP16/BF16 tensor to symmetric per-row `(w_int8, scale_f16)`."""
    w_f32 = weight_2d.to(torch.float32)
    amax = w_f32.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 1e-12, amax / 127.0, torch.ones_like(amax))
    w_i8 = torch.round(w_f32 / scale).clamp(-127.0, 127.0).to(torch.int8).contiguous()
    s_f16 = torch.where(amax > 1e-12, scale, torch.zeros_like(scale)).to(torch.float16).contiguous()
    return w_i8, s_f16


def quantize_2d_q4_torch(
    weight_2d: "torch.Tensor", group_size: int = 64
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Quantizes `[Out, In]` FP16/BF16 tensor to `(qw_u8, scales_f16, biases_f16)`."""
    out_dim, in_dim = weight_2d.shape
    w_f32 = weight_2d.to(torch.float32).view(out_dim, in_dim // group_size, group_size)
    w_min, w_max = w_f32.amin(dim=-1), w_f32.amax(dim=-1)
    scales = ((w_max - w_min) / 15.0).to(torch.float16)
    biases = w_min.to(torch.float16)
    s_f32, b_f32 = scales.to(torch.float32).unsqueeze(-1), biases.to(torch.float32).unsqueeze(-1)
    nz = s_f32.abs() > 1e-12
    safe_s = torch.where(nz, s_f32, torch.ones_like(s_f32))
    q = torch.where(nz, torch.round((w_f32 - b_f32) / safe_s), torch.zeros_like(w_f32))
    q_u8 = q.clamp(0.0, 15.0).to(torch.uint8).view(out_dim, in_dim)
    return (q_u8[:, 0::2] | (q_u8[:, 1::2] << 4)).contiguous(), scales.contiguous(), biases.contiguous()


def triton_q4_dense_linear(
    x: "torch.Tensor", qw_u8: "torch.Tensor", qs_f16: "torch.Tensor", qb_f16: "torch.Tensor", group_size: int = 64
) -> "torch.Tensor":
    """Vectorized `uint32` Q4 dense linear `x @ W_q4.T` (`--backbone q4` on CUDA & CPU)."""
    orig_prefix = x.shape[:-1]
    in_dim = int(x.shape[-1])
    x_2d = x.contiguous().view(-1, in_dim)
    N, out_dim = int(x_2d.shape[0]), int(qw_u8.shape[0])

    if N <= 16 and x_2d.device.type == "cuda":
        fast = cuda_q4_dense_linear(x_2d, qw_u8, qs_f16, qb_f16, group_size=group_size)
        if fast is not None:
            return fast.view(*orig_prefix, out_dim)

    if triton_kernels_available(x_2d.device) and N <= 8 and group_size in (32, 64, 128):
        w32 = qw_u8.view(torch.int32)
        words_in = int(w32.shape[1])
        words_g = group_size // 8
        words_step = words_g
        for cand in (32, 16, 8, 4):
            if cand >= words_g and words_in >= cand and words_in % cand == 0:
                words_step = cand
                break
        out = torch.empty((N, out_dim), device=x_2d.device, dtype=x_2d.dtype)
        block_m = 16
        _q4_dense_gemv_kernel[(N, triton.cdiv(out_dim, block_m))](
            x_2d, w32, qs_f16, qb_f16, out, out_dim, words_in // words_step,
            int(x_2d.stride(0)), int(w32.stride(0)), int(qs_f16.stride(0)), int(out.stride(0)),
            WORDS_G=words_g, WORDS_STEP=words_step, BLOCK_M=block_m, num_warps=4,
        )
        return out.view(*orig_prefix, out_dim)

    chunks = [
        x_2d @ unpack_q4_torch(
            qw_u8[r0 : min(out_dim, r0 + 2048)].unsqueeze(0),
            qs_f16[r0 : min(out_dim, r0 + 2048)].unsqueeze(0),
            qb_f16[r0 : min(out_dim, r0 + 2048)].unsqueeze(0),
            group_size=group_size, out_dtype=x_2d.dtype,
        ).squeeze(0).T
        for r0 in range(0, out_dim, 2048)
    ]
    return (chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=-1)).view(*orig_prefix, out_dim)
