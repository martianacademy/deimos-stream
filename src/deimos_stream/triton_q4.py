#!/usr/bin/env python3
"""
DeimosStream Fused CUDA Triton Paged 4-Bit (`Q4` Group-64) MoE Gathered Kernels
(`src/deimos_stream/triton_q4.py`)
===============================================================================
Warp-coalesced `WORDS_STEP = 32` (`32 × uint32 = 128 contiguous bytes = 256 weights = 4 groups`
per 32-thread warp, `BLOCK_M = 16, num_warps = 4` -> 1 exact warp per row, 100% 128-byte DRAM
burst coalescing, and only `2-8` runtime loop iterations with zero register spilling on Tesla T4):
  1. `_q4_gather_swiglu_kernel`: Fuses `gate` + `up` Q4 gathered GEMVs + `silu(gate) * up`.
  2. `_q4_gather_gemv_kernel`: Warp-coalesced Q4 gathered GEMV with inline `scores` weighting.
"""

from __future__ import annotations

from typing import Optional

from deimos_stream.cuda_kernels import cuda_paged_q4_swiglu
from deimos_stream.quant_q4 import unpack_q4_torch
from deimos_stream.triton_dense import (
    HAS_TORCH,
    HAS_TRITON,
    quantize_2d_int8_torch,
    quantize_2d_q4_torch,
    triton_int8_gemv,
    triton_kernels_available,
    triton_q4_dense_linear,
    triton_swiglu_gated,
)

if HAS_TORCH:
    import torch
    import torch.nn.functional as F
else:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

if HAS_TRITON:
    import triton
    import triton.language as tl
else:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


if HAS_TRITON:

    @triton.jit
    def _q4_gather_swiglu_kernel(
        x_ptr, w32_ptr, s_ptr, b_ptr, slots_ptr, out_ptr,
        inter_dim, num_steps, top_k,
        swiglu_limit, swiglu_alpha, swiglu_up_offset,
        stride_xn, stride_ws, stride_wr, stride_ss, stride_sr, stride_ob,
        HAS_CUSTOM_ACT: tl.constexpr,
        WORDS_G: tl.constexpr, WORDS_STEP: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        slot = tl.load(slots_ptr + pid_b).to(tl.int64)
        x_row = (pid_b // top_k).to(tl.int64)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < inter_dim
        rows_u = rows + inter_dim
        w_idx = tl.arange(0, WORDS_STEP)
        sub_g = w_idx // WORDS_G
        groups_step: tl.constexpr = WORDS_STEP // WORDS_G
        step_elems: tl.constexpr = WORDS_STEP * 8

        wg_base = slot * stride_ws + rows.to(tl.int64)[:, None] * stride_wr
        wu_base = slot * stride_ws + rows_u.to(tl.int64)[:, None] * stride_wr
        sbg_base = slot * stride_ss + rows.to(tl.int64) * stride_sr
        sbu_base = slot * stride_ss + rows_u.to(tl.int64) * stride_sr
        x_row_ptr = x_ptr + x_row * stride_xn

        acc_g = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M,), dtype=tl.float32)

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

            wg = tl.load(w32_ptr + wg_base + col_off, mask=row_mask[:, None], other=0).to(tl.uint32)
            dg = (
                (wg & 0xF).to(tl.float32) * x0[None, :]
                + ((wg >> 4) & 0xF).to(tl.float32) * x1[None, :]
                + ((wg >> 8) & 0xF).to(tl.float32) * x2[None, :]
                + ((wg >> 12) & 0xF).to(tl.float32) * x3[None, :]
                + ((wg >> 16) & 0xF).to(tl.float32) * x4[None, :]
                + ((wg >> 20) & 0xF).to(tl.float32) * x5[None, :]
                + ((wg >> 24) & 0xF).to(tl.float32) * x6[None, :]
                + ((wg >> 28) & 0xF).to(tl.float32) * x7[None, :]
            )
            sg = tl.load(s_ptr + sbg_base[:, None] + grp_off, mask=row_mask[:, None], other=0.0).to(tl.float32)
            bg = tl.load(b_ptr + sbg_base[:, None] + grp_off, mask=row_mask[:, None], other=0.0).to(tl.float32)
            acc_g += tl.sum(dg * sg + x_wsum[None, :] * bg, axis=1)

            wu = tl.load(w32_ptr + wu_base + col_off, mask=row_mask[:, None], other=0).to(tl.uint32)
            du = (
                (wu & 0xF).to(tl.float32) * x0[None, :]
                + ((wu >> 4) & 0xF).to(tl.float32) * x1[None, :]
                + ((wu >> 8) & 0xF).to(tl.float32) * x2[None, :]
                + ((wu >> 12) & 0xF).to(tl.float32) * x3[None, :]
                + ((wu >> 16) & 0xF).to(tl.float32) * x4[None, :]
                + ((wu >> 20) & 0xF).to(tl.float32) * x5[None, :]
                + ((wu >> 24) & 0xF).to(tl.float32) * x6[None, :]
                + ((wu >> 28) & 0xF).to(tl.float32) * x7[None, :]
            )
            su = tl.load(s_ptr + sbu_base[:, None] + grp_off, mask=row_mask[:, None], other=0.0).to(tl.float32)
            bu = tl.load(b_ptr + sbu_base[:, None] + grp_off, mask=row_mask[:, None], other=0.0).to(tl.float32)
            acc_u += tl.sum(du * su + x_wsum[None, :] * bu, axis=1)

        if HAS_CUSTOM_ACT:
            if swiglu_limit > 0.0:
                acc_g = tl.minimum(acc_g, swiglu_limit)
                acc_u = tl.minimum(tl.maximum(acc_u, -swiglu_limit), swiglu_limit)
            act = (acc_g * tl.sigmoid(swiglu_alpha * acc_g)) * (acc_u + swiglu_up_offset)
        else:
            act = (acc_g * tl.sigmoid(acc_g)) * acc_u
        tl.store(out_ptr + pid_b.to(tl.int64) * stride_ob + rows, act.to(out_ptr.dtype.element_ty), mask=row_mask)

    @triton.jit
    def _q4_gather_gemv_kernel(
        x_ptr, w32_ptr, s_ptr, b_ptr, slots_ptr, scores_ptr, out_ptr,
        out_dim, num_steps, top_k,
        stride_xn, stride_ws, stride_wr, stride_ss, stride_sr, stride_ob, stride_om,
        IS_SHARED_X: tl.constexpr, HAS_SCORES: tl.constexpr,
        WORDS_G: tl.constexpr, WORDS_STEP: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        slot = tl.load(slots_ptr + pid_b).to(tl.int64)
        x_row = (pid_b // top_k).to(tl.int64) if IS_SHARED_X else pid_b.to(tl.int64)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < out_dim
        w_idx = tl.arange(0, WORDS_STEP)
        sub_g = w_idx // WORDS_G
        groups_step: tl.constexpr = WORDS_STEP // WORDS_G
        step_elems: tl.constexpr = WORDS_STEP * 8

        w_row_base = slot * stride_ws + rows.to(tl.int64)[:, None] * stride_wr
        sb_row_base = slot * stride_ss + rows.to(tl.int64) * stride_sr
        x_row_ptr = x_ptr + x_row * stride_xn

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

        if HAS_SCORES:
            acc = acc * tl.load(scores_ptr + pid_b).to(tl.float32)
        tl.store(
            out_ptr + pid_b.to(tl.int64) * stride_ob + rows * stride_om,
            acc.to(out_ptr.dtype.element_ty),
            mask=row_mask,
        )


def _pick_words_step(words_in: int, words_g: int) -> int:
    for cand in (32, 16, 8, 4):
        if cand >= words_g and words_in >= cand and words_in % cand == 0:
            return cand
    return words_g


def triton_q4_gather_gemv(
    x: "torch.Tensor",
    w_pool_u8: "torch.Tensor",
    s_pool_f16: "torch.Tensor",
    b_pool_f16: "torch.Tensor",
    flat_slots: "torch.Tensor",
    top_k: int,
    group_size: int = 64,
    is_shared_x: bool = True,
    scores_flat: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Warp-coalesced 4-bit gathered GEMV reading 128-byte `uint32` sectors from arena pools."""
    assert HAS_TORCH and HAS_TRITON
    if group_size not in (32, 64, 128):
        raise ValueError(f"Unsupported Triton Q4 group_size={group_size}")

    x_c = x.contiguous()
    slots_c = flat_slots.contiguous()
    w32 = w_pool_u8.view(torch.int32)
    B = int(slots_c.shape[0])
    _, out_dim, words_in = w32.shape
    words_g = group_size // 8
    words_step = _pick_words_step(int(words_in), words_g)
    num_steps = int(words_in) // words_step

    out = torch.empty((B, int(out_dim)), device=x_c.device, dtype=x_c.dtype)
    block_m = 16
    sc_c = scores_flat.contiguous() if scores_flat is not None else x_c
    _q4_gather_gemv_kernel[(B, triton.cdiv(int(out_dim), block_m))](
        x_c, w32, s_pool_f16, b_pool_f16, slots_c, sc_c, out,
        int(out_dim), num_steps, int(top_k),
        int(x_c.stride(0)), int(w32.stride(0)), int(w32.stride(1)),
        int(s_pool_f16.stride(0)), int(s_pool_f16.stride(1)),
        int(out.stride(0)), int(out.stride(1)),
        IS_SHARED_X=bool(is_shared_x), HAS_SCORES=(scores_flat is not None),
        WORDS_G=words_g, WORDS_STEP=words_step, BLOCK_M=block_m, num_warps=4,
    )
    return out


def run_paged_q4_swiglu(
    x_chunk: "torch.Tensor",
    gu_w_pool: "torch.Tensor",
    gu_s_pool: "torch.Tensor",
    gu_b_pool: "torch.Tensor",
    dn_w_pool: "torch.Tensor",
    dn_s_pool: "torch.Tensor",
    dn_b_pool: "torch.Tensor",
    flat_slots: "torch.Tensor",
    scores: "torch.Tensor",
    group_size: int = 64,
    swiglu_limit: float = 0.0,
    swiglu_alpha: float = 1.0,
    swiglu_up_offset: float = 0.0,
) -> "torch.Tensor":
    """Executes paged 4-bit SwiGLU MoE in 2 warp-coalesced `WORDS_STEP=32` Triton launches."""
    N, D = x_chunk.shape
    K = int(scores.shape[1])
    dt = x_chunk.dtype
    lim = float(swiglu_limit)
    alp = float(swiglu_alpha)
    uoff = float(swiglu_up_offset)
    custom_act = lim > 0.0 or alp != 1.0 or uoff != 0.0

    if N <= 16 and x_chunk.device.type == "cuda":
        fast = cuda_paged_q4_swiglu(
            x_chunk, gu_w_pool, gu_s_pool, gu_b_pool,
            dn_w_pool, dn_s_pool, dn_b_pool,
            flat_slots, scores, group_size=group_size,
            swiglu_limit=lim, swiglu_alpha=alp, swiglu_up_offset=uoff,
        )
        if fast is not None:
            return fast

    if triton_kernels_available(x_chunk.device) and group_size in (32, 64, 128):
        x_c = x_chunk.contiguous()
        slots_c = flat_slots.contiguous()
        B = int(slots_c.shape[0])
        gu_w32 = gu_w_pool.view(torch.int32)
        _, two_i, words_d = gu_w32.shape
        inter = int(two_i // 2)
        words_g = group_size // 8
        words_step = _pick_words_step(int(words_d), words_g)
        block_m = 16
        act = torch.empty((B, inter), device=x_c.device, dtype=dt)
        _q4_gather_swiglu_kernel[(B, triton.cdiv(inter, block_m))](
            x_c, gu_w32, gu_s_pool, gu_b_pool, slots_c, act,
            inter, int(words_d) // words_step, K,
            lim, alp, uoff,
            int(x_c.stride(0)), int(gu_w32.stride(0)), int(gu_w32.stride(1)),
            int(gu_s_pool.stride(0)), int(gu_s_pool.stride(1)), int(act.stride(0)),
            HAS_CUSTOM_ACT=custom_act,
            WORDS_G=words_g, WORDS_STEP=words_step, BLOCK_M=block_m, num_warps=4,
        )
        dn_weighted = triton_q4_gather_gemv(
            act, dn_w_pool, dn_s_pool, dn_b_pool, slots_c,
            top_k=K, group_size=group_size, is_shared_x=False,
            scores_flat=scores.reshape(-1),
        )
        return dn_weighted.view(N, K, D).sum(dim=1)

    gu_w = unpack_q4_torch(
        gu_w_pool[flat_slots], gu_s_pool[flat_slots], gu_b_pool[flat_slots],
        group_size=group_size, out_dtype=dt,
    )
    x_rep = x_chunk.unsqueeze(1).expand(N, K, D).reshape(N * K, D, 1)
    gate, up = torch.bmm(gu_w, x_rep).squeeze(-1).chunk(2, dim=-1)
    if custom_act:
        gf, uf = gate.float(), up.float()
        if lim > 0.0:
            gf = gf.clamp(max=lim)
            uf = uf.clamp(min=-lim, max=lim)
        act_cpu = ((gf * torch.sigmoid(alp * gf)) * (uf + uoff)).to(dt).unsqueeze(-1)
    else:
        act_cpu = (F.silu(gate) * up).unsqueeze(-1)
    dn_w = unpack_q4_torch(
        dn_w_pool[flat_slots], dn_s_pool[flat_slots], dn_b_pool[flat_slots],
        group_size=group_size, out_dtype=dt,
    )
    dn_out = torch.bmm(dn_w, act_cpu).squeeze(-1).view(N, K, D)
    return (dn_out * scores.to(dt).unsqueeze(-1)).sum(dim=1)


__all__ = [
    "quantize_2d_int8_torch",
    "quantize_2d_q4_torch",
    "run_paged_q4_swiglu",
    "triton_int8_gemv",
    "triton_kernels_available",
    "triton_q4_dense_linear",
    "triton_q4_gather_gemv",
    "triton_swiglu_gated",
]
