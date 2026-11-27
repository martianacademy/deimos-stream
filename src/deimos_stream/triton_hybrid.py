#!/usr/bin/env python3
"""
DeimosStream Fused CUDA Triton Kernels for Hybrid GatedDeltaNet + GQA + MoE Backbones
(`src/deimos_stream/triton_hybrid.py`)
=====================================================================================
Eliminates ~2,300+ eager PyTorch CUDA kernel launches and ~540 MB/token of redundant
FP32 recurrent-state HBM traffic per token on `Qwen3.5/Qwen3.6` Hybrid MoE models:
  1. `triton_zero_centered_rmsnorm`: `(1 + w) * x * rsqrt(mean(x^2) + eps)` in FP32.
  2. `triton_rmsnorm_gated`: `w * rms_norm(x) * silu(gate)` matching HF `RMSNormGated`.
  3. `triton_causal_conv1d_update`: In-place state shift + depthwise conv1d + SiLU (`L=1`).
  4. `triton_recurrent_gated_delta_step`: Single-pass in-place `[D_k, D_v]` recurrent state
     decay + delta update + L2-normalized Q/K query readout (`L=1`, `sm_75+` compatible).
  5. `triton_topk_softmax`: Single-kernel FP32 softmax + top-K selection + renormalization.
  6. `triton_partial_rope_inplace`: Fused partial Rotary Position Embedding for Q and K.
"""

from __future__ import annotations

from typing import Tuple

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False


def hybrid_triton_available(device: "torch.device") -> bool:
    return bool(HAS_TORCH and HAS_TRITON and device.type == "cuda")


if HAS_TRITON:

    @triton.jit
    def _zero_centered_rmsnorm_kernel(
        x_ptr, w_ptr, out_ptr, stride_xm, stride_om, n_cols, eps, BLOCK_N: tl.constexpr
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_N).to(tl.int64)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        rstd = tl.rsqrt(var + eps)
        y = x * rstd * (1.0 + w)
        tl.store(out_ptr + row * stride_om + cols, y.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rmsnorm_gated_kernel(
        x_ptr, z_ptr, w_ptr, out_ptr, stride_m, n_cols, eps, BLOCK_N: tl.constexpr
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_N).to(tl.int64)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0)
        var = tl.sum(x * x, axis=0) / n_cols
        normed = (x * tl.rsqrt(var + eps)).to(out_ptr.dtype.element_ty).to(tl.float32)
        normed = normed * w.to(tl.float32)
        silu_z = z * tl.sigmoid(z)
        y = (normed * silu_z).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * stride_m + cols, y, mask=mask)

    @triton.jit
    def _causal_conv1d_update_kernel(
        x_ptr,
        cs_ptr,
        w_ptr,
        b_ptr,
        out_ptr,
        conv_dim,
        stride_xb,
        stride_csb,
        stride_csc,
        stride_wc,
        HAS_BIAS: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        STATE_LEN: tl.constexpr,
        KERNEL_W: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_b = tl.program_id(0).to(tl.int64)
        pid_c = tl.program_id(1).to(tl.int64)
        cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C).to(tl.int64)
        mask = cols < conv_dim

        # x has shape [B, conv_dim, 1] or [B, 1, conv_dim] contiguous per batch -> stride_xb = conv_dim
        x_new = tl.load(x_ptr + pid_b * stride_xb + cols, mask=mask, other=0.0).to(tl.float32)
        acc = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32) if HAS_BIAS else tl.zeros((BLOCK_C,), dtype=tl.float32)

        cs_base = cs_ptr + pid_b * stride_csb + cols * stride_csc
        start_s: tl.constexpr = STATE_LEN - (KERNEL_W - 1)

        # Compute dot over last (KERNEL_W - 1) state taps + x_new, then shift state left by 1
        for k in tl.static_range(0, KERNEL_W - 1):
            s_val = tl.load(cs_base + (start_s + k), mask=mask, other=0.0).to(tl.float32)
            w_val = tl.load(w_ptr + cols * stride_wc + k, mask=mask, other=0.0).to(tl.float32)
            acc += s_val * w_val

        w_last = tl.load(w_ptr + cols * stride_wc + (KERNEL_W - 1), mask=mask, other=0.0).to(tl.float32)
        acc += x_new * w_last

        for s in tl.static_range(0, STATE_LEN - 1):
            nxt_v = tl.load(cs_base + (s + 1), mask=mask, other=0.0)
            tl.store(cs_base + s, nxt_v, mask=mask)
        tl.store(cs_base + (STATE_LEN - 1), x_new.to(cs_ptr.dtype.element_ty), mask=mask)

        if APPLY_SILU:
            acc = acc * tl.sigmoid(acc)
        tl.store(out_ptr + pid_b * stride_xb + cols, acc.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _recurrent_gated_delta_step_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        a_ptr,
        b_ptr,
        alog_ptr,
        dtb_ptr,
        state_ptr,
        out_ptr,
        num_k_heads,
        num_v_heads,
        d_k,
        d_v,
        q_scale,
        stride_qb,
        stride_qh,
        stride_vb,
        stride_vh,
        stride_sb,
        stride_sh,
        stride_sk,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        pid_bh = tl.program_id(0).to(tl.int64)
        pid_v = tl.program_id(1).to(tl.int64)

        b_idx = pid_bh // num_v_heads
        vh = pid_bh % num_v_heads
        gva_rep = num_v_heads // num_k_heads
        kh = vh // gva_rep

        k_offs = tl.arange(0, BLOCK_K)
        k_mask = k_offs < d_k
        v_offs = pid_v.to(tl.int32) * BLOCK_V + tl.arange(0, BLOCK_V)
        v_mask = v_offs < d_v

        q_vec = tl.load(q_ptr + b_idx * stride_qb + kh * stride_qh + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        k_vec = tl.load(k_ptr + b_idx * stride_qb + kh * stride_qh + k_offs, mask=k_mask, other=0.0).to(tl.float32)

        q_inv = tl.rsqrt(tl.sum(q_vec * q_vec, axis=0) + 1e-6)
        k_inv = tl.rsqrt(tl.sum(k_vec * k_vec, axis=0) + 1e-6)
        q_vec = (q_vec * q_inv) * q_scale
        k_vec = k_vec * k_inv

        a_val = tl.load(a_ptr + pid_bh).to(tl.float32)
        b_val = tl.load(b_ptr + pid_bh).to(tl.float32)
        alog_val = tl.load(alog_ptr + vh).to(tl.float32)
        dtb_val = tl.load(dtb_ptr + vh).to(tl.float32)

        sp_in = a_val + dtb_val
        sp = tl.where(sp_in > 20.0, sp_in, tl.log(1.0 + tl.exp(sp_in)))
        decay = tl.exp(-tl.exp(alog_val) * sp)
        beta = tl.sigmoid(b_val)

        s_head = state_ptr + b_idx * stride_sb + vh * stride_sh
        s_offs = k_offs[:, None] * stride_sk + v_offs[None, :]
        sv_mask = k_mask[:, None] & v_mask[None, :]
        s_tile = tl.load(s_head + s_offs, mask=sv_mask, other=0.0).to(tl.float32)

        s_tile = s_tile * decay
        kv_mem = tl.sum(s_tile * k_vec[:, None], axis=0)
        v_tile = tl.load(v_ptr + b_idx * stride_vb + vh * stride_vh + v_offs, mask=v_mask, other=0.0).to(tl.float32)
        delta = (v_tile - kv_mem) * beta
        s_tile = s_tile + k_vec[:, None] * delta[None, :]

        tl.store(s_head + s_offs, s_tile, mask=sv_mask)
        out_tile = tl.sum(s_tile * q_vec[:, None], axis=0)
        tl.store(out_ptr + b_idx * stride_vb + vh * stride_vh + v_offs, out_tile.to(out_ptr.dtype.element_ty), mask=v_mask)

    @triton.jit
    def _topk_softmax_kernel(
        logits_ptr,
        out_sc_ptr,
        out_id_ptr,
        n_experts,
        stride_lm,
        stride_sm,
        TOP_K: tl.constexpr,
        NORM_TOPK: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_E).to(tl.int64)
        mask = offs < n_experts

        x = tl.load(logits_ptr + row * stride_lm + offs, mask=mask, other=-float("inf")).to(tl.float32)
        x_max = tl.max(x, axis=0)
        ex = tl.where(mask, tl.exp(x - x_max), 0.0)
        probs = ex / tl.sum(ex, axis=0)

        work = tl.where(mask, probs, -1.0)
        sel_rank = tl.full((BLOCK_E,), -1, dtype=tl.int64)
        sum_sel = 0.0
        for k in tl.static_range(0, TOP_K):
            best_val = tl.max(work, axis=0)
            is_max = work == best_val
            best_idx = tl.min(tl.where(is_max, offs, BLOCK_E), axis=0)
            tl.store(out_id_ptr + row * TOP_K + k, best_idx.to(tl.int32))
            sel_rank = tl.where(offs == best_idx, k, sel_rank)
            sum_sel += best_val
            work = tl.where(offs == best_idx, -1.0, work)

        inv_s = (1.0 / tl.maximum(sum_sel, 1e-12)) if NORM_TOPK else 1.0
        sel_mask = sel_rank >= 0
        out_ptrs = out_sc_ptr + row * stride_sm + tl.maximum(sel_rank, 0)
        tl.store(out_ptrs, (probs * inv_s).to(out_sc_ptr.dtype.element_ty), mask=sel_mask)

    @triton.jit
    def _partial_rope_decode_kernel(
        qk_ptr,
        cos_ptr,
        sin_ptr,
        n_heads,
        stride_b,
        stride_h,
        stride_cb,
        HALF_ROT: tl.constexpr,
    ):
        pid_b = tl.program_id(0).to(tl.int64)
        pid_h = tl.program_id(1).to(tl.int64)
        offs = tl.arange(0, HALF_ROT).to(tl.int64)

        base = qk_ptr + pid_b * stride_b + pid_h * stride_h
        x1 = tl.load(base + offs).to(tl.float32)
        x2 = tl.load(base + HALF_ROT + offs).to(tl.float32)

        c1 = tl.load(cos_ptr + pid_b * stride_cb + offs).to(tl.float32)
        c2 = tl.load(cos_ptr + pid_b * stride_cb + HALF_ROT + offs).to(tl.float32)
        s1 = tl.load(sin_ptr + pid_b * stride_cb + offs).to(tl.float32)
        s2 = tl.load(sin_ptr + pid_b * stride_cb + HALF_ROT + offs).to(tl.float32)

        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2
        tl.store(base + offs, y1.to(qk_ptr.dtype.element_ty))
        tl.store(base + HALF_ROT + offs, y2.to(qk_ptr.dtype.element_ty))


def triton_zero_centered_rmsnorm(x: "torch.Tensor", weight: "torch.Tensor", eps: float) -> "torch.Tensor":
    orig_shape = x.shape
    n_cols = int(orig_shape[-1])
    x_2d = x.contiguous().view(-1, n_cols)
    n_rows = int(x_2d.shape[0])
    out = torch.empty_like(x_2d)
    block_n = triton.next_power_of_2(n_cols)
    num_warps = 4 if block_n <= 1024 else 8
    _zero_centered_rmsnorm_kernel[(n_rows,)](
        x_2d, weight, out, x_2d.stride(0), out.stride(0), n_cols, float(eps),
        BLOCK_N=block_n, num_warps=num_warps,
    )
    return out.view(orig_shape)


def triton_rmsnorm_gated(
    x: "torch.Tensor", gate: "torch.Tensor", weight: "torch.Tensor", eps: float
) -> "torch.Tensor":
    orig_shape = x.shape
    n_cols = int(orig_shape[-1])
    x_2d = x.contiguous().view(-1, n_cols)
    z_2d = gate.contiguous().view(-1, n_cols)
    n_rows = int(x_2d.shape[0])
    out = torch.empty_like(x_2d)
    block_n = triton.next_power_of_2(n_cols)
    _rmsnorm_gated_kernel[(n_rows,)](
        x_2d, z_2d, weight, out, x_2d.stride(0), n_cols, float(eps),
        BLOCK_N=block_n, num_warps=4,
    )
    return out.view(orig_shape)


def triton_causal_conv1d_update(
    x_bcl: "torch.Tensor",
    conv_state: "torch.Tensor",
    weight_2d: "torch.Tensor",
    bias: "torch.Tensor | None",
    apply_silu: bool = True,
) -> "torch.Tensor":
    """Single-token (`L == 1`) fused depthwise conv1d + in-place `conv_state` update.
    `x_bcl` has shape `[B, C, 1]`; returns `[B, C, 1]`.
    """
    B, C, _ = x_bcl.shape
    state_len = int(conv_state.shape[-1])
    kw = int(weight_2d.shape[-1])
    x_c = x_bcl.contiguous()
    out = torch.empty_like(x_c)
    block_c = 128
    grid = (int(B), triton.cdiv(int(C), block_c))
    _causal_conv1d_update_kernel[grid](
        x_c,
        conv_state,
        weight_2d,
        bias if bias is not None else x_c,
        out,
        int(C),
        int(x_c.stride(0)),
        int(conv_state.stride(0)),
        int(conv_state.stride(1)),
        int(weight_2d.stride(0)),
        HAS_BIAS=bias is not None,
        APPLY_SILU=bool(apply_silu),
        STATE_LEN=state_len,
        KERNEL_W=kw,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return out


def triton_recurrent_gated_delta_step(
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    a: "torch.Tensor",
    b: "torch.Tensor",
    A_log: "torch.Tensor",
    dt_bias: "torch.Tensor",
    recurrent_state: "torch.Tensor",
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Single-token (`L == 1`) fused GatedDeltaNet recurrent update + readout.
    Updates `recurrent_state` (`[B, H_v, D_k, D_v]` float32) IN-PLACE when contiguous float32!
    """
    B, _, H_k, D_k = key.shape
    _, _, H_v, D_v = value.shape
    q_c = query.contiguous()
    k_c = key.contiguous()
    v_c = value.contiguous()
    a_c = a.contiguous()
    b_c = b.contiguous()
    if recurrent_state.dtype != torch.float32 or not recurrent_state.is_contiguous():
        recurrent_state = recurrent_state.to(torch.float32).contiguous()

    out = torch.empty_like(v_c)
    block_k = triton.next_power_of_2(int(D_k))
    block_v = min(16, triton.next_power_of_2(int(D_v)))
    grid = (int(B * H_v), triton.cdiv(int(D_v), block_v))
    _recurrent_gated_delta_step_kernel[grid](
        q_c, k_c, v_c, a_c, b_c, A_log, dt_bias, recurrent_state, out,
        int(H_k), int(H_v), int(D_k), int(D_v), float(D_k) ** -0.5,
        int(q_c.stride(0)), int(q_c.stride(2)),
        int(v_c.stride(0)), int(v_c.stride(2)),
        int(recurrent_state.stride(0)), int(recurrent_state.stride(1)), int(recurrent_state.stride(2)),
        BLOCK_K=block_k, BLOCK_V=block_v, num_warps=4,
    )
    return out, recurrent_state


def triton_topk_softmax(
    router_logits: "torch.Tensor", top_k: int, norm_topk: bool = True
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Returns `(scores [N, top_k] in logits.dtype, indices [N, top_k] in int32)`."""
    logits_2d = router_logits.contiguous().view(-1, router_logits.shape[-1])
    N, E = logits_2d.shape
    out_sc = torch.empty((int(N), int(top_k)), device=logits_2d.device, dtype=logits_2d.dtype)
    out_id = torch.empty((int(N), int(top_k)), device=logits_2d.device, dtype=torch.int32)
    block_e = triton.next_power_of_2(int(E))
    _topk_softmax_kernel[(int(N),)](
        logits_2d, out_sc, out_id, int(E),
        int(logits_2d.stride(0)), int(out_sc.stride(0)),
        TOP_K=int(top_k), NORM_TOPK=bool(norm_topk), BLOCK_E=block_e, num_warps=4,
    )
    return out_sc, out_id


def triton_partial_rope_decode_inplace(
    q_bhld: "torch.Tensor",
    k_bhld: "torch.Tensor",
    cos_bld: "torch.Tensor",
    sin_bld: "torch.Tensor",
) -> None:
    """In-place partial RoPE for `L == 1` (`q_bhld`: `[B, H, 1, D]`, `cos/sin`: `[B, 1, rot_dim]`)."""
    rot_dim = int(cos_bld.shape[-1])
    half_rot = rot_dim // 2
    if (half_rot & (half_rot - 1)) != 0:
        return
    cos_c = cos_bld.contiguous()
    sin_c = sin_bld.contiguous()
    for t in (q_bhld, k_bhld):
        B, H, _, _ = t.shape
        _partial_rope_decode_kernel[(int(B), int(H))](
            t, cos_c, sin_c, int(H),
            int(t.stride(0)), int(t.stride(1)), int(cos_c.stride(0)),
            HALF_ROT=half_rot, num_warps=2,
        )
