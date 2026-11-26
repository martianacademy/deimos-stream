#!/usr/bin/env python3
"""
DeimosStream Fused CUDA Triton 4-Bit (`Q4` Group-64) MoE & INT8 `lm_head` GEMV Kernels
(`src/deimos_stream/triton_q4.py`)
======================================================================================
Eliminates the ~31x HBM memory amplification and temporary FP32/FP16 dequantization spikes
of PyTorch `unpack_q4_torch` + `torch.bmm`:
  1. `triton_q4_gather_gemv`: Reads packed `uint8` 4-bit nibbles + `float16` scales/biases
     directly in-place from `TorchPagedExpertArena` (`[C, Out, In // 2]`) via `slot_ids`,
     unpacking and accumulating in registers using pure FP32 FMA (`sm_75` Turing Tesla T4
     through Hopper/Blackwell compatible — zero `tl.dot` `sm_80` restriction!).
     Algebraic factorization per group `g`:
       sum_j ((q_lo*s + b)*x_even + (q_hi*s + b)*x_odd)
         = s * sum_j (q_lo*x_even + q_hi*x_odd) + b * sum_j (x_even + x_odd)
  2. `triton_int8_gemv`: Fused row-wise INT8 `lm_head` projection with zero FP16 weight copies.
"""

from __future__ import annotations

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
    def _q4_gather_gemv_kernel(
        x_ptr,
        w_ptr,
        s_ptr,
        b_ptr,
        slots_ptr,
        out_ptr,
        out_dim,
        stride_xn,
        stride_ws,
        stride_wr,
        stride_ss,
        stride_sr,
        stride_ob,
        stride_om,
        TOP_K: tl.constexpr,
        IS_SHARED_X: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        HALF_G: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)

        slot = tl.load(slots_ptr + pid_b).to(tl.int64)
        if IS_SHARED_X:
            x_row = (pid_b // TOP_K).to(tl.int64)
        else:
            x_row = pid_b.to(tl.int64)

        rows = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        row_mask = rows < out_dim
        j = tl.arange(0, HALF_G).to(tl.int64)

        w_row_base = slot * stride_ws + rows[:, None] * stride_wr
        sb_row_base = slot * stride_ss + rows * stride_sr
        x_row_base = x_row * stride_xn

        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        group_size: tl.constexpr = HALF_G * 2

        for g in range(0, NUM_GROUPS):
            x_g = x_ptr + x_row_base + g * group_size
            x_even = tl.load(x_g + j * 2).to(tl.float32)
            x_odd = tl.load(x_g + j * 2 + 1).to(tl.float32)
            sum_x = tl.sum(x_even + x_odd, axis=0)

            packed = tl.load(w_ptr + w_row_base + (g * HALF_G + j)[None, :], mask=row_mask[:, None], other=0)
            q_lo = (packed & 0x0F).to(tl.float32)
            q_hi = ((packed >> 4) & 0x0F).to(tl.float32)

            sc = tl.load(s_ptr + sb_row_base + g, mask=row_mask, other=0.0).to(tl.float32)
            bi = tl.load(b_ptr + sb_row_base + g, mask=row_mask, other=0.0).to(tl.float32)

            dot_q = tl.sum(q_lo * x_even[None, :] + q_hi * x_odd[None, :], axis=1)
            acc += dot_q * sc + sum_x * bi

        tl.store(
            out_ptr + pid_b.to(tl.int64) * stride_ob + rows * stride_om,
            acc.to(out_ptr.dtype.element_ty),
            mask=row_mask,
        )

    @triton.jit
    def _int8_gemv_kernel(
        x_ptr,
        w_ptr,
        s_ptr,
        out_ptr,
        vocab_size,
        in_dim,
        stride_xn,
        stride_wv,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
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
        tl.store(
            out_ptr + pid_n * stride_on + rows,
            (acc * sc).to(out_ptr.dtype.element_ty),
            mask=row_mask,
        )


def triton_q4_gather_gemv(
    x: "torch.Tensor",
    w_pool_u8: "torch.Tensor",
    s_pool_f16: "torch.Tensor",
    b_pool_f16: "torch.Tensor",
    flat_slots: "torch.Tensor",
    top_k: int,
    group_size: int = 64,
    is_shared_x: bool = True,
) -> "torch.Tensor":
    """
    Fused 4-bit gathered GEMV reading directly from `[C, Out, In // 2]` arena pools.
    - `is_shared_x=True`: `x` has shape `[N, In]`, `flat_slots` has shape `[N * top_k]`.
    - `is_shared_x=False`: `x` has shape `[N * top_k, In]`, `flat_slots` has shape `[N * top_k]`.
    Returns `[N * top_k, Out]` in `x.dtype` with ZERO unpacked weight memory allocated.
    """
    assert HAS_TORCH and HAS_TRITON
    if group_size not in (32, 64, 128):
        raise ValueError(f"Unsupported Triton Q4 group_size={group_size}")

    x_c = x.contiguous()
    slots_c = flat_slots.contiguous()
    B = int(slots_c.shape[0])
    _, out_dim, half_in = w_pool_u8.shape
    in_dim = int(half_in * 2)
    num_groups = in_dim // group_size
    half_g = group_size // 2

    out = torch.empty((B, int(out_dim)), device=x_c.device, dtype=x_c.dtype)
    block_m = 32
    grid = (B, triton.cdiv(int(out_dim), block_m))

    _q4_gather_gemv_kernel[grid](
        x_c,
        w_pool_u8,
        s_pool_f16,
        b_pool_f16,
        slots_c,
        out,
        int(out_dim),
        int(x_c.stride(0)),
        int(w_pool_u8.stride(0)),
        int(w_pool_u8.stride(1)),
        int(s_pool_f16.stride(0)),
        int(s_pool_f16.stride(1)),
        int(out.stride(0)),
        int(out.stride(1)),
        TOP_K=int(top_k),
        IS_SHARED_X=bool(is_shared_x),
        NUM_GROUPS=int(num_groups),
        HALF_G=int(half_g),
        BLOCK_M=block_m,
        num_warps=4,
    )
    return out


def triton_int8_gemv(
    x_2d: "torch.Tensor",
    w_i8: "torch.Tensor",
    scale_1d: "torch.Tensor",
) -> "torch.Tensor":
    """Fused INT8 `lm_head` GEMV `[N, D] @ [V, D].T * scale[V]` with zero FP16 weight copies."""
    assert HAS_TORCH and HAS_TRITON
    x_c = x_2d.contiguous()
    sc_c = scale_1d.reshape(-1).contiguous()
    N, in_dim = x_c.shape
    vocab_size = int(w_i8.shape[0])

    out = torch.empty((int(N), vocab_size), device=x_c.device, dtype=x_c.dtype)
    block_m = 64
    block_k = 128
    grid = (int(N), triton.cdiv(vocab_size, block_m))

    _int8_gemv_kernel[grid](
        x_c,
        w_i8,
        sc_c,
        out,
        vocab_size,
        int(in_dim),
        int(x_c.stride(0)),
        int(w_i8.stride(0)),
        int(out.stride(0)),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=4,
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
) -> "torch.Tensor":
    """
    Executes paged 4-bit SwiGLU MoE projection `sum_k scores[n,k] * (W_dn[slot] @ (silu(gate) * up))`.
    Automatically uses zero-scratch fused CUDA Triton Q4 GEMV on NVIDIA GPUs and falls back
    cleanly to vectorized `unpack_q4_torch` + `torch.bmm` on CPU or non-Triton hosts.
    """
    N, D = x_chunk.shape
    K = int(scores.shape[1])
    dt = x_chunk.dtype

    if triton_kernels_available(x_chunk.device) and group_size in (32, 64, 128):
        gu_out = triton_q4_gather_gemv(
            x_chunk,
            gu_w_pool,
            gu_s_pool,
            gu_b_pool,
            flat_slots,
            top_k=K,
            group_size=group_size,
            is_shared_x=True,
        )
        gate, up = gu_out.chunk(2, dim=-1)
        act = F.silu(gate) * up
        dn_out = triton_q4_gather_gemv(
            act,
            dn_w_pool,
            dn_s_pool,
            dn_b_pool,
            flat_slots,
            top_k=K,
            group_size=group_size,
            is_shared_x=False,
        ).view(N, K, D)
        return (dn_out * scores.to(dt).unsqueeze(-1)).sum(dim=1)

    gu_w = unpack_q4_torch(
        gu_w_pool[flat_slots], gu_s_pool[flat_slots], gu_b_pool[flat_slots],
        group_size=group_size, out_dtype=dt,
    )
    x_rep = x_chunk.unsqueeze(1).expand(N, K, D).reshape(N * K, D, 1)
    gate, up = torch.bmm(gu_w, x_rep).squeeze(-1).chunk(2, dim=-1)
    act = (F.silu(gate) * up).unsqueeze(-1)
    dn_w = unpack_q4_torch(
        dn_w_pool[flat_slots], dn_s_pool[flat_slots], dn_b_pool[flat_slots],
        group_size=group_size, out_dtype=dt,
    )
    dn_out = torch.bmm(dn_w, act).squeeze(-1).view(N, K, D)
    return (dn_out * scores.to(dt).unsqueeze(-1)).sum(dim=1)
