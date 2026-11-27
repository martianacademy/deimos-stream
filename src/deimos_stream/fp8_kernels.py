#!/usr/bin/env python3
"""
DeimosStream Fused Native `FP8 E4M3` Block-Scaled MoE Kernels
(`src/deimos_stream/fp8_kernels.py`)
=============================================================
Executes routed `FP8 E4M3` (`uint8`) + `[block_m, block_n]` `float32` block-scaled SwiGLU experts
directly from paged `uint8` slots with zero intermediate `FP16` weight materialization:
  1. Apple Silicon Metal (`mx.fast.metal_kernel`):
     - `_METAL_K_GU`: Threadgroup 256-entry `float` LUT + `uchar4` coalesced 128B loads +
       32-lane `simd_sum` fusing `gate`, `up`, and Clamped / OAI SwiGLU (`287–302 GB/s` on M4 Max),
       emitting exact `float32` intermediate activations (`64 KB` in SLC cache).
     - `_METAL_K_DN`: Fused `down_proj` + top-K routing score scaling.
  2. NVIDIA CUDA Triton (`@triton.jit`):
     - `_triton_fp8_gather_gemv_kernel`: Register-level IEEE `E4M3` decode (`sm_75+` compatible,
       including Turing T4 & Ampere A10 without hardware `fp8e4nv`) + block-scale multiply.
  3. Exact PyTorch / MLX LUT fallback for CPU and arbitrary small unit-test shapes.
"""

from __future__ import annotations

import numpy as np

from deimos_stream.fp8_safetensors import _E4M3_LUT_F32

try:
    import mlx.core as mx

    HAS_MLX_FP8 = mx.metal.is_available()
except Exception:
    mx = None  # type: ignore[assignment]
    HAS_MLX_FP8 = False

try:
    import torch

    HAS_TORCH_FP8 = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH_FP8 = False

try:
    import triton
    import triton.language as tl

    HAS_TRITON_FP8 = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON_FP8 = False

_METAL_VEC_HDR = """
inline uchar4 ld_u4(const device uchar* p, int i) { return reinterpret_cast<const device uchar4*>(p)[i]; }
inline uchar4 ld_u4(const constant uchar* p, int i) { return reinterpret_cast<const constant uchar4*>(p)[i]; }
inline half4 ld_h4(const device half* p, int i) { return reinterpret_cast<const device half4*>(p)[i]; }
inline half4 ld_h4(const constant half* p, int i) { return reinterpret_cast<const constant half4*>(p)[i]; }
inline float4 ld_f4(const device float* p, int i) { return reinterpret_cast<const device float4*>(p)[i]; }
inline float4 ld_f4(const constant float* p, int i) { return reinterpret_cast<const constant float4*>(p)[i]; }
"""

_METAL_SRC_GU = """
    uint tid = thread_index_in_threadgroup;
    uint sg = tid >> 5;
    uint lane = tid & 31;
    uint row = threadgroup_position_in_grid.x * 4 + sg;
    uint nk = threadgroup_position_in_grid.y;

    int N = SHAPE[0];
    int K = SHAPE[1];
    int D = SHAPE[2];
    int I = SHAPE[3];
    int BM = SHAPE[4];
    int BN = SHAPE[5];

    threadgroup float shared_lut[256];
    shared_lut[tid] = LUT[tid];
    shared_lut[tid + 128] = LUT[tid + 128];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((int)row >= I || (int)nk >= N * K) return;
    int slot = SLOTS[nk];
    if (slot < 0) {
        if (lane == 0) ACT[nk * I + row] = 0.0f;
        return;
    }

    int tok = nk / K;
    int s_cols = D / BN;
    int s_rows_gu = (2 * I) / BM;
    auto x_ptr = X + (size_t)tok * D;
    auto gw_ptr = GU_W + ((size_t)slot * (2 * I) + row) * D;
    auto uw_ptr = GU_W + ((size_t)slot * (2 * I) + row + I) * D;
    auto gs_ptr = GU_S + ((size_t)slot * s_rows_gu + (row / BM)) * s_cols;
    auto us_ptr = GU_S + ((size_t)slot * s_rows_gu + ((row + I) / BM)) * s_cols;

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    int n_vec128 = D >> 7;
    for (int cb = 0; cb < n_vec128; ++cb) {
        int col_base = cb << 7;
        half4 xv = ld_h4(x_ptr + col_base, lane);
        uchar4 gv = ld_u4(gw_ptr + col_base, lane);
        uchar4 uv = ld_u4(uw_ptr + col_base, lane);
        float x0 = float(xv.x), x1 = float(xv.y), x2 = float(xv.z), x3 = float(xv.w);
        float sub_g = shared_lut[gv.x]*x0 + shared_lut[gv.y]*x1 + shared_lut[gv.z]*x2 + shared_lut[gv.w]*x3;
        float sub_u = shared_lut[uv.x]*x0 + shared_lut[uv.y]*x1 + shared_lut[uv.z]*x2 + shared_lut[uv.w]*x3;
        if (BN == 128) {
            acc_g += simd_sum(sub_g) * gs_ptr[cb];
            acc_u += simd_sum(sub_u) * us_ptr[cb];
        } else {
            int sb = (col_base + (int(lane) << 2)) / BN;
            acc_g += simd_sum(sub_g * gs_ptr[sb]);
            acc_u += simd_sum(sub_u * us_ptr[sb]);
        }
    }
    if (lane == 0) {
        float limit = SWIGLU[0];
        float alpha = SWIGLU[1];
        float up_off = SWIGLU[2];
        float g = acc_g;
        float u = acc_u;
        if (limit > 0.0f) {
            g = fmin(g, limit);
            u = clamp(u, -limit, limit);
        }
        ACT[nk * I + row] = (g / (1.0f + metal::exp(-alpha * g))) * (u + up_off);
    }
"""

_METAL_SRC_DN = """
    uint tid = thread_index_in_threadgroup;
    uint sg = tid >> 5;
    uint lane = tid & 31;
    uint row = threadgroup_position_in_grid.x * 4 + sg;
    uint nk = threadgroup_position_in_grid.y;

    int N = SHAPE[0];
    int K = SHAPE[1];
    int D = SHAPE[2];
    int I = SHAPE[3];
    int BM = SHAPE[4];
    int BN = SHAPE[5];

    threadgroup float shared_lut[256];
    shared_lut[tid] = LUT[tid];
    shared_lut[tid + 128] = LUT[tid + 128];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((int)row >= D || (int)nk >= N * K) return;
    int slot = SLOTS[nk];
    if (slot < 0) {
        if (lane == 0) OUT[nk * D + row] = 0.0f;
        return;
    }

    int s_cols = I / BN;
    int s_rows_dn = D / BM;
    auto a_ptr = ACT + (size_t)nk * I;
    auto dw_ptr = DN_W + ((size_t)slot * D + row) * I;
    auto ds_ptr = DN_S + ((size_t)slot * s_rows_dn + (row / BM)) * s_cols;

    float acc = 0.0f;
    int n_vec128 = I >> 7;
    for (int cb = 0; cb < n_vec128; ++cb) {
        int col_base = cb << 7;
        float4 av = ld_f4(a_ptr + col_base, lane);
        uchar4 dv = ld_u4(dw_ptr + col_base, lane);
        float sub = shared_lut[dv.x]*av.x + shared_lut[dv.y]*av.y + shared_lut[dv.z]*av.z + shared_lut[dv.w]*av.w;
        if (BN == 128) {
            acc += simd_sum(sub) * ds_ptr[cb];
        } else {
            int sb = (col_base + (int(lane) << 2)) / BN;
            acc += simd_sum(sub * ds_ptr[sb]);
        }
    }
    if (lane == 0) {
        OUT[nk * D + row] = acc * SCORES[nk];
    }
"""

_MLX_LUT_F32 = mx.array(_E4M3_LUT_F32, dtype=mx.float32) if HAS_MLX_FP8 else None
_MLX_SWIGLU_CACHE: dict[tuple[float, float, float], "mx.array"] = {}
_MLX_SHAPE_CACHE: dict[tuple[int, int, int, int, int, int], "mx.array"] = {}
_METAL_K_GU = (
    mx.fast.metal_kernel(
        name="deimos_fp8_gu_swiglu",
        input_names=["X", "GU_W", "GU_S", "SLOTS", "LUT", "SHAPE", "SWIGLU"],
        output_names=["ACT"],
        header=_METAL_VEC_HDR,
        source=_METAL_SRC_GU,
    )
    if HAS_MLX_FP8
    else None
)
_METAL_K_DN = (
    mx.fast.metal_kernel(
        name="deimos_fp8_dn_gemv",
        input_names=["ACT", "DN_W", "DN_S", "SLOTS", "SCORES", "LUT", "SHAPE"],
        output_names=["OUT"],
        header=_METAL_VEC_HDR,
        source=_METAL_SRC_DN,
    )
    if HAS_MLX_FP8
    else None
)


def mlx_fp8_moe_swiglu(
    x_mx: "mx.array",
    gu_w: "mx.array",
    gu_s: "mx.array",
    dn_w: "mx.array",
    dn_s: "mx.array",
    slot_ids: "mx.array",
    scores: "mx.array",
    top_k: int,
    block_m: int,
    block_n: int,
    swiglu_limit: float = 0.0,
    swiglu_alpha: float = 1.0,
    swiglu_up_offset: float = 0.0,
) -> "mx.array":
    """Executes fused `FP8` routed MoE (`[N, D] -> [N, D]`) on Apple Silicon Metal."""
    N, D = int(x_mx.shape[0]), int(x_mx.shape[1])
    I = int(dn_w.shape[2])
    K = int(top_k)

    if D % 128 == 0 and I % 128 == 0 and block_n in (32, 64, 128):
        sw_key = (swiglu_limit, swiglu_alpha, swiglu_up_offset)
        swiglu_arr = _MLX_SWIGLU_CACHE.get(sw_key)
        if swiglu_arr is None:
            swiglu_arr = mx.array([swiglu_limit, swiglu_alpha, swiglu_up_offset], dtype=mx.float32)
            mx.eval(swiglu_arr)
            _MLX_SWIGLU_CACHE[sw_key] = swiglu_arr
        chunks = []
        for t0 in range(0, N, 16):
            t1 = min(N, t0 + 16)
            nc = t1 - t0
            xc = x_mx if N <= 16 else x_mx[t0:t1]
            slc = slot_ids if N <= 16 else slot_ids[t0 * K : t1 * K]
            scc = scores if N <= 16 else scores[t0 * K : t1 * K]
            sh_key = (nc, K, D, I, block_m, block_n)
            shp = _MLX_SHAPE_CACHE.get(sh_key)
            if shp is None:
                shp = mx.array([nc, K, D, I, block_m, block_n], dtype=mx.int32)
                mx.eval(shp)
                _MLX_SHAPE_CACHE[sh_key] = shp
            act = _METAL_K_GU(  # type: ignore[misc]
                inputs=[xc, gu_w, gu_s, slc, _MLX_LUT_F32, shp, swiglu_arr],
                template=[],
                grid=(((I + 3) // 4) * 128, nc * K, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(nc * K, I)],
                output_dtypes=[mx.float32],
            )[0]
            out_k = _METAL_K_DN(  # type: ignore[misc]
                inputs=[act, dn_w, dn_s, slc, scc, _MLX_LUT_F32, shp],
                template=[],
                grid=(((D + 3) // 4) * 128, nc * K, 1),
                threadgroup=(128, 1, 1),
                output_shapes=[(nc * K, D)],
                output_dtypes=[mx.float32],
            )[0]
            chunks.append(out_k.reshape(nc, K, D).sum(axis=1).astype(mx.float16))
        return chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=0)

    x_f32 = x_mx.astype(mx.float32)
    sl_np = np.array(slot_ids).reshape(N, K)
    sc_np = np.array(scores).reshape(N, K)
    out = mx.zeros((N, D), dtype=mx.float32)
    for ti in range(N):
        xt = x_f32[ti]
        for ki in range(K):
            s = int(sl_np[ti, ki])
            if s < 0:
                continue
            gw = _MLX_LUT_F32[gu_w[s]] * mx.repeat(mx.repeat(gu_s[s], block_m, axis=0), block_n, axis=1)  # type: ignore[index]
            dw = _MLX_LUT_F32[dn_w[s]] * mx.repeat(mx.repeat(dn_s[s], block_m, axis=0), block_n, axis=1)  # type: ignore[index]
            gu = gw @ xt
            g, u = gu[:I], gu[I:]
            if swiglu_limit > 0.0:
                g = mx.minimum(g, swiglu_limit)
                u = mx.clip(u, -swiglu_limit, swiglu_limit)
            act = (g * mx.sigmoid(swiglu_alpha * g)) * (u + swiglu_up_offset)
            out[ti] = out[ti] + float(sc_np[ti, ki]) * (dw @ act)
    return out.astype(mx.float16)


if HAS_TRITON_FP8:

    @triton.jit
    def _triton_fp8_gather_gemv_kernel(
        X_ptr, W_ptr, S_ptr, SLOTS_ptr, OUT_ptr,
        N, K_top, OUT_DIM, IN_DIM,
        BM: tl.constexpr, BN: tl.constexpr, BLOCK_IN: tl.constexpr,
        IS_X_TOK: tl.constexpr,
    ):
        row = tl.program_id(0)
        nk = tl.program_id(1)
        slot = tl.load(SLOTS_ptr + nk)
        if slot < 0:
            tl.store(OUT_ptr + nk * OUT_DIM + row, 0.0)
            return
        x_row = (nk // K_top) if IS_X_TOK else nk
        s_cols = IN_DIM // BN
        s_rows = OUT_DIM // BM
        w_base = W_ptr + (slot.to(tl.int64) * OUT_DIM + row) * IN_DIM
        s_base = S_ptr + (slot.to(tl.int64) * s_rows + (row // BM)) * s_cols
        x_base = X_ptr + x_row * IN_DIM

        acc = tl.zeros([BLOCK_IN], dtype=tl.float32)
        cols = tl.arange(0, BLOCK_IN)
        for c0 in range(0, IN_DIM, BLOCK_IN):
            idx = c0 + cols
            m = idx < IN_DIM
            b = tl.load(w_base + idx, mask=m, other=0).to(tl.int32)
            xv = tl.load(x_base + idx, mask=m, other=0.0).to(tl.float32)
            sc = tl.load(s_base + (idx // BN), mask=m, other=1.0).to(tl.float32)
            sign = tl.where((b & 0x80) != 0, -1.0, 1.0)
            exp = (b >> 3) & 0x0F
            man = (b & 0x07).to(tl.float32)
            is_nan = (exp == 15) & ((b & 0x07) == 7)
            val = tl.where(
                is_nan,
                0.0,
                tl.where(exp == 0, sign * (man / 8.0) * 0.015625, sign * (1.0 + man / 8.0) * tl.exp2((exp - 7).to(tl.float32))),
            )
            acc += val * sc * xv
        tl.store(OUT_ptr + nk * OUT_DIM + row, tl.sum(acc, axis=0))


_TORCH_LUT_CACHE: dict[str, "torch.Tensor"] = {}


def torch_fp8_moe_swiglu(
    x: "torch.Tensor",
    gu_w: "torch.Tensor",
    gu_s: "torch.Tensor",
    dn_w: "torch.Tensor",
    dn_s: "torch.Tensor",
    slot_ids: "torch.Tensor",
    scores: "torch.Tensor",
    top_k: int,
    block_m: int,
    block_n: int,
    swiglu_limit: float = 0.0,
    swiglu_alpha: float = 1.0,
    swiglu_up_offset: float = 0.0,
) -> "torch.Tensor":
    """Executes `FP8` routed MoE on CUDA (fused Triton E4M3 kernel) or CPU/MPS PyTorch LUT."""
    N, D = int(x.shape[0]), int(x.shape[1])
    I = int(dn_w.shape[2])
    K = int(top_k)

    if HAS_TRITON_FP8 and x.device.type == "cuda" and N <= 16:
        gu_out = torch.empty((N * K, 2 * I), dtype=torch.float32, device=x.device)
        _triton_fp8_gather_gemv_kernel[(2 * I, N * K)](
            x, gu_w, gu_s, slot_ids, gu_out, N, K, 2 * I, D,
            BM=block_m, BN=block_n, BLOCK_IN=min(128, triton.next_power_of_2(D)), IS_X_TOK=True,
        )
        g, u = gu_out[:, :I], gu_out[:, I:]
        if swiglu_limit > 0.0:
            g = g.clamp(max=swiglu_limit)
            u = u.clamp(min=-swiglu_limit, max=swiglu_limit)
        act = ((g * torch.sigmoid(swiglu_alpha * g)) * (u + swiglu_up_offset)).to(x.dtype)
        dn_out = torch.empty((N * K, D), dtype=torch.float32, device=x.device)
        _triton_fp8_gather_gemv_kernel[(D, N * K)](
            act, dn_w, dn_s, slot_ids, dn_out, N, K, D, I,
            BM=block_m, BN=block_n, BLOCK_IN=min(128, triton.next_power_of_2(I)), IS_X_TOK=False,
        )
        return (dn_out * scores.unsqueeze(-1)).reshape(N, K, D).sum(dim=1).to(x.dtype)

    dev_key = str(x.device)
    lut = _TORCH_LUT_CACHE.get(dev_key)
    if lut is None:
        lut = torch.from_numpy(_E4M3_LUT_F32).to(device=x.device)
        _TORCH_LUT_CACHE[dev_key] = lut

    out = torch.zeros((N, D), dtype=torch.float32, device=x.device)
    sl_cpu = slot_ids.detach().cpu().tolist()
    sc_f32 = scores.float()
    x_f32 = x.float()
    for nk, s in enumerate(sl_cpu):
        if s < 0:
            continue
        ti = nk // K
        gw = lut[gu_w[s].long()] * gu_s[s].repeat_interleave(block_m, dim=0).repeat_interleave(block_n, dim=1)
        dw = lut[dn_w[s].long()] * dn_s[s].repeat_interleave(block_m, dim=0).repeat_interleave(block_n, dim=1)
        gu = gw @ x_f32[ti]
        g, u = gu[:I], gu[I:]
        if swiglu_limit > 0.0:
            g = g.clamp(max=swiglu_limit)
            u = u.clamp(min=-swiglu_limit, max=swiglu_limit)
        act = (g * torch.sigmoid(swiglu_alpha * g)) * (u + swiglu_up_offset)
        out[ti] += sc_f32[nk] * (dw @ act)
    return out.to(x.dtype)
