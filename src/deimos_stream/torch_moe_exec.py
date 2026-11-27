#!/usr/bin/env python3
"""
DeimosStream Hybrid CUDA/CPU MoE Execution Dispatcher (`src/deimos_stream/torch_moe_exec.py`)
=============================================================================================
Executes paged Q4 MoE forward passes over `TorchPagedExpertArena`:
  - Fully pinned (`slots == num_experts`): 100% GPU execution with zero CPU syncs.
  - CUDA decode (`N == 1`): Single burst D2H copy (`x`, `top_k_ids`, `top_k_scores`, optional
    piggybacked lookahead `la_ids`) -> TinyLFU split -> parallel GPU hit GEMV (`triton_q4`) +
    `shared_expert` overlapped with C++17 `AVX2/NEON` `cas_mmap` miss compute + non-evicting
    `POSIX_MADV_WILLNEED` lookahead + async `_dma_stream` warm slot promotion.
  - CUDA prefill (`N > 1`) & CPU fallback: chunked batch execution.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np

from deimos_stream.arena import DeimosMerkleIntegrityError
from deimos_stream.triton_q4 import run_paged_q4_swiglu

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from deimos_stream.torch_arena import TorchPagedExpertArena


def bind_hybrid_c_api(lib: ctypes.CDLL) -> None:
    """Idempotently binds C++17 ring-batch, hybrid decode, and SIMD Q4 MoE signatures."""
    if getattr(lib, "_deimos_hybrid_bound", False):
        return
    i32p = ctypes.POINTER(ctypes.c_int32)
    f32p = ctypes.POINTER(ctypes.c_float)
    u16p = ctypes.POINTER(ctypes.c_uint16)
    ip = ctypes.POINTER(ctypes.c_int)

    lib.deimos_stage_q4_batch.argtypes = [
        ctypes.c_void_p, ctypes.c_int, i32p, i32p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.deimos_stage_q4_batch.restype = ctypes.c_int

    lib.deimos_arena_resolve_ring_batch.argtypes = [
        ctypes.c_void_p, ctypes.c_int, i32p, f32p, ctypes.c_int, ctypes.c_int,
        i32p, i32p, i32p, ip,
    ]
    lib.deimos_arena_resolve_ring_batch.restype = ctypes.c_int

    lib.deimos_arena_resolve_hybrid_decode.argtypes = [
        ctypes.c_void_p, ctypes.c_int, i32p, f32p, ctypes.c_int,
        ctypes.c_float, ctypes.c_int, ctypes.c_int,
        i32p, f32p, ip, i32p, f32p, ip, i32p, i32p, ip,
    ]
    lib.deimos_arena_resolve_hybrid_decode.restype = ctypes.c_int

    lib.deimos_cpu_q4_swiglu_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_int, i32p, f32p, ctypes.c_int,
        u16p, u16p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    lib.deimos_cpu_q4_swiglu_forward.restype = ctypes.c_int
    lib.deimos_warm_cpu_moe_pool.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.deimos_warm_cpu_moe_pool.restype = None
    lib.deimos_arena_mark_preloaded_prefix.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.deimos_arena_mark_preloaded_prefix.restype = ctypes.c_int
    lib.deimos_evict_expert.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.deimos_evict_expert.restype = ctypes.c_int
    lib.deimos_prefetch_layer.argtypes = [ctypes.c_void_p, ctypes.c_int, i32p, ctypes.c_int]
    lib.deimos_prefetch_layer.restype = None
    lib._deimos_hybrid_bound = True  # type: ignore[attr-defined]


def execute_torch_moe_layer(
    arena: "TorchPagedExpertArena",
    layer_idx: int,
    x_chunk: "torch.Tensor",
    top_k_index: "torch.Tensor",
    top_k_weights: "torch.Tensor",
    miss_tail_tau: float = 1.0,
    min_k: int = 4,
    shared_fn: Optional[Callable[[], "torch.Tensor"]] = None,
    la_ids_fn: Optional[Callable[[], "torch.Tensor"]] = None,
) -> "torch.Tensor":
    """Dispatches `layer_idx` MoE across GPU slot cache and zero-PCIe CPU SIMD miss engine."""
    N, D = x_chunk.shape
    K = int(top_k_index.shape[-1])
    l_i = int(layer_idx)

    sw_kw = {
        "group_size": arena.group_size,
        "swiglu_limit": arena.swiglu_limit,
        "swiglu_alpha": arena.swiglu_alpha,
        "swiglu_up_offset": arena.swiglu_up_offset,
    }
    if arena.is_fully_pinned:
        y_sh = shared_fn() if shared_fn is not None else None
        y = run_paged_q4_swiglu(
            x_chunk,
            arena.gu_w[l_i], arena.gu_s[l_i], arena.gu_b[l_i],
            arena.dn_w[l_i], arena.dn_s[l_i], arena.dn_b[l_i],
            top_k_index.reshape(-1).to(torch.long),
            top_k_weights,
            **sw_kw,
        )
        return (y + y_sh) if y_sh is not None else y

    if not arena.is_cuda:
        y_sh = shared_fn() if shared_fn is not None else None
        pred_np: Optional[np.ndarray] = None
        if N == 1 and la_ids_fn is not None and (l_i + 1 < arena.num_layers):
            pred_np = la_ids_fn().detach().cpu().numpy().astype(np.int32)
        ind_np = top_k_index.detach().cpu().numpy().astype(np.int32)
        sc_np = top_k_weights.detach().float().cpu().numpy()
        max_c = max(1, arena.slots_per_layer // max(1, K))
        outs: List[torch.Tensor] = []
        for s0 in range(0, N, max_c):
            s1 = min(N, s0 + max_c)
            phys_np, eff_sc = arena.resolve_and_prefetch(
                l_i, ind_np[s0:s1], sc_np[s0:s1],
                miss_tail_tau=miss_tail_tau if N == 1 else 1.0,
                min_k=min_k,
                next_layer_idx=(l_i + 1) if (N == 1 and pred_np is not None) else -1,
                next_pred_ids_np=pred_np if N == 1 else None,
            )
            sc_t = (
                top_k_weights[s0:s1]
                if (N > 1 or miss_tail_tau >= 0.9999)
                else torch.from_numpy(eff_sc).to(dtype=x_chunk.dtype)
            )
            outs.append(run_paged_q4_swiglu(
                x_chunk[s0:s1],
                arena.gu_w[l_i], arena.gu_s[l_i], arena.gu_b[l_i],
                arena.dn_w[l_i], arena.dn_s[l_i], arena.dn_b[l_i],
                torch.from_numpy(phys_np.reshape(-1)).long(),
                sc_t,
                **sw_kw,
            ))
        y = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
        return (y + y_sh) if y_sh is not None else y

    compute_stream = torch.cuda.current_stream(arena.device)
    promo_ev = arena._layer_promo_ev[l_i]
    if promo_ev is not None:
        compute_stream.wait_event(promo_ev)
        arena._layer_promo_ev[l_i] = None

    if N == 1 and K <= 32 and not arena.use_nocache_pread:
        x_src = x_chunk if x_chunk.dtype == torch.float16 else x_chunk.to(torch.float16)
        id_src = top_k_index.reshape(-1)
        if id_src.dtype != torch.int32:
            id_src = id_src.to(torch.int32)
        sc_src = top_k_weights.reshape(-1)
        if sc_src.dtype == torch.float16:
            arena._pin_x.copy_(x_src, non_blocking=True)
            arena._pin_ids[:K].copy_(id_src, non_blocking=True)
            arena._pin_sc_f16[:K].copy_(sc_src, non_blocking=True)
            compute_stream.synchronize()
            arena._pin_sc.numpy()[:K] = arena._pin_sc_f16.numpy()[:K]
        else:
            arena._pin_x.copy_(x_src, non_blocking=True)
            arena._pin_ids[:K].copy_(id_src, non_blocking=True)
            arena._pin_sc[:K].copy_(sc_src.to(torch.float32), non_blocking=True)
            compute_stream.synchronize()

        sc_buf = getattr(arena, "_hyb_scratch", None)
        if sc_buf is None:
            sc_buf = (
                np.empty((32,), dtype=np.int32),
                np.empty((32,), dtype=np.float32),
                np.empty((32,), dtype=np.int32),
                np.empty((32,), dtype=np.float32),
                np.empty((32,), dtype=np.int32),
                np.empty((32,), dtype=np.int32),
            )
            arena._hyb_scratch = sc_buf  # type: ignore[attr-defined]
        hit_slots, hit_sc, miss_e, miss_sc, promo_e, promo_slots = sc_buf
        n_hits, n_miss, n_promos = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)

        max_p = 0
        if arena._ring_events:
            ev0 = arena._ring_events[arena._ring_cursor]
            if ev0 is None or ev0.query():
                ev1 = arena._ring_events[(arena._ring_cursor + 1) % arena._ring_slots]
                max_p = 2 if (ev1 is None or ev1.query()) else 1

        rc = arena.bridge.lib.deimos_arena_resolve_hybrid_decode(
            arena.arena_handle, l_i,
            ctypes.cast(arena._pin_ids.data_ptr(), ctypes.POINTER(ctypes.c_int32)),
            ctypes.cast(arena._pin_sc.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            K, float(miss_tail_tau), int(min_k), max_p,
            hit_slots.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            hit_sc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(n_hits),
            miss_e.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            miss_sc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(n_miss),
            promo_e.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            promo_slots.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.byref(n_promos),
        )
        if rc != 0:
            raise RuntimeError(f"deimos_arena_resolve_hybrid_decode failed (rc={rc})")

        nh, nm, np_cnt = int(n_hits.value), int(n_miss.value), int(n_promos.value)
        y_sh = shared_fn() if shared_fn is not None else None
        y_gpu: Optional[torch.Tensor] = None
        if nh > 0:
            arena._pin_hit_slots.numpy()[:nh] = hit_slots[:nh]
            arena._pin_hit_sc.numpy()[0, :nh] = hit_sc[:nh]
            s_gpu = arena._gpu_hit_slots[:nh]
            w_gpu = arena._gpu_hit_sc[:, :nh]
            s_gpu.copy_(arena._pin_hit_slots[:nh], non_blocking=True)
            w_gpu.copy_(arena._pin_hit_sc[:, :nh], non_blocking=True)
            w_use = w_gpu if x_chunk.dtype == torch.float16 else w_gpu.to(x_chunk.dtype)
            y_gpu = run_paged_q4_swiglu(
                x_chunk,
                arena.gu_w[l_i], arena.gu_s[l_i], arena.gu_b[l_i],
                arena.dn_w[l_i], arena.dn_s[l_i], arena.dn_b[l_i],
                s_gpu, w_use, **sw_kw,
            )

        if nm > 0:
            rc_cpu = arena.bridge.lib.deimos_cpu_q4_swiglu_forward(
                arena.bridge.handle, l_i,
                miss_e.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                miss_sc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                nm,
                ctypes.cast(arena._pin_x.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
                ctypes.cast(arena._pin_cpu_out.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
                D, arena.intermediate_size, arena.group_size,
                1 if arena.verify_merkle_on_cold else 0,
                float(arena.swiglu_limit), float(arena.swiglu_alpha), float(arena.swiglu_up_offset),
            )
            if rc_cpu == -2:
                raise DeimosMerkleIntegrityError(f"Corrupted Q4 Merkle leaf rejected in layer {l_i}!")
            if rc_cpu != 0:
                raise RuntimeError(f"deimos_cpu_q4_swiglu_forward failed (rc={rc_cpu})")
            arena._gpu_cpu_out.copy_(arena._pin_cpu_out, non_blocking=True)
            y_cpu = arena._gpu_cpu_out if x_chunk.dtype == torch.float16 else arena._gpu_cpu_out.to(x_chunk.dtype)
            y_out = (y_gpu + y_cpu) if y_gpu is not None else y_cpu
        else:
            y_out = y_gpu if y_gpu is not None else torch.zeros_like(x_chunk)

        if y_sh is not None:
            y_out = y_out + y_sh
        if np_cnt > 0:
            arena._stage_via_ring(
                l_i, promo_e[:np_cnt].tolist(), promo_slots[:np_cnt].tolist(),
                wait_compute_first=True, block_compute_after=False,
            )
        return y_out

    y_sh = shared_fn() if shared_fn is not None else None
    ind_np = top_k_index.detach().to(torch.int32).cpu().numpy()
    sc_np = top_k_weights.detach().to(torch.float32).cpu().numpy()
    max_c = min(32, max(1, arena.slots_per_layer // max(1, K)))
    outs_b: List[torch.Tensor] = []
    for s0 in range(0, N, max_c):
        s1 = min(N, s0 + max_c)
        phys_np, _ = arena.resolve_and_prefetch(l_i, ind_np[s0:s1], sc_np[s0:s1], miss_tail_tau=1.0)
        outs_b.append(run_paged_q4_swiglu(
            x_chunk[s0:s1],
            arena.gu_w[l_i], arena.gu_s[l_i], arena.gu_b[l_i],
            arena.dn_w[l_i], arena.dn_s[l_i], arena.dn_b[l_i],
            torch.from_numpy(phys_np.reshape(-1)).to(device=arena.device, dtype=torch.long),
            top_k_weights[s0:s1],
            **sw_kw,
        ))
    y_b = outs_b[0] if len(outs_b) == 1 else torch.cat(outs_b, dim=0)
    return (y_b + y_sh) if y_sh is not None else y_b
