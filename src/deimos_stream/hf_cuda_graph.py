#!/usr/bin/env python3
"""
DeimosStream Hybrid-MoE `CUDAGraph` Decode Accelerator (`src/deimos_stream/hf_cuda_graph.py`)
=============================================================================================
Eliminates ~80 ms/tok of HuggingFace Python + `@triton.jit` argument-key dispatch overhead
across 40 hybrid `GatedDeltaNet` + `GQA` + `PagedMoE` layers during single-token decode (`S == 1`):
  - Captures `pre_graphs[l]` (`input_norm + linear_attn/self_attn + residual + post_norm + router
    + shared_expert + pinned D2H copy`), `moe_graphs[l]` (`fixed-K hit Q4 SwiGLU + residual`),
    and `head_graph` (`final_norm + lm_head`) into a single shared CUDA Graph memory pool.
  - Missed experts receive `score = 0.0` in `moe_graphs[l]` while computing simultaneously in
    parallel on C++17 `AVX2` worker threads over zero-copy `cas_mmap`.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from deimos_stream.arena import DeimosMerkleIntegrityError
from deimos_stream.triton_hybrid import hybrid_triton_available, triton_partial_rope_decode_inplace
from deimos_stream.triton_q4 import run_paged_q4_swiglu

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from deimos_stream.hf_streamer import DeimosHFHybridMoEEngine

MAX_GRAPH_SEQ = 1024


class _StaticLinearCacheProxy:
    """Minimal zero-overhead proxy exposing static `conv` and `rec` buffers to `_fast_linear_attn`."""

    def __init__(self, conv_map: Dict[int, "torch.Tensor"], rec_map: Dict[int, "torch.Tensor"]):
        self._conv = conv_map
        self._rec = rec_map

    def get_conv_state(self, layer_idx: int, state_idx: int = 0) -> "torch.Tensor":
        return self._conv[layer_idx]

    def get_recurrent_state(self, layer_idx: int, state_idx: int = 0) -> "torch.Tensor":
        return self._rec[layer_idx]

    def is_recording_past(self, layer_idx: int) -> bool:
        return False


class DeimosHFCudaGraphDecode:
    """Captures and replays per-layer `CUDAGraph` segments for `DeimosHFHybridMoEEngine` (`S == 1`)."""

    def __init__(self, eng: "DeimosHFHybridMoEEngine", max_seq: int = MAX_GRAPH_SEQ):
        self.eng = eng
        self.arena = eng.arena
        self.device = eng.device
        self.dtype = eng.compute_dtype
        self.num_layers = eng.num_layers
        self.D = eng.hidden_size
        self.K = eng.top_k
        self.max_seq = int(max_seq)

        inner = eng.model.model
        if hasattr(inner, "language_model"):
            inner = inner.language_model
        self.inner = inner
        self.layers = inner.layers[: self.num_layers]

        pkw = {"device": "cpu", "pin_memory": True}
        self._pin_emb_in = torch.empty((self.D,), dtype=self.dtype, **pkw)
        self.static_h = torch.zeros((1, 1, self.D), dtype=self.dtype, device=self.device)
        self.static_res = torch.zeros((1, 1, self.D), dtype=self.dtype, device=self.device)
        self.static_normed = torch.zeros((1, 1, self.D), dtype=self.dtype, device=self.device)
        self.static_x_flat = torch.zeros((1, self.D), dtype=self.dtype, device=self.device)
        self.static_sh_out = torch.zeros((1, self.D), dtype=self.dtype, device=self.device)
        self.static_logits = torch.zeros((1, eng.vocab_size), dtype=self.dtype, device=self.device)
        self.static_pos_i64 = torch.zeros((1,), dtype=torch.long, device=self.device)
        self.static_arange = torch.arange(self.max_seq, dtype=torch.long, device=self.device).view(1, 1, 1, self.max_seq)

        pos_grid = torch.arange(self.max_seq, device=self.device).view(1, 1, -1).expand(3, 1, -1)
        cos_all, sin_all = inner.rotary_emb(self.static_h, pos_grid)
        self.rope_cos = cos_all.to(self.dtype).contiguous()
        self.rope_sin = sin_all.to(self.dtype).contiguous()
        rot_dim = int(self.rope_cos.shape[-1])
        self.static_cos = torch.zeros((1, 1, rot_dim), dtype=self.dtype, device=self.device)
        self.static_sin = torch.zeros((1, 1, rot_dim), dtype=self.dtype, device=self.device)

        self.static_conv: Dict[int, torch.Tensor] = {}
        self.static_rec: Dict[int, torch.Tensor] = {}
        self.static_k: Dict[int, torch.Tensor] = {}
        self.static_v: Dict[int, torch.Tensor] = {}

        for l_i, lyr in enumerate(self.layers):
            if getattr(lyr, "block_type", "") == "linear_attention":
                la = lyr.linear_attn
                c_dim, c_k = int(la.conv_dim), int(la.conv_kernel_size)
                n_vh, hk, hv = int(la.num_v_heads), int(la.head_k_dim), int(la.head_v_dim)
                self.static_conv[l_i] = torch.zeros((1, c_dim, c_k), dtype=self.dtype, device=self.device)
                self.static_rec[l_i] = torch.zeros((1, n_vh, hk, hv), dtype=torch.float32, device=self.device)
            else:
                att = lyr.self_attn
                hd = int(att._head_dim)
                n_kv = int(att._k_out) // hd
                self.static_k[l_i] = torch.zeros((1, n_kv, self.max_seq, hd), dtype=self.dtype, device=self.device)
                self.static_v[l_i] = torch.zeros((1, n_kv, self.max_seq, hd), dtype=self.dtype, device=self.device)

        self._proxy = _StaticLinearCacheProxy(self.static_conv, self.static_rec)
        self._pre_graphs: List[torch.cuda.CUDAGraph] = []
        self._moe_graphs: List[torch.cuda.CUDAGraph] = []
        self._head_graph: Optional[torch.cuda.CUDAGraph] = None
        self._bound_cache_id: int = -1
        self._cur_pos: int = 0
        self._capture_all()

    def _run_gqa_static(self, l_i: int, attn_mod: Any) -> "torch.Tensor":
        qkv = attn_mod._qkv_fn(self.static_normed)
        q_out, k_out, hd = attn_mod._q_out, attn_mod._k_out, attn_mod._head_dim
        q_raw, gate = torch.chunk(qkv[..., :q_out].view(1, 1, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(1, 1, -1)
        q = attn_mod.q_norm(q_raw).transpose(1, 2).contiguous()
        k = attn_mod.k_norm(qkv[..., q_out : q_out + k_out].view(1, 1, -1, hd)).transpose(1, 2).contiguous()
        v = qkv[..., q_out + k_out :].view(1, 1, -1, hd).transpose(1, 2).contiguous()
        triton_partial_rope_decode_inplace(q, k, self.static_cos, self.static_sin)

        kb, vb = self.static_k[l_i], self.static_v[l_i]
        kb.index_copy_(2, self.static_pos_i64, k)
        vb.index_copy_(2, self.static_pos_i64, v)
        n_kv = int(kb.shape[1])
        n_rep = int(q.shape[1]) // n_kv
        qg = q.view(1, n_kv, n_rep, hd)
        scores = torch.matmul(qg, kb.transpose(-1, -2)) * attn_mod._scaling
        scores = scores.masked_fill(self.static_arange > self.static_pos_i64, float("-inf"))
        probs = torch.softmax(scores.float(), dim=-1).to(self.dtype)
        att_out = torch.matmul(probs, vb).reshape(1, 1, -1) * torch.sigmoid(gate)
        return attn_mod._o_fn(att_out)

    def _pre_op(self, l_i: int) -> None:
        if l_i == 0:
            self.static_h[0, 0].copy_(self._pin_emb_in, non_blocking=True)
            self.static_cos.copy_(self.rope_cos.index_select(1, self.static_pos_i64))
            self.static_sin.copy_(self.rope_sin.index_select(1, self.static_pos_i64))
        lyr = self.layers[l_i]
        self.static_res.copy_(self.static_h)
        self.static_normed.copy_(lyr.input_layernorm(self.static_h))
        if getattr(lyr, "block_type", "") == "linear_attention":
            mix = lyr.linear_attn(self.static_normed, cache=self._proxy)
        else:
            mix = self._run_gqa_static(l_i, lyr.self_attn)
        self.static_h.copy_(self.static_res + mix)
        self.static_x_flat.copy_(lyr.post_attention_layernorm(self.static_h).view(1, self.D))
        _, rw, se = lyr.mlp.gate(self.static_x_flat)
        self.arena._pin_x.copy_(self.static_x_flat, non_blocking=True)
        self.arena._pin_ids[: self.K].copy_(se.view(-1), non_blocking=True)
        self.arena._pin_sc_f16[: self.K].copy_(rw.view(-1), non_blocking=True)

    def _moe_op(self, l_i: int) -> None:
        K = self.K
        lyr = self.layers[l_i]
        s_gpu = self.arena._gpu_hit_slots[:K]
        w_gpu = self.arena._gpu_hit_sc[:, :K]
        s_gpu.copy_(self.arena._pin_hit_slots[:K], non_blocking=True)
        w_gpu.copy_(self.arena._pin_hit_sc[:, :K], non_blocking=True)
        y_sh = lyr.mlp._run_shared_fn(self.static_x_flat)
        y_hit = run_paged_q4_swiglu(
            self.static_x_flat,
            self.arena.gu_w[l_i], self.arena.gu_s[l_i], self.arena.gu_b[l_i],
            self.arena.dn_w[l_i], self.arena.dn_s[l_i], self.arena.dn_b[l_i],
            s_gpu, w_gpu, group_size=self.arena.group_size,
        )
        self.static_h.add_((y_hit + y_sh).view(1, 1, self.D))

    def _head_op(self) -> None:
        h_n = self.inner.norm(self.static_h)
        self.static_logits.copy_(self.eng.model.lm_head(h_n).view(1, self.eng.vocab_size))

    def _capture_all(self) -> None:
        self.arena._pin_hit_slots.zero_()
        self.arena._pin_hit_sc.zero_()
        self._pin_emb_in.zero_()
        self.static_pos_i64.zero_()

        cap_stream = torch.cuda.Stream(device=self.device)
        cap_stream.wait_stream(torch.cuda.current_stream(self.device))
        pool = torch.cuda.graph_pool_handle()

        with torch.cuda.stream(cap_stream):
            for _ in range(2):
                for l_i in range(self.num_layers):
                    self._pre_op(l_i)
                    self._moe_op(l_i)
                self._head_op()
            cap_stream.synchronize()

            for l_i in range(self.num_layers):
                g_pre = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_pre, pool=pool, stream=cap_stream):
                    self._pre_op(l_i)
                self._pre_graphs.append(g_pre)

                g_moe = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_moe, pool=pool, stream=cap_stream):
                    self._moe_op(l_i)
                self._moe_graphs.append(g_moe)

            g_head = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_head, pool=pool, stream=cap_stream):
                self._head_op()
            self._head_graph = g_head

        torch.cuda.current_stream(self.device).wait_stream(cap_stream)
        torch.cuda.synchronize(self.device)

    def _sync_from_hf_cache(self, cache: Any) -> bool:
        seq_len = int(cache.get_seq_length())
        if seq_len <= 0 or seq_len >= self.max_seq - 2:
            return False
        for l_i in self.static_conv:
            self.static_conv[l_i].copy_(cache.get_conv_state(l_i, state_idx=0))
            self.static_rec[l_i].copy_(cache.get_recurrent_state(l_i, state_idx=0).float())
        for l_i in self.static_k:
            c_lyr = cache.layers[l_i]
            self.static_k[l_i][:, :, :seq_len, :].copy_(c_lyr.keys[:, :, :seq_len, :])
            self.static_v[l_i][:, :, :seq_len, :].copy_(c_lyr.values[:, :, :seq_len, :])
        self._bound_cache_id = id(cache)
        self._cur_pos = seq_len
        cache._deimos_gpos = seq_len
        return True

    def _sync_to_hf_cache(self, cache: Any) -> None:
        pos = self._cur_pos
        for l_i in self.static_conv:
            cache.layers[l_i].conv_states[0].copy_(self.static_conv[l_i])
            cache.layers[l_i].recurrent_states[0].copy_(self.static_rec[l_i].to(cache.layers[l_i].recurrent_states[0].dtype))
        for l_i in self.static_k:
            cache.layers[l_i].keys = self.static_k[l_i][:, :, :pos, :].clone()
            cache.layers[l_i].values = self.static_v[l_i][:, :, :pos, :].clone()
        self._bound_cache_id = -1

    def step(self, tok_id: int, cache: Any) -> Optional["torch.Tensor"]:
        if id(cache) != self._bound_cache_id or getattr(cache, "_deimos_gpos", -1) != self._cur_pos:
            if not self._sync_from_hf_cache(cache):
                return None
        if self._cur_pos >= self.max_seq - 1:
            self._sync_to_hf_cache(cache)
            return None

        arena = self.arena
        K = self.K
        compute_stream = torch.cuda.current_stream(self.device)
        self._pin_emb_in.numpy()[:] = self.eng._mmap_emb.view[int(tok_id)]
        self.static_pos_i64.fill_(self._cur_pos)

        sc_buf = getattr(arena, "_hyb_scratch", None)
        if sc_buf is None:
            sc_buf = tuple(np.empty((32,), dtype=dt) for dt in (np.int32, np.float32, np.int32, np.float32, np.int32, np.int32))
            arena._hyb_scratch = sc_buf  # type: ignore[attr-defined]
        hit_slots, hit_sc, miss_e, miss_sc, promo_e, promo_slots = sc_buf
        n_hits, n_miss, n_promos = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
        pin_hs_np = arena._pin_hit_slots.numpy()
        pin_hsc_np = arena._pin_hit_sc.numpy()
        pin_sc_f32 = arena._pin_sc.numpy()
        pin_sc_f16 = arena._pin_sc_f16.numpy()

        for l_i in range(self.num_layers):
            promo_ev = arena._layer_promo_ev[l_i]
            if promo_ev is not None:
                compute_stream.wait_event(promo_ev)
                arena._layer_promo_ev[l_i] = None

            self._pre_graphs[l_i].replay()
            compute_stream.synchronize()
            pin_sc_f32[:K] = pin_sc_f16[:K]

            max_p = 0
            if arena._ring_events and not getattr(arena, "_disjoint_ram", False):
                ev0 = arena._ring_events[arena._ring_cursor]
                if ev0 is None or ev0.query():
                    ev1 = arena._ring_events[(arena._ring_cursor + 1) % arena._ring_slots]
                    max_p = 2 if (ev1 is None or ev1.query()) else 1

            rc = arena.bridge.lib.deimos_arena_resolve_hybrid_decode(
                arena.arena_handle, l_i,
                ctypes.cast(arena._pin_ids.data_ptr(), ctypes.POINTER(ctypes.c_int32)),
                ctypes.cast(arena._pin_sc.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                K, float(self.eng.miss_tail_tau), int(self.eng.min_k), max_p,
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
            if nh > 0:
                pin_hs_np[:nh] = hit_slots[:nh]
                pin_hsc_np[0, :nh] = hit_sc[:nh]
            if nh < K:
                pin_hs_np[nh:K] = 0
                pin_hsc_np[0, nh:K] = 0.0

            self._moe_graphs[l_i].replay()

            if nm > 0:
                rc_cpu = arena.bridge.lib.deimos_cpu_q4_swiglu_forward(
                    arena.bridge.handle, l_i,
                    miss_e.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    miss_sc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    nm,
                    ctypes.cast(arena._pin_x.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
                    ctypes.cast(arena._pin_cpu_out.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
                    self.D, arena.intermediate_size, arena.group_size,
                    1 if arena.verify_merkle_on_cold else 0,
                )
                if rc_cpu == -2:
                    raise DeimosMerkleIntegrityError(f"Corrupted Q4 Merkle leaf rejected in layer {l_i}!")
                if rc_cpu != 0:
                    raise RuntimeError(f"deimos_cpu_q4_swiglu_forward failed (rc={rc_cpu})")
                arena._gpu_cpu_out.copy_(arena._pin_cpu_out, non_blocking=True)
                self.static_h.view(1, self.D).add_(arena._gpu_cpu_out)

            if np_cnt > 0:
                arena._stage_via_ring(
                    l_i, promo_e[:np_cnt].tolist(), promo_slots[:np_cnt].tolist(),
                    wait_compute_first=True, block_compute_after=False,
                )

        assert self._head_graph is not None
        self._head_graph.replay()
        self._cur_pos += 1
        cache._deimos_gpos = self._cur_pos
        return self.static_logits

    def close(self) -> None:
        self._pre_graphs.clear()
        self._moe_graphs.clear()
        self._head_graph = None
        self.static_conv.clear()
        self.static_rec.clear()
        self.static_k.clear()
        self.static_v.clear()


def can_use_hf_cuda_graph(eng: "DeimosHFHybridMoEEngine") -> bool:
    mtype = str(eng.config.get("model_type", "")).lower()
    moe_ids = getattr(eng, "moe_layer_ids", list(range(eng.num_layers)))
    return bool(
        torch is not None
        and mtype in {"qwen3_5_moe", "qwen3_5", "qwen3_5_moe_text", "qwen3_5_text"}
        and len(moe_ids) == eng.num_layers
        and eng.device.type == "cuda"
        and hybrid_triton_available(eng.device)
        and not eng.arena.use_nocache_pread
        and not eng.arena.is_fully_pinned
        and eng.top_k <= 32
    )
