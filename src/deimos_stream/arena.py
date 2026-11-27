#!/usr/bin/env python3
"""
DeimosStream Pre-Allocated Quantized Expert VRAM Pool (`src/deimos_stream/arena.py`)
====================================================================================
Wraps `libdeimos_stream.dylib` (`DeimosMerkleBridge` + C++17 `DeimosArenaManager`):
  - Allocates ONE fixed per-layer 4-bit Expert Pool in Unified GPU Memory.
  - Executes LFU+LRU page mapping, zero-GIL background Residual Lookahead staging,
    and Cache-Aware Miss-Only Nucleus Dynamic-K filtering in native C++17.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None  # type: ignore[assignment]

from deimos_stream.heat_profile import save_arena_heat, seed_arena_from_heat
from deimos_stream.merkle_builder import compile_deimos_dylib, compute_q4_leaf_layout


class DeimosMerkleIntegrityError(RuntimeError):
    """Raised when `DEIMIDX2` Tier-1 or Tier-2 cryptographic Merkle verification fails."""


class _CStats(ctypes.Structure):
    _fields_ = [
        ("xor_proofs_verified", ctypes.c_uint64),
        ("byte_proofs_verified", ctypes.c_uint64),
        ("dma_slot_stages", ctypes.c_uint64),
        ("dma_bytes_staged", ctypes.c_uint64),
        ("evictions_issued", ctypes.c_uint64),
        ("prefetches_issued", ctypes.c_uint64),
    ]


class _CArenaTelemetry(ctypes.Structure):
    _fields_ = [
        ("cache_hits", ctypes.c_uint64),
        ("cache_misses", ctypes.c_uint64),
        ("evictions", ctypes.c_uint64),
        ("lookahead_submitted", ctypes.c_uint64),
        ("lookahead_staged", ctypes.c_uint64),
        ("lookahead_useful_hits", ctypes.c_uint64),
        ("tail_misses_skipped", ctypes.c_uint64),
        ("total_experts_evaluated", ctypes.c_uint64),
    ]


class _CHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_char * 8),
        ("version", ctypes.c_uint32),
        ("num_layers", ctypes.c_uint32),
        ("num_experts", ctypes.c_uint32),
        ("tree_leaf_cap", ctypes.c_uint32),
        ("vocab_size", ctypes.c_uint32),
        ("hidden_size", ctypes.c_uint32),
        ("intermediate_size", ctypes.c_uint32),
        ("quant_bits", ctypes.c_uint32),
        ("quant_group_size", ctypes.c_uint32),
        ("num_cow_leaves", ctypes.c_uint32),
        ("expert_stride_bytes", ctypes.c_uint64),
        ("expert_payload_bytes", ctypes.c_uint64),
        ("offset_xor_tree", ctypes.c_uint64),
        ("offset_expert_table", ctypes.c_uint64),
        ("offset_token_table", ctypes.c_uint64),
        ("offset_kv_table", ctypes.c_uint64),
        ("total_idx_bytes", ctypes.c_uint64),
        ("orig_safetensors_bytes", ctypes.c_uint64),
        ("base_model_root", ctypes.c_uint8 * 32),
        ("active_model_root", ctypes.c_uint8 * 32),
        ("embedding_root", ctypes.c_uint8 * 32),
        ("shared_core_root", ctypes.c_uint8 * 32),
        ("expert_tree_root", ctypes.c_uint8 * 32),
        ("output_head_root", ctypes.c_uint8 * 32),
        ("kv_table_slots", ctypes.c_uint32),
        ("reserved", ctypes.c_uint8 * 12),
    ]


class ArenaStats(TypedDict):
    slots_per_layer: int
    total_physical_slots: int
    pool_vram_mb: float
    cache_hits: int
    cache_misses: int
    hit_rate_pct: float
    evictions: int
    lookahead_submitted: int
    lookahead_staged: int
    lookahead_useful_hits: int
    tail_misses_skipped: int
    avg_active_k: float
    xor_proofs_verified: int
    byte_proofs_verified: int
    dma_bytes_staged_mb: float


class DeimosMerkleBridge:
    """Typed `ctypes` binding to `libdeimos_stream.dylib` (`DEIMIDX2` C++17 core)."""

    def __init__(
        self,
        store_dir: Path,
        cas_filename: str = "cas_experts_q4.bin",
        enable_nocache_fd: bool = True,
    ):
        dylib_path = compile_deimos_dylib(force=False)
        self.lib = ctypes.CDLL(str(dylib_path))
        self._bind_signatures()

        self.store_dir: Path = Path(store_dir).resolve()
        self.handle: Optional[int] = self.lib.deimos_open(
            str(self.store_dir).encode("utf-8"),
            cas_filename.encode("utf-8"),
            1 if enable_nocache_fd else 0,
        )
        if not self.handle:
            raise DeimosMerkleIntegrityError(
                f"DEIMIDX2 Tier-1 Merkle verification or mmap init failed at {self.store_dir}"
            )
        hdr = _CHeader()
        rc = self.lib.deimos_get_header(self.handle, ctypes.byref(hdr))
        if rc != 0:
            self.close()
            raise DeimosMerkleIntegrityError(f"Failed to read DEIMIDX2 header (rc={rc})")
        self.header: _CHeader = hdr

    def _bind_signatures(self) -> None:
        L = self.lib
        L.deimos_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        L.deimos_open.restype = ctypes.c_void_p
        L.deimos_close.argtypes = [ctypes.c_void_p]
        L.deimos_close.restype = None

        L.deimos_get_header.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CHeader)]
        L.deimos_get_header.restype = ctypes.c_int
        L.deimos_verify_leaf.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        L.deimos_verify_leaf.restype = ctypes.c_int
        L.deimos_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CStats)]
        L.deimos_get_stats.restype = None

        L.deimos_kv_lookup.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
        L.deimos_kv_lookup.restype = ctypes.c_int
        L.deimos_kv_insert.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_double,
            ctypes.c_char_p,
        ]
        L.deimos_kv_insert.restype = ctypes.c_int

        L.deimos_arena_create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
        ]
        L.deimos_arena_create.restype = ctypes.c_void_p
        L.deimos_arena_destroy.argtypes = [ctypes.c_void_p]
        L.deimos_arena_destroy.restype = None

        L.deimos_arena_bind_layer_ptrs.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        L.deimos_arena_bind_layer_ptrs.restype = ctypes.c_int
        L.deimos_arena_preload_all.argtypes = [ctypes.c_void_p]
        L.deimos_arena_preload_all.restype = ctypes.c_int

        L.deimos_arena_resolve_and_prefetch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
        ]
        L.deimos_arena_resolve_and_prefetch.restype = ctypes.c_int

        L.deimos_arena_wait_idle.argtypes = [ctypes.c_void_p]
        L.deimos_arena_wait_idle.restype = None
        L.deimos_arena_get_slot_gens.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int,
        ]
        L.deimos_arena_get_slot_gens.restype = ctypes.c_int
        L.deimos_arena_get_telemetry.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_CArenaTelemetry),
        ]
        L.deimos_arena_get_telemetry.restype = None
        L.deimos_arena_reset_telemetry.argtypes = [ctypes.c_void_p]
        L.deimos_arena_reset_telemetry.restype = None

    def verify_leaf(self, layer_idx: int, expert_idx: int, verify_bytes: bool = True) -> None:
        rc = self.lib.deimos_verify_leaf(
            self.handle, int(layer_idx), int(expert_idx), 1 if verify_bytes else 0
        )
        if rc != 0:
            raise DeimosMerkleIntegrityError(
                f"Merkle proof failed for L{layer_idx}E{expert_idx} (rc={rc})"
            )

    def kv_lookup(self, prefix_hash32: bytes, now_ts: float) -> bool:
        return bool(self.lib.deimos_kv_lookup(self.handle, prefix_hash32, float(now_ts)) == 1)

    def kv_insert(
        self, prefix_hash32: bytes, parent_hash32: bytes, token_end: int, now_ts: float
    ) -> Optional[str]:
        evicted_buf = ctypes.create_string_buffer(32)
        rc = self.lib.deimos_kv_insert(
            self.handle, prefix_hash32, parent_hash32, int(token_end), float(now_ts), evicted_buf
        )
        if rc == 1:
            return bytes(evicted_buf.raw).hex()
        return None

    def get_stats(self) -> Dict[str, int]:
        st = _CStats()
        if self.handle:
            self.lib.deimos_get_stats(self.handle, ctypes.byref(st))
        return {
            "xor_proofs_verified": int(st.xor_proofs_verified),
            "byte_proofs_verified": int(st.byte_proofs_verified),
            "dma_slot_stages": int(st.dma_slot_stages),
            "dma_bytes_staged": int(st.dma_bytes_staged),
            "evictions_issued": int(st.evictions_issued),
            "prefetches_issued": int(st.prefetches_issued),
        }

    def close(self) -> None:
        if self.handle:
            self.lib.deimos_close(self.handle)
            self.handle = None


class PagedExpertArena:
    """
    Fixed-VRAM Per-Layer Q4 Expert Pool backed by C++17 `DeimosArenaManager`
    for O(1) page mapping, zero-GIL Residual Lookahead prefetching, and Miss-Only Dynamic-K.
    """

    def __init__(
        self,
        bridge: DeimosMerkleBridge,
        slots_per_layer: int = 16,
        verify_merkle_on_cold: bool = True,
        use_nocache_pread: bool = False,
    ):
        self.bridge: DeimosMerkleBridge = bridge
        h = bridge.header
        self.num_layers: int = int(h.num_layers)
        self.num_experts: int = int(h.num_experts)
        self.hidden_size: int = int(h.hidden_size)
        self.intermediate_size: int = int(h.intermediate_size)
        self.group_size: int = int(h.quant_group_size)
        self.slots_per_layer: int = max(1, min(self.num_experts, int(slots_per_layer)))
        self.is_fully_pinned: bool = self.slots_per_layer == self.num_experts
        self.use_nocache_pread: bool = bool(use_nocache_pread)

        layout = compute_q4_leaf_layout(
            self.hidden_size, self.intermediate_size, group_size=self.group_size
        )
        self.payload_bytes: int = layout["payload_bytes"]

        C = self.slots_per_layer
        gu_out = 2 * self.intermediate_size
        self.gu_w: List[mx.array] = [
            mx.zeros((C, gu_out, self.hidden_size // 8), dtype=mx.uint32)
            for _ in range(self.num_layers)
        ]
        self.gu_s: List[mx.array] = [
            mx.zeros((C, gu_out, self.hidden_size // self.group_size), dtype=mx.float16)
            for _ in range(self.num_layers)
        ]
        self.gu_b: List[mx.array] = [
            mx.zeros((C, gu_out, self.hidden_size // self.group_size), dtype=mx.float16)
            for _ in range(self.num_layers)
        ]
        self.dn_w: List[mx.array] = [
            mx.zeros((C, self.hidden_size, self.intermediate_size // 8), dtype=mx.uint32)
            for _ in range(self.num_layers)
        ]
        self.dn_s: List[mx.array] = [
            mx.zeros(
                (C, self.hidden_size, self.intermediate_size // self.group_size), dtype=mx.float16
            )
            for _ in range(self.num_layers)
        ]
        self.dn_b: List[mx.array] = [
            mx.zeros(
                (C, self.hidden_size, self.intermediate_size // self.group_size), dtype=mx.float16
            )
            for _ in range(self.num_layers)
        ]
        mx.eval(self.gu_w, self.gu_s, self.gu_b, self.dn_w, self.dn_s, self.dn_b)
        mx.synchronize()

        self.arena_handle: Optional[int] = self.bridge.lib.deimos_arena_create(
            self.bridge.handle,
            self.num_layers,
            self.num_experts,
            self.slots_per_layer,
            layout["gu_w_bytes"],
            layout["gu_s_bytes"],
            layout["dn_w_bytes"],
            layout["dn_s_bytes"],
            1 if verify_merkle_on_cold else 0,
            1 if use_nocache_pread else 0,
        )
        if not self.arena_handle:
            raise RuntimeError("Failed to initialize C++17 DeimosArenaManager")

        self._views_keepalive: List[Tuple[np.ndarray, ...]] = []
        for l_idx in range(self.num_layers):
            vw = np.frombuffer(memoryview(self.gu_w[l_idx]), dtype=np.uint8)
            vs = np.frombuffer(memoryview(self.gu_s[l_idx]), dtype=np.uint8)
            vb = np.frombuffer(memoryview(self.gu_b[l_idx]), dtype=np.uint8)
            vdw = np.frombuffer(memoryview(self.dn_w[l_idx]), dtype=np.uint8)
            vds = np.frombuffer(memoryview(self.dn_s[l_idx]), dtype=np.uint8)
            vdb = np.frombuffer(memoryview(self.dn_b[l_idx]), dtype=np.uint8)
            self._views_keepalive.append((vw, vs, vb, vdw, vds, vdb))
            self.bridge.lib.deimos_arena_bind_layer_ptrs(
                self.arena_handle,
                l_idx,
                int(vw.ctypes.data),
                int(vs.ctypes.data),
                int(vb.ctypes.data),
                int(vdw.ctypes.data),
                int(vds.ctypes.data),
                int(vdb.ctypes.data),
            )

        if self.is_fully_pinned:
            rc = self.bridge.lib.deimos_arena_preload_all(self.arena_handle)
            if rc == -2:
                raise DeimosMerkleIntegrityError("Corrupted Q4 Merkle leaf detected during preload!")
            if rc != 0:
                raise RuntimeError(f"deimos_arena_preload_all failed (rc={rc})")
        else:
            seed_arena_from_heat(self, stage_bound_pools=True)

    def resolve_and_prefetch(
        self,
        layer_idx: int,
        routed_ids_np: np.ndarray,
        routed_scores_np: np.ndarray,
        miss_tail_tau: float = 1.0,
        min_k: int = 4,
        next_layer_idx: int = -1,
        next_pred_ids_np: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single C++17 call joining `layer_idx` lookahead, staging unpredicted misses,
        filtering cold tail misses (`miss_tail_tau`), and dispatching `next_layer_idx` lookahead.
        """
        num_tokens, top_k = routed_ids_np.shape
        ids_c = np.ascontiguousarray(routed_ids_np, dtype=np.int32)
        sc_c = np.ascontiguousarray(routed_scores_np, dtype=np.float32)

        out_phys = np.empty((num_tokens, top_k), dtype=np.int32)
        out_sc = np.empty((num_tokens, top_k), dtype=np.float32)
        eff_k = ctypes.c_int(top_k)

        pred_ptr = None
        pred_cnt = 0
        pred_c: Optional[np.ndarray] = None
        if next_layer_idx >= 0 and next_pred_ids_np is not None:
            pred_c = np.ascontiguousarray(next_pred_ids_np.reshape(-1), dtype=np.int32)
            pred_ptr = pred_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            pred_cnt = int(pred_c.size)

        rc = self.bridge.lib.deimos_arena_resolve_and_prefetch(
            self.arena_handle,
            int(layer_idx),
            ids_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            sc_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(num_tokens),
            int(top_k),
            float(miss_tail_tau),
            int(min_k),
            int(next_layer_idx),
            pred_ptr,
            pred_cnt,
            out_phys.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            out_sc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(eff_k),
        )
        if rc == -2:
            raise DeimosMerkleIntegrityError(
                f"Corrupted Q4 Merkle leaf rejected in layer {layer_idx}!"
            )
        if rc != 0:
            raise RuntimeError(f"deimos_arena_resolve_and_prefetch failed (rc={rc})")

        k_final = int(eff_k.value)
        if k_final < top_k and num_tokens == 1:
            return out_phys[:, :k_final], out_sc[:, :k_final]
        return out_phys, out_sc

    def wait_idle(self) -> None:
        if self.arena_handle:
            self.bridge.lib.deimos_arena_wait_idle(self.arena_handle)

    def reset_telemetry(self) -> None:
        if self.arena_handle:
            self.bridge.lib.deimos_arena_reset_telemetry(self.arena_handle)

    def pool_vram_mb(self) -> float:
        return (self.num_layers * self.slots_per_layer * self.payload_bytes) / (1024.0 * 1024.0)

    def get_stats(self, nominal_top_k: int = 8) -> ArenaStats:
        c_st = self.bridge.get_stats()
        a_st = _CArenaTelemetry()
        if self.arena_handle:
            self.bridge.lib.deimos_arena_get_telemetry(
                self.arena_handle, ctypes.byref(a_st)
            )
        hits = int(a_st.cache_hits)
        misses = int(a_st.cache_misses)
        tot = hits + misses
        skipped = int(a_st.tail_misses_skipped)
        eval_e = int(a_st.total_experts_evaluated)
        denom_calls = (eval_e + skipped) / max(1, nominal_top_k)
        avg_k = (eval_e / denom_calls) if denom_calls > 0 else float(nominal_top_k)
        return {
            "slots_per_layer": self.slots_per_layer,
            "total_physical_slots": self.num_layers * self.slots_per_layer,
            "pool_vram_mb": round(self.pool_vram_mb(), 2),
            "cache_hits": hits,
            "cache_misses": misses,
            "hit_rate_pct": round(100.0 * hits / tot, 2) if tot > 0 else 100.0,
            "evictions": int(a_st.evictions),
            "lookahead_submitted": int(a_st.lookahead_submitted),
            "lookahead_staged": int(a_st.lookahead_staged),
            "lookahead_useful_hits": int(a_st.lookahead_useful_hits),
            "tail_misses_skipped": skipped,
            "avg_active_k": round(avg_k, 2),
            "xor_proofs_verified": c_st["xor_proofs_verified"],
            "byte_proofs_verified": c_st["byte_proofs_verified"],
            "dma_bytes_staged_mb": round(c_st["dma_bytes_staged"] / (1024.0 * 1024.0), 2),
        }

    def close(self) -> None:
        if self.arena_handle:
            save_arena_heat(self)
            self.bridge.lib.deimos_arena_destroy(self.arena_handle)
            self.arena_handle = None
        self._views_keepalive.clear()
        self.gu_w.clear()
        self.gu_s.clear()
        self.gu_b.clear()
        self.dn_w.clear()
        self.dn_s.clear()
        self.dn_b.clear()
