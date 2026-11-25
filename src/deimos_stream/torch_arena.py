#!/usr/bin/env python3
"""
DeimosStream PyTorch (`CUDA` / `CPU`) Paged Q4 Expert Arena (`src/deimos_stream/torch_arena.py`)
===============================================================================================
Implements fixed-VRAM 4-bit Expert Slot Pooling for discrete NVIDIA CUDA GPUs (and CPU CI):
  - On `cuda`: Allocates ONE fixed Q4 slot pool in GPU HBM (`[C, out_dim, in_dim // 2]` `uint8` +
    `float16` scales/biases) AND a page-locked (`pin_memory=True`) host staging shadow bound to
    C++17 `DeimosArenaManager` (`F_NOCACHE` `pread` + SHA-256 Merkle proofs + Residual Lookahead).
    Uses ABA-safe `deimos_arena_get_slot_gens` to stream ONLY newly staged slots over PCIe via
    `non_blocking=True` DMA on a dedicated `torch.cuda.Stream`.
  - On `cpu`: Binds CPU tensors directly to `DeimosArenaManager` with zero shadow memory overhead.
"""

from __future__ import annotations

import ctypes
from typing import List, Optional, Sequence, Tuple

import numpy as np

from deimos_stream.arena import (
    ArenaStats,
    DeimosMerkleBridge,
    DeimosMerkleIntegrityError,
    _CArenaTelemetry,
)
from deimos_stream.merkle_builder import compute_q4_leaf_layout

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False


class TorchPagedExpertArena:
    """Fixed-VRAM Q4 Expert Slot Pool for PyTorch `cuda` (pinned async PCIe DMA) and `cpu`."""

    def __init__(
        self,
        bridge: DeimosMerkleBridge,
        device: "torch.device",
        slots_per_layer: int = 16,
        verify_merkle_on_cold: bool = True,
        use_nocache_pread: bool = False,
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for TorchPagedExpertArena (`pip install deimos-stream[cuda]`).")
        self.bridge: DeimosMerkleBridge = bridge
        self.device: torch.device = device
        self.is_cuda: bool = device.type == "cuda"

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
        gu_half_in = self.hidden_size // 2
        gu_groups = self.hidden_size // self.group_size
        dn_half_in = self.intermediate_size // 2
        dn_groups = self.intermediate_size // self.group_size

        def _alloc_layer_set(dev: torch.device, pinned: bool) -> Tuple[torch.Tensor, ...]:
            kw = {"device": dev, "pin_memory": pinned}
            return (
                torch.zeros((C, gu_out, gu_half_in), dtype=torch.uint8, **kw),
                torch.zeros((C, gu_out, gu_groups), dtype=torch.float16, **kw),
                torch.zeros((C, gu_out, gu_groups), dtype=torch.float16, **kw),
                torch.zeros((C, self.hidden_size, dn_half_in), dtype=torch.uint8, **kw),
                torch.zeros((C, self.hidden_size, dn_groups), dtype=torch.float16, **kw),
                torch.zeros((C, self.hidden_size, dn_groups), dtype=torch.float16, **kw),
            )

        self.gu_w: List[torch.Tensor] = []
        self.gu_s: List[torch.Tensor] = []
        self.gu_b: List[torch.Tensor] = []
        self.dn_w: List[torch.Tensor] = []
        self.dn_s: List[torch.Tensor] = []
        self.dn_b: List[torch.Tensor] = []

        self._host_shadow: List[Tuple[torch.Tensor, ...]] = []
        self._gpu_slot_gen: List[np.ndarray] = [
            np.zeros((C,), dtype=np.uint64) for _ in range(self.num_layers)
        ]
        self._dma_stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=self.device) if self.is_cuda else None
        )

        for _ in range(self.num_layers):
            if self.is_cuda:
                gw, gs, gb, dw, ds, db = _alloc_layer_set(self.device, pinned=False)
                hw = _alloc_layer_set(torch.device("cpu"), pinned=True)
                self._host_shadow.append(hw)
            else:
                gw, gs, gb, dw, ds, db = _alloc_layer_set(torch.device("cpu"), pinned=False)
                self._host_shadow.append((gw, gs, gb, dw, ds, db))
            self.gu_w.append(gw)
            self.gu_s.append(gs)
            self.gu_b.append(gb)
            self.dn_w.append(dw)
            self.dn_s.append(ds)
            self.dn_b.append(db)

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
            raise RuntimeError("Failed to initialize C++17 DeimosArenaManager for PyTorch")

        for l_idx in range(self.num_layers):
            hw = self._host_shadow[l_idx]
            self.bridge.lib.deimos_arena_bind_layer_ptrs(
                self.arena_handle,
                l_idx,
                int(hw[0].data_ptr()),
                int(hw[1].data_ptr()),
                int(hw[2].data_ptr()),
                int(hw[3].data_ptr()),
                int(hw[4].data_ptr()),
                int(hw[5].data_ptr()),
            )

        if self.is_fully_pinned:
            rc = self.bridge.lib.deimos_arena_preload_all(self.arena_handle)
            if rc == -2:
                raise DeimosMerkleIntegrityError("Corrupted Q4 Merkle leaf detected during preload!")
            if rc != 0:
                raise RuntimeError(f"deimos_arena_preload_all failed (rc={rc})")
            if self.is_cuda:
                for l_idx in range(self.num_layers):
                    self._sync_dirty_slots_to_cuda(l_idx, range(self.slots_per_layer))
                torch.cuda.synchronize(self.device)
                # Free pinned host shadow once 100% pinned in CUDA HBM!
                self._host_shadow.clear()

    def _sync_dirty_slots_to_cuda(self, l_idx: int, needed_slots: Sequence[int]) -> None:
        if not self.is_cuda or not self._host_shadow:
            return
        cpp_gens = np.empty((self.slots_per_layer,), dtype=np.uint64)
        self.bridge.lib.deimos_arena_get_slot_gens(
            self.arena_handle,
            int(l_idx),
            cpp_gens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            self.slots_per_layer,
        )
        gpu_gens = self._gpu_slot_gen[l_idx]
        dirty = [int(s) for s in set(needed_slots) if gpu_gens[int(s)] != cpp_gens[int(s)]]
        if not dirty:
            return

        hw = self._host_shadow[l_idx]
        assert self._dma_stream is not None
        with torch.cuda.stream(self._dma_stream):
            for ps in dirty:
                self.gu_w[l_idx][ps].copy_(hw[0][ps], non_blocking=True)
                self.gu_s[l_idx][ps].copy_(hw[1][ps], non_blocking=True)
                self.gu_b[l_idx][ps].copy_(hw[2][ps], non_blocking=True)
                self.dn_w[l_idx][ps].copy_(hw[3][ps], non_blocking=True)
                self.dn_s[l_idx][ps].copy_(hw[4][ps], non_blocking=True)
                self.dn_b[l_idx][ps].copy_(hw[5][ps], non_blocking=True)
                gpu_gens[ps] = cpp_gens[ps]
        torch.cuda.current_stream(self.device).wait_stream(self._dma_stream)

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
        phys_res = out_phys[:, :k_final] if (k_final < top_k and num_tokens == 1) else out_phys
        sc_res = out_sc[:, :k_final] if (k_final < top_k and num_tokens == 1) else out_sc

        if self.is_cuda and not self.is_fully_pinned:
            self._sync_dirty_slots_to_cuda(layer_idx, phys_res.reshape(-1).tolist())

        return phys_res, sc_res

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
            self.bridge.lib.deimos_arena_get_telemetry(self.arena_handle, ctypes.byref(a_st))
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
            self.bridge.lib.deimos_arena_destroy(self.arena_handle)
            self.arena_handle = None
        self._host_shadow.clear()
        self.gu_w.clear()
        self.gu_s.clear()
        self.gu_b.clear()
        self.dn_w.clear()
        self.dn_s.clear()
        self.dn_b.clear()
        if self.is_cuda:
            torch.cuda.empty_cache()
