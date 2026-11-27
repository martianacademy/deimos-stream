#!/usr/bin/env python3
"""
DeimosStream TinyLFU Paged `FP8 E4M3` Expert Slot Arena
(`src/deimos_stream/fp8_arena.py`)
=======================================================
Manages physical `[S, 2*I, D]` `uint8` + `[S, 2*I/bm, D/bn]` `float32` slot pools per MoE layer
in Apple Silicon Metal Unified Memory (`mlx.core`) or PyTorch (`cuda` / `cpu`), paging cold
experts directly from raw `.safetensors` shards via `F_NOCACHE` `os.pread` (`0` file-cache swap)
and executing hit/staged slots through fused Metal / Triton `FP8` SwiGLU kernels.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from deimos_stream.arena import ArenaStats
from deimos_stream.fp8_kernels import HAS_MLX_FP8, mlx_fp8_moe_swiglu, torch_fp8_moe_swiglu
from deimos_stream.fp8_safetensors import SafetensorsZeroCopyIndex
from deimos_stream.torch_arena import HAS_TORCH

if HAS_MLX_FP8:
    import mlx.core as mx
else:
    mx = None  # type: ignore[assignment]

if HAS_TORCH:
    import torch
else:
    torch = None  # type: ignore[assignment]


class DeimosFP8PagedArena:
    """TinyLFU-managed paged `FP8 E4M3` expert slot pool backed by `F_NOCACHE` `.safetensors` reads."""

    def __init__(
        self,
        index: SafetensorsZeroCopyIndex,
        device: "torch.device",
        slots_per_layer: int,
        use_mlx_metal: bool = False,
        use_nocache_pread: bool = True,
    ):
        self.index = index
        self.device = device
        self.use_mlx_metal = bool(use_mlx_metal and HAS_MLX_FP8)
        self.use_nocache_pread = bool(use_nocache_pread)
        self.num_layers = index.num_moe_layers
        self.num_experts = index.num_experts
        self.slots_per_layer = max(1, min(self.num_experts, int(slots_per_layer)))
        self.is_fully_pinned = False

        D, I = index.hidden_size, index.inter_size
        bm, bn = index.block_m, index.block_n
        S = self.slots_per_layer
        self.swiglu_limit = 0.0
        self.swiglu_alpha = 1.0
        self.swiglu_up_offset = 0.0

        if self.use_mlx_metal:
            self.gu_w = [mx.zeros((S, 2 * I, D), dtype=mx.uint8) for _ in range(self.num_layers)]
            self.gu_s = [mx.zeros((S, (2 * I) // bm, D // bn), dtype=mx.float32) for _ in range(self.num_layers)]
            self.dn_w = [mx.zeros((S, D, I), dtype=mx.uint8) for _ in range(self.num_layers)]
            self.dn_s = [mx.zeros((S, D // bm, I // bn), dtype=mx.float32) for _ in range(self.num_layers)]
            mx.eval(*self.gu_w, *self.gu_s, *self.dn_w, *self.dn_s)
        else:
            self.gu_w = [torch.zeros((S, 2 * I, D), dtype=torch.uint8, device=device) for _ in range(self.num_layers)]
            self.gu_s = [
                torch.zeros((S, (2 * I) // bm, D // bn), dtype=torch.float32, device=device)
                for _ in range(self.num_layers)
            ]
            self.dn_w = [torch.zeros((S, D, I), dtype=torch.uint8, device=device) for _ in range(self.num_layers)]
            self.dn_s = [
                torch.zeros((S, D // bm, I // bn), dtype=torch.float32, device=device)
                for _ in range(self.num_layers)
            ]

        self.exp_to_slot = np.full((self.num_layers, self.num_experts), -1, dtype=np.int32)
        self.slot_to_exp = np.full((self.num_layers, S), -1, dtype=np.int32)
        self.slot_freq = np.zeros((self.num_layers, S), dtype=np.int32)
        self.efreq = np.zeros((self.num_layers, self.num_experts), dtype=np.int32)
        self.reset_telemetry()

    def reset_telemetry(self) -> None:
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        self.tail_misses_skipped = 0
        self.active_expert_evals = 0
        self.token_steps = 0
        self.dma_bytes_staged = 0

    def wait_idle(self) -> None:
        if self.use_mlx_metal:
            mx.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def pool_vram_mb(self) -> float:
        return (self.num_layers * self.slots_per_layer * self.index.expert_bytes) / (1024.0 * 1024.0)

    def get_stats(self, nominal_top_k: int = 8) -> ArenaStats:
        total = self.cache_hits + self.cache_misses
        hr = (100.0 * self.cache_hits / total) if total > 0 else 0.0
        avg_k = (self.active_expert_evals / self.token_steps) if self.token_steps > 0 else float(nominal_top_k)
        return {
            "slots_per_layer": self.slots_per_layer,
            "pool_vram_mb": round(self.pool_vram_mb(), 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_pct": round(hr, 2),
            "evictions": self.evictions,
            "lookahead_staged": 0,
            "lookahead_useful_hits": 0,
            "tail_misses_skipped": self.tail_misses_skipped,
            "avg_active_k": round(avg_k, 2),
            "xor_proofs_verified": 0,
            "byte_proofs_verified": 0,
            "dma_bytes_staged_mb": round(self.dma_bytes_staged / (1024.0 * 1024.0), 2),
        }

    def _stage_span(self, l_idx: int, ids_span: np.ndarray) -> np.ndarray:
        e2s = self.exp_to_slot[l_idx]
        s2e = self.slot_to_exp[l_idx]
        sfreq = self.slot_freq[l_idx]
        ef = self.efreq[l_idx]

        active_set = set()
        miss_list: List[int] = []
        miss_seen = set()
        for e in ids_span.ravel():
            ei = int(e)
            if ei < 0:
                continue
            active_set.add(ei)
            ef[ei] = min(int(ef[ei]) + 1, 1_000_000)
            ps = int(e2s[ei])
            if ps >= 0:
                self.cache_hits += 1
                sfreq[ps] = ef[ei]
            else:
                self.cache_misses += 1
                if ei not in miss_seen:
                    miss_seen.add(ei)
                    miss_list.append(ei)

        if miss_list:
            loaded = (
                [self.index.read_expert_fp8(l_idx, miss_list[0])]
                if len(miss_list) == 1
                else list(self.index._io_pool.map(lambda ex: self.index.read_expert_fp8(l_idx, ex), miss_list))
            )
            protected_slots = {int(e2s[ex]) for ex in active_set if int(e2s[ex]) >= 0}
            for ei, (gu_u8, gu_sc, dn_u8, dn_sc) in zip(miss_list, loaded):
                free_slots = [s for s in range(self.slots_per_layer) if s2e[s] < 0 and s not in protected_slots]
                if free_slots:
                    ps = free_slots[0]
                else:
                    cands = [s for s in range(self.slots_per_layer) if s not in protected_slots]
                    ps = min(cands, key=lambda s: int(sfreq[s]))
                    old_e = int(s2e[ps])
                    if old_e >= 0:
                        e2s[old_e] = -1
                    self.evictions += 1
                protected_slots.add(ps)
                s2e[ps] = ei
                e2s[ei] = ps
                sfreq[ps] = ef[ei]
                self.dma_bytes_staged += self.index.expert_bytes
                if self.use_mlx_metal:
                    self.gu_w[l_idx][ps] = mx.array(gu_u8)
                    self.gu_s[l_idx][ps] = mx.array(gu_sc)
                    self.dn_w[l_idx][ps] = mx.array(dn_u8)
                    self.dn_s[l_idx][ps] = mx.array(dn_sc)
                else:
                    self.gu_w[l_idx][ps].copy_(torch.from_numpy(gu_u8))
                    self.gu_s[l_idx][ps].copy_(torch.from_numpy(gu_sc))
                    self.dn_w[l_idx][ps].copy_(torch.from_numpy(dn_u8))
                    self.dn_s[l_idx][ps].copy_(torch.from_numpy(dn_sc))
            if self.use_mlx_metal:
                mx.eval(self.gu_w[l_idx], self.gu_s[l_idx], self.dn_w[l_idx], self.dn_s[l_idx])

        slots_out = np.full(ids_span.shape, -1, dtype=np.int32)
        valid = ids_span >= 0
        slots_out[valid] = e2s[ids_span[valid]]
        return slots_out

    def forward_moe(
        self,
        layer_idx: int,
        hidden_states: "torch.Tensor",
        top_k_index: "torch.Tensor",
        top_k_weights: "torch.Tensor",
        miss_tail_tau: float = 1.0,
        min_k: int = 4,
        shared_fn: Optional[Callable[[], "torch.Tensor"]] = None,
    ) -> "torch.Tensor":
        N = int(hidden_states.shape[0])
        K = int(top_k_index.shape[1])
        ids_np = top_k_index.detach().cpu().numpy().astype(np.int32, copy=True)
        wts_np = top_k_weights.detach().float().cpu().numpy().astype(np.float32, copy=True)

        if miss_tail_tau < 0.9999:
            e2s = self.exp_to_slot[layer_idx]
            for ti in range(N):
                order = np.argsort(-wts_np[ti])
                row_sum = float(np.sum(wts_np[ti]))
                cum = 0.0
                kept = 0
                for ki in order:
                    ei = int(ids_np[ti, ki])
                    w = float(wts_np[ti, ki])
                    if e2s[ei] >= 0 or kept < min_k or cum < miss_tail_tau * row_sum:
                        cum += w
                        kept += 1
                    else:
                        ids_np[ti, ki] = -1
                        wts_np[ti, ki] = 0.0
                        self.tail_misses_skipped += 1
                new_sum = float(np.sum(wts_np[ti]))
                if new_sum > 1e-7 and new_sum < row_sum:
                    wts_np[ti] *= row_sum / new_sum

        self.token_steps += N
        self.active_expert_evals += int(np.sum(ids_np >= 0))
        sh_out = shared_fn() if shared_fn is not None else None

        spans: List[Tuple[int, int]] = []
        t0 = 0
        while t0 < N:
            uniq = set(int(x) for x in ids_np[t0] if x >= 0)
            t1 = t0 + 1
            while t1 < N:
                cand = uniq | set(int(x) for x in ids_np[t1] if x >= 0)
                if len(cand) > self.slots_per_layer:
                    break
                uniq = cand
                t1 += 1
            spans.append((t0, t1))
            t0 = t1

        bm, bn = self.index.block_m, self.index.block_n
        out_parts: List[torch.Tensor] = []
        for s0, s1 in spans:
            slots_span = self._stage_span(layer_idx, ids_span=ids_np[s0:s1])
            sc_flat = wts_np[s0:s1].reshape(-1)
            sl_flat = slots_span.reshape(-1)
            if self.use_mlx_metal:
                x_np = hidden_states[s0:s1].detach().to(torch.float16).cpu().numpy()
                y_mx = mlx_fp8_moe_swiglu(
                    mx.array(x_np),
                    self.gu_w[layer_idx],
                    self.gu_s[layer_idx],
                    self.dn_w[layer_idx],
                    self.dn_s[layer_idx],
                    mx.array(sl_flat),
                    mx.array(sc_flat),
                    top_k=K,
                    block_m=bm,
                    block_n=bn,
                    swiglu_limit=self.swiglu_limit,
                    swiglu_alpha=self.swiglu_alpha,
                    swiglu_up_offset=self.swiglu_up_offset,
                )
                mx.eval(y_mx)
                y_part = torch.from_numpy(np.array(y_mx, copy=True)).to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
            else:
                y_part = torch_fp8_moe_swiglu(
                    hidden_states[s0:s1],
                    self.gu_w[layer_idx],
                    self.gu_s[layer_idx],
                    self.dn_w[layer_idx],
                    self.dn_s[layer_idx],
                    torch.from_numpy(sl_flat).to(device=self.device, dtype=torch.int32),
                    torch.from_numpy(sc_flat).to(device=self.device, dtype=torch.float32),
                    top_k=K,
                    block_m=bm,
                    block_n=bn,
                    swiglu_limit=self.swiglu_limit,
                    swiglu_alpha=self.swiglu_alpha,
                    swiglu_up_offset=self.swiglu_up_offset,
                ).to(hidden_states.dtype)
            out_parts.append(y_part)

        y = out_parts[0] if len(out_parts) == 1 else torch.cat(out_parts, dim=0)
        return (y + sh_out) if sh_out is not None else y

    def close(self) -> None:
        self.gu_w.clear()
        self.gu_s.clear()
        self.dn_w.clear()
        self.dn_s.clear()
        if self.use_mlx_metal:
            mx.clear_cache()
