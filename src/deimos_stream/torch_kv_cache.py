#!/usr/bin/env python3
"""
DeimosStream PyTorch (`CUDA` / `CPU`) Merkle KV-Prefix Cache (`src/deimos_stream/torch_kv_cache.py`)
====================================================================================================
`DEIMIDX2` Robin-Hood 16-token chained SHA-256 KV-Prefix Cache using `.safetensors` blocks
100% cross-compatible between `DeimosPagedMoEEngine` (`MLX`) and `TorchDeimosPagedMoEEngine` (`CUDA`).
"""

from __future__ import annotations

import collections
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

from deimos_stream.arena import DeimosMerkleBridge
from deimos_stream.kv_cache import BLOCK_TOKENS, chain_kv_block_hash
from deimos_stream.torch_arena import HAS_TORCH

if HAS_TORCH:
    from safetensors.torch import load_file as st_load_torch, save_file as st_save_torch
    import torch
else:
    torch = None  # type: ignore[assignment]


class TorchKVCache:
    """Per-layer `[1, n_kv_heads, seq_len, head_dim]` KV cache matching MLX `.safetensors` layout."""

    def __init__(self) -> None:
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.offset: int = 0

    def update_and_fetch(
        self, k: "torch.Tensor", v: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        if self.keys is None or self.values is None:
            self.keys = k
            self.values = v
        else:
            self.keys = torch.cat([self.keys, k], dim=2)
            self.values = torch.cat([self.values, v], dim=2)
        self.offset = int(self.keys.shape[2])
        return self.keys, self.values


class TorchMerkleKVPrefixCache:
    """Bounded RAM + Disk `.safetensors` KV-Prefix Cache indexed by `DEIMIDX2`."""

    def __init__(
        self,
        store_dir: Path,
        bridge: DeimosMerkleBridge,
        num_layers: int,
        seed_root_hex: str,
        device: "torch.device",
        compute_dtype: "torch.dtype",
        max_ram_blocks: int = 32,
        max_disk_blocks: int = 256,
    ):
        self.kv_dir: Path = Path(store_dir).resolve() / "kv_cache_blocks"
        self.kv_dir.mkdir(parents=True, exist_ok=True)
        self.bridge = bridge
        self.num_layers = int(num_layers)
        self.seed_root32 = bytes.fromhex(seed_root_hex)
        self.device = device
        self.compute_dtype = compute_dtype
        self.max_ram_blocks = max(1, int(max_ram_blocks))
        self.max_disk_blocks = max(4, int(max_disk_blocks))
        self.ram_lru: collections.OrderedDict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = (
            collections.OrderedDict()
        )
        self.disk_lru: collections.OrderedDict[str, Path] = collections.OrderedDict()

    def _block_path(self, hash_hex: str) -> Path:
        return self.kv_dir / f"{hash_hex[:32]}_kv.safetensors"

    def _evict_hash(self, hash_hex: str) -> None:
        self.ram_lru.pop(hash_hex, None)
        p = self.disk_lru.pop(hash_hex, None) or self._block_path(hash_hex)
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    def match_longest_prefix(self, token_ids: Sequence[int], caches: List[TorchKVCache]) -> int:
        max_cacheable = max(0, len(token_ids) - 1)
        num_blocks = max_cacheable // BLOCK_TOKENS
        if num_blocks == 0:
            return 0
        now_ts = time.time()
        prev_h = self.seed_root32
        matched: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []

        for b_idx in range(num_blocks):
            blk = token_ids[b_idx * BLOCK_TOKENS : (b_idx + 1) * BLOCK_TOKENS]
            curr_h = chain_kv_block_hash(prev_h, blk)
            h_hex = curr_h.hex()
            if h_hex in self.ram_lru:
                self.ram_lru.move_to_end(h_hex)
                self.bridge.kv_lookup(curr_h, now_ts)
                matched.append(self.ram_lru[h_hex])
                prev_h = curr_h
                continue
            fpath = self._block_path(h_hex)
            if self.bridge.kv_lookup(curr_h, now_ts) and fpath.exists():
                loaded = st_load_torch(str(fpath), device="cpu")
                kv_layers = [
                    (
                        loaded[f"k_{l_i}"].to(device=self.device, dtype=self.compute_dtype),
                        loaded[f"v_{l_i}"].to(device=self.device, dtype=self.compute_dtype),
                    )
                    for l_i in range(self.num_layers)
                ]
                self.ram_lru[h_hex] = kv_layers
                while len(self.ram_lru) > self.max_ram_blocks:
                    self.ram_lru.popitem(last=False)
                matched.append(kv_layers)
                prev_h = curr_h
            else:
                break

        if not matched:
            return 0
        for l_i in range(self.num_layers):
            k_cat = torch.cat([b[l_i][0] for b in matched], dim=2)
            v_cat = torch.cat([b[l_i][1] for b in matched], dim=2)
            caches[l_i].update_and_fetch(k_cat, v_cat)
        return len(matched) * BLOCK_TOKENS

    def save_prefix_blocks(self, token_ids: Sequence[int], caches: List[TorchKVCache]) -> int:
        max_cacheable = max(0, len(token_ids) - 1)
        num_blocks = max_cacheable // BLOCK_TOKENS
        if num_blocks == 0:
            return 0
        now_ts = time.time()
        prev_h = self.seed_root32
        saved = 0
        for b_idx in range(num_blocks):
            start = b_idx * BLOCK_TOKENS
            end = start + BLOCK_TOKENS
            curr_h = chain_kv_block_hash(prev_h, token_ids[start:end])
            h_hex = curr_h.hex()
            fpath = self._block_path(h_hex)
            if h_hex not in self.ram_lru or not fpath.exists():
                kv_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
                tensors_cpu: Dict[str, torch.Tensor] = {}
                for l_i in range(self.num_layers):
                    assert caches[l_i].keys is not None and caches[l_i].values is not None
                    k_sl = caches[l_i].keys[:, :, start:end, :].contiguous()
                    v_sl = caches[l_i].values[:, :, start:end, :].contiguous()
                    kv_layers.append((k_sl, v_sl))
                    tensors_cpu[f"k_{l_i}"] = k_sl.to(device="cpu", dtype=torch.float16).contiguous()
                    tensors_cpu[f"v_{l_i}"] = v_sl.to(device="cpu", dtype=torch.float16).contiguous()
                self.ram_lru[h_hex] = kv_layers
                while len(self.ram_lru) > self.max_ram_blocks:
                    self.ram_lru.popitem(last=False)
                if not fpath.exists():
                    st_save_torch(tensors_cpu, str(fpath))
                    self.disk_lru[h_hex] = fpath
                    while len(self.disk_lru) > self.max_disk_blocks:
                        old_h, _ = self.disk_lru.popitem(last=False)
                        self._evict_hash(old_h)
                ev = self.bridge.kv_insert(curr_h, prev_h, end, now_ts)
                if ev is not None and ev != h_hex:
                    self._evict_hash(ev)
                saved += 1
            prev_h = curr_h
        return saved
