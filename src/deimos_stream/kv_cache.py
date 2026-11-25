#!/usr/bin/env python3
"""
DeimosStream Cryptographic Merkle KV-Prefix Cache (`src/deimos_stream/kv_cache.py`)
===================================================================================
Chains 16-token prompt blocks cryptographically from `model_merkle_root`:
  `H_b = SHA256(0x02 || H_{b-1} || int32_le(tok_0..tok_15))`
Probes the `DEIMIDX2` C++ Robin-Hood hash table in `O(1)` (`deimos_kv_lookup`),
restores cached `.safetensors` KV blocks directly into MLX `KVCache` instances, and
automatically cleans up evicted `.safetensors` files when Robin-Hood slots rotate.
"""

from __future__ import annotations

import collections
import hashlib
from pathlib import Path
import struct
import time
from typing import Dict, List, Sequence, Tuple

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from deimos_stream.arena import DeimosMerkleBridge

BLOCK_TOKENS: int = 16
_BLOCK_PACK = struct.Struct("<16i")


def chain_kv_block_hash(prev_hash32: bytes, block_tokens: Sequence[int]) -> bytes:
    """Computes `SHA256(0x02 || prev_hash32 || 16 x int32_le)`."""
    h = hashlib.sha256()
    h.update(b"\x02")
    h.update(prev_hash32)
    h.update(_BLOCK_PACK.pack(*[int(t) for t in block_tokens]))
    return h.digest()


class MerkleKVPrefixCache:
    """Bounded RAM + Disk `.safetensors` KV-Prefix Cache indexed by `DEIMIDX2`."""

    def __init__(
        self,
        store_dir: Path,
        bridge: DeimosMerkleBridge,
        num_layers: int,
        seed_root_hex: str,
        max_ram_blocks: int = 32,
        max_disk_blocks: int = 256,
    ):
        self.kv_dir: Path = Path(store_dir).resolve() / "kv_cache_blocks"
        self.kv_dir.mkdir(parents=True, exist_ok=True)
        self.bridge: DeimosMerkleBridge = bridge
        self.num_layers: int = int(num_layers)
        self.seed_root32: bytes = bytes.fromhex(seed_root_hex)
        self.max_ram_blocks: int = max(1, int(max_ram_blocks))
        self.max_disk_blocks: int = max(4, int(max_disk_blocks))

        self.ram_lru: collections.OrderedDict[str, List[Tuple[mx.array, mx.array]]] = (
            collections.OrderedDict()
        )
        self.disk_lru: collections.OrderedDict[str, Path] = collections.OrderedDict()
        self.tokens_skipped: int = 0
        self.ram_hits: int = 0
        self.disk_hits: int = 0
        self.blocks_saved: int = 0
        self.evictions: int = 0

    def _block_path(self, hash_hex: str) -> Path:
        return self.kv_dir / f"{hash_hex[:32]}_kv.safetensors"

    def _evict_hash(self, hash_hex: str) -> None:
        self.ram_lru.pop(hash_hex, None)
        p = self.disk_lru.pop(hash_hex, None) or self._block_path(hash_hex)
        if p.exists():
            try:
                p.unlink()
                self.evictions += 1
            except OSError:
                pass

    def _put_ram(self, hash_hex: str, kv_layers: List[Tuple[mx.array, mx.array]]) -> None:
        self.ram_lru[hash_hex] = kv_layers
        self.ram_lru.move_to_end(hash_hex)
        while len(self.ram_lru) > self.max_ram_blocks:
            self.ram_lru.popitem(last=False)

    def match_longest_prefix(
        self, token_ids: Sequence[int], caches: List[KVCache]
    ) -> int:
        """
        Matches as many 16-token prefix blocks as possible (leaving >= 1 token to produce logits),
        populating `caches` in a single concatenated update per layer. Returns matched token count.
        """
        max_cacheable = max(0, len(token_ids) - 1)
        num_blocks = max_cacheable // BLOCK_TOKENS
        if num_blocks == 0:
            return 0

        now_ts = time.time()
        prev_h = self.seed_root32
        matched_blocks: List[List[Tuple[mx.array, mx.array]]] = []

        for b_idx in range(num_blocks):
            blk = token_ids[b_idx * BLOCK_TOKENS : (b_idx + 1) * BLOCK_TOKENS]
            curr_h = chain_kv_block_hash(prev_h, blk)
            h_hex = curr_h.hex()

            if h_hex in self.ram_lru:
                self.ram_lru.move_to_end(h_hex)
                self.bridge.kv_lookup(curr_h, now_ts)
                matched_blocks.append(self.ram_lru[h_hex])
                self.ram_hits += 1
                prev_h = curr_h
                continue

            fpath = self._block_path(h_hex)
            if self.bridge.kv_lookup(curr_h, now_ts) and fpath.exists():
                loaded = mx.load(str(fpath))
                kv_layers = [
                    (loaded[f"k_{l_idx}"], loaded[f"v_{l_idx}"])
                    for l_idx in range(self.num_layers)
                ]
                mx.eval(kv_layers)
                self._put_ram(h_hex, kv_layers)
                matched_blocks.append(kv_layers)
                self.disk_hits += 1
                prev_h = curr_h
            else:
                break

        if not matched_blocks:
            return 0

        for l_idx in range(self.num_layers):
            k_cat = (
                matched_blocks[0][l_idx][0]
                if len(matched_blocks) == 1
                else mx.concatenate([b[l_idx][0] for b in matched_blocks], axis=2)
            )
            v_cat = (
                matched_blocks[0][l_idx][1]
                if len(matched_blocks) == 1
                else mx.concatenate([b[l_idx][1] for b in matched_blocks], axis=2)
            )
            caches[l_idx].update_and_fetch(k_cat, v_cat)

        mx.eval([c.state for c in caches])
        matched_toks = len(matched_blocks) * BLOCK_TOKENS
        self.tokens_skipped += matched_toks
        return matched_toks

    def save_prefix_blocks(self, token_ids: Sequence[int], caches: List[KVCache]) -> int:
        """Persists newly computed 16-token KV blocks into RAM, Disk `.safetensors`, and `DEIMIDX2`."""
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
                kv_layers: List[Tuple[mx.array, mx.array]] = []
                arrays: Dict[str, mx.array] = {}
                for l_idx in range(self.num_layers):
                    k_sl = mx.contiguous(caches[l_idx].keys[..., start:end, :]).astype(mx.float16)
                    v_sl = mx.contiguous(caches[l_idx].values[..., start:end, :]).astype(mx.float16)
                    kv_layers.append((k_sl, v_sl))
                    arrays[f"k_{l_idx}"] = k_sl
                    arrays[f"v_{l_idx}"] = v_sl
                mx.eval(arrays)
                self._put_ram(h_hex, kv_layers)

                if not fpath.exists():
                    mx.save_safetensors(str(fpath), arrays)
                    self.disk_lru[h_hex] = fpath
                    self.disk_lru.move_to_end(h_hex)
                    while len(self.disk_lru) > self.max_disk_blocks:
                        old_h, _ = self.disk_lru.popitem(last=False)
                        self._evict_hash(old_h)
                    self.blocks_saved += 1

                evicted_hex = self.bridge.kv_insert(curr_h, prev_h, end, now_ts)
                if evicted_hex is not None and evicted_hex != h_hex:
                    self._evict_hash(evicted_hex)
                saved += 1

            prev_h = curr_h
        return saved
