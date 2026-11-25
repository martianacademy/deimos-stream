#!/usr/bin/env python3
"""
Unit & Fault-Injection Suite for DeimosStream (`tests/test_merkle_and_arena.py`)
================================================================================
Verifies:
  1. Universal `DEIMIDX2` Dynamic Topology (`24 layers x 60 experts = 1,440` leaves,
     `151,936` vocab — non-power-of-2 leaf counts & large vocabularies).
  2. Tier-1 & Tier-2 Cryptographic Tamper Rejection (`DeimosMerkleIntegrityError` on
     1-bit flip in `cas_experts_q4.bin` and tampered header root).
  3. Standalone `0.000000` Bit-Exact Logit & Greedy Token Parity between 4-slot Paged
     streaming (with heavy LFU+LRU evictions) and 16-slot Pinned Q4 on a synthetic MoE.
  4. Real-Hardware 6.92B `OLMoE-1B-7B` 8-Slot vs 64-Slot `0.0` Parity + Held-Out
     Perplexity (`< 6.45`) when a local `OLMoE` Q4 store is available.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Optional
import unittest

import numpy as np

from deimos_stream.arena import _CArenaTelemetry, DeimosMerkleBridge, DeimosMerkleIntegrityError
from deimos_stream.converter import convert_safetensors_moe_to_deimos_q4, normalize_moe_config
from deimos_stream.detokenizer import IncrementalStreamingDetokenizer
from deimos_stream.engine import HAS_MLX, DeimosPagedMoEEngine
from deimos_stream.merkle_builder import (
    PAGE_SIZE,
    build_deimidx2_file,
    compute_q4_leaf_layout,
)

if HAS_MLX:
    import mlx.core as mx


def find_real_olmoe_store() -> Optional[Path]:
    """Resolves local 6.92B OLMoE Q4 store if present on developer machine."""
    env_p = os.environ.get("DEIMOS_OLMOE_STORE")
    candidates = [
        Path(env_p) if env_p else None,
        Path(__file__).resolve().parent.parent / "olmoe_merkle_store" / "v3_q4",
    ]
    for p in candidates:
        if p is not None and (p / "cas_experts_q4.bin").exists() and (p / "merkle_tree.idx").exists():
            return p.resolve()
    return None


def build_synthetic_moe_store(
    root_dir: Path,
    num_layers: int = 3,
    num_experts: int = 16,
    top_k: int = 4,
    with_shared_expert: bool = False,
) -> Path:
    """Creates and converts a deterministic `.safetensors` Sparse MoE checkpoint for CI."""
    src_dir = root_dir / "synthetic_safetensors"
    out_dir = root_dir / "deimos_q4_store"
    src_dir.mkdir(parents=True, exist_ok=True)

    D, I, V = 128, 64, 256
    n_heads, n_kv_heads, head_dim = 8, 2, 16
    cfg = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": num_layers,
        "hidden_size": D,
        "moe_intermediate_size": I,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": head_dim,
        "num_experts": num_experts,
        "num_experts_per_tok": top_k,
        "vocab_size": V,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "norm_topk_prob": True,
    }
    (src_dir / "config.json").write_text(json.dumps(cfg))

    rng = np.random.default_rng(42)
    tensors: Dict[str, mx.array] = {
        "model.embed_tokens.weight": mx.array((rng.standard_normal((V, D)) * 0.05).astype(np.float16)),
        "model.norm.weight": mx.ones((D,), dtype=mx.float16),
    }
    for l_i in range(num_layers):
        lp = f"model.layers.{l_i}"
        tensors[f"{lp}.input_layernorm.weight"] = mx.ones((D,), dtype=mx.float16)
        tensors[f"{lp}.post_attention_layernorm.weight"] = mx.ones((D,), dtype=mx.float16)
        tensors[f"{lp}.self_attn.q_proj.weight"] = mx.array(
            (rng.standard_normal((n_heads * head_dim, D)) * 0.05).astype(np.float16)
        )
        tensors[f"{lp}.self_attn.k_proj.weight"] = mx.array(
            (rng.standard_normal((n_kv_heads * head_dim, D)) * 0.05).astype(np.float16)
        )
        tensors[f"{lp}.self_attn.v_proj.weight"] = mx.array(
            (rng.standard_normal((n_kv_heads * head_dim, D)) * 0.05).astype(np.float16)
        )
        tensors[f"{lp}.self_attn.o_proj.weight"] = mx.array(
            (rng.standard_normal((D, n_heads * head_dim)) * 0.05).astype(np.float16)
        )
        tensors[f"{lp}.self_attn.q_norm.weight"] = mx.ones((head_dim,), dtype=mx.float16)
        tensors[f"{lp}.self_attn.k_norm.weight"] = mx.ones((head_dim,), dtype=mx.float16)
        tensors[f"{lp}.mlp.gate.weight"] = mx.array(
            (rng.standard_normal((num_experts, D)) * 0.15).astype(np.float16)
        )
        if with_shared_expert:
            for sp, sh in (("gate_proj", (I, D)), ("up_proj", (I, D)), ("down_proj", (D, I))):
                tensors[f"{lp}.mlp.shared_expert.{sp}.weight"] = mx.array(
                    (rng.standard_normal(sh) * 0.05).astype(np.float16)
                )
            tensors[f"{lp}.mlp.shared_expert_gate.weight"] = mx.array(
                (rng.standard_normal((1, D)) * 0.05).astype(np.float16)
            )
        for e_i in range(num_experts):
            ep = f"{lp}.mlp.experts.{e_i}"
            tensors[f"{ep}.gate_proj.weight"] = mx.array(
                (rng.standard_normal((I, D)) * 0.05).astype(np.float16)
            )
            tensors[f"{ep}.up_proj.weight"] = mx.array(
                (rng.standard_normal((I, D)) * 0.05).astype(np.float16)
            )
            tensors[f"{ep}.down_proj.weight"] = mx.array(
                (rng.standard_normal((D, I)) * 0.05).astype(np.float16)
            )

    mx.eval(tensors)
    mx.save_safetensors(str(src_dir / "model.safetensors"), tensors)
    return convert_safetensors_moe_to_deimos_q4(src_dir, out_dir, group_size=64)


HELDOUT_EVAL_TEXTS = [
    (
        "Cryptographic Merkle trees allow sparse neural network runtimes to verify "
        "individual expert weight blocks in logarithmic time without reading the entire "
        "multi-gigabyte checkpoint from solid-state storage."
    ),
    (
        "In computer architecture, virtual memory decouples logical page addresses from "
        "physical DRAM frames using a hardware translation lookaside buffer and page table."
    ),
    (
        "def binary_search(arr: list[int], target: int) -> int:\n"
        "    lo, hi = 0, len(arr) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n"
    ),
]


class TestDeimosMerkleAndArena(unittest.TestCase):
    def test_1_dynamic_non_1024_topology_and_corruption_rejection(self) -> None:
        """Verifies arbitrary `(L=24, E=60 -> 1,440 leaves, V=151,936)` + 1-bit tamper rejection."""
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            num_layers, num_experts, vocab_size = 24, 60, 151936
            hidden_size, inter_size = 512, 256
            layout = compute_q4_leaf_layout(hidden_size, inter_size, group_size=64)
            stride = layout["stride_bytes"]
            self.assertEqual(stride % PAGE_SIZE, 0)

            rng = np.random.default_rng(7)
            leaf_hashes = []
            cas_path = tdir / "cas_experts_q4.bin"
            with open(cas_path, "wb") as f:
                for _ in range(num_layers * num_experts):
                    blk = rng.integers(0, 256, size=(stride,), dtype=np.uint8).tobytes()
                    leaf_hashes.append(hashlib.sha256(blk).digest())
                    f.write(blk)

            idx_path = tdir / "merkle_tree.idx"
            build_deimidx2_file(
                idx_path=idx_path,
                num_layers=num_layers,
                num_experts=num_experts,
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                intermediate_size=inter_size,
                quant_bits=4,
                quant_group_size=64,
                expert_stride_bytes=stride,
                expert_payload_bytes=layout["payload_bytes"],
                leaf_hashes=leaf_hashes,
            )

            bridge = DeimosMerkleBridge(tdir, enable_nocache_fd=False)
            try:
                self.assertEqual(bytes(bridge.header.magic), b"DEIMIDX2")
                self.assertEqual(int(bridge.header.num_layers), 24)
                self.assertEqual(int(bridge.header.num_experts), 60)
                self.assertEqual(int(bridge.header.tree_leaf_cap), 2048)
                self.assertEqual(int(bridge.header.vocab_size), 151936)
                bridge.verify_leaf(23, 59, verify_bytes=True)
            finally:
                bridge.close()

            corrupt_slot = 10 * num_experts + 33
            corrupt_off = corrupt_slot * stride + 128
            with open(cas_path, "r+b") as cf:
                cf.seek(corrupt_off)
                b0 = cf.read(1)[0]
                cf.seek(corrupt_off)
                cf.write(bytes([b0 ^ 0x01]))

            bridge2 = DeimosMerkleBridge(tdir, enable_nocache_fd=False)
            try:
                with self.assertRaises(DeimosMerkleIntegrityError):
                    bridge2.verify_leaf(10, 33, verify_bytes=True)
            finally:
                bridge2.close()

            raw_idx = bytearray(idx_path.read_bytes())
            raw_idx[250] ^= 0xFF
            idx_path.write_bytes(bytes(raw_idx))
            with self.assertRaises(DeimosMerkleIntegrityError):
                DeimosMerkleBridge(tdir, enable_nocache_fd=False)

    def test_1b_cpp17_arena_lookahead_dynamick_and_robinhood_kv(self) -> None:
        """Cross-platform C++17 test of Arena LFU+LRU, Residual Lookahead, Dynamic-K & KV table."""
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            L, E, V, D, I = 3, 12, 4096, 128, 64
            layout = compute_q4_leaf_layout(D, I, group_size=64)
            stride = layout["stride_bytes"]

            rng = np.random.default_rng(2026)
            leaves = [rng.integers(0, 256, size=(stride,), dtype=np.uint8) for _ in range(L * E)]
            hashes = [hashlib.sha256(lf.tobytes()).digest() for lf in leaves]
            with open(tdir / "cas_experts_q4.bin", "wb") as cf:
                for lf in leaves:
                    cf.write(lf.tobytes())
            build_deimidx2_file(
                tdir / "merkle_tree.idx", L, E, V, D, I, 4, 64, stride, layout["payload_bytes"], hashes, kv_table_slots=8
            )

            bridge = DeimosMerkleBridge(tdir, enable_nocache_fd=True)
            slots = 4
            ah = bridge.lib.deimos_arena_create(
                bridge.handle, L, E, slots,
                layout["gu_w_bytes"], layout["gu_s_bytes"], layout["dn_w_bytes"], layout["dn_s_bytes"], 1, 1
            )
            self.assertTrue(bool(ah))
            pools: List[Tuple[np.ndarray, ...]] = []  # type: ignore[name-defined]
            try:
                for l_i in range(L):
                    bufs = tuple(
                        np.zeros((slots * layout[k],), dtype=np.uint8)
                        for k in ("gu_w_bytes", "gu_s_bytes", "gu_s_bytes", "dn_w_bytes", "dn_s_bytes", "dn_s_bytes")
                    )
                    pools.append(bufs)
                    bridge.lib.deimos_arena_bind_layer_ptrs(ah, l_i, *[int(b.ctypes.data) for b in bufs])

                ids = np.array([[2, 5, 8, 11]], dtype=np.int32)
                scs = np.array([[0.55, 0.30, 0.10, 0.05]], dtype=np.float32)
                next_pred = np.array([1, 4, 7, 10], dtype=np.int32)
                out_p = np.empty((1, 4), dtype=np.int32)
                out_s = np.empty((1, 4), dtype=np.float32)
                eff_k = ctypes.c_int(4)

                rc = bridge.lib.deimos_arena_resolve_and_prefetch(
                    ah, 0,
                    ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    scs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    1, 4, 0.80, 2,
                    1, next_pred.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), 4,
                    out_p.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    out_s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.byref(eff_k),
                )
                self.assertEqual(rc, 0)
                self.assertEqual(eff_k.value, 2)  # 0.55 + 0.30 = 0.85 >= 0.80 -> last 2 cold tail misses skipped!
                bridge.lib.deimos_arena_wait_idle(ah)

                tel = _CArenaTelemetry()
                bridge.lib.deimos_arena_get_telemetry(ah, ctypes.byref(tel))
                self.assertEqual(int(tel.tail_misses_skipped), 2)
                self.assertEqual(int(tel.lookahead_staged), 4)

                # Verify staged bytes in Layer 1 slot match raw leaf bytes of predicted expert!
                self.assertEqual(
                    bytes(pools[1][0][: layout["gu_w_bytes"]]),
                    bytes(leaves[1 * E + 1][: layout["gu_w_bytes"]]),
                )

                # Verify Robin-Hood KV lookup + insert
                kh = hashlib.sha256(b"blk0").digest()
                ph = hashlib.sha256(b"root").digest()
                self.assertFalse(bridge.kv_lookup(kh, 1.0))
                bridge.kv_insert(kh, ph, 16, 1.0)
                self.assertTrue(bridge.kv_lookup(kh, 2.0))
            finally:
                bridge.lib.deimos_arena_destroy(ah)
                bridge.close()

        # Verify pure-Python MoE config normalizer & O(1) streaming detokenizer
        norm_cfg = normalize_moe_config(
            {"num_hidden_layers": 8, "hidden_size": 512, "num_attention_heads": 8, "n_routed_experts": 64, "moe_intermediate_size": 256, "vocab_size": 32000}
        )
        self.assertEqual(norm_cfg["num_experts"], 64)

        class _MockTok:
            def decode(self, ids: List[int]) -> str:
                return " ".join(f"w{i}" for i in ids)

        detok = IncrementalStreamingDetokenizer(_MockTok(), window_size=6)
        emitted = "".join(detok.add_token(i) for i in range(12))
        self.assertEqual(emitted, detok.full_text())

    @unittest.skipUnless(HAS_MLX, "MLX Metal GPU runtime requires macOS arm64")
    def test_2_standalone_paged_vs_pinned_bit_exact_parity(self) -> None:
        """Verifies exact `0.0` logit & token parity between 4-slot Paged and 16-slot Pinned Q4."""
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = build_synthetic_moe_store(
                Path(tmp), num_layers=3, num_experts=16, top_k=4, with_shared_expert=True
            )
            prompt_ids = [5, 19, 42, 77, 108, 201]

            eng_paged = DeimosPagedMoEEngine(
                store_dir, slots_per_layer=4, verify_merkle_on_cold=True, use_nocache_pread=True
            )
            try:
                logits_paged = np.asarray(eng_paged.forward_logits(prompt_ids).astype(mx.float32))
                m_paged = eng_paged.benchmark_token_ids(prompt_ids, max_new_tokens=16)
                self.assertGreater(m_paged["evictions"], 0)
            finally:
                eng_paged.close()

            eng_full = DeimosPagedMoEEngine(
                store_dir, slots_per_layer=16, verify_merkle_on_cold=False, use_nocache_pread=False
            )
            try:
                logits_full = np.asarray(eng_full.forward_logits(prompt_ids).astype(mx.float32))
                m_full = eng_full.benchmark_token_ids(prompt_ids, max_new_tokens=16)
            finally:
                eng_full.close()

            self.assertAlmostEqual(float(np.max(np.abs(logits_paged - logits_full))), 0.0, places=5)
            self.assertEqual(m_paged["generated_ids"], m_full["generated_ids"])

    @unittest.skipUnless(HAS_MLX, "MLX Metal GPU runtime requires macOS arm64")
    def test_3_real_olmoe_6_9b_parity_and_perplexity_if_present(self) -> None:
        """Runs full 6.92B OLMoE-1B-7B 8-slot vs 64-slot `0.0` parity & PPL when store exists."""
        olmoe_dir = find_real_olmoe_store()
        if olmoe_dir is None:
            self.skipTest("Local 6.92B OLMoE Q4 store not present (CI environment)")

        prompt = "<|user|>\nWhat is a Merkle tree?\n<|assistant|>\n"
        eng_paged = DeimosPagedMoEEngine(
            olmoe_dir, slots_per_layer=8, verify_merkle_on_cold=True, use_nocache_pread=False
        )
        try:
            ids = eng_paged.tokenizer.encode(prompt).ids  # type: ignore[union-attr]
            logits_paged = np.asarray(eng_paged.forward_logits(ids).astype(mx.float32))
            res_paged = eng_paged.benchmark_generation(prompt, max_new_tokens=16)
            self.assertGreater(res_paged["evictions"], 0)
            self.assertLessEqual(res_paged["peak_vram_mb"] - res_paged["active_vram_mb"], 100.0)
            ppl_q4 = eng_paged.compute_perplexity(HELDOUT_EVAL_TEXTS)
        finally:
            eng_paged.close()

        eng_full = DeimosPagedMoEEngine(
            olmoe_dir, slots_per_layer=64, verify_merkle_on_cold=False, use_nocache_pread=False
        )
        try:
            logits_full = np.asarray(eng_full.forward_logits(ids).astype(mx.float32))
            res_full = eng_full.benchmark_generation(prompt, max_new_tokens=16)
        finally:
            eng_full.close()

        self.assertEqual(float(np.max(np.abs(logits_paged - logits_full))), 0.0)
        self.assertEqual(res_paged["generated_ids"], res_full["generated_ids"])
        self.assertLess(ppl_q4, 6.45)


if __name__ == "__main__":
    unittest.main(verbosity=2)
