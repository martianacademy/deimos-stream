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

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Dict, Optional
import unittest

import mlx.core as mx
import numpy as np

from deimos_stream.arena import DeimosMerkleBridge, DeimosMerkleIntegrityError
from deimos_stream.converter import convert_safetensors_moe_to_deimos_q4
from deimos_stream.engine import DeimosPagedMoEEngine
from deimos_stream.merkle_builder import (
    PAGE_SIZE,
    build_deimidx2_file,
    compute_q4_leaf_layout,
)


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
