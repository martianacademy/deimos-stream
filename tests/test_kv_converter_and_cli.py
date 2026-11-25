#!/usr/bin/env python3
"""
Integration Tests for `MerkleKVPrefixCache`, Universal `.safetensors` Converter & CLI
(`tests/test_kv_converter_and_cli.py`)
=====================================================================================
Verifies:
  1. Standalone `DEIMIDX2` Robin-Hood `MerkleKVPrefixCache`: warm restart from disk
     `.safetensors` skipping `>= 32` tokens with `0.0` exact token parity.
  2. Mixtral-style (`block_sparse_moe.experts.{e}.w1/w2/w3`, absent QK-Norm) `.safetensors`
     conversion + `deimos-stream inspect --verify-leaves` & `deimos-stream bench` CLI.
  3. Real-Hardware 6.92B `OLMoE-1B-7B` multi-turn `DeimosChatSession` streaming & warm KV
     reuse when local `OLMoE` Q4 store is present.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Dict
import unittest

import numpy as np

from deimos_stream.cli import DeimosChatSession, inspect_store, main as cli_main
from deimos_stream.engine import HAS_MLX, DeimosPagedMoEEngine
from tests.test_merkle_and_arena import build_synthetic_moe_store, find_real_olmoe_store

if HAS_MLX:
    import mlx.core as mx


@unittest.skipUnless(HAS_MLX, "MLX Metal GPU runtime requires macOS arm64")
class TestDeimosKVConverterAndCLI(unittest.TestCase):
    def test_1_standalone_merkle_kv_prefix_cache_across_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = build_synthetic_moe_store(
                Path(tmp), num_layers=3, num_experts=12, top_k=4, with_shared_expert=True
            )
            long_ids = [(i * 7 + 3) % 250 for i in range(40)]

            eng_nokv = DeimosPagedMoEEngine(
                store_dir, slots_per_layer=6, enable_merkle_kv=False, miss_tail_tau=1.0
            )
            try:
                m_cold_ref = eng_nokv.benchmark_token_ids(long_ids, max_new_tokens=10)
            finally:
                eng_nokv.close()

            eng_kv1 = DeimosPagedMoEEngine(
                store_dir, slots_per_layer=6, enable_merkle_kv=True, miss_tail_tau=1.0
            )
            try:
                _ = eng_kv1.benchmark_token_ids(long_ids, max_new_tokens=4)
                m_ram_warm = eng_kv1.benchmark_token_ids(long_ids, max_new_tokens=10)
                self.assertEqual(m_ram_warm["kv_skipped_tokens"], 32)
                self.assertEqual(m_ram_warm["generated_ids"], m_cold_ref["generated_ids"])
            finally:
                eng_kv1.close()

            eng_kv2 = DeimosPagedMoEEngine(
                store_dir, slots_per_layer=6, enable_merkle_kv=True, miss_tail_tau=1.0
            )
            try:
                m_disk_warm = eng_kv2.benchmark_token_ids(long_ids, max_new_tokens=10)
                self.assertEqual(m_disk_warm["kv_skipped_tokens"], 32)
                self.assertEqual(m_disk_warm["generated_ids"], m_cold_ref["generated_ids"])
            finally:
                eng_kv2.close()

    def test_2_mixtral_layout_converter_and_cli_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "mixtral_src"
            out_dir = Path(tmp) / "mixtral_deimos"
            src_dir.mkdir(parents=True)

            D, I, V, L, E, K = 128, 64, 192, 2, 8, 2
            n_heads, n_kv_heads, head_dim = 4, 2, 32
            cfg = {
                "model_type": "mixtral",
                "num_hidden_layers": L,
                "hidden_size": D,
                "intermediate_size": I,
                "num_attention_heads": n_heads,
                "num_key_value_heads": n_kv_heads,
                "num_local_experts": E,
                "num_experts_per_tok": K,
                "vocab_size": V,
                "rms_norm_eps": 1e-5,
            }
            (src_dir / "config.json").write_text(json.dumps(cfg))

            rng = np.random.default_rng(99)
            tensors: Dict[str, mx.array] = {
                "model.embed_tokens.weight": mx.array(
                    (rng.standard_normal((V, D)) * 0.05).astype(np.float16)
                ),
                "model.norm.weight": mx.ones((D,), dtype=mx.float16),
                "lm_head.weight": mx.array(
                    (rng.standard_normal((V, D)) * 0.05).astype(np.float16)
                ),
            }
            for l_i in range(L):
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
                tensors[f"{lp}.block_sparse_moe.gate.weight"] = mx.array(
                    (rng.standard_normal((E, D)) * 0.05).astype(np.float16)
                )
                for e_i in range(E):
                    ep = f"{lp}.block_sparse_moe.experts.{e_i}"
                    tensors[f"{ep}.w1.weight"] = mx.array(
                        (rng.standard_normal((I, D)) * 0.05).astype(np.float16)
                    )
                    tensors[f"{ep}.w3.weight"] = mx.array(
                        (rng.standard_normal((I, D)) * 0.05).astype(np.float16)
                    )
                    tensors[f"{ep}.w2.weight"] = mx.array(
                        (rng.standard_normal((D, I)) * 0.05).astype(np.float16)
                    )
            mx.eval(tensors)
            mx.save_safetensors(str(src_dir / "model.safetensors"), tensors)

            rc_conv = cli_main(["convert", "--model-dir", str(src_dir), "--output-dir", str(out_dir)])
            self.assertEqual(rc_conv, 0)

            info = inspect_store(out_dir, verify_all_leaves=True)
            self.assertEqual(info["magic"], "DEIMIDX2")
            self.assertEqual(info["total_experts"], L * E)
            self.assertEqual(info["byte_proofs_verified"], L * E)

            rc_bench = cli_main(["bench", "--store-dir", str(out_dir), "--slots", "4", "--tokens", "8"])
            self.assertEqual(rc_bench, 0)

    def test_3_real_olmoe_multi_turn_chat_session_if_present(self) -> None:
        olmoe_dir = find_real_olmoe_store()
        if olmoe_dir is None:
            self.skipTest("Local 6.92B OLMoE Q4 store not present (CI environment)")

        sess = DeimosChatSession(olmoe_dir, slots_per_layer=16, miss_tail_tau=0.85)
        try:
            t1 = sess.stream_turn(
                "Give a one-sentence definition of virtual memory paging in operating systems.",
                max_new_tokens=20,
                print_live=False,
            )
            self.assertGreater(int(t1["num_generated_tokens"]), 5)  # type: ignore[arg-type]
            t2 = sess.stream_turn(
                "Now give one sentence on why page tables matter.",
                max_new_tokens=20,
                print_live=False,
            )
            self.assertGreaterEqual(int(t2["kv_skipped_tokens"]), 16)  # type: ignore[arg-type]
        finally:
            sess.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
