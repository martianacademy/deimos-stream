#!/usr/bin/env python3
"""
Unit & Ablation Suite for Residual Lookahead & Miss-Only Dynamic-K
(`tests/test_lookahead_and_dynamick.py`)
==================================================================
Verifies:
  1. Standalone CI-safe Residual Lookahead (`enable_lookahead=True, miss_tail_tau=1.0`)
     bit-exact parity against full pinned Q4 (`0` race conditions) + positive background
     staging (`lookahead_staged > 0`), plus Miss-Only Dynamic-K (`miss_tail_tau=0.80`)
     skipping cold tail misses (`tail_misses_skipped > 0`, `avg_active_k < top_k`).
  2. Real-Hardware 6.92B `OLMoE-1B-7B` Residual Lookahead miss reduction & single-token
     streaming held-out Perplexity (`< 6.45`) when local `OLMoE` Q4 store is present.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from deimos_stream.engine import HAS_MLX, DeimosPagedMoEEngine
from tests.test_merkle_and_arena import (
    HELDOUT_EVAL_TEXTS,
    build_synthetic_moe_store,
    find_real_olmoe_store,
)


@unittest.skipUnless(HAS_MLX, "MLX Metal GPU runtime requires macOS arm64")
class TestDeimosLookaheadAndDynamicK(unittest.TestCase):
    def test_1_standalone_lookahead_exact_parity_and_dynamick_tail_skip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = build_synthetic_moe_store(
                Path(tmp), num_layers=4, num_experts=16, top_k=6, with_shared_expert=False
            )
            prompt_ids = [11, 29, 64, 105, 199]

            eng_la = DeimosPagedMoEEngine(
                store_dir,
                slots_per_layer=6,
                use_nocache_pread=True,
                enable_lookahead=True,
                miss_tail_tau=1.0,
            )
            try:
                m_la = eng_la.benchmark_token_ids(prompt_ids, max_new_tokens=20)
            finally:
                eng_la.close()

            eng_ref = DeimosPagedMoEEngine(store_dir, slots_per_layer=16, miss_tail_tau=1.0)
            try:
                m_ref = eng_ref.benchmark_token_ids(prompt_ids, max_new_tokens=20)
            finally:
                eng_ref.close()

            self.assertEqual(m_la["generated_ids"], m_ref["generated_ids"])
            self.assertGreater(m_la["lookahead_staged"], 0)

            eng_dyn = DeimosPagedMoEEngine(
                store_dir,
                slots_per_layer=6,
                use_nocache_pread=True,
                enable_lookahead=True,
                miss_tail_tau=0.75,
                min_k=2,
            )
            try:
                m_dyn = eng_dyn.benchmark_token_ids(prompt_ids, max_new_tokens=24)
                self.assertGreater(m_dyn["tail_misses_skipped"], 0)
                self.assertLess(m_dyn["avg_active_k"], 6.0)
            finally:
                eng_dyn.close()

    def test_2_real_olmoe_lookahead_recall_and_dynamick_ppl_if_present(self) -> None:
        olmoe_dir = find_real_olmoe_store()
        if olmoe_dir is None:
            self.skipTest("Local 6.92B OLMoE Q4 store not present (CI environment)")

        prompt = (
            "<|user|>\nExplain Merkle trees and virtual memory paging in simple words.\n"
            "<|assistant|>\n"
        )
        eng_nola = DeimosPagedMoEEngine(
            olmoe_dir,
            slots_per_layer=16,
            use_nocache_pread=True,
            enable_lookahead=False,
            miss_tail_tau=1.0,
        )
        try:
            m_nola = eng_nola.benchmark_generation(prompt, max_new_tokens=32)
        finally:
            eng_nola.close()

        eng_la = DeimosPagedMoEEngine(
            olmoe_dir,
            slots_per_layer=16,
            use_nocache_pread=True,
            enable_lookahead=True,
            miss_tail_tau=1.0,
        )
        try:
            m_la = eng_la.benchmark_generation(prompt, max_new_tokens=32)
        finally:
            eng_la.close()

        eng_ref = DeimosPagedMoEEngine(olmoe_dir, slots_per_layer=64, miss_tail_tau=1.0)
        try:
            m_ref = eng_ref.benchmark_generation(prompt, max_new_tokens=32)
        finally:
            eng_ref.close()

        self.assertEqual(m_la["generated_ids"], m_ref["generated_ids"])
        self.assertEqual(m_nola["generated_ids"], m_ref["generated_ids"])
        self.assertGreater(m_la["lookahead_useful_hits"], 100)
        self.assertLess(m_la["cache_misses"], m_nola["cache_misses"])

        eng_dyn = DeimosPagedMoEEngine(
            olmoe_dir,
            slots_per_layer=16,
            use_nocache_pread=True,
            enable_lookahead=True,
            miss_tail_tau=0.85,
            min_k=4,
        )
        try:
            m_dyn = eng_dyn.benchmark_generation(prompt, max_new_tokens=32)
            self.assertGreater(m_dyn["tail_misses_skipped"], 0)
            self.assertLess(m_dyn["avg_active_k"], 8.0)
            ppl_dyn = eng_dyn.compute_perplexity(HELDOUT_EVAL_TEXTS, single_token_mode=True)
            self.assertLess(ppl_dyn, 6.45)
        finally:
            eng_dyn.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
