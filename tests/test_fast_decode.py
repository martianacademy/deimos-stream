#!/usr/bin/env python3
"""
Fast Decode Suite (`tests/test_fast_decode.py`)
===============================================
Verifies `fast_decode.py` on the MLX dense engine:
  1. `forward_logits(mx.array)` matches `forward_logits(list)` exactly (GPU embedding copy).
  2. The pipelined loop (pinned arena) yields the same greedy ids as the synchronous loop.
  3. Stop ids and an early caller `break` leave the KV caches at the synchronous loop's offset.
  4. A paged arena falls back to the synchronous loop (`supports_pipeline` is False).
  5. `--backbone q4` quantizes attention + lm_head, shrinks memory, keeps logits close, generates.
  6. Real OLMoE (when `DEIMOS_OLMOE_STORE` is set): 64 pipelined tokens == synchronous tokens.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import List, Optional, Set
import unittest

import numpy as np

from deimos_stream.engine import HAS_MLX, DeimosPagedMoEEngine
from deimos_stream.factory import create_paged_moe_engine
from deimos_stream.fast_decode import iter_decode_tokens, supports_pipeline
from tests.test_merkle_and_arena import build_synthetic_moe_store, find_real_olmoe_store

if HAS_MLX:
    import mlx.core as mx

PROMPT = [(i * 7 + 3) % 250 for i in range(20)]


def _run(eng: DeimosPagedMoEEngine, n: int, stops: Optional[Set[int]] = None, brk: int = -1):
    caches = eng.make_caches()
    logits, _ = eng.prefill_prompt(PROMPT, caches)
    out: List[int] = []
    gen = iter_decode_tokens(eng, logits, caches, n, 0.0, stops)
    try:
        for t in gen:
            out.append(t)
            if len(out) == brk:
                break
    finally:
        gen.close()
    return out, caches[0].offset


@unittest.skipUnless(HAS_MLX, "MLX Metal GPU runtime requires macOS arm64")
class TestFastDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.store = build_synthetic_moe_store(Path(cls._tmp.name), num_layers=3, num_experts=16, top_k=4)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _eng(self, slots: int = 16, backbone: str = "fp16") -> DeimosPagedMoEEngine:
        return create_paged_moe_engine(  # type: ignore[return-value]
            self.store, backend="mlx", slots_per_layer=slots, miss_tail_tau=1.0,
            enable_merkle_kv=False, backbone=backbone,
        )

    def test_1_mx_token_input_matches_list_input(self) -> None:
        eng = self._eng()
        try:
            a = np.asarray(eng.forward_logits(PROMPT).astype(mx.float32))
            b = np.asarray(eng.forward_logits(mx.array(PROMPT)).astype(mx.float32))
            self.assertEqual(float(np.max(np.abs(a - b))), 0.0)
        finally:
            eng.close()

    def test_2_pipelined_equals_sync_with_stops_and_breaks(self) -> None:
        eng = self._eng()
        try:
            self.assertTrue(supports_pipeline(eng))
            pipe, off_p = _run(eng, 24)
            eng.pipelined_decode = False
            self.assertFalse(supports_pipeline(eng))
            sync, off_s = _run(eng, 24)
            self.assertEqual(pipe, sync)
            self.assertEqual(off_p, off_s)
            self.assertEqual(off_s, len(PROMPT) + 23)  # last token is not fed

            stop = {sync[9]}
            first = sync.index(sync[9])
            for flag in (True, False):
                eng.pipelined_decode = flag
                ids, off = _run(eng, 24, stops=stop)
                self.assertEqual(ids, sync[:first])
                self.assertEqual(off, len(PROMPT) + first)  # stop token never fed
                ids, off = _run(eng, 24, brk=5)
                self.assertEqual(ids, sync[:5])
                self.assertEqual(off, len(PROMPT) + 4)  # broken-off token never fed
        finally:
            eng.close()

    def test_3_paged_arena_uses_sync_loop_with_same_tokens(self) -> None:
        pinned, paged = self._eng(16), self._eng(4)
        try:
            self.assertFalse(supports_pipeline(paged))
            self.assertEqual(_run(pinned, 16)[0], _run(paged, 16)[0])
        finally:
            pinned.close()
            paged.close()

    def test_4_q4_backbone(self) -> None:
        fp, q4 = self._eng(), self._eng(backbone="q4")
        try:
            self.assertEqual((q4.backbone, q4.lm_head_bits, fp.lm_head_bits), ("q4", 4, 8))
            self.assertIsNotNone(q4.attns[0].qkv_q)
            self.assertNotIn("qkv_w", q4.attns[0])  # FP16 copy released
            a = np.asarray(fp.forward_logits(PROMPT).astype(mx.float32)).ravel()
            b = np.asarray(q4.forward_logits(PROMPT).astype(mx.float32)).ravel()
            self.assertGreater(float(np.corrcoef(a, b)[0, 1]), 0.98)
            ids, _ = _run(q4, 12)
            self.assertEqual(len(ids), 12)
        finally:
            fp.close()
            q4.close()

        eng_cpu_q4 = create_paged_moe_engine(
            self.store, backend="cpu", slots_per_layer=8, miss_tail_tau=1.0,
            enable_merkle_kv=False, backbone="q4",
        )
        try:
            self.assertEqual(eng_cpu_q4.backbone, "q4")  # type: ignore[attr-defined]
            m = eng_cpu_q4.benchmark_token_ids(PROMPT, max_new_tokens=6)
            self.assertEqual(len(m["generated_ids"]), 7)
        finally:
            eng_cpu_q4.close()

    def test_5_sampling_path_runs(self) -> None:
        eng = self._eng()
        try:
            caches = eng.make_caches()
            logits, _ = eng.prefill_prompt(PROMPT, caches)
            gen = iter_decode_tokens(eng, logits, caches, 10, temperature=0.8)
            ids = list(gen)
            self.assertEqual(len(ids), 10)
            self.assertTrue(all(0 <= t < eng.vocab_size for t in ids))
        finally:
            eng.close()

    def test_6_real_olmoe_pipelined_equals_sync_if_present(self) -> None:
        store = find_real_olmoe_store()
        if store is None:
            self.skipTest("Local 6.92B OLMoE Q4 store not present (CI environment)")
        eng = create_paged_moe_engine(store, backend="mlx", slots_per_layer=64, miss_tail_tau=1.0, enable_merkle_kv=False)
        try:
            ids = eng.tokenizer.encode("<|user|>\nWhat is a Merkle tree?\n<|assistant|>\n").ids  # type: ignore[union-attr]
            pipe = eng.benchmark_token_ids(ids, max_new_tokens=64)["generated_ids"]
            eng.pipelined_decode = False
            sync = eng.benchmark_token_ids(ids, max_new_tokens=64)["generated_ids"]
            self.assertEqual(pipe, sync)
        finally:
            eng.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
