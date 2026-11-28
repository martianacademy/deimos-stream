#!/usr/bin/env python3
"""
Memory-Aware Slot Planner Suite (`tests/test_memory_budget.py`)
===============================================================
Verifies `--slots auto`:
  1. Pins every expert when backbone + lm_head + all experts + reserve fit.
  2. Otherwise picks floor((usable - fixed) / (layers * stride)) slots.
  3. Clamps to top_k (flagged) when memory is below the routing minimum.
  4. Honours the `max_memory_gb` cap and the per-backend fraction + reserve defaults.
  5. Reads top_k from nested `text_config` and counts `hf_backbone.safetensors` (hybrid stores).
  6. `create_paged_moe_engine(slots_per_layer="auto")` builds a pinned / clamped arena end to end.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from deimos_stream.memory_budget import GIB, parse_slots_arg, plan_slots, store_layout
from deimos_stream.torch_arena import HAS_TORCH

MIB = 1024**2


def _fake_store(root: Path, hybrid: bool = False) -> Path:
    """4 layers x 16 experts x 1 MiB stride, 10 MiB backbone + 2 MiB lm_head, top_k = 4."""
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"num_layers": 4, "num_experts_per_layer": 16, "expert_stride_bytes": MIB}
    if hybrid:
        manifest["store_kind"] = "hf_hybrid_q4"
    (root / "merkle_manifest.json").write_text(json.dumps(manifest))
    with open(root / ("hf_backbone.safetensors" if hybrid else "shared_core.bin"), "wb") as f:
        f.truncate(10 * MIB)
    with open(root / "lm_head_int8.bin", "wb") as f:
        f.truncate(2 * MIB)
    with open(root / "embed_tokens.bin", "wb") as f:  # CPU mmap: must NOT count as device memory
        f.truncate(50 * MIB)
    cfg = {"text_config": {"num_experts_per_tok": 4}} if hybrid else {"num_experts_per_tok": 4}
    (root / "config.json").write_text(json.dumps(cfg))
    return root


class TestMemoryBudget(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = _fake_store(Path(self._tmp.name) / "store")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _plan(self, avail_mib: float, **kw):  # type: ignore[no-untyped-def]
        kw.setdefault("memory_fraction", 1.0)
        kw.setdefault("reserve_bytes", 0)
        return plan_slots(self.store, "mlx", available_bytes=int(avail_mib * MIB), **kw)

    def test_1_layout_counts_device_weights_only(self) -> None:
        self.assertEqual(store_layout(self.store), (4, 16, MIB, 12 * MIB, 4))

    def test_2_pins_all_experts_when_everything_fits(self) -> None:
        p = self._plan(100)  # fixed 12 + experts 64 = 76 MiB <= 100
        self.assertEqual(p.slots_per_layer, 16)
        self.assertTrue(p.fully_pinned)
        self.assertFalse(p.forced_minimum)
        self.assertEqual(self._plan(76).slots_per_layer, 16)  # exact fit still pins

    def test_3_fits_partial_slots_from_free_memory(self) -> None:
        p = self._plan(40)  # (40 - 12) / 4 = 7
        self.assertEqual(p.slots_per_layer, 7)
        self.assertFalse(p.fully_pinned)
        self.assertEqual(self._plan(75.9).slots_per_layer, 15)

    def test_4_clamps_to_top_k_and_flags_it(self) -> None:
        p = self._plan(20)  # (20 - 12) / 4 = 2 < top_k = 4
        self.assertEqual(p.slots_per_layer, 4)
        self.assertTrue(p.forced_minimum)
        self.assertIn("WARNING", p.summary())

    def test_5_max_memory_cap_and_default_fraction_reserve(self) -> None:
        self.assertEqual(self._plan(1024, max_memory_gb=40 / 1024).slots_per_layer, 7)
        # cuda default: usable = 0.90 * avail - 1 GiB reserve
        p = plan_slots(self.store, "cuda", available_bytes=2 * GIB)
        self.assertEqual(p.usable_bytes, int(2 * GIB * 0.90) - GIB)
        self.assertTrue(p.fully_pinned)
        # reserve larger than the budget never goes negative
        self.assertEqual(plan_slots(self.store, "cuda", available_bytes=GIB).usable_bytes, 0)

    def test_6_hybrid_store_nested_top_k_and_backbone(self) -> None:
        h = _fake_store(Path(self._tmp.name) / "hybrid", hybrid=True)
        self.assertEqual(store_layout(h), (4, 16, MIB, 12 * MIB, 4))
        p = plan_slots(h, "cuda", available_bytes=40 * MIB, memory_fraction=1.0, reserve_bytes=0)
        self.assertEqual(p.slots_per_layer, 7)

    def test_6b_q4_backbone_shrinks_fixed_bytes(self) -> None:
        # shared_core / hf_backbone 10 MiB FP16 -> 10 * 4.5/16, lm_head 2 MiB INT8 -> 2 * 4.5/8
        expected_fixed = int(10 * MIB * 4.5 / 16) + int(2 * MIB * 4.5 / 8)
        p = self._plan(40, backbone="q4")
        self.assertEqual(p.fixed_bytes, expected_fixed)
        self.assertEqual(p.slots_per_layer, int((40 * MIB - p.fixed_bytes) // (4 * MIB)))

        h = _fake_store(Path(self._tmp.name) / "hybrid_q4", hybrid=True)
        ph = plan_slots(
            h, "cuda", available_bytes=40 * MIB, memory_fraction=1.0, reserve_bytes=0, backbone="q4"
        )
        # The PyTorch HF-hybrid engine (cuda/cpu) keeps its fused INT8 lm_head under --backbone q4,
        # so the planner must not shrink it there (under-planning VRAM otherwise).
        self.assertEqual(ph.fixed_bytes, int(10 * MIB * 7.5 / 16) + 2 * MIB)
        # MLX hybrid engines quantize the lm_head to Q4, and an explicit override is honoured.
        q4_head = int(10 * MIB * 7.5 / 16) + int(2 * MIB * 4.5 / 8)
        pm = plan_slots(
            h, "mlx", available_bytes=40 * MIB, memory_fraction=1.0, reserve_bytes=0, backbone="q4"
        )
        self.assertEqual(pm.fixed_bytes, q4_head)
        po = plan_slots(
            h, "cuda", available_bytes=40 * MIB, memory_fraction=1.0, reserve_bytes=0, backbone="q4",
            lm_head_q4=True,
        )
        self.assertEqual(po.fixed_bytes, q4_head)

    def test_6c_cuda_ring_buffer_not_throttled_by_host_ram(self) -> None:
        # CUDA TorchPagedExpertArena uses a fixed 32-slot (~58 MB) ring buffer instead of a full
        # host shadow, so tight host RAM containers still get full GPU VRAM slot capacity.
        p = plan_slots(self.store, "cuda", available_bytes=GIB, memory_fraction=1.0, reserve_bytes=0,
                       host_available_bytes=2 * GIB)
        self.assertEqual((p.slots_per_layer, p.host_limited), (16, False))

    def test_7_parse_slots_arg(self) -> None:
        self.assertEqual(parse_slots_arg("auto"), "auto")
        self.assertEqual(parse_slots_arg(" AUTO "), "auto")
        self.assertEqual(parse_slots_arg("16"), 16)
        self.assertEqual(parse_slots_arg(8), 8)
        for bad in ("0", "-2", "abc", 0):
            with self.assertRaises(ValueError):
                parse_slots_arg(bad)  # type: ignore[arg-type]


@unittest.skipUnless(HAS_TORCH, "PyTorch (`deimos-stream[cuda]`) not installed")
class TestAutoSlotsEngine(unittest.TestCase):
    def test_auto_slots_end_to_end(self) -> None:
        import torch

        from deimos_stream.factory import create_paged_moe_engine
        from tests.test_torch_cuda_backend import _build_mlx_free_moe_store

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = _build_mlx_free_moe_store(Path(tmp))  # 3 layers x 12 experts, top_k 4
            prompt_ids = [(i * 5 + 7) % 240 for i in range(20)]

            eng_auto = create_paged_moe_engine(store_dir, backend=dev, miss_tail_tau=1.0)
            eng_tight = create_paged_moe_engine(store_dir, backend=dev, miss_tail_tau=1.0, max_memory_gb=0.5)
            try:
                self.assertEqual(eng_auto.arena.slots_per_layer, 12)
                self.assertTrue(eng_auto.arena.is_fully_pinned)
                self.assertTrue(eng_auto.slot_plan.fully_pinned)
                # 0.5 GiB cap minus the 1 GiB reserve leaves nothing, so it clamps to top_k = 4 and flags it
                self.assertEqual(eng_tight.arena.slots_per_layer, 4)
                self.assertTrue(eng_tight.slot_plan.forced_minimum)
                m_auto = eng_auto.benchmark_token_ids(prompt_ids, max_new_tokens=8)
                m_tight = eng_tight.benchmark_token_ids(prompt_ids, max_new_tokens=8)
                self.assertEqual(m_auto["generated_ids"], m_tight["generated_ids"])
            finally:
                eng_auto.close()
                eng_tight.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
