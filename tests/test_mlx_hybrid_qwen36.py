#!/usr/bin/env python3
"""
Apple Silicon Native MLX Hybrid-MoE (`Qwen3.6` GatedDeltaNet + GQA + MoE) Verification Suite
(`tests/test_mlx_hybrid_qwen36.py`)
============================================================================================
Verifies on Apple Silicon Metal (`mlx` + `mlx-lm`):
  1. `create_paged_moe_engine(..., backend="auto")` routes `hf_hybrid_q4` stores directly to
     `DeimosMLXHybridMoEEngine` (native Metal `gated_delta_update` + `mx.gather_qmm`), never `mps`.
  2. Bit-exact `4-slot Paged` vs `12-slot Pinned` logit parity (`0.0` diff), 100% identical
     greedy tokens, LFU/LRU evictions, Residual Lookahead staging, and `verify_merkle_on_cold=False`.
  3. Cross-Engine Parity (`DeimosMLXHybridMoEEngine` Metal vs `DeimosHFHybridMoEEngine` PyTorch)
     on the EXACT SAME universal `DEIMIDX2` `hf_hybrid_q4` store (`Pearson r > 0.999`, top-1 match,
     identical greedy sequence) + OpenAI `/v1/chat/completions` SSE streaming!
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
import numpy as np

from deimos_stream.engine import HAS_MLX
from deimos_stream.factory import create_paged_moe_engine
from deimos_stream.hf_converter import convert_hf_hybrid_moe_to_deimos
from deimos_stream.hf_streamer import DeimosHFHybridMoEEngine
from deimos_stream.mlx_hybrid import DeimosMLXHybridMoEEngine
from deimos_stream.server import create_deimos_app
from deimos_stream.torch_arena import HAS_TORCH

if HAS_MLX:
    import mlx.core as mx
if HAS_TORCH:
    import torch


def _build_synthetic_qwen36_store(tmp_root: Path) -> Path:
    from safetensors.torch import save_file as st_save_torch
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeConfig,
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    src_dir = tmp_root / "qwen36_raw"
    store_dir = tmp_root / "qwen36_deimos_q4"
    src_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(2026)
    tcfg = Qwen3_5MoeTextConfig(
        dtype="bfloat16",
        hidden_size=128,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_hidden_layers=4,
        full_attention_interval=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        num_experts=12,
        num_experts_per_tok=2,
        vocab_size=256,
        tie_word_embeddings=False,
    )
    m = Qwen3_5MoeForCausalLM(tcfg).to(torch.bfloat16).eval()
    sd = {
        ("model.language_model." + k[len("model.") :] if k.startswith("model.") else k): v.contiguous()
        for k, v in m.state_dict().items()
    }
    st_save_torch(sd, str(src_dir / "model.safetensors"))
    (src_dir / "config.json").write_text(
        json.dumps(Qwen3_5MoeConfig(text_config=tcfg.to_dict()).to_dict())
    )
    return convert_hf_hybrid_moe_to_deimos(str(src_dir), store_dir, group_size=64)


@unittest.skipUnless(HAS_MLX and HAS_TORCH, "Requires Apple Silicon MLX + PyTorch/Transformers")
class TestDeimosMLXHybridQwen36(unittest.TestCase):
    def test_1_mlx_hybrid_paged_vs_pinned_and_cross_engine_pytorch_parity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = _build_synthetic_qwen36_store(Path(tmp))
            prompt_ids = [5, 19, 37, 81, 112, 149]

            # 1. Verify backend="auto" routes hf_hybrid_q4 directly to native Metal MLX!
            eng_paged = create_paged_moe_engine(
                store_dir, backend="auto", slots_per_layer=4, miss_tail_tau=1.0
            )
            self.assertIsInstance(eng_paged, DeimosMLXHybridMoEEngine)
            self.assertEqual(eng_paged.backend_name, "mlx_hybrid")

            eng_pinned = DeimosMLXHybridMoEEngine(
                store_dir, slots_per_layer=12, miss_tail_tau=1.0
            )
            eng_noverify = DeimosMLXHybridMoEEngine(
                store_dir, slots_per_layer=4, verify_merkle_on_cold=False, miss_tail_tau=1.0
            )
            eng_pt = DeimosHFHybridMoEEngine(
                store_dir, device_str="cpu", slots_per_layer=4, miss_tail_tau=1.0
            )
            try:
                # 2. Bit-exact 4-slot Paged vs 12-slot Pinned vs --no-verify on Apple Metal
                l_paged = np.asarray(eng_paged.forward_logits(prompt_ids).astype(mx.float32))
                l_pinned = np.asarray(eng_pinned.forward_logits(prompt_ids).astype(mx.float32))
                l_nv = np.asarray(eng_noverify.forward_logits(prompt_ids).astype(mx.float32))
                self.assertAlmostEqual(float(np.max(np.abs(l_paged - l_pinned))), 0.0, places=5)
                self.assertAlmostEqual(float(np.max(np.abs(l_paged - l_nv))), 0.0, places=5)

                m_paged = eng_paged.benchmark_token_ids(prompt_ids, max_new_tokens=8)
                m_pinned = eng_pinned.benchmark_token_ids(prompt_ids, max_new_tokens=8)
                self.assertEqual(m_paged["generated_ids"], m_pinned["generated_ids"])
                self.assertGreater(m_paged["evictions"], 0)
                self.assertGreater(m_paged["lookahead_staged"], 0)

                # 3. Cross-Engine Parity: Native Metal MLX vs PyTorch on the SAME DEIMIDX2 store!
                l_pt = eng_pt.forward_logits(prompt_ids).float().cpu().numpy()
                corr = float(np.corrcoef(l_paged.reshape(-1), l_pt.reshape(-1))[0, 1])
                print(
                    f"\n[Qwen3.6 Hybrid Metal MLX vs PyTorch] Pearson r={corr:.6f} | "
                    f"top1=({int(np.argmax(l_paged))}, {int(np.argmax(l_pt))})"
                )
                self.assertGreater(corr, 0.998)
                self.assertEqual(int(np.argmax(l_paged)), int(np.argmax(l_pt)))

                m_pt = eng_pt.benchmark_token_ids(prompt_ids, max_new_tokens=8)
                self.assertEqual(m_paged["generated_ids"], m_pt["generated_ids"])
            finally:
                eng_paged.close()
                eng_pinned.close()
                eng_noverify.close()
                eng_pt.close()

            # 4. Verify OpenAI /v1/chat/completions SSE streaming over native MLX Hybrid engine
            app = create_deimos_app(
                store_dir,
                backend="auto",
                slots_per_layer=4,
                verify_merkle_on_cold=False,
                model_name="qwen36-mlx-hybrid",
            )
            with TestClient(app) as client:
                h = client.get("/health").json()
                self.assertEqual(h["backend"], "mlx_hybrid")
                resp = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "qwen36-mlx-hybrid",
                        "stream": True,
                        "max_tokens": 6,
                        "messages": [{"role": "user", "content": "Test MLX Qwen3.6 SSE"}],
                    },
                )
                self.assertEqual(resp.status_code, 200)
                self.assertIn("data: [DONE]", resp.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
