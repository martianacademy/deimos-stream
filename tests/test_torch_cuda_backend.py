#!/usr/bin/env python3
"""
PyTorch (`CUDA` / `CPU`) Backend & Cross-Engine `MLX <-> PyTorch` Parity Suite
(`tests/test_torch_cuda_backend.py`)
==============================================================================
Verifies:
  1. `resolve_backend("auto")` always prefers `mlx` on Apple Silicon (`darwin/arm64`) and
     never degrades to `mps`, while selecting `cuda` on NVIDIA hosts (or `cpu` fallback).
  2. MLX-free `.safetensors` -> `DEIMIDX2` Q4 conversion (`quantize_fp16_np_to_q4`),
     `TorchDeimosPagedMoEEngine` 4-slot Paged vs 12-slot Pinned `0.0` exact parity,
     Residual Lookahead, Miss-Only Dynamic-K, `TorchMerkleKVPrefixCache` warm restart,
     and OpenAI `/v1/chat/completions` SSE streaming on PyTorch!
  3. Cross-Engine Parity (`DeimosPagedMoEEngine` MLX vs `TorchDeimosPagedMoEEngine` PyTorch)
     on BOTH synthetic GQA+Shared-Expert MoE AND real 6.92B `OLMoE-1B-7B`!
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Dict
import unittest

from fastapi.testclient import TestClient
import numpy as np
from safetensors.numpy import save_file as st_save_np

import deimos_stream.converter as conv_mod
from deimos_stream import inspect_store
from deimos_stream.engine import HAS_MLX, DeimosPagedMoEEngine
from deimos_stream.factory import create_paged_moe_engine, resolve_backend
from deimos_stream.hf_converter import convert_hf_hybrid_moe_to_deimos
from deimos_stream.hf_streamer import DeimosHFHybridMoEEngine
from deimos_stream.server import create_deimos_app
from deimos_stream.torch_arena import HAS_TORCH
from deimos_stream.torch_engine import TorchDeimosPagedMoEEngine
from tests.test_merkle_and_arena import find_real_olmoe_store

if HAS_TORCH:
    import torch


def _build_mlx_free_moe_store(root_dir: Path) -> Path:
    """Builds a Qwen3-style GQA + per-head QK-Norm + Shared-Expert MoE store using pure NumPy."""
    src_dir = root_dir / "np_safetensors"
    out_dir = root_dir / "np_deimos_q4"
    src_dir.mkdir(parents=True, exist_ok=True)

    D, I, V, L, E, K = 128, 64, 256, 3, 12, 4
    n_heads, n_kv_heads, head_dim = 8, 2, 16
    cfg = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": L,
        "hidden_size": D,
        "moe_intermediate_size": I,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": head_dim,
        "num_experts": E,
        "num_experts_per_tok": K,
        "vocab_size": V,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "norm_topk_prob": True,
    }
    (src_dir / "config.json").write_text(json.dumps(cfg))

    rng = np.random.default_rng(777)
    tensors: Dict[str, np.ndarray] = {
        "model.embed_tokens.weight": (rng.standard_normal((V, D)) * 0.05).astype(np.float16),
        "model.norm.weight": np.ones((D,), dtype=np.float16),
    }
    for l_i in range(L):
        lp = f"model.layers.{l_i}"
        tensors[f"{lp}.input_layernorm.weight"] = np.ones((D,), dtype=np.float16)
        tensors[f"{lp}.post_attention_layernorm.weight"] = np.ones((D,), dtype=np.float16)
        tensors[f"{lp}.self_attn.q_proj.weight"] = (
            rng.standard_normal((n_heads * head_dim, D)) * 0.05
        ).astype(np.float16)
        tensors[f"{lp}.self_attn.k_proj.weight"] = (
            rng.standard_normal((n_kv_heads * head_dim, D)) * 0.05
        ).astype(np.float16)
        tensors[f"{lp}.self_attn.v_proj.weight"] = (
            rng.standard_normal((n_kv_heads * head_dim, D)) * 0.05
        ).astype(np.float16)
        tensors[f"{lp}.self_attn.o_proj.weight"] = (
            rng.standard_normal((D, n_heads * head_dim)) * 0.05
        ).astype(np.float16)
        tensors[f"{lp}.self_attn.q_norm.weight"] = np.ones((head_dim,), dtype=np.float16)
        tensors[f"{lp}.self_attn.k_norm.weight"] = np.ones((head_dim,), dtype=np.float16)
        tensors[f"{lp}.mlp.gate.weight"] = (rng.standard_normal((E, D)) * 0.15).astype(np.float16)
        for sp, sh in (("gate_proj", (I, D)), ("up_proj", (I, D)), ("down_proj", (D, I))):
            tensors[f"{lp}.mlp.shared_expert.{sp}.weight"] = (
                rng.standard_normal(sh) * 0.05
            ).astype(np.float16)
        tensors[f"{lp}.mlp.shared_expert_gate.weight"] = (
            rng.standard_normal((1, D)) * 0.05
        ).astype(np.float16)
        for e_i in range(E):
            ep = f"{lp}.mlp.experts.{e_i}"
            tensors[f"{ep}.gate_proj.weight"] = (rng.standard_normal((I, D)) * 0.05).astype(np.float16)
            tensors[f"{ep}.up_proj.weight"] = (rng.standard_normal((I, D)) * 0.05).astype(np.float16)
            tensors[f"{ep}.down_proj.weight"] = (rng.standard_normal((D, I)) * 0.05).astype(np.float16)

    st_save_np(tensors, str(src_dir / "model.safetensors"))

    # Exercise the MLX-free Linux/CUDA quantization & conversion path explicitly!
    saved_mx = conv_mod.mx
    conv_mod.mx = None  # type: ignore[assignment]
    try:
        return conv_mod.convert_safetensors_moe_to_deimos_q4(src_dir, out_dir, group_size=64)
    finally:
        conv_mod.mx = saved_mx


@unittest.skipUnless(HAS_TORCH, "PyTorch (`deimos-stream[cuda]`) not installed")
class TestDeimosTorchCudaBackend(unittest.TestCase):
    def _torch_dev(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_1_backend_priority_prefers_mlx_on_mac_and_never_mps(self) -> None:
        resolved = resolve_backend("auto")
        self.assertNotEqual(resolved, "mps")
        if HAS_MLX:
            self.assertEqual(resolved, "mlx")
        elif torch.cuda.is_available():
            self.assertEqual(resolved, "cuda")
        else:
            self.assertEqual(resolved, "cpu")

    def test_2_torch_engine_paged_vs_pinned_kv_dynamick_and_openai_sse(self) -> None:
        dev = self._torch_dev()
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = _build_mlx_free_moe_store(Path(tmp))
            prompt_ids = [(i * 5 + 7) % 240 for i in range(36)]

            eng_paged = TorchDeimosPagedMoEEngine(
                store_dir, device_str=dev, slots_per_layer=4, use_nocache_pread=True, miss_tail_tau=1.0
            )
            eng_pinned = TorchDeimosPagedMoEEngine(
                store_dir, device_str=dev, slots_per_layer=12, miss_tail_tau=1.0
            )
            try:
                l_paged = eng_paged.forward_logits(prompt_ids[:6]).float().cpu().numpy()
                l_pinned = eng_pinned.forward_logits(prompt_ids[:6]).float().cpu().numpy()
                self.assertAlmostEqual(float(np.max(np.abs(l_paged - l_pinned))), 0.0, places=5)

                m_cold = eng_paged.benchmark_token_ids(prompt_ids, max_new_tokens=12)
                m_ref = eng_pinned.benchmark_token_ids(prompt_ids, max_new_tokens=12)
                self.assertEqual(m_cold["generated_ids"], m_ref["generated_ids"])
                self.assertGreater(m_cold["evictions"], 0)
                self.assertGreater(m_cold["lookahead_staged"], 0)

                m_warm = eng_paged.benchmark_token_ids(prompt_ids, max_new_tokens=12)
                self.assertEqual(m_warm["kv_skipped_tokens"], 32)
                self.assertEqual(m_warm["generated_ids"], m_ref["generated_ids"])
            finally:
                eng_paged.close()
                eng_pinned.close()

            # Verify OpenAI SSE server over PyTorch backend
            app = create_deimos_app(store_dir, backend=dev, slots_per_layer=4, model_name="torch-moe")
            with TestClient(app) as client:
                h = client.get("/health").json()
                self.assertEqual(h["backend"], dev)
                r = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "torch-moe",
                        "stream": True,
                        "max_tokens": 6,
                        "messages": [{"role": "user", "content": "Test PyTorch SSE stream"}],
                    },
                )
                self.assertEqual(r.status_code, 200)
                self.assertIn("data: [DONE]", r.text)

    @unittest.skipUnless(HAS_MLX, "Cross-engine MLX <-> PyTorch parity requires macOS arm64")
    def test_3_cross_engine_mlx_vs_pytorch_parity_synthetic_and_real_olmoe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = _build_mlx_free_moe_store(Path(tmp))
            seq = [4, 18, 33, 91, 140]
            e_mlx = DeimosPagedMoEEngine(store_dir, slots_per_layer=4, miss_tail_tau=1.0)
            e_pt = TorchDeimosPagedMoEEngine(store_dir, device_str="cpu", slots_per_layer=4, miss_tail_tau=1.0)
            try:
                import mlx.core as mx

                out_mlx = np.asarray(e_mlx.forward_logits(seq).astype(mx.float32))
                out_pt = e_pt.forward_logits(seq).float().cpu().numpy()
                corr = float(np.corrcoef(out_mlx.reshape(-1), out_pt.reshape(-1))[0, 1])
                self.assertGreater(corr, 0.999)
                self.assertEqual(int(np.argmax(out_mlx)), int(np.argmax(out_pt)))
            finally:
                e_mlx.close()
                e_pt.close()

        olmoe_dir = find_real_olmoe_store()
        if olmoe_dir is not None:
            prompt = "<|user|>\nWhat is virtual memory?\n<|assistant|>\n"
            e_mlx2 = DeimosPagedMoEEngine(olmoe_dir, slots_per_layer=8, miss_tail_tau=1.0)
            e_pt2 = TorchDeimosPagedMoEEngine(olmoe_dir, device_str="cpu", slots_per_layer=8, miss_tail_tau=1.0)
            try:
                import mlx.core as mx

                ids = e_mlx2.tokenizer.encode(prompt).ids  # type: ignore[union-attr]
                l_m = np.asarray(e_mlx2.forward_logits(ids).astype(mx.float32))
                l_p = e_pt2.forward_logits(ids).float().cpu().numpy()
                corr_real = float(np.corrcoef(l_m.reshape(-1), l_p.reshape(-1))[0, 1])
                print(f"\n[Real 6.9B OLMoE MLX vs PyTorch Q4] Pearson r={corr_real:.6f}, top1=({int(np.argmax(l_m))}, {int(np.argmax(l_p))})")
                self.assertGreater(corr_real, 0.998)
                self.assertEqual(int(np.argmax(l_m)), int(np.argmax(l_p)))
            finally:
                e_mlx2.close()
                e_pt2.close()

    def test_4_hf_hybrid_qwen36_moe_zero_ram_meta_streamer(self) -> None:
        """Verifies 2026 Qwen3.6 Hybrid GatedDeltaNet+GQA+MoE conversion, Merkle proofs, and 0.0 parity."""
        from safetensors.torch import save_file as st_save_torch
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
            Qwen3_5MoeConfig,
            Qwen3_5MoeTextConfig,
        )
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

        dev = self._torch_dev()
        with tempfile.TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "qwen36_raw"
            store_dir = Path(tmp) / "qwen36_deimos_q4"
            src_dir.mkdir()

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
            sd["model.visual.dummy.weight"] = torch.zeros((4, 4), dtype=torch.bfloat16)
            sd["mtp.layers.0.dummy.weight"] = torch.zeros((4, 4), dtype=torch.bfloat16)
            st_save_torch(sd, str(src_dir / "model.safetensors"))
            (src_dir / "config.json").write_text(
                json.dumps(Qwen3_5MoeConfig(text_config=tcfg.to_dict()).to_dict())
            )

            convert_hf_hybrid_moe_to_deimos(str(src_dir), store_dir, group_size=64)
            info = inspect_store(store_dir, verify_all_leaves=True)
            self.assertEqual(info["xor_proofs_verified"], 96)
            self.assertEqual(info["byte_proofs_verified"], 48)

            eng_paged = create_paged_moe_engine(store_dir, backend=dev, slots_per_layer=4, miss_tail_tau=1.0)
            eng_pinned = DeimosHFHybridMoEEngine(store_dir, device_str=dev, slots_per_layer=12, miss_tail_tau=1.0)
            try:
                self.assertIsInstance(eng_paged, DeimosHFHybridMoEEngine)
                p_ids = [5, 19, 37, 81, 112, 149]
                lp = eng_paged.forward_logits(p_ids).float().cpu().numpy()
                lr = eng_pinned.forward_logits(p_ids).float().cpu().numpy()
                self.assertAlmostEqual(float(np.max(np.abs(lp - lr))), 0.0, places=5)

                m_p = eng_paged.benchmark_token_ids(p_ids, max_new_tokens=8)
                m_r = eng_pinned.benchmark_token_ids(p_ids, max_new_tokens=8)
                self.assertEqual(m_p["generated_ids"], m_r["generated_ids"])
                self.assertGreater(m_p["evictions"], 0)
                self.assertGreater(m_p["lookahead_staged"], 0)
            finally:
                eng_paged.close()
                eng_pinned.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
