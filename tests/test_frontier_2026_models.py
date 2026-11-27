#!/usr/bin/env python3
"""
2026 Frontier Sparse & Hybrid MoE End-to-End Parity Suite (`tests/test_frontier_2026_models.py`)
================================================================================================
Tests conversion (`convert_hf_hybrid_moe_to_deimos`), `DEIMIDX2` Merkle proof auditing, C++17
`AVX2/NEON` clamped/OAI SwiGLU (`deimos_cpu_q4_swiglu_forward`), PyTorch paged execution, and
Apple Silicon `MLX` (`DeimosMLXUniversalMoEEngine`) across late-2026 frontier architectures:
  1. `DeepSeek-V4-Flash` (`deepseek_v4`: raw `layers.*.ffn` + `UE8M0` `.scale` FP8 + `swiglu_limit=10.0`)
  2. `GLM-5.3-Flash` (`glm5_next`: multimodal VLM extraction + `FP8` + `swiglu_limit=10.0`)
  3. `Qwen3.8-Flash-Next` (`qwen4_exp_text`: Hybrid `GatedDeltaNet` + `QSA` + gated shared expert)
  4. `MiniMax-M3.1-Flash` (`minimax_m3_vl_text`: `SwiGLU-OAI` `alpha=1.702, limit=7.0, up+1` on Torch & MLX)
  5. `Kimi-K3 / Kimi-Linear` (`kimi_linear`: `KDA` + `MLA` + `num_experts_per_token` on Torch & MLX)
  6. `LiquidAI LFM2.5-MoE` (`lfm2_moe`: `LIV ShortConv` + `num_dense_layers` on Torch & MLX)
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from deimos_stream.arena import DeimosMerkleBridge
from deimos_stream.cli import inspect_store
from deimos_stream.engine import HAS_MLX
from deimos_stream.factory import create_paged_moe_engine
from deimos_stream.hf_converter import convert_hf_hybrid_moe_to_deimos
from deimos_stream.quant_q4 import unpack_q4_torch
from deimos_stream.torch_arena import HAS_TORCH, TorchPagedExpertArena
from deimos_stream.torch_moe_exec import bind_hybrid_c_api

if HAS_TORCH:
    from safetensors.torch import save_file as st_save_torch
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.astype(np.float64).ravel(), b.astype(np.float64).ravel())[0, 1])


def _init_small_weights(model: "torch.nn.Module", std: float = 0.04) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.is_floating_point() and "A_log" not in name and "hc_" not in name:
                p.normal_(mean=0.0, std=std)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch & transformers required")
def test_deepseek_v4_ue8m0_clamped_swiglu_and_cpu_simd_parity(tmp_path: Path) -> None:
    torch.manual_seed(2026)
    cfg = AutoConfig.for_model(
        "deepseek_v4",
        vocab_size=256, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        hidden_size=128, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        kv_lora_rank=32, q_lora_rank=32, qk_rope_head_dim=16, qk_nope_head_dim=16, v_head_dim=32,
        index_n_heads=2, index_head_dim=16, index_topk=8,
        num_local_experts=4, num_experts_per_tok=2,
        mlp_layer_types=["moe", "moe"], scoring_func="sqrtsoftplus", swiglu_limit=2.5,
    )
    ref = AutoModelForCausalLM.from_config(cfg).eval()
    _init_small_weights(ref, std=0.18)  # Triggers active clamping at swiglu_limit=2.5!

    toks = [4, 17, 33, 91]
    with torch.no_grad():
        ref_logits = ref(torch.tensor([toks], dtype=torch.long)).logits[:, -1, :].float().numpy().ravel()

    # Serialize using DeepSeek-V4-Flash raw `layers.*.ffn.experts.*.w1/w2/w3.{weight,scale}` UE8M0 format
    raw_sd: Dict[str, torch.Tensor] = {}
    bs = 64
    has_ue8m0 = hasattr(torch, "float8_e8m0fnu") and hasattr(torch, "float8_e4m3fn")
    for k, v in ref.state_dict().items():
        rk = k
        if rk.startswith("model."):
            rk = rk[len("model.") :]
        rk = rk.replace("embed_tokens.weight", "embed.weight").replace("lm_head.weight", "head.weight")
        rk = rk.replace(".input_layernorm.", ".attn_norm.").replace(".post_attention_layernorm.", ".ffn_norm.")
        rk = rk.replace(".self_attn.", ".attn.").replace(".mlp.", ".ffn.")
        rk = rk.replace(".shared_experts.gate_proj.", ".shared_experts.w1.")
        rk = rk.replace(".shared_experts.down_proj.", ".shared_experts.w2.")
        rk = rk.replace(".shared_experts.up_proj.", ".shared_experts.w3.")

        if rk.endswith(".ffn.experts.gate_up_proj"):
            stem = rk[: -len(".gate_up_proj")]
            for e in range(4):
                gw, uw = v[e].chunk(2, dim=0)
                for pn, wt in (("w1", gw), ("w3", uw)):
                    if has_ue8m0:
                        wf = wt.float().view(1, bs, 2, bs)
                        exp_s = (wf.abs().amax(dim=(1, 3)).clamp(min=1e-4) / 224.0).to(torch.float8_e8m0fnu)
                        q = (wf / exp_s.float()[:, None, :, None]).clamp(-448.0, 448.0).reshape(64, 128)
                        raw_sd[f"{stem}.{e}.{pn}.weight"] = q.to(torch.float8_e4m3fn)
                        raw_sd[f"{stem}.{e}.{pn}.scale"] = exp_s
                    else:
                        raw_sd[f"{stem}.{e}.{pn}.weight"] = wt.contiguous()
        elif rk.endswith(".ffn.experts.down_proj"):
            stem = rk[: -len(".down_proj")]
            for e in range(4):
                raw_sd[f"{stem}.{e}.w2.weight"] = v[e].contiguous()
        else:
            raw_sd[rk] = v.contiguous()

    ckpt_dir = tmp_path / "dsv4_raw_ue8m0"
    store_dir = tmp_path / "dsv4_deimos_q4"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg_d = cfg.to_dict()
    cfg_d["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3", "scale_fmt": "ue8m0", "weight_block_size": [64, 64]}
    (ckpt_dir / "config.json").write_text(json.dumps(cfg_d, indent=2))
    st_save_torch(raw_sd, str(ckpt_dir / "model.safetensors"))

    convert_hf_hybrid_moe_to_deimos(str(ckpt_dir), store_dir, group_size=64)
    assert inspect_store(store_dir, verify_all_leaves=True)["byte_proofs_verified"] == 8

    eng = create_paged_moe_engine(store_dir, backend="cpu", slots_per_layer=2, miss_tail_tau=1.0)
    try:
        out_logits = eng.forward_logits(toks).float().cpu().numpy().ravel()  # type: ignore[union-attr]
        assert _pearson_r(ref_logits, out_logits) > 0.992
        assert len(eng.benchmark_token_ids(toks, max_new_tokens=2)["generated_ids"]) == 3
    finally:
        eng.close()

    # Direct C++17 SIMD (`deimos_cpu_q4_swiglu_forward`) SwiGLU-OAI / Clamped SwiGLU parity check
    bridge = DeimosMerkleBridge(store_dir, cas_filename="cas_experts_q4.bin", enable_nocache_fd=False)
    arena = TorchPagedExpertArena(bridge, device=torch.device("cpu"), slots_per_layer=4, verify_merkle_on_cold=True)
    bind_hybrid_c_api(bridge.lib)
    try:
        x_t = torch.randn((1, 128), dtype=torch.float16) * 2.5
        e_ids = np.array([0, 2], dtype=np.int32)
        e_sc = np.array([0.6, 0.4], dtype=np.float32)
        out_u16 = np.zeros((128,), dtype=np.uint16)
        rc = bridge.lib.deimos_cpu_q4_swiglu_forward(
            bridge.handle, 0,
            e_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            e_sc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            2,
            x_t.numpy().view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            out_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            128, 64, 64, 1,
            7.0, 1.702, 1.0,
        )
        assert rc == 0
        c_out = out_u16.view(np.float16).astype(np.float32)
        gu_w = unpack_q4_torch(arena.gu_w[0][e_ids], arena.gu_s[0][e_ids], arena.gu_b[0][e_ids], 64, torch.float32)
        dn_w = unpack_q4_torch(arena.dn_w[0][e_ids], arena.dn_s[0][e_ids], arena.dn_b[0][e_ids], 64, torch.float32)
        xf = x_t.float().expand(2, 128).unsqueeze(-1)
        g, u = torch.bmm(gu_w, xf).squeeze(-1).chunk(2, dim=-1)
        g = g.clamp(max=7.0)
        u = u.clamp(min=-7.0, max=7.0)
        act = ((g * torch.sigmoid(1.702 * g)) * (u + 1.0)).unsqueeze(-1)
        ref_y = (torch.bmm(dn_w, act).squeeze(-1) * torch.tensor([[0.6], [0.4]])).sum(dim=0).numpy()
        assert _pearson_r(ref_y, c_out) > 0.999
    finally:
        arena.close()
        bridge.close()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch & transformers required")
def test_glm53_flash_multimodal_fp8_clamped_swiglu_parity(tmp_path: Path) -> None:
    torch.manual_seed(53)
    cfg = AutoConfig.for_model(
        "glm5_next",
        text_config={
            "vocab_size": 256, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
            "hidden_size": 128, "intermediate_size": 256, "moe_intermediate_size": 64,
            "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
            "mlp_layer_types": ["dense", "sparse"],
            "first_k_dense_replace": 1, "num_attention_heads": 4, "num_key_value_heads": 4, "head_dim": 32,
            "kv_lora_rank": 32, "q_lora_rank": 32, "qk_rope_head_dim": 0, "qk_nope_head_dim": 32, "v_head_dim": 32,
            "index_n_heads": 2, "index_head_dim": 16, "index_topk": 16,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1, "swiglu_limit": 3.0,
        },
    )
    ref = AutoModelForImageTextToText.from_config(cfg).eval()
    ref.model.visual = None
    _init_small_weights(ref, std=0.12)

    toks = [6, 15, 29, 71]
    with torch.no_grad():
        ref_logits = ref(input_ids=torch.tensor([toks], dtype=torch.long)).logits[:, -1, :].float().numpy().ravel()

    sd: Dict[str, torch.Tensor] = {"model.visual.dummy.weight": torch.zeros((16, 16), dtype=torch.float16)}
    for k, v in ref.state_dict().items():
        if k.endswith(".mlp.experts.gate_up_proj"):
            stem = k[: -len(".gate_up_proj")]
            for e in range(4):
                gw, uw = v[e].chunk(2, dim=0)
                sd[f"{stem}.{e}.gate_proj.weight"] = gw.contiguous()
                sd[f"{stem}.{e}.up_proj.weight"] = uw.contiguous()
        elif k.endswith(".mlp.experts.down_proj"):
            stem = k[: -len(".down_proj")]
            for e in range(4):
                sd[f"{stem}.{e}.down_proj.weight"] = v[e].contiguous()
        else:
            sd[k] = v.contiguous()

    ckpt_dir, store_dir = tmp_path / "glm53_src", tmp_path / "glm53_q4"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    st_save_torch(sd, str(ckpt_dir / "model.safetensors"))

    convert_hf_hybrid_moe_to_deimos(str(ckpt_dir), store_dir, group_size=64)
    eng = create_paged_moe_engine(store_dir, backend="auto", slots_per_layer=2, miss_tail_tau=1.0)
    try:
        logits = eng.forward_logits(toks).float().cpu().numpy().ravel()  # type: ignore[union-attr]
        assert _pearson_r(ref_logits, logits) > 0.993
    finally:
        eng.close()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch & transformers required")
def test_qwen38_flash_next_qsa_hybrid_moe_parity(tmp_path: Path) -> None:
    torch.manual_seed(38)
    cfg = AutoConfig.for_model(
        "qwen4_exp_text",
        vocab_size=256, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        hidden_size=128, intermediate_size=256, moe_intermediate_size=64, shared_expert_intermediate_size=64,
        num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
        num_experts=4, num_experts_per_tok=2,
        ple_layer_ids=[], ngram_vocab_size_base=64,
        indexer_budget=16, indexer_compress_ratio=4, indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=32,
    )
    ref = AutoModelForCausalLM.from_config(cfg).eval()
    _init_small_weights(ref, std=0.04)

    toks = [2, 18, 44, 99]
    with torch.no_grad():
        ref_logits = ref(torch.tensor([toks], dtype=torch.long)).logits[:, -1, :].float().numpy().ravel()

    ckpt_dir, store_dir = tmp_path / "qwen38_src", tmp_path / "qwen38_q4"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    st_save_torch({k: v.contiguous() for k, v in ref.state_dict().items()}, str(ckpt_dir / "model.safetensors"))

    convert_hf_hybrid_moe_to_deimos(str(ckpt_dir), store_dir, group_size=64)
    eng = create_paged_moe_engine(store_dir, backend="auto", slots_per_layer=2, miss_tail_tau=1.0)
    try:
        logits = eng.forward_logits(toks).float().cpu().numpy().ravel()  # type: ignore[union-attr]
        assert _pearson_r(ref_logits, logits) > 0.995
    finally:
        eng.close()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch & transformers required")
def test_minimax_m31_flash_swiglu_oai_pytorch_and_mlx_parity(tmp_path: Path) -> None:
    torch.manual_seed(31)
    cfg = AutoConfig.for_model(
        "minimax_m3_vl_text",
        vocab_size=256, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        hidden_size=128, intermediate_size=64, dense_intermediate_size=128, shared_intermediate_size=64,
        num_hidden_layers=2, first_k_dense_replace=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        num_local_experts=4, num_experts_per_tok=2, n_shared_experts=1,
        routed_scaling_factor=1.25, swiglu_alpha=1.702, swiglu_limit=7.0,
    )
    ref = AutoModelForCausalLM.from_config(cfg).eval()
    _init_small_weights(ref, std=0.08)

    toks = [8, 27, 51, 84]
    with torch.no_grad():
        ref_logits = ref(torch.tensor([toks], dtype=torch.long)).logits[:, -1, :].float().numpy().ravel()

    ckpt_dir, store_dir = tmp_path / "mm31_src", tmp_path / "mm31_q4"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    st_save_torch({k: v.contiguous() for k, v in ref.state_dict().items()}, str(ckpt_dir / "model.safetensors"))

    convert_hf_hybrid_moe_to_deimos(str(ckpt_dir), store_dir, group_size=64)
    eng_cpu = create_paged_moe_engine(store_dir, backend="cpu", slots_per_layer=2, miss_tail_tau=1.0)
    try:
        cpu_logits = eng_cpu.forward_logits(toks).float().cpu().numpy().ravel()  # type: ignore[union-attr]
        assert _pearson_r(ref_logits, cpu_logits) > 0.995
    finally:
        eng_cpu.close()

    if HAS_MLX:
        eng_mlx = create_paged_moe_engine(store_dir, backend="mlx", slots_per_layer=2, miss_tail_tau=1.0)
        try:
            mlx_logits = np.asarray(eng_mlx.forward_logits(toks), dtype=np.float32).ravel()
            assert _pearson_r(ref_logits, mlx_logits) > 0.995
            assert len(eng_mlx.benchmark_token_ids(toks, max_new_tokens=2)["generated_ids"]) == 3
        finally:
            eng_mlx.close()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch & transformers required")
def test_kimi_k3_linear_and_liquid_lfm25_moe_pytorch_and_mlx_parity(tmp_path: Path) -> None:
    torch.manual_seed(93)
    # 1. Kimi-K3 / Kimi-Linear (`KDA` + `MLA` + `num_experts_per_token`)
    kcfg = AutoConfig.for_model(
        "kimi_linear",
        vocab_size=256, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        hidden_size=128, intermediate_size=256, moe_intermediate_size=64,
        num_hidden_layers=2, layer_types=["linear_attention", "full_attention"], first_k_dense_replace=1,
        num_attention_heads=4, num_key_value_heads=4,
        kv_lora_rank=32, q_lora_rank=32, qk_rope_head_dim=16, qk_nope_head_dim=16, v_head_dim=32,
        linear_num_heads=4, linear_head_k_dim=32, linear_head_v_dim=32,
        num_experts=4, num_experts_per_token=2, num_shared_experts=1,
    )
    kref = AutoModelForCausalLM.from_config(kcfg).eval()
    _init_small_weights(kref, std=0.04)
    toks = [11, 32, 64, 112]
    with torch.no_grad():
        k_logits = kref(torch.tensor([toks], dtype=torch.long)).logits[:, -1, :].float().numpy().ravel()

    k_ckpt, k_store = tmp_path / "kimi_k3_src", tmp_path / "kimi_k3_q4"
    k_ckpt.mkdir(parents=True, exist_ok=True)
    (k_ckpt / "config.json").write_text(json.dumps(kcfg.to_dict(), indent=2))
    st_save_torch({k: v.contiguous() for k, v in kref.state_dict().items()}, str(k_ckpt / "model.safetensors"))
    convert_hf_hybrid_moe_to_deimos(str(k_ckpt), k_store, group_size=64)

    eng_k = create_paged_moe_engine(k_store, backend="cpu", slots_per_layer=2, miss_tail_tau=1.0)
    try:
        assert _pearson_r(k_logits, eng_k.forward_logits(toks).float().cpu().numpy().ravel()) > 0.995  # type: ignore[union-attr]
    finally:
        eng_k.close()
    if HAS_MLX:
        eng_k_mlx = create_paged_moe_engine(k_store, backend="mlx", slots_per_layer=2, miss_tail_tau=1.0)
        try:
            assert _pearson_r(k_logits, np.asarray(eng_k_mlx.forward_logits(toks), dtype=np.float32).ravel()) > 0.994
        finally:
            eng_k_mlx.close()

    # 2. LiquidAI LFM2.5-MoE (`LIV ShortConv` + `num_dense_layers=1` + `feed_forward`)
    lcfg = AutoConfig.for_model(
        "lfm2_moe",
        vocab_size=256, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        hidden_size=128, intermediate_size=256, moe_intermediate_size=64,
        num_hidden_layers=3, num_dense_layers=1,
        num_attention_heads=4, num_key_value_heads=2,
        num_experts=4, num_experts_per_tok=2,
        layer_types=["conv", "full_attention", "full_attention"],
    )
    lref = AutoModelForCausalLM.from_config(lcfg).eval()
    _init_small_weights(lref, std=0.04)
    with torch.no_grad():
        l_logits = lref(torch.tensor([toks], dtype=torch.long)).logits[:, -1, :].float().numpy().ravel()

    l_ckpt, l_store = tmp_path / "lfm25_src", tmp_path / "lfm25_q4"
    l_ckpt.mkdir(parents=True, exist_ok=True)
    (l_ckpt / "config.json").write_text(json.dumps(lcfg.to_dict(), indent=2))
    st_save_torch({k: v.clone() for k, v in lref.state_dict().items() if k != "lm_head.weight"}, str(l_ckpt / "model.safetensors"))
    convert_hf_hybrid_moe_to_deimos(str(l_ckpt), l_store, group_size=64)

    eng_l = create_paged_moe_engine(l_store, backend="cpu", slots_per_layer=2, miss_tail_tau=1.0)
    try:
        assert _pearson_r(l_logits, eng_l.forward_logits(toks).float().cpu().numpy().ravel()) > 0.995  # type: ignore[union-attr]
    finally:
        eng_l.close()
    if HAS_MLX:
        eng_l_mlx = create_paged_moe_engine(l_store, backend="mlx", slots_per_layer=2, miss_tail_tau=1.0)
        try:
            assert _pearson_r(l_logits, np.asarray(eng_l_mlx.forward_logits(toks), dtype=np.float32).ravel()) > 0.995
        finally:
            eng_l_mlx.close()
