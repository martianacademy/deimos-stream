#!/usr/bin/env python3
"""
End-to-End Tests for Native Checkpoint MTP (`mtp.*` & `NextN`) Drafter
(`tests/test_native_mtp_drafter.py`)
======================================================================
Verifies:
  1. `Qwen3.6` (`qwen_mtp`): `hf_converter` exports `mtp_head.safetensors` + `mtp_manifest.json`
     without altering `DEIMIDX2` leaves, and both `DeimosHFHybridMoEEngine` (PyTorch) and
     `DeimosMLXHybridMoEEngine` (Apple Silicon MLX) auto-attach native MTP drafters that draft
     tokens on non-repeating prompts while preserving 100% exact greedy output tokens.
  2. `GLM-5.3-Flash` (`nextn_layer`, `FP8` + `mHC` + `swiglu_limit=10.0`): both zero-copy raw
     `.safetensors` `DeimosFP8SafetensorsEngine` and converted `DEIMIDX2` `DeimosHFHybridMoEEngine`
     auto-load `layers.{num_hidden_layers}` MTP weights and preserve 100% exact greedy parity.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from deimos_stream.fp8_engine import DeimosFP8SafetensorsEngine
from deimos_stream.hf_converter import convert_hf_hybrid_moe_to_deimos
from deimos_stream.hf_streamer import DeimosHFHybridMoEEngine
from deimos_stream.mlx_hybrid import HAS_MLX, DeimosMLXHybridMoEEngine
from deimos_stream.mtp_drafter import MLXNativeMTPDrafter, TorchNativeMTPDrafter


def _quant_fp8_block(w_f32: torch.Tensor, block: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    orig = w_f32.shape
    m, n = orig[-2], orig[-1]
    lead = orig[:-2]
    grouped = w_f32.reshape(*lead, m // block, block, n // block, block)
    amax = grouped.abs().amax(dim=(-3, -1)).clamp(min=1e-6)
    scale_inv = (amax / 400.0).to(torch.float32)
    s_exp = scale_inv.repeat_interleave(block, dim=-2).repeat_interleave(block, dim=-1)
    q_fp8 = (w_f32 / s_exp).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q_fp8.contiguous(), scale_inv.contiguous()


def _build_qwen36_with_mtp_checkpoint(ckpt_dir: Path) -> None:
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer,
        Qwen3_5MoeForCausalLM,
    )

    torch.manual_seed(17)
    cfg_dict = {
        "model_type": "qwen3_5_moe_text",
        "vocab_size": 96,
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "rms_norm_eps": 1e-6,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "full_attention_interval": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "mtp_num_hidden_layers": 1,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
        },
    }
    cfg = Qwen3_5MoeTextConfig(**cfg_dict)
    model = Qwen3_5MoeForCausalLM(cfg).eval()
    for p in model.parameters():
        if p.ndim >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=0.04)

    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}

    # Build a real full-attention Qwen3_5MoeDecoderLayer for `mtp.layers.0.*`
    mtp_cfg = Qwen3_5MoeTextConfig(**{**cfg_dict, "num_hidden_layers": 1, "layer_types": ["full_attention"]})
    mtp_layer = Qwen3_5MoeDecoderLayer(mtp_cfg, layer_idx=0).eval()
    for p in mtp_layer.parameters():
        if p.ndim >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=0.04)
    for lk, lv in mtp_layer.state_dict().items():
        sd[f"mtp.layers.0.{lk}"] = lv.to(torch.bfloat16).contiguous()

    d = cfg_dict["hidden_size"]
    sd["mtp.pre_fc_norm_embedding.weight"] = torch.zeros(d, dtype=torch.bfloat16)
    sd["mtp.pre_fc_norm_hidden.weight"] = torch.zeros(d, dtype=torch.bfloat16)
    # Bias eh_proj toward forwarding embedding + hidden smoothly
    fc = torch.cat([0.5 * torch.eye(d), 0.5 * torch.eye(d)], dim=1)
    sd["mtp.fc.weight"] = fc.to(torch.bfloat16).contiguous()
    sd["mtp.norm.weight"] = torch.zeros(d, dtype=torch.bfloat16)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    save_file(sd, str(ckpt_dir / "model.safetensors"))


def _build_glm53_with_nextn_fp8_checkpoint(ckpt_dir: Path) -> None:
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextForConditionalGeneration,
        Glm5NextTextDecoderLayer,
    )

    torch.manual_seed(29)
    raw_cfg = {
        "model_type": "glm5_next",
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [64, 64]},
        "vision_config": {
            "model_type": "glm5_next_vision",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "depth": 1,
            "out_hidden_size": 128,
        },
        "text_config": {
            "model_type": "glm5_next_text",
            "vocab_size": 96,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "bos_token_id": 2,
            "hidden_size": 128,
            "intermediate_size": 128,
            "moe_intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_nextn_predict_layers": 1,
            "first_k_dense_replace": 1,
            "mlp_layer_types": ["dense", "sparse"],
            "layer_types": ["linear_attention", "indexed_attention"],
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 2.5,
            "swiglu_limit": 10.0,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "q_lora_rank": 64,
            "kv_lora_rank": 64,
            "qk_nope_head_dim": 32,
            "qk_rope_head_dim": 0,
            "v_head_dim": 32,
            "mla_use_nope": True,
            "index_n_heads": 2,
            "index_head_dim": 32,
            "index_topk": 16,
            "index_kpool": 2,
            "indexer_types": ["none", "full"],
            "mhc": True,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 4,
            "hc_eps": 1e-6,
            "rms_norm_eps": 1e-6,
            "linear_attn_config": {
                "num_heads": 4,
                "head_dim": 32,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": 0.0,
            },
        },
    }
    clean_cfg = {k: v for k, v in raw_cfg.items() if k != "quantization_config"}
    hf_cfg = Glm5NextConfig.from_dict(clean_cfg)
    model = Glm5NextForConditionalGeneration(hf_cfg).eval()
    for p in model.parameters():
        if p.ndim >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=0.03)

    sd: dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if k.startswith("model.visual."):
            continue
        if ".experts." in k and v.ndim == 2 and v.shape[0] % 64 == 0 and v.shape[1] % 64 == 0:
            q8, s_inv = _quant_fp8_block(v.float(), block=64)
            sd[k] = q8
            sd[f"{k}_scale_inv"] = s_inv
        else:
            sd[k] = v.to(torch.bfloat16 if v.dtype == torch.float32 and "A_log" not in k else v.dtype).contiguous()

    # Append NextN MTP block at `model.language_model.layers.2.*`
    mtp_tcfg = copy.deepcopy(hf_cfg.text_config)
    mtp_tcfg.num_hidden_layers = 3
    mtp_tcfg.mlp_layer_types = ["dense", "sparse", "sparse"]
    mtp_tcfg.layer_types = ["linear_attention", "indexed_attention", "indexed_attention"]
    mtp_tcfg.indexer_types = ["none", "full", "full"]
    nextn_lyr = Glm5NextTextDecoderLayer(mtp_tcfg, layer_idx=2).eval()
    for p in nextn_lyr.parameters():
        if p.ndim >= 2:
            torch.nn.init.normal_(p, mean=0.0, std=0.03)

    pfx = "model.language_model.layers.2"
    for lk, lv in nextn_lyr.state_dict().items():
        full_k = f"{pfx}.{lk}"
        if ".experts." in full_k and lv.ndim == 2 and lv.shape[0] % 64 == 0 and lv.shape[1] % 64 == 0:
            q8, s_inv = _quant_fp8_block(lv.float(), block=64)
            sd[full_k] = q8
            sd[f"{full_k}_scale_inv"] = s_inv
        else:
            sd[full_k] = lv.to(torch.bfloat16 if lv.dtype == torch.float32 and "A_log" not in lk else lv.dtype).contiguous()

    d = 128
    sd[f"{pfx}.enorm.weight"] = torch.ones(d, dtype=torch.bfloat16)
    sd[f"{pfx}.hnorm.weight"] = torch.ones(d, dtype=torch.bfloat16)
    sd[f"{pfx}.eh_proj.weight"] = torch.cat([0.5 * torch.eye(d), 0.5 * torch.eye(d)], dim=1).to(torch.bfloat16)
    sd[f"{pfx}.shared_head.norm.weight"] = torch.ones(d, dtype=torch.bfloat16)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(json.dumps(raw_cfg, indent=2))
    save_file(sd, str(ckpt_dir / "model-00001-of-00001.safetensors"))
    (ckpt_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1024}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in sd}}, indent=2)
    )


def test_qwen36_native_mtp_pytorch_and_mlx_exact_parity(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "qwen36_mtp_ckpt"
    store_dir = tmp_path / "qwen36_mtp_store"
    _build_qwen36_with_mtp_checkpoint(ckpt_dir)

    convert_hf_hybrid_moe_to_deimos(str(ckpt_dir), store_dir, group_size=64, delete_downloaded_shards=False)
    assert (store_dir / "mtp_head.safetensors").exists()
    assert (store_dir / "mtp_manifest.json").exists()
    man = json.loads((store_dir / "merkle_manifest.json").read_text())
    assert man["has_mtp_head"] is True
    assert man["total_experts"] == 4 * 4  # DEIMIDX2 leaves untouched

    # Non-repeating prompt so SuffixNgramDrafter alone cannot draft anything
    prompt = [3, 11, 19, 27, 35, 43, 51, 59]

    eng_hf = DeimosHFHybridMoEEngine(store_dir, device_str="cpu", slots_per_layer=2, verify_merkle_on_cold=True)
    try:
        eng_hf.enable_speculative = False
        base_hf = eng_hf.benchmark_token_ids(prompt, max_new_tokens=16)["generated_ids"]

        eng_hf.enable_speculative = True
        eng_hf.spec_max_draft = 3
        spec_hf = eng_hf.benchmark_token_ids(prompt, max_new_tokens=16)["generated_ids"]

        assert isinstance(eng_hf.mtp_drafter, TorchNativeMTPDrafter)
        assert eng_hf.last_spec_stats["mtp_active"] is True
        assert eng_hf.last_spec_stats["drafted_tokens"] > 0
        assert spec_hf == base_hf
    finally:
        eng_hf.close()

    if HAS_MLX:
        eng_mlx = DeimosMLXHybridMoEEngine(store_dir, slots_per_layer=2, verify_merkle_on_cold=True)
        try:
            eng_mlx.enable_speculative = False
            base_mlx = eng_mlx.benchmark_token_ids(prompt, max_new_tokens=16)["generated_ids"]

            eng_mlx.enable_speculative = True
            eng_mlx.spec_max_draft = 3
            spec_mlx = eng_mlx.benchmark_token_ids(prompt, max_new_tokens=16)["generated_ids"]

            assert isinstance(eng_mlx.mtp_drafter, MLXNativeMTPDrafter)
            assert eng_mlx.last_spec_stats["mtp_active"] is True
            assert eng_mlx.last_spec_stats["drafted_tokens"] > 0
            assert spec_mlx == base_mlx
        finally:
            eng_mlx.close()


def test_glm53_nextn_mtp_raw_fp8_and_converted_q4_exact_parity(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "glm53_mtp_fp8"
    store_dir = tmp_path / "glm53_mtp_q4"
    _build_glm53_with_nextn_fp8_checkpoint(ckpt_dir)

    prompt = [5, 14, 23, 32, 41, 50, 61]

    # 1. Zero-copy raw `.safetensors` FP8 engine auto-loads `layers.2.*` MTP head directly!
    eng_fp8 = DeimosFP8SafetensorsEngine(ckpt_dir, device_str="cpu", slots_per_layer=2)
    try:
        eng_fp8.enable_speculative = False
        base_fp8 = eng_fp8.benchmark_token_ids(prompt, max_new_tokens=14)["generated_ids"]

        eng_fp8.enable_speculative = True
        eng_fp8.spec_max_draft = 2
        spec_fp8 = eng_fp8.benchmark_token_ids(prompt, max_new_tokens=14)["generated_ids"]

        assert isinstance(eng_fp8.mtp_drafter, TorchNativeMTPDrafter)
        assert eng_fp8.last_spec_stats["mtp_active"] is True
        assert eng_fp8.last_spec_stats["drafted_tokens"] > 0
        assert spec_fp8 == base_fp8
    finally:
        eng_fp8.close()

    # 2. Converted `DEIMIDX2` Q4 store exports `mtp_head.safetensors` & runs bit-exact!
    convert_hf_hybrid_moe_to_deimos(str(ckpt_dir), store_dir, group_size=64, delete_downloaded_shards=False)
    assert (store_dir / "mtp_head.safetensors").exists()

    eng_q4 = DeimosHFHybridMoEEngine(store_dir, device_str="cpu", slots_per_layer=2, verify_merkle_on_cold=True)
    try:
        eng_q4.enable_speculative = False
        base_q4 = eng_q4.benchmark_token_ids(prompt, max_new_tokens=14)["generated_ids"]

        eng_q4.enable_speculative = True
        eng_q4.spec_max_draft = 2
        spec_q4 = eng_q4.benchmark_token_ids(prompt, max_new_tokens=14)["generated_ids"]

        assert isinstance(eng_q4.mtp_drafter, TorchNativeMTPDrafter)
        assert eng_q4.last_spec_stats["mtp_active"] is True
        assert eng_q4.last_spec_stats["drafted_tokens"] > 0
        assert spec_q4 == base_q4
    finally:
        eng_q4.close()


def test_native_mtp_mlx_vs_torch_math_parity_and_multi_token_jumps(tmp_path: Path) -> None:
    """Verifies Torch vs MLX MTP hidden/draft agreement (`r > 0.999`) and `target_calls < max_new_tokens`."""
    import numpy as np

    ckpt_dir = tmp_path / "qwen36_calibrated_ckpt"
    store_dir = tmp_path / "qwen36_calibrated_store"
    _build_qwen36_with_mtp_checkpoint(ckpt_dir)

    # Align the checkpoint's `lm_head` & `mtp` residual on a subset of tokens so MTP drafts hit >60%
    from safetensors.torch import load_file

    sd = load_file(str(ckpt_dir / "model.safetensors"))
    # Zero MTP attn/moe projections so `z1 + 0` passes `0.5*emb + 0.5*h_prev` cleanly, and align target
    for k in list(sd.keys()):
        if k.startswith("mtp.layers.0.") and ("o_proj" in k or "down_proj" in k):
            sd[k] = torch.zeros_like(sd[k])
        if k.startswith("model.layers.") and ("o_proj" in k or "out_proj" in k or "down_proj" in k):
            sd[k] = (sd[k].float() * 0.02).to(torch.bfloat16)
    d = 128
    sd["mtp.fc.weight"] = torch.cat([torch.eye(d), torch.zeros(d, d)], dim=1).to(torch.bfloat16)
    save_file(sd, str(ckpt_dir / "model.safetensors"))

    convert_hf_hybrid_moe_to_deimos(str(ckpt_dir), store_dir, group_size=64, delete_downloaded_shards=False)
    prompt = [4, 12, 20, 28, 36, 44]

    eng_hf = DeimosHFHybridMoEEngine(store_dir, device_str="cpu", slots_per_layer=4, verify_merkle_on_cold=False)
    try:
        eng_hf.enable_speculative = False
        base_ids = eng_hf.benchmark_token_ids(prompt, max_new_tokens=18)["generated_ids"]

        eng_hf.enable_speculative = True
        eng_hf.spec_max_draft = 3
        spec_ids = eng_hf.benchmark_token_ids(prompt, max_new_tokens=18)["generated_ids"]
        assert spec_ids == base_ids
        assert eng_hf.last_spec_stats["accepted_tokens"] >= 6
        assert eng_hf.last_spec_stats["target_calls"] < 18

        # Capture Torch MTP step-1 fused+layer output on fresh prefill
        c_hf = eng_hf.make_caches()
        _ = eng_hf.prefill_prompt(prompt, c_hf)
        eng_hf.mtp_drafter.on_prefill(prompt, eng_hf._last_hidden_states)
        t_drafts = eng_hf.mtp_drafter.draft_step(prompt[-1], 3)
        t_h = eng_hf.mtp_drafter._pending_h.detach().float().cpu().numpy().reshape(-1)
    finally:
        eng_hf.close()

    if HAS_MLX:
        eng_mlx = DeimosMLXHybridMoEEngine(store_dir, slots_per_layer=4, verify_merkle_on_cold=False)
        try:
            eng_mlx.enable_speculative = False
            base_mlx = eng_mlx.benchmark_token_ids(prompt, max_new_tokens=18)["generated_ids"]

            eng_mlx.enable_speculative = True
            eng_mlx.spec_max_draft = 3
            spec_mlx = eng_mlx.benchmark_token_ids(prompt, max_new_tokens=18)["generated_ids"]
            assert spec_mlx == base_mlx
            assert eng_mlx.last_spec_stats["accepted_tokens"] >= 6
            assert eng_mlx.last_spec_stats["target_calls"] < 18

            c_mx = eng_mlx.make_caches()
            _ = eng_mlx.prefill_prompt(prompt, c_mx)
            eng_mlx.mtp_drafter.on_prefill(prompt, eng_mlx._last_hidden_states)
            m_drafts = eng_mlx.mtp_drafter.draft_step(prompt[-1], 3)
            m_h = np.array(eng_mlx.mtp_drafter._pending_h, dtype=np.float32).reshape(-1)
            corr = float(np.corrcoef(t_h, m_h)[0, 1])
            assert corr > 0.999
            assert m_drafts[0] == t_drafts[0]
        finally:
            eng_mlx.close()

