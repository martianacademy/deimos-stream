#!/usr/bin/env python3
"""
Zero-Copy Raw `.safetensors` Native `FP8` Engine Parity Suite
(`tests/test_fp8_zero_copy_engine.py`)
=============================================================
Verifies:
  1. Custom Apple Silicon Metal `deimos_fp8_gu_swiglu` + `deimos_fp8_dn_gemv` (`uchar4` + `float`
     LUT + `simd_sum`) numerical parity against exact `FP32` reference across `BM/BN = 128` & `64`,
     clamped SwiGLU, `SwiGLU-OAI`, multi-chunk prefill (`N=20`), and skipped slots (`slot = -1`).
  2. Multi-shard unconverted `zai-org/GLM-5.3-Flash` (`glm5_next` multimodal VLM, `F8_E4M3` +
     `weight_scale_inv` `[128, 128]`, `BF16` mmap embeddings + INT8 `lm_head`) running directly via
     `create_paged_moe_engine`, `plan_slots`, and `inspect_store` with **0 extra files written**.
  3. Unconverted `DeepSeek-V4-Flash` (`deepseek_v4`, `UE8M0` `.scale` + `layers.*.ffn`) parity
     and Miss-Only Nucleus Dynamic-K (`miss_tail_tau < 1.0`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from deimos_stream.cli import inspect_store
from deimos_stream.factory import create_paged_moe_engine
from deimos_stream.fp8_kernels import HAS_MLX_FP8, mlx_fp8_moe_swiglu
from deimos_stream.fp8_safetensors import (
    _E4M3_LUT_F32,
    _encode_f32_to_fp8_block,
    is_raw_fp8_moe_checkpoint,
)
from deimos_stream.memory_budget import plan_slots
from deimos_stream.torch_arena import HAS_TORCH

if HAS_MLX_FP8:
    import mlx.core as mx

if HAS_TORCH:
    from safetensors.torch import save_file as st_save_torch
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.astype(np.float64).ravel(), b.astype(np.float64).ravel())[0, 1])


def _init_weights(model: "torch.nn.Module", std: float = 0.05) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.is_floating_point() and "A_log" not in name and "hc_" not in name:
                p.normal_(mean=0.0, std=std)


@pytest.mark.skipif(not HAS_MLX_FP8, reason="Apple Silicon Metal MLX required")
@pytest.mark.parametrize("bm,bn,limit,alpha,up_off", [(128, 128, 3.5, 1.0, 0.0), (64, 64, 7.0, 1.702, 1.0)])
def test_mlx_metal_fp8_swiglu_kernel_parity(bm: int, bn: int, limit: float, alpha: float, up_off: float) -> None:
    rng = np.random.default_rng(101 + bm)
    N, K, S, D, I = 20, 3, 5, 256, 128
    x_np = (rng.standard_normal((N, D)) * 0.6).astype(np.float16)

    gu_u8 = np.zeros((S, 2 * I, D), dtype=np.uint8)
    gu_sc = np.zeros((S, (2 * I) // bm, D // bn), dtype=np.float32)
    dn_u8 = np.zeros((S, D, I), dtype=np.uint8)
    dn_sc = np.zeros((S, D // bm, I // bn), dtype=np.float32)
    for s in range(S):
        gu_u8[s], gu_sc[s] = _encode_f32_to_fp8_block((rng.standard_normal((2 * I, D)) * 0.35).astype(np.float32), bm, bn)
        dn_u8[s], dn_sc[s] = _encode_f32_to_fp8_block((rng.standard_normal((D, I)) * 0.35).astype(np.float32), bm, bn)

    slots_np = rng.integers(0, S, size=(N, K), dtype=np.int32)
    slots_np[3, 2] = -1  # Verify Dynamic-K skipped tail slot handling
    scores_np = rng.uniform(0.1, 0.6, size=(N, K)).astype(np.float32)
    scores_np[3, 2] = 0.0

    y_mx = mlx_fp8_moe_swiglu(
        mx.array(x_np),
        mx.array(gu_u8),
        mx.array(gu_sc),
        mx.array(dn_u8),
        mx.array(dn_sc),
        mx.array(slots_np.reshape(-1)),
        mx.array(scores_np.reshape(-1)),
        top_k=K,
        block_m=bm,
        block_n=bn,
        swiglu_limit=limit,
        swiglu_alpha=alpha,
        swiglu_up_offset=up_off,
    )
    mx.eval(y_mx)
    got = np.array(y_mx, dtype=np.float32)

    ref = np.zeros((N, D), dtype=np.float32)
    for ti in range(N):
        xt = x_np[ti].astype(np.float32)
        for ki in range(K):
            s = int(slots_np[ti, ki])
            if s < 0:
                continue
            gw = _E4M3_LUT_F32[gu_u8[s]] * np.repeat(np.repeat(gu_sc[s], bm, axis=0), bn, axis=1)
            dw = _E4M3_LUT_F32[dn_u8[s]] * np.repeat(np.repeat(dn_sc[s], bm, axis=0), bn, axis=1)
            gu = gw @ xt
            g = np.minimum(gu[:I], limit)
            u = np.clip(gu[I:], -limit, limit)
            act = (g / (1.0 + np.exp(-alpha * g))) * (u + up_off)
            ref[ti] += scores_np[ti, ki] * (dw @ act)

    assert _pearson_r(ref, got) > 0.99998
    assert float(np.max(np.abs(ref.astype(np.float16).astype(np.float32) - got))) <= 0.0625


@pytest.mark.skipif(not (HAS_TORCH and hasattr(torch, "float8_e4m3fn")), reason="PyTorch with float8_e4m3fn required")
def test_raw_fp8_glm53_flash_multishard_zero_copy_engine(tmp_path: Path) -> None:
    torch.manual_seed(530)
    cfg = AutoConfig.for_model(
        "glm5_next",
        text_config={
            "vocab_size": 256,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_size": 256,
            "intermediate_size": 512,
            "moe_intermediate_size": 128,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "mlp_layer_types": ["dense", "sparse"],
            "first_k_dense_replace": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "kv_lora_rank": 64,
            "q_lora_rank": 64,
            "qk_rope_head_dim": 0,
            "qk_nope_head_dim": 64,
            "v_head_dim": 64,
            "index_n_heads": 2,
            "index_head_dim": 32,
            "index_topk": 16,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "swiglu_limit": 3.0,
        },
    )
    ref = AutoModelForImageTextToText.from_config(cfg).eval()
    ref.model.visual = None
    _init_weights(ref, std=0.08)

    toks = [5, 19, 42, 88, 121]
    with torch.no_grad():
        ref_logits = ref(input_ids=torch.tensor([toks], dtype=torch.long)).logits[:, -1, :].float().numpy().ravel()

    shard1: Dict[str, torch.Tensor] = {"model.visual.dummy.weight": torch.zeros((16, 16), dtype=torch.bfloat16)}
    shard2: Dict[str, torch.Tensor] = {}
    bm, bn = 128, 128

    def _put_fp8(target: Dict[str, torch.Tensor], key: str, w: torch.Tensor) -> None:
        u8, sc = _encode_f32_to_fp8_block(w.float().numpy(), bm, bn)
        target[key] = torch.from_numpy(u8).view(torch.float8_e4m3fn)
        target[f"{key}_scale_inv"] = torch.from_numpy(sc)

    for k, v in ref.state_dict().items():
        if k.endswith(".mlp.experts.gate_up_proj"):
            stem = k[: -len(".gate_up_proj")]
            for e in range(4):
                gw, uw = v[e].chunk(2, dim=0)
                dst = shard1 if e < 2 else shard2
                _put_fp8(dst, f"{stem}.{e}.gate_proj.weight", gw)
                _put_fp8(dst, f"{stem}.{e}.up_proj.weight", uw)
        elif k.endswith(".mlp.experts.down_proj"):
            stem = k[: -len(".down_proj")]
            for e in range(4):
                dst = shard1 if e < 2 else shard2
                _put_fp8(dst, f"{stem}.{e}.down_proj.weight", v[e])
        elif "embed_tokens" in k or "lm_head" in k:
            shard1[k] = v.to(torch.bfloat16).contiguous()
        elif v.dtype == torch.float32 and ("A_log" in k or "dt_bias" in k or "hc_" in k or "gate" in k or "conv1d" in k):
            shard1[k] = v.contiguous()
        else:
            shard1[k] = v.to(torch.bfloat16).contiguous()

    ckpt_dir = tmp_path / "glm53_raw_fp8"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = cfg.to_dict()
    cfg_dict["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}
    (ckpt_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    s1_name, s2_name = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    st_save_torch(shard1, str(ckpt_dir / s1_name))
    st_save_torch(shard2, str(ckpt_dir / s2_name))
    wm = {k: s1_name for k in shard1}
    wm.update({k: s2_name for k in shard2})
    (ckpt_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}, indent=2))

    files_before = {p.name for p in ckpt_dir.iterdir()}
    assert is_raw_fp8_moe_checkpoint(ckpt_dir)

    info = inspect_store(ckpt_dir)
    assert info["store_kind"] == "raw_safetensors_fp8"
    assert info["num_shards"] == 2
    assert info["block_shape"] == [128, 128]

    plan = plan_slots(ckpt_dir, backend="mlx" if HAS_MLX_FP8 else "cpu", available_bytes=4 * 1024**3)
    assert plan.slots_per_layer == 4

    eng = create_paged_moe_engine(ckpt_dir, backend="auto", slots_per_layer=2, miss_tail_tau=1.0)
    try:
        if HAS_MLX_FP8:
            assert eng.backend_name == "fp8_metal"
        logits = eng.forward_logits(toks).float().cpu().numpy().ravel()  # type: ignore[union-attr]
        assert _pearson_r(ref_logits, logits) > 0.998
        bench = eng.benchmark_token_ids(toks, max_new_tokens=3)
        assert len(bench["generated_ids"]) == 4
        assert bench["slots_per_layer"] == 2
    finally:
        eng.close()

    assert {p.name for p in ckpt_dir.iterdir()} == files_before


@pytest.mark.skipif(not (HAS_TORCH and hasattr(torch, "float8_e8m0fnu")), reason="PyTorch with float8_e8m0fnu required")
def test_raw_fp8_deepseek_v4_ue8m0_and_dynamic_k(tmp_path: Path) -> None:
    torch.manual_seed(404)
    cfg = AutoConfig.for_model(
        "deepseek_v4",
        vocab_size=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=128,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        kv_lora_rank=32,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=32,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        num_local_experts=4,
        num_experts_per_tok=3,
        mlp_layer_types=["moe", "moe"],
        scoring_func="sqrtsoftplus",
        swiglu_limit=2.5,
    )
    ref = AutoModelForCausalLM.from_config(cfg).eval()
    _init_weights(ref, std=0.14)

    toks = [7, 22, 61, 105]
    with torch.no_grad():
        ref_logits = ref(torch.tensor([toks], dtype=torch.long)).logits[:, -1, :].float().numpy().ravel()

    raw_sd: Dict[str, torch.Tensor] = {}
    bs = 64
    for k, v in ref.state_dict().items():
        rk = k[len("model.") :] if k.startswith("model.") else k
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
                    wf = wt.float().view(1, bs, 2, bs)
                    exp_s = (wf.abs().amax(dim=(1, 3)).clamp(min=1e-4) / 224.0).to(torch.float8_e8m0fnu)
                    q = (wf / exp_s.float()[:, None, :, None]).clamp(-448.0, 448.0).reshape(64, 128)
                    raw_sd[f"{stem}.{e}.{pn}.weight"] = q.to(torch.float8_e4m3fn)
                    raw_sd[f"{stem}.{e}.{pn}.scale"] = exp_s
        elif rk.endswith(".ffn.experts.down_proj"):
            stem = rk[: -len(".down_proj")]
            for e in range(4):
                wf = v[e].float().view(2, bs, 1, bs)
                exp_s = (wf.abs().amax(dim=(1, 3)).clamp(min=1e-4) / 224.0).to(torch.float8_e8m0fnu)
                q = (wf / exp_s.float()[:, None, :, None]).clamp(-448.0, 448.0).reshape(128, 64)
                raw_sd[f"{stem}.{e}.w2.weight"] = q.to(torch.float8_e4m3fn)
                raw_sd[f"{stem}.{e}.w2.scale"] = exp_s
        else:
            raw_sd[rk] = v.contiguous()

    ckpt_dir = tmp_path / "dsv4_raw_fp8"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg_d = cfg.to_dict()
    cfg_d["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3", "scale_fmt": "ue8m0", "weight_block_size": [64, 64]}
    (ckpt_dir / "config.json").write_text(json.dumps(cfg_d, indent=2))
    st_save_torch(raw_sd, str(ckpt_dir / "model.safetensors"))

    eng = create_paged_moe_engine(ckpt_dir, backend="cpu", slots_per_layer=3, miss_tail_tau=1.0)
    try:
        out_logits = eng.forward_logits(toks).float().cpu().numpy().ravel()  # type: ignore[union-attr]
        assert _pearson_r(ref_logits, out_logits) > 0.997
    finally:
        eng.close()

    eng_tau = create_paged_moe_engine(ckpt_dir, backend="cpu", slots_per_layer=3, miss_tail_tau=0.35, min_k=1)
    try:
        _ = eng_tau.forward_logits(toks)
        st = eng_tau.arena.get_stats(nominal_top_k=3)
        assert st["tail_misses_skipped"] > 0
    finally:
        eng_tau.close()
