#!/usr/bin/env python3
"""
End-to-End Multi-Family Sparse & Hybrid MoE Adapter Parity Suite (`tests/test_model_adapters.py`)
=================================================================================================
Verifies real conversion (`convert_hf_hybrid_moe_to_deimos`), cryptographic `DEIMIDX2` leaf
auditing (`inspect_store`), PyTorch paged execution (`DeimosHFHybridMoEEngine`), AND Apple Silicon
MLX execution (`DeimosMLXUniversalMoEEngine`) across heterogeneous 2025–2026 MoE families:
  1. `DeepSeek-V3` (`MLA` + `first_k_dense_replace=1` + `NoAux` sigmoid router + `FP8` block scales)
  2. `GLM-4.5` (`Glm4MoeForCausalLM` + `first_k_dense_replace=1` + additive `shared_experts`)
  3. `Llama-4` (`Llama4ForCausalLM` + interleaved `moe_layers` + transposed `[E, D, 2I]` 3D experts)
  4. Architecture validation & `deimos-stream models` CLI subcommand
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from deimos_stream.cli import inspect_store, main as cli_main
from deimos_stream.engine import HAS_MLX
from deimos_stream.factory import create_paged_moe_engine
from deimos_stream.hf_converter import convert_hf_hybrid_moe_to_deimos
from deimos_stream.model_adapters import validate_convertible_architecture
from deimos_stream.torch_arena import HAS_TORCH

if HAS_TORCH:
    from safetensors.torch import save_file as st_save_torch
    import torch
    from transformers import (
        AutoModelForCausalLM,
        DeepseekV3Config,
        Glm4MoeConfig,
        Llama4TextConfig,
    )


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    return float(np.corrcoef(af, bf)[0, 1])


def _save_split_experts_checkpoint(
    ckpt_dir: Path,
    cfg_dict: Dict[str, object],
    state_dict: Dict[str, "torch.Tensor"],
    num_experts: int,
    use_fp8_blocks: bool = False,
) -> None:
    """Writes a realistic HF checkpoint (splitting 3D `gate_up_proj` into 2D per-expert keys if needed)."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    flat_sd: Dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        if k.endswith(".mlp.experts.gate_up_proj"):
            base = k[: -len(".gate_up_proj")]
            for e in range(num_experts):
                g_w, u_w = v[e].chunk(2, dim=0)
                flat_sd[f"{base}.{e}.gate_proj.weight"] = g_w.contiguous()
                flat_sd[f"{base}.{e}.up_proj.weight"] = u_w.contiguous()
        elif k.endswith(".mlp.experts.down_proj"):
            base = k[: -len(".down_proj")]
            for e in range(num_experts):
                flat_sd[f"{base}.{e}.down_proj.weight"] = v[e].contiguous()
        else:
            flat_sd[k] = v.contiguous()

    if use_fp8_blocks and hasattr(torch, "float8_e4m3fn"):
        fp8_sd: Dict[str, torch.Tensor] = {}
        bs = 64
        for k, t in flat_sd.items():
            if k.endswith(".weight") and t.ndim == 2 and t.shape[0] % bs == 0 and t.shape[1] % bs == 0:
                m, n = t.shape
                tf = t.float().view(m // bs, bs, n // bs, bs)
                amax = tf.abs().amax(dim=(1, 3)).clamp(min=1e-5)
                scale_inv = (amax / 448.0).to(torch.float32)
                q = (tf / scale_inv[:, None, :, None]).clamp(-448.0, 448.0).reshape(m, n)
                fp8_sd[k] = q.to(torch.float8_e4m3fn)
                fp8_sd[f"{k}_scale_inv"] = scale_inv
            else:
                fp8_sd[k] = t
        # Inject dummy MTP extra layer & vision weight to verify converter filtering
        fp8_sd["model.layers.99.mlp.down_proj.weight"] = torch.zeros((64, 64), dtype=torch.float16)
        fp8_sd["model.visual.patch.weight"] = torch.zeros((16, 16), dtype=torch.float16)
        cfg_dict = dict(cfg_dict)
        cfg_dict["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [64, 64]}
        flat_sd = fp8_sd

    (ckpt_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    st_save_torch(flat_sd, str(ckpt_dir / "model.safetensors"))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch & transformers required")
def test_deepseek_v3_fp8_mla_mixed_layers_parity(tmp_path: Path) -> None:
    torch.manual_seed(7)
    cfg = DeepseekV3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=64,
        q_lora_rank=64,
        qk_rope_head_dim=16,
        v_head_dim=16,
        qk_nope_head_dim=16,
        n_routed_experts=6,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
        routed_scaling_factor=1.5,
    )
    ref = AutoModelForCausalLM.from_config(cfg).eval()
    with torch.no_grad():
        for p in ref.parameters():
            if p.is_floating_point():
                p.normal_(mean=0.0, std=0.04)
        # Ensure non-zero router correction bias
        ref.model.layers[1].mlp.gate.e_score_correction_bias.copy_(torch.linspace(-0.05, 0.05, 6))

    toks = [3, 19, 42, 77, 105]
    with torch.no_grad():
        ref_logits = ref(torch.tensor([toks], dtype=torch.long)).logits[:, -1, :].float().numpy().ravel()

    ckpt_dir = tmp_path / "dsv3_fp8_src"
    store_dir = tmp_path / "dsv3_deimos_q4"
    _save_split_experts_checkpoint(ckpt_dir, cfg.to_dict(), ref.state_dict(), num_experts=6, use_fp8_blocks=True)

    convert_hf_hybrid_moe_to_deimos(str(ckpt_dir), store_dir, group_size=64)
    info = inspect_store(store_dir, verify_all_leaves=True)
    assert info["byte_proofs_verified"] == 6
    mf = json.loads((store_dir / "merkle_manifest.json").read_text())
    assert mf["moe_layer_ids"] == [1]
    assert mf["num_layers"] == 1
    assert mf["num_total_layers"] == 2
    assert "quantization_config" not in json.loads((store_dir / "config.json").read_text())

    eng_cpu = create_paged_moe_engine(store_dir, backend="cpu", slots_per_layer=2, miss_tail_tau=1.0)
    try:
        cpu_logits = eng_cpu.forward_logits(toks).float().cpu().numpy().ravel()  # type: ignore[union-attr]
        assert _pearson_r(ref_logits, cpu_logits) > 0.995
        bm_cpu = eng_cpu.benchmark_token_ids(toks, max_new_tokens=3)
        assert len(bm_cpu["generated_ids"]) == 4
    finally:
        eng_cpu.close()

    if HAS_MLX:
        eng_mlx = create_paged_moe_engine(store_dir, backend="mlx", slots_per_layer=2, miss_tail_tau=1.0)
        try:
            mlx_logits = np.asarray(eng_mlx.forward_logits(toks), dtype=np.float32).ravel()
            assert _pearson_r(ref_logits, mlx_logits) > 0.995
            bm_mlx = eng_mlx.benchmark_token_ids(toks, max_new_tokens=3)
            assert len(bm_mlx["generated_ids"]) == 4
        finally:
            eng_mlx.close()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch & transformers required")
def test_glm4_moe_mixed_layers_parity(tmp_path: Path) -> None:
    torch.manual_seed(11)
    cfg = Glm4MoeConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        n_routed_experts=6,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
    )
    ref = AutoModelForCausalLM.from_config(cfg).eval()
    with torch.no_grad():
        for p in ref.parameters():
            if p.is_floating_point():
                p.normal_(mean=0.0, std=0.04)

    toks = [5, 14, 28, 63]
    with torch.no_grad():
        ref_logits = ref(torch.tensor([toks], dtype=torch.long)).logits[:, -1, :].float().numpy().ravel()

    ckpt_dir = tmp_path / "glm4_src"
    store_dir = tmp_path / "glm4_deimos_q4"
    _save_split_experts_checkpoint(ckpt_dir, cfg.to_dict(), ref.state_dict(), num_experts=6, use_fp8_blocks=False)

    # Exercise CLI auto-routing (`should_use_hf_hybrid_converter` without `--hf-hybrid`)
    rc = cli_main(["convert", "--model-dir", str(ckpt_dir), "--output-dir", str(store_dir)])
    assert rc == 0

    eng_cpu = create_paged_moe_engine(store_dir, backend="cpu", slots_per_layer=2, miss_tail_tau=1.0)
    try:
        cpu_logits = eng_cpu.forward_logits(toks).float().cpu().numpy().ravel()  # type: ignore[union-attr]
        assert _pearson_r(ref_logits, cpu_logits) > 0.995
        bm_cpu = eng_cpu.benchmark_token_ids(toks, max_new_tokens=3)
        assert len(bm_cpu["generated_ids"]) == 4
    finally:
        eng_cpu.close()

    if HAS_MLX:
        eng_mlx = create_paged_moe_engine(store_dir, backend="mlx", slots_per_layer=2, miss_tail_tau=1.0)
        try:
            mlx_logits = np.asarray(eng_mlx.forward_logits(toks), dtype=np.float32).ravel()
            assert _pearson_r(ref_logits, mlx_logits) > 0.995
            bm_mlx = eng_mlx.benchmark_token_ids(toks, max_new_tokens=3)
            assert len(bm_mlx["generated_ids"]) == 4
        finally:
            eng_mlx.close()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch & transformers required")
def test_llama4_interleaved_transposed_3d_experts_parity(tmp_path: Path) -> None:
    torch.manual_seed(23)
    cfg = Llama4TextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=64,
        intermediate_size_mlp=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_local_experts=4,
        num_experts_per_tok=1,
        interleave_moe_layer_step=2,
    )
    ref = AutoModelForCausalLM.from_config(cfg).eval()
    with torch.no_grad():
        for p in ref.parameters():
            if p.is_floating_point():
                p.normal_(mean=0.0, std=0.04)

    toks = [9, 21, 55, 88]
    with torch.no_grad():
        ref_logits = ref(torch.tensor([toks], dtype=torch.long)).logits[:, -1, :].float().numpy().ravel()

    ckpt_dir = tmp_path / "llama4_src"
    store_dir = tmp_path / "llama4_deimos_q4"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    st_save_torch({k: v.contiguous() for k, v in ref.state_dict().items()}, str(ckpt_dir / "model.safetensors"))

    convert_hf_hybrid_moe_to_deimos(str(ckpt_dir), store_dir, group_size=64)
    mf = json.loads((store_dir / "merkle_manifest.json").read_text())
    assert mf["moe_layer_ids"] == [1, 3]
    assert mf["moe_containers"] == ["feed_forward", "feed_forward"]

    eng_cpu = create_paged_moe_engine(store_dir, backend="cpu", slots_per_layer=2, miss_tail_tau=1.0)
    try:
        cpu_logits = eng_cpu.forward_logits(toks).float().cpu().numpy().ravel()  # type: ignore[union-attr]
        assert _pearson_r(ref_logits, cpu_logits) > 0.995
        bm_cpu = eng_cpu.benchmark_token_ids(toks, max_new_tokens=3)
        assert len(bm_cpu["generated_ids"]) == 4
    finally:
        eng_cpu.close()

    if HAS_MLX:
        eng_mlx = create_paged_moe_engine(store_dir, backend="mlx", slots_per_layer=2, miss_tail_tau=1.0)
        try:
            mlx_logits = np.asarray(eng_mlx.forward_logits(toks), dtype=np.float32).ravel()
            assert _pearson_r(ref_logits, mlx_logits) > 0.995
            bm_mlx = eng_mlx.benchmark_token_ids(toks, max_new_tokens=3)
            assert len(bm_mlx["generated_ids"]) == 4
        finally:
            eng_mlx.close()


def test_roadmap_validation_and_models_cli(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError, match="DeepSeek-V4.1-Flash"):
        validate_convertible_architecture(
            {"model_type": "deepseek_v41", "text_config": {"model_type": "deepseek_v41_text", "engram_layer_ids": [1, 14]}}
        )
    with pytest.raises(ValueError, match="gpt_oss"):
        validate_convertible_architecture({"model_type": "gpt_oss"})
    with pytest.raises(ValueError, match="inkling2_loop"):
        validate_convertible_architecture({"model_type": "inkling2_loop"})

    rc = cli_main(["models"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["supported"]) >= 8
    assert any("DeepSeek-V4" in f["family"] for f in out["supported"])
    assert any("GLM-5.3-Flash" in f["family"] for f in out["supported"])
    assert any("Qwen3.8" in f["family"] for f in out["supported"])
    assert any("Kimi-K3" in f["family"] for f in out["supported"])
    assert any("MiniMax-M3.1" in f["family"] for f in out["supported"])
    assert any("DeepSeek-V4.1-Flash" in r["family"] for r in out["roadmap"])
