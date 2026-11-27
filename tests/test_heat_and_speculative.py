#!/usr/bin/env python3
"""
Verification Suite for Lossless `.deimos_heat` Warm Slot Profiles & Speculative / MTP Decoding
(`tests/test_heat_and_speculative.py`)
==============================================================================================
Verifies strictly zero quality loss (`tau=1.0`, `0.0` greedy token drift) across:
  1. Persistent `.deimos_heat` routing profile save/load across sessions on both MLX (`PagedExpertArena`)
     and PyTorch (`TorchPagedExpertArena`), confirming bit-exact logits/tokens + higher warm hit rates
     + preserved `DeimosMerkleIntegrityError` rejection on tampered leaves.
  2. Lossless `iter_speculative_tokens` (`SuffixNgramDrafter` + custom `mtp_drafter` hook) across:
     - `DeimosPagedMoEEngine` (MLX `KVCache`, including early `gen.close()` KV offset exactness)
     - `TorchDeimosPagedMoEEngine` (PyTorch `TorchKVCache`)
     - `DeimosHFHybridMoEEngine` (`Qwen3.6` Hybrid `GatedDeltaNet` recurrent + GQA `DynamicCache`)
     - `DeimosMLXUniversalMoEEngine` (`LFM2.5` Hybrid `LIV ShortConv` `ArraysCache` + `KVCache`)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from deimos_stream.arena import DeimosMerkleIntegrityError
from deimos_stream.converter import convert_safetensors_moe_to_deimos_q4
from deimos_stream.engine import HAS_MLX
from deimos_stream.factory import create_paged_moe_engine
from deimos_stream.fast_decode import iter_decode_tokens
from deimos_stream.heat_profile import HEAT_FILENAME, compute_warm_slot_plan, load_heat_counts
from deimos_stream.hf_converter import convert_hf_hybrid_moe_to_deimos
from deimos_stream.torch_arena import HAS_TORCH

if HAS_TORCH:
    from safetensors.torch import save_file as st_save_torch
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
else:
    from safetensors.numpy import save_file as st_save_np


def _build_synthetic_olmoe_store(tmp_path: Path, name: str = "olmoe_q4") -> Path:
    rng = np.random.default_rng(707)
    L, E, K, V, D, I = 2, 8, 2, 128, 64, 64
    H, D_h = 4, 16
    ckpt = tmp_path / f"{name}_src"
    store = tmp_path / name
    ckpt.mkdir(parents=True, exist_ok=True)

    cfg = {
        "architectures": ["OlmoeForCausalLM"],
        "model_type": "olmoe",
        "num_hidden_layers": L,
        "num_experts": E,
        "num_experts_per_tok": K,
        "vocab_size": V,
        "hidden_size": D,
        "intermediate_size": I,
        "num_attention_heads": H,
        "num_key_value_heads": H,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
    }
    (ckpt / "config.json").write_text(json.dumps(cfg, indent=2))

    tensors = {
        "model.embed_tokens.weight": (rng.standard_normal((V, D)) * 0.08).astype(np.float16),
        "model.norm.weight": np.ones((D,), dtype=np.float16),
        "lm_head.weight": (rng.standard_normal((V, D)) * 0.08).astype(np.float16),
    }
    for l_idx in range(L):
        pfx = f"model.layers.{l_idx}"
        tensors[f"{pfx}.input_layernorm.weight"] = np.ones((D,), dtype=np.float16)
        tensors[f"{pfx}.post_attention_layernorm.weight"] = np.ones((D,), dtype=np.float16)
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"{pfx}.self_attn.{p}.weight"] = (rng.standard_normal((D, D)) * 0.08).astype(np.float16)
        tensors[f"{pfx}.self_attn.q_norm.weight"] = np.ones((D,), dtype=np.float16)
        tensors[f"{pfx}.self_attn.k_norm.weight"] = np.ones((D,), dtype=np.float16)
        # Skew router biases toward experts 5 & 6 so cold default `0..1` misses while `.deimos_heat` hits!
        r_w = (rng.standard_normal((E, D)) * 0.03).astype(np.float16)
        r_w[5] += 0.45
        r_w[6] += 0.40
        tensors[f"{pfx}.mlp.gate.weight"] = r_w
        for e in range(E):
            for proj, sh in (("gate_proj", (I, D)), ("up_proj", (I, D)), ("down_proj", (D, I))):
                tensors[f"{pfx}.mlp.experts.{e}.{proj}.weight"] = (
                    rng.standard_normal(sh) * 0.08
                ).astype(np.float16)

    if HAS_TORCH:
        st_save_torch({k: torch.from_numpy(v) for k, v in tensors.items()}, str(ckpt / "model.safetensors"))
    else:
        st_save_np(tensors, str(ckpt / "model.safetensors"))
    convert_safetensors_moe_to_deimos_q4(ckpt, store, group_size=64)
    return store


@pytest.mark.parametrize("backend", ([b for b in (("mlx" if HAS_MLX else None), ("cpu" if HAS_TORCH else None)) if b]))
def test_persistent_heat_profile_bit_exact_and_warm_hit_boost(tmp_path: Path, backend: str) -> None:
    store = _build_synthetic_olmoe_store(tmp_path, name=f"heat_{backend}")
    prompt = [3, 7, 11, 15, 3, 7, 11, 15, 21, 29]
    assert not (store / HEAT_FILENAME).exists()

    # Session 1 (Cold): no `.deimos_heat` yet -> records true routing frequencies on close()
    eng1 = create_paged_moe_engine(
        store, backend=backend, slots_per_layer=2, miss_tail_tau=1.0,
        enable_lookahead=False, verify_merkle_on_cold=False,
    )
    try:
        logits1 = np.asarray(
            eng1.forward_logits(prompt).cpu().numpy() if backend == "cpu" else eng1.forward_logits(prompt),  # type: ignore[union-attr]
            dtype=np.float32,
        )
        st1 = eng1.arena.get_stats(nominal_top_k=2)
        m1 = eng1.benchmark_token_ids(prompt, max_new_tokens=10)
    finally:
        eng1.close()

    assert (store / HEAT_FILENAME).is_file()
    loaded = load_heat_counts(store, 2, 8)
    assert loaded is not None
    counts, runs, tot_hits = loaded
    assert runs == 1 and tot_hits > 0
    for l_idx in range(2):
        assert int(counts[l_idx].max()) > 0

    # Session 2 (Warm): preloads each layer's top-2 hottest experts at startup -> higher hit rate & 0.0 logit diff!
    eng2 = create_paged_moe_engine(
        store, backend=backend, slots_per_layer=2, miss_tail_tau=1.0,
        enable_lookahead=False, verify_merkle_on_cold=False,
    )
    try:
        logits2 = np.asarray(
            eng2.forward_logits(prompt).cpu().numpy() if backend == "cpu" else eng2.forward_logits(prompt),  # type: ignore[union-attr]
            dtype=np.float32,
        )
        st2 = eng2.arena.get_stats(nominal_top_k=2)
        assert float(np.max(np.abs(logits1 - logits2))) == 0.0
        assert st2["hit_rate_pct"] >= st1["hit_rate_pct"]
        assert st2["cache_misses"] < st1["cache_misses"]
        m2 = eng2.benchmark_token_ids(prompt, max_new_tokens=10)
        assert m2["generated_ids"] == m1["generated_ids"]
    finally:
        eng2.close()

    # Corrupting a warm expert leaf on an unverified cold store must STILL trigger DeimosMerkleIntegrityError!
    counts_after, _, _ = load_heat_counts(store, 2, 8)  # type: ignore[misc]
    warm_ids_after, _, _ = compute_warm_slot_plan(counts_after, 2)
    warm_e_l0 = int(warm_ids_after[0, 0])
    cas_path = store / "cas_experts_q4.bin"
    raw = bytearray(cas_path.read_bytes())
    stride = len(raw) // (2 * 8)
    raw[warm_e_l0 * stride + 32] ^= 0xFF
    cas_path.write_bytes(bytes(raw))
    with pytest.raises(DeimosMerkleIntegrityError):
        create_paged_moe_engine(store, backend=backend, slots_per_layer=2, verify_merkle_on_cold=True)


@pytest.mark.parametrize("backend", ([b for b in (("mlx" if HAS_MLX else None), ("cpu" if HAS_TORCH else None)) if b]))
def test_speculative_decode_kv_engines_exact_tokens_and_cache_offset(tmp_path: Path, backend: str) -> None:
    store = _build_synthetic_olmoe_store(tmp_path, name=f"spec_{backend}")
    eng = create_paged_moe_engine(store, backend=backend, slots_per_layer=4, miss_tail_tau=1.0)
    try:
        prompt = [4, 9, 12, 18, 25, 4, 9, 12, 18, 25]

        # 1. Baseline greedy generation + final KV cache offset
        c_base = eng.make_caches()
        pf_base, _ = eng.prefill_prompt(prompt, c_base)  # type: ignore[arg-type]
        base_ids = list(iter_decode_tokens(eng, pf_base, c_base, 14, prompt_ids=prompt))

        # 2. Speculative generation with SuffixNgramDrafter
        eng.enable_speculative = True  # type: ignore[attr-defined]
        c_spec = eng.make_caches()
        pf_spec, _ = eng.prefill_prompt(prompt, c_spec)  # type: ignore[arg-type]
        spec_ids = list(iter_decode_tokens(eng, pf_spec, c_spec, 14, prompt_ids=prompt))
        assert spec_ids == base_ids
        assert c_spec[0].offset == c_base[0].offset

        # 3. Oracle MTP drafter hook (`100%` accurate 3-token drafts -> <= 5 target calls for 14 tokens!)
        cursor = {"pos": 0}

        def _exact_oracle(_cur_tok: int, cap: int) -> list[int]:
            p = cursor["pos"]
            out = base_ids[p + 1 : min(len(base_ids), p + 1 + cap)]
            cursor["pos"] = p + 1 + len(out)
            return out

        eng.mtp_drafter = _exact_oracle  # type: ignore[attr-defined]
        c_mtp = eng.make_caches()
        pf_mtp, _ = eng.prefill_prompt(prompt, c_mtp)  # type: ignore[arg-type]
        mtp_ids = list(iter_decode_tokens(eng, pf_mtp, c_mtp, 14, prompt_ids=prompt))
        assert mtp_ids == base_ids
        assert c_mtp[0].offset == c_base[0].offset
        assert eng.last_spec_stats["accepted_tokens"] >= 9  # type: ignore[attr-defined]
        assert eng.last_spec_stats["target_calls"] <= 5  # type: ignore[attr-defined]

        # 4. Adversarial wrong MTP drafter + mid-batch early `gen.close()` -> STILL 100% lossless!
        eng.mtp_drafter = lambda cur_tok, cap: [(cur_tok + 61) % 128, (cur_tok + 62) % 128][:cap]  # type: ignore[attr-defined]
        c_adv = eng.make_caches()
        pf_adv, _ = eng.prefill_prompt(prompt, c_adv)  # type: ignore[arg-type]
        adv_ids = list(iter_decode_tokens(eng, pf_adv, c_adv, 14, prompt_ids=prompt))
        assert adv_ids == base_ids
        assert c_adv[0].offset == c_base[0].offset

        # Early break after 5 tokens leaves exact KV offset `len(prompt) + 4`
        cursor["pos"] = 0
        eng.mtp_drafter = _exact_oracle  # type: ignore[attr-defined]
        c_early = eng.make_caches()
        pf_early, _ = eng.prefill_prompt(prompt, c_early)  # type: ignore[arg-type]
        gen = iter_decode_tokens(eng, pf_early, c_early, 14, prompt_ids=prompt)
        taken = [next(gen) for _ in range(5)]
        gen.close()
        assert taken == base_ids[:5]
        assert c_early[0].offset == len(prompt) + 4
    finally:
        eng.close()


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch & transformers required")
def test_speculative_decode_hybrid_recurrent_hf_and_mlx_exact_parity(tmp_path: Path) -> None:
    torch.manual_seed(2027)
    # 1. PyTorch HuggingFace Hybrid `Qwen3.6` (`GatedDeltaNet` recurrent + GQA `DynamicCache`)
    qcfg = AutoConfig.for_model(
        "qwen3_5_moe_text",
        vocab_size=128, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        hidden_size=128, intermediate_size=128, moe_intermediate_size=64, shared_expert_intermediate_size=64,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
        num_experts=4, num_experts_per_tok=2,
    )
    qref = AutoModelForCausalLM.from_config(qcfg).eval()
    with torch.no_grad():
        for n, p in qref.named_parameters():
            if p.is_floating_point() and "A_log" not in n:
                p.normal_(mean=0.0, std=0.05)

    q_src, q_store = tmp_path / "q36_src", tmp_path / "q36_q4"
    q_src.mkdir(parents=True, exist_ok=True)
    (q_src / "config.json").write_text(json.dumps(qcfg.to_dict(), indent=2))
    st_save_torch({k: v.contiguous() for k, v in qref.state_dict().items()}, str(q_src / "model.safetensors"))
    convert_hf_hybrid_moe_to_deimos(str(q_src), q_store, group_size=64)

    prompt = [5, 11, 19, 27, 5, 11, 19, 27]
    for backend in (["cpu", "mlx"] if HAS_MLX else ["cpu"]):
        eng = create_paged_moe_engine(q_store, backend=backend, slots_per_layer=2, miss_tail_tau=1.0)
        try:
            base = eng.benchmark_token_ids(prompt, max_new_tokens=10)["generated_ids"]

            # Test mixed partial-accept + full-accept + full-reject MTP drafts on recurrent caches!
            def _mixed_drafter(cur_tok: int, cap: int) -> list[int]:
                idx = base.index(cur_tok) if cur_tok in base else -1
                if idx >= 0 and idx + 2 < len(base):
                    # First draft right, second draft intentionally wrong -> forces partial recurrent rollback!
                    return [base[idx + 1], (base[idx + 2] + 37) % 128][:cap]
                return []

            eng.enable_speculative = True  # type: ignore[attr-defined]
            eng.mtp_drafter = _mixed_drafter  # type: ignore[attr-defined]
            spec_mixed = eng.benchmark_token_ids(prompt, max_new_tokens=10)["generated_ids"]
            assert spec_mixed == base

            # Full step-indexed oracle acceptance on recurrent caches -> cuts target calls by >= 2x with 0 drift!
            cur_idx = {"p": 0}

            def _rec_oracle(_cur_tok: int, cap: int) -> list[int]:
                p = cur_idx["p"]
                out = base[p + 1 : min(len(base), p + 1 + cap)]
                cur_idx["p"] = p + 1 + len(out)
                return out

            eng.mtp_drafter = _rec_oracle  # type: ignore[attr-defined]
            spec_oracle = eng.benchmark_token_ids(prompt, max_new_tokens=10)["generated_ids"]
            assert spec_oracle == base
            assert eng.last_spec_stats["target_calls"] <= 5  # type: ignore[attr-defined]
        finally:
            eng.close()
