#!/usr/bin/env python3
"""
Universal `.safetensors` Sparse MoE -> DeimosStream (`DEIMIDX2` Q4) Converter
(`src/deimos_stream/converter.py`)
=============================================================================
Converts HuggingFace or MLX `.safetensors` Sparse MoE checkpoints directly into a
page-aligned `DEIMIDX2` Q4 Merkle Store (`cas_experts_q4.bin`, `shared_core.bin`,
`core_manifest.json`, `embed_tokens.bin`, `lm_head_int8.bin`, `merkle_tree.idx`).

Supported MoE Families & Shard Formats:
  - `OLMoE`, `Qwen2MoE / Qwen3MoE`, `Mixtral`, `DeepSeek-MoE`
  - Arbitrary GQA (`num_key_value_heads <= num_attention_heads`), full/per-head/absent QK-Norm
  - Both 2D per-expert `.safetensors` (`mlp.experts.{e}.gate_proj.weight` or `w1/w2/w3`),
    3D stacked `switch_mlp` FP16/BF16, AND pre-quantized MLX 4-bit `switch_mlp`
  - Optional `shared_expert` (`Qwen2MoE` / `DeepSeek-MoE`) + tied or untied `lm_head`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np

from deimos_stream.merkle_builder import (
    build_deimidx2_file,
    compile_deimos_dylib,
    compute_q4_leaf_layout,
)


class ShardTensorLoader:
    """Lazy shard-cached MLX `.safetensors` tensor loader avoiding full-model RAM spikes."""

    def __init__(self, model_dir: Path):
        self.model_dir: Path = Path(model_dir).resolve()
        idx_file = self.model_dir / "model.safetensors.index.json"
        if idx_file.exists():
            raw_idx = json.loads(idx_file.read_text())
            self.weight_map: Dict[str, str] = dict(raw_idx["weight_map"])
        else:
            single = self.model_dir / "model.safetensors"
            if not single.exists():
                shards = sorted(self.model_dir.glob("*.safetensors"))
                if not shards:
                    raise FileNotFoundError(f"No .safetensors files found in {self.model_dir}")
                single = shards[0]
            loaded = mx.load(str(single))
            self.weight_map = {k: single.name for k in loaded.keys()}
        self._active_shard_name: Optional[str] = None
        self._active_shard_tensors: Dict[str, mx.array] = {}

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def total_source_bytes(self) -> int:
        seen = {self.model_dir / v for v in self.weight_map.values()}
        return sum(p.stat().st_size for p in seen if p.exists())

    def get_fp16_np(self, key: str) -> np.ndarray:
        arr = self.get_mlx(key)
        if arr.dtype != mx.float16:
            arr = arr.astype(mx.float16)
        mx.eval(arr)
        return np.asarray(arr, dtype=np.float16)

    def get_mlx(self, key: str) -> mx.array:
        sname = self.weight_map[key]
        if sname != self._active_shard_name:
            self._active_shard_tensors.clear()
            mx.clear_cache()
            self._active_shard_tensors = mx.load(str(self.model_dir / sname))
            self._active_shard_name = sname
        return self._active_shard_tensors[key]

    def get_dequant_or_fp16_np(
        self, prefix: str, group_size: int = 64, bits: int = 4
    ) -> np.ndarray:
        """Loads `{prefix}.weight` either directly (FP16/BF16) or dequantized if MLX-quantized."""
        if self.has(f"{prefix}.scales"):
            w = self.get_mlx(f"{prefix}.weight")
            s = self.get_mlx(f"{prefix}.scales").astype(mx.float16)
            b = (
                self.get_mlx(f"{prefix}.biases").astype(mx.float16)
                if self.has(f"{prefix}.biases")
                else None
            )
            deq = mx.dequantize(w, s, b, group_size=group_size, bits=bits).astype(mx.float16)
            mx.eval(deq)
            return np.asarray(deq, dtype=np.float16)
        return self.get_fp16_np(f"{prefix}.weight")


def normalize_moe_config(raw_cfg: Dict[str, object]) -> Dict[str, object]:
    """Standardizes HuggingFace / MLX MoE config keys (`OLMoE`, `Qwen2/3-MoE`, `Mixtral`, `DeepSeek`)."""
    t_cfg = raw_cfg.get("text_config", raw_cfg)
    assert isinstance(t_cfg, dict)

    num_layers = int(t_cfg["num_hidden_layers"])  # type: ignore[arg-type]
    hidden_size = int(t_cfg["hidden_size"])  # type: ignore[arg-type]
    num_heads = int(t_cfg["num_attention_heads"])  # type: ignore[arg-type]
    num_kv_heads = int(t_cfg.get("num_key_value_heads", num_heads))  # type: ignore[arg-type]
    head_dim = int(t_cfg.get("head_dim", hidden_size // num_heads))  # type: ignore[arg-type]

    num_experts = int(
        t_cfg.get("num_experts")
        or t_cfg.get("num_local_experts")
        or t_cfg.get("n_routed_experts")
        or 0
    )
    if num_experts <= 0:
        raise ValueError("Checkpoint is not a Sparse MoE model (`num_experts == 0`).")

    inter_size = int(
        t_cfg.get("moe_intermediate_size")
        or t_cfg.get("intermediate_size")
        or 0
    )
    top_k = int(t_cfg.get("num_experts_per_tok", 2))  # type: ignore[arg-type]
    vocab_size = int(t_cfg["vocab_size"])  # type: ignore[arg-type]
    rms_eps = float(t_cfg.get("rms_norm_eps", 1e-5))  # type: ignore[arg-type]
    rope_theta = float(t_cfg.get("rope_theta", 10000.0))  # type: ignore[arg-type]
    norm_topk = bool(t_cfg.get("norm_topk_prob", True))

    q_cfg = raw_cfg.get("quantization") or t_cfg.get("quantization")
    src_q_bits = int(q_cfg["bits"]) if isinstance(q_cfg, dict) and "bits" in q_cfg else 0
    src_q_group = int(q_cfg.get("group_size", 64)) if isinstance(q_cfg, dict) else 64

    out = dict(t_cfg)
    out.update(
        {
            "num_hidden_layers": num_layers,
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "num_experts": num_experts,
            "intermediate_size": inter_size,
            "num_experts_per_tok": top_k,
            "vocab_size": vocab_size,
            "rms_norm_eps": rms_eps,
            "rope_theta": rope_theta,
            "norm_topk_prob": norm_topk,
            "_src_quant_bits": src_q_bits,
            "_src_quant_group": src_q_group,
        }
    )
    return out


def convert_safetensors_moe_to_deimos_q4(
    source_model_dir: Path,
    output_store_dir: Path,
    group_size: int = 64,
) -> Path:
    """
    Converts any supported HuggingFace / MLX `.safetensors` Sparse MoE directory into a
    ready-to-run `DEIMIDX2` Q4 store (`output_store_dir`).
    """
    t0 = time.perf_counter()
    src_dir = Path(source_model_dir).expanduser().resolve()
    out_dir = Path(output_store_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    compile_deimos_dylib(force=False)

    raw_cfg = json.loads((src_dir / "config.json").read_text())
    cfg = normalize_moe_config(raw_cfg)
    src_q_bits = int(cfg.pop("_src_quant_bits"))  # type: ignore[arg-type]
    src_q_group = int(cfg.pop("_src_quant_group"))  # type: ignore[arg-type]

    num_layers = int(cfg["num_hidden_layers"])  # type: ignore[arg-type]
    num_experts = int(cfg["num_experts"])  # type: ignore[arg-type]
    hidden_size = int(cfg["hidden_size"])  # type: ignore[arg-type]
    inter_size = int(cfg["intermediate_size"])  # type: ignore[arg-type]
    vocab_size = int(cfg["vocab_size"])  # type: ignore[arg-type]

    for tok_f in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        if (src_dir / tok_f).exists():
            shutil.copy2(src_dir / tok_f, out_dir / tok_f)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    loader = ShardTensorLoader(src_dir)
    orig_bytes = loader.total_source_bytes()
    pfx = (
        "language_model.model"
        if loader.has("language_model.model.embed_tokens.weight")
        else "model"
    )

    emb_np = loader.get_dequant_or_fp16_np(
        f"{pfx}.embed_tokens", src_q_group, max(4, src_q_bits)
    )
    emb_bytes = np.ascontiguousarray(emb_np, dtype=np.float16).tobytes()
    (out_dir / "embed_tokens.bin").write_bytes(emb_bytes)
    emb_root = hashlib.sha256(emb_bytes).digest()

    lm_pfx = (
        "language_model.lm_head" if loader.has("language_model.lm_head.weight") else "lm_head"
    )
    if loader.has(f"{lm_pfx}.weight"):
        lm_np = loader.get_dequant_or_fp16_np(lm_pfx, src_q_group, max(4, src_q_bits))
    else:
        lm_np = emb_np
    del emb_np
    lm_f32 = lm_np.astype(np.float32)
    del lm_np
    max_abs = np.maximum(np.max(np.abs(lm_f32), axis=-1, keepdims=True), 1e-6)
    scales_f16 = (max_abs / 127.0).astype(np.float16)
    w_i8 = np.clip(np.round(lm_f32 / scales_f16.astype(np.float32)), -127, 127).astype(np.int8)
    del lm_f32
    head_bytes = w_i8.tobytes() + scales_f16.tobytes()
    (out_dir / "lm_head_int8.bin").write_bytes(head_bytes)
    head_root = hashlib.sha256(head_bytes).digest()
    del w_i8, scales_f16

    core_entries: Dict[str, Dict[str, object]] = {}
    core_hasher = hashlib.sha256()
    core_off = 0
    with open(out_dir / "shared_core.bin", "wb") as core_f:

        def _write_core(canonical_name: str, arr_fp16: np.ndarray) -> None:
            nonlocal core_off
            contig = np.ascontiguousarray(arr_fp16, dtype=np.float16)
            raw = contig.tobytes()
            core_entries[canonical_name] = {
                "offset": core_off,
                "nbytes": len(raw),
                "shape": list(contig.shape),
            }
            core_f.write(raw)
            core_hasher.update(raw)
            core_off += len(raw)

        for l_idx in range(num_layers):
            lp = f"{pfx}.layers.{l_idx}"
            _write_core(
                f"layers.{l_idx}.in_norm", loader.get_fp16_np(f"{lp}.input_layernorm.weight")
            )
            _write_core(
                f"layers.{l_idx}.post_norm",
                loader.get_fp16_np(f"{lp}.post_attention_layernorm.weight"),
            )
            for p_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                w_p = loader.get_dequant_or_fp16_np(
                    f"{lp}.self_attn.{p_name}", src_q_group, max(4, src_q_bits)
                )
                _write_core(f"layers.{l_idx}.{p_name}", w_p)
            for n_name in ("q_norm", "k_norm"):
                nk = f"{lp}.self_attn.{n_name}.weight"
                if loader.has(nk):
                    _write_core(f"layers.{l_idx}.{n_name}", loader.get_fp16_np(nk))

            rk_candidates = (f"{lp}.mlp.gate.weight", f"{lp}.block_sparse_moe.gate.weight")
            rk = next(k for k in rk_candidates if loader.has(k))
            _write_core(f"layers.{l_idx}.router", loader.get_fp16_np(rk))

            for sh_tag in ("shared_expert", "shared_experts"):
                if loader.has(f"{lp}.mlp.{sh_tag}.gate_proj.weight"):
                    for sp in ("gate_proj", "up_proj", "down_proj"):
                        sw = loader.get_dequant_or_fp16_np(
                            f"{lp}.mlp.{sh_tag}.{sp}", src_q_group, max(4, src_q_bits)
                        )
                        _write_core(f"layers.{l_idx}.shared_{sp}", sw)
                    if loader.has(f"{lp}.mlp.shared_expert_gate.weight"):
                        _write_core(
                            f"layers.{l_idx}.shared_gate",
                            loader.get_fp16_np(f"{lp}.mlp.shared_expert_gate.weight"),
                        )

        _write_core("final_norm", loader.get_fp16_np(f"{pfx}.norm.weight"))

    core_root = core_hasher.digest()
    (out_dir / "core_manifest.json").write_text(json.dumps({"tensors": core_entries}, indent=2))

    layout = compute_q4_leaf_layout(hidden_size, inter_size, group_size=group_size)
    stride_bytes = layout["stride_bytes"]
    payload_bytes = layout["payload_bytes"]
    pad_tail = b"\x00" * (stride_bytes - payload_bytes)
    leaf_hashes: List[bytes] = []

    with open(out_dir / "cas_experts_q4.bin", "wb") as cas_f:
        for l_idx in range(num_layers):
            lp = f"{pfx}.layers.{l_idx}"
            sw_pfx = f"{lp}.mlp.switch_mlp"

            if (
                loader.has(f"{sw_pfx}.gate_proj.scales")
                and src_q_bits == 4
                and src_q_group == group_size
            ):
                gw = loader.get_mlx(f"{sw_pfx}.gate_proj.weight")
                uw = loader.get_mlx(f"{sw_pfx}.up_proj.weight")
                gs = loader.get_mlx(f"{sw_pfx}.gate_proj.scales").astype(mx.float16)
                us = loader.get_mlx(f"{sw_pfx}.up_proj.scales").astype(mx.float16)
                gb = loader.get_mlx(f"{sw_pfx}.gate_proj.biases").astype(mx.float16)
                ub = loader.get_mlx(f"{sw_pfx}.up_proj.biases").astype(mx.float16)
                gu_qw = mx.concatenate([gw, uw], axis=1)
                gu_qs = mx.concatenate([gs, us], axis=1)
                gu_qb = mx.concatenate([gb, ub], axis=1)
                dn_qw = loader.get_mlx(f"{sw_pfx}.down_proj.weight")
                dn_qs = loader.get_mlx(f"{sw_pfx}.down_proj.scales").astype(mx.float16)
                dn_qb = loader.get_mlx(f"{sw_pfx}.down_proj.biases").astype(mx.float16)
                mx.eval(gu_qw, gu_qs, gu_qb, dn_qw, dn_qs, dn_qb)
            else:
                if loader.has(f"{sw_pfx}.gate_proj.weight"):
                    g3 = loader.get_dequant_or_fp16_np(
                        f"{sw_pfx}.gate_proj", src_q_group, max(4, src_q_bits)
                    )
                    u3 = loader.get_dequant_or_fp16_np(
                        f"{sw_pfx}.up_proj", src_q_group, max(4, src_q_bits)
                    )
                    d3 = loader.get_dequant_or_fp16_np(
                        f"{sw_pfx}.down_proj", src_q_group, max(4, src_q_bits)
                    )
                    gu_np = np.concatenate([g3, u3], axis=1)
                    dn_np = d3
                else:
                    gu_np = np.empty((num_experts, 2 * inter_size, hidden_size), dtype=np.float16)
                    dn_np = np.empty((num_experts, hidden_size, inter_size), dtype=np.float16)
                    is_mixtral = loader.has(f"{lp}.block_sparse_moe.experts.0.w1.weight")
                    for e_idx in range(num_experts):
                        if is_mixtral:
                            ep = f"{lp}.block_sparse_moe.experts.{e_idx}"
                            gw_e = loader.get_fp16_np(f"{ep}.w1.weight")
                            uw_e = loader.get_fp16_np(f"{ep}.w3.weight")
                            dw_e = loader.get_fp16_np(f"{ep}.w2.weight")
                        else:
                            ep = f"{lp}.mlp.experts.{e_idx}"
                            gw_e = loader.get_fp16_np(f"{ep}.gate_proj.weight")
                            uw_e = loader.get_fp16_np(f"{ep}.up_proj.weight")
                            dw_e = loader.get_fp16_np(f"{ep}.down_proj.weight")
                        gu_np[e_idx, :inter_size] = gw_e
                        gu_np[e_idx, inter_size:] = uw_e
                        dn_np[e_idx] = dw_e
                gu_qw, gu_qs, gu_qb = mx.quantize(mx.array(gu_np), group_size=group_size, bits=4)
                dn_qw, dn_qs, dn_qb = mx.quantize(mx.array(dn_np), group_size=group_size, bits=4)
                mx.eval(gu_qw, gu_qs, gu_qb, dn_qw, dn_qs, dn_qb)

            gu_qw_np, gu_qs_np, gu_qb_np = np.asarray(gu_qw), np.asarray(gu_qs), np.asarray(gu_qb)
            dn_qw_np, dn_qs_np, dn_qb_np = np.asarray(dn_qw), np.asarray(dn_qs), np.asarray(dn_qb)
            for e_idx in range(num_experts):
                raw_leaf = (
                    gu_qw_np[e_idx].tobytes()
                    + gu_qs_np[e_idx].tobytes()
                    + gu_qb_np[e_idx].tobytes()
                    + dn_qw_np[e_idx].tobytes()
                    + dn_qs_np[e_idx].tobytes()
                    + dn_qb_np[e_idx].tobytes()
                    + pad_tail
                )
                leaf_hashes.append(hashlib.sha256(raw_leaf).digest())
                cas_f.write(raw_leaf)
            mx.clear_cache()
            sys.stdout.write(
                f"[DeimosStream] Layer {l_idx + 1}/{num_layers} "
                f"({(l_idx + 1) * num_experts}/{num_layers * num_experts} Q4 Merkle leaves)\n"
            )
            sys.stdout.flush()

    model_root_hex, expert_root_hex = build_deimidx2_file(
        idx_path=out_dir / "merkle_tree.idx",
        num_layers=num_layers,
        num_experts=num_experts,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=inter_size,
        quant_bits=4,
        quant_group_size=group_size,
        expert_stride_bytes=stride_bytes,
        expert_payload_bytes=payload_bytes,
        leaf_hashes=leaf_hashes,
        embedding_root=emb_root,
        shared_core_root=core_root,
        output_head_root=head_root,
        orig_safetensors_bytes=orig_bytes,
    )
    (out_dir / "merkle_manifest.json").write_text(
        json.dumps(
            {
                "format": "DEIMIDX2",
                "version": 2,
                "source_model_dir": str(src_dir),
                "quant_bits": 4,
                "quant_group_size": group_size,
                "num_layers": num_layers,
                "num_experts_per_layer": num_experts,
                "total_experts": num_layers * num_experts,
                "expert_stride_bytes": stride_bytes,
                "expert_payload_bytes": payload_bytes,
                "model_merkle_root": model_root_hex,
                "expert_tree_root": expert_root_hex,
                "conversion_time_sec": round(time.perf_counter() - t0, 2),
            },
            indent=2,
        )
    )
    return out_dir
