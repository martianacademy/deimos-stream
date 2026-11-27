#!/usr/bin/env python3
"""
DeimosStream HuggingFace Universal Sparse & Hybrid-MoE -> `DEIMIDX2` Q4 Converter
(`src/deimos_stream/hf_converter.py`)
=================================================================================
Converts modern 2025–2026 HuggingFace Sparse, Hybrid & Multimodal MoE checkpoints
(`Qwen3.6 / 3.5 / Next`, `Qwen2/3-MoE`, `DeepSeek-V2 / V3 / R1 / V3.2`, `Kimi-K2 / Moonlight`,
`GLM-4.5`, `Llama-4 Scout / Maverick`, `Mixtral`, `OLMoE`, `MiniMax-M2`, `ERNIE-4.5`, `Jamba`) into:
  - `cas_experts_q4.bin` + `merkle_tree.idx` (`DEIMIDX2` 16 KB page-aligned Q4 Merkle leaves)
  - `hf_backbone.safetensors` (FP16 non-expert text layers + dense `first_k_dense_replace` layers)
  - `embed_tokens.bin` (zero-VRAM `mmap` FP16 row table)
  - `lm_head_int8.bin` (tiled INT8 output head saving ~50% head VRAM + ~840 MiB temp spikes)

Automatically dequantizes block-scaled `FP8` (`weight_scale_inv`) shards, handles mixed
dense + MoE layer schedules (`moe_layer_ids`), transposes `Llama-4` `[E, D, 2I]` experts,
and deletes downloaded shards via reference counting (`delete_downloaded_shards=True`).
"""

from __future__ import annotations

from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from safetensors import safe_open

from deimos_stream.merkle_builder import (
    build_deimidx2_file,
    compile_deimos_dylib,
    compute_q4_leaf_layout,
)
from deimos_stream.model_adapters import (
    canonicalize_hf_key,
    detect_lm_prefix,
    discover_hf_moe_layers,
    extract_layer_experts_gu_dn,
    is_ignored_checkpoint_key,
    pop_tensor_dequant,
    validate_convertible_architecture,
)
from deimos_stream.torch_arena import HAS_TORCH

if HAS_TORCH:
    from safetensors.torch import save_file as st_save_torch
    import torch
else:
    torch = None  # type: ignore[assignment]


def quantize_torch_to_q4_bytes(
    weight_t: "torch.Tensor", group_size: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU-accelerated 4-bit affine group-64 quantizer producing `(qw_u32, scales_f16, biases_f16)`."""
    orig_shape = tuple(weight_t.shape)
    w_f32 = weight_t.to(torch.float32)
    grouped = w_f32.reshape(*orig_shape[:-1], orig_shape[-1] // group_size, group_size)

    w_min = grouped.amin(dim=-1)
    w_max = grouped.amax(dim=-1)
    scales_f16 = ((w_max - w_min) / 15.0).to(torch.float16)
    biases_f16 = w_min.to(torch.float16)

    s_f32 = scales_f16.to(torch.float32).unsqueeze(-1)
    b_f32 = biases_f16.to(torch.float32).unsqueeze(-1)
    nz = s_f32.abs() > 1e-12
    safe_s = torch.where(nz, s_f32, torch.ones_like(s_f32))
    q = torch.where(nz, torch.round((grouped - b_f32) / safe_s), torch.zeros_like(grouped))
    q_u8 = q.clamp(0.0, 15.0).to(torch.uint8).reshape(*orig_shape)

    packed_u8 = (q_u8[..., 0::2] | (q_u8[..., 1::2] << 4)).contiguous().cpu().numpy()
    qw_u32 = packed_u8.view(np.uint32).reshape(*orig_shape[:-1], orig_shape[-1] // 8)
    return (
        qw_u32,
        scales_f16.contiguous().cpu().numpy(),
        biases_f16.contiguous().cpu().numpy(),
    )


class _RefCountedShardFetcher:
    """Downloads `.safetensors` shards on demand and deletes them once refcount reaches 0."""

    def __init__(
        self,
        repo_or_dir: str,
        cache_dir: Path,
        needed_keys: Set[str],
        weight_map: Dict[str, str],
        delete_shards: bool,
    ):
        p = Path(repo_or_dir).expanduser()
        self.local_dir: Optional[Path] = p.resolve() if p.exists() else None
        self.repo_id: str = repo_or_dir
        self.cache_dir: Path = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.weight_map = weight_map
        self.delete_shards = bool(delete_shards and self.local_dir is None)
        self.refcounts: Counter[str] = Counter(weight_map[k] for k in needed_keys)
        self._shard_paths: Dict[str, Path] = {}

    def _ensure_shard(self, shard_name: str) -> Path:
        if self.local_dir is not None:
            return self.local_dir / shard_name
        if shard_name not in self._shard_paths or not self._shard_paths[shard_name].exists():
            from huggingface_hub import hf_hub_download

            dl = hf_hub_download(
                repo_id=self.repo_id, filename=shard_name, local_dir=str(self.cache_dir)
            )
            self._shard_paths[shard_name] = Path(dl)
        return self._shard_paths[shard_name]

    def pop_tensor(self, key: str, device: str = "cpu") -> "torch.Tensor":
        shard_name = self.weight_map[key]
        spath = self._ensure_shard(shard_name)
        with safe_open(str(spath), framework="pt", device=device) as sf:
            t = sf.get_tensor(key)
        self.refcounts[shard_name] -= 1
        if self.delete_shards and self.refcounts[shard_name] <= 0:
            try:
                spath.unlink()
            except OSError:
                pass
            self._shard_paths.pop(shard_name, None)
        return t


def _fetch_metadata_files(
    repo_or_dir: str, out_dir: Path, tmp_dir: Path
) -> Tuple[Dict[str, object], Dict[str, str], int]:
    p = Path(repo_or_dir).expanduser()
    meta_files = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
    )
    if p.exists():
        src_dir = p.resolve()
        for fn in meta_files:
            if (src_dir / fn).exists():
                shutil.copy2(src_dir / fn, out_dir / fn)
        raw_cfg = json.loads((src_dir / "config.json").read_text())
        idx_p = src_dir / "model.safetensors.index.json"
        if idx_p.exists():
            idx_d = json.loads(idx_p.read_text())
            wm = dict(idx_d["weight_map"])
            tot_b = sum((src_dir / s).stat().st_size for s in set(wm.values()) if (src_dir / s).exists())
            return raw_cfg, wm, tot_b
        shards = sorted(src_dir.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No .safetensors files found in {src_dir}")
        wm = {}
        tot_b = 0
        for sp in shards:
            tot_b += sp.stat().st_size
            with safe_open(str(sp), framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    wm[k] = sp.name
        return raw_cfg, wm, tot_b

    from huggingface_hub import hf_hub_download

    tmp_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(hf_hub_download(repo_id=repo_or_dir, filename="config.json", local_dir=str(tmp_dir)))
    shutil.copy2(cfg_path, out_dir / "config.json")
    for fn in meta_files[1:]:
        try:
            fp = Path(hf_hub_download(repo_id=repo_or_dir, filename=fn, local_dir=str(tmp_dir)))
            shutil.copy2(fp, out_dir / fn)
        except Exception:
            pass
    raw_cfg = json.loads(cfg_path.read_text())
    try:
        idx_p = Path(
            hf_hub_download(
                repo_id=repo_or_dir, filename="model.safetensors.index.json", local_dir=str(tmp_dir)
            )
        )
        idx_d = json.loads(idx_p.read_text())
        tot_b = int(idx_d.get("metadata", {}).get("total_size", 0))
        return raw_cfg, dict(idx_d["weight_map"]), tot_b
    except Exception:
        sp = Path(hf_hub_download(repo_id=repo_or_dir, filename="model.safetensors", local_dir=str(tmp_dir)))
        with safe_open(str(sp), framework="pt", device="cpu") as sf:
            wm = {k: "model.safetensors" for k in sf.keys()}
        return raw_cfg, wm, sp.stat().st_size


def _cast_backbone_tensor(key: str, t_cpu: "torch.Tensor") -> "torch.Tensor":
    """Casts BF16 backbone weights to FP16 while preserving genuine FP32 buffers/norms/biases."""
    del key
    if t_cpu.dtype == torch.bfloat16:
        return t_cpu.to(torch.float16).contiguous()
    return t_cpu.contiguous()


def convert_hf_hybrid_moe_to_deimos(
    repo_id_or_dir: str,
    output_store_dir: Path,
    group_size: int = 64,
    delete_downloaded_shards: bool = True,
) -> Path:
    """Converts any supported HuggingFace Sparse/Hybrid/Multimodal MoE into a `DEIMIDX2` Q4 store."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for convert_hf_hybrid_moe_to_deimos.")
    t0 = time.perf_counter()
    out_dir = Path(output_store_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dl = out_dir / "_dl_cache"
    compile_deimos_dylib(force=False)

    raw_cfg, weight_map, orig_bytes = _fetch_metadata_files(repo_id_or_dir, out_dir, tmp_dl)
    validate_convertible_architecture(raw_cfg)
    t_cfg = dict(raw_cfg.get("text_config", raw_cfg))  # type: ignore[arg-type]

    num_layers = int(t_cfg["num_hidden_layers"])
    hidden_size = int(t_cfg["hidden_size"])
    inter_size = int(
        t_cfg.get("moe_intermediate_size") or t_cfg.get("ffn_hidden_size") or t_cfg["intermediate_size"]
    )
    num_experts = int(
        t_cfg.get("num_experts") or t_cfg.get("num_local_experts") or t_cfg["n_routed_experts"]
    )
    vocab_size = int(t_cfg["vocab_size"])
    q_dev = "cuda" if torch.cuda.is_available() else "cpu"

    active_keys = {k for k in weight_map if not is_ignored_checkpoint_key(k, num_layers)}
    lp_root = detect_lm_prefix(active_keys)
    moe_specs = discover_hf_moe_layers(num_layers, lp_root, active_keys)
    if not moe_specs:
        raise ValueError(f"No routed MoE expert tensors discovered in {repo_id_or_dir!r}.")
    moe_map: Dict[int, str] = dict(moe_specs)
    moe_layer_ids = [l_i for l_i, _ in moe_specs]
    moe_containers = [attr for _, attr in moe_specs]
    num_moe_layers = len(moe_layer_ids)

    fetcher = _RefCountedShardFetcher(
        repo_id_or_dir, tmp_dl, active_keys, weight_map, delete_shards=delete_downloaded_shards
    )
    used_special: Set[str] = set()

    emb_cands = (
        f"{lp_root}.embed_tokens.weight" if lp_root else "embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed.weight",
    )
    emb_key = next(k for k in emb_cands if k in active_keys)
    emb_t = pop_tensor_dequant(fetcher, emb_key, active_keys, used_special, device="cpu").to(torch.float16).contiguous()
    emb_bytes = emb_t.numpy().tobytes()
    (out_dir / "embed_tokens.bin").write_bytes(emb_bytes)
    emb_root = hashlib.sha256(emb_bytes).digest()
    del emb_bytes

    lm_key_cands = ("lm_head.weight", "language_model.lm_head.weight", "head.weight")
    lm_key = next((k for k in lm_key_cands if k in active_keys), None)
    if lm_key is not None:
        lm_f32 = pop_tensor_dequant(fetcher, lm_key, active_keys, used_special, device=q_dev).to(torch.float32)
    else:
        lm_f32 = emb_t.to(device=q_dev, dtype=torch.float32)
    del emb_t

    max_abs = torch.clamp(lm_f32.abs().amax(dim=-1, keepdim=True), min=1e-6)
    scales_f16 = (max_abs / 127.0).to(torch.float16)
    w_i8 = torch.clamp(torch.round(lm_f32 / scales_f16.to(torch.float32)), -127, 127).to(torch.int8)
    head_bytes = w_i8.contiguous().cpu().numpy().tobytes() + scales_f16.contiguous().cpu().numpy().tobytes()
    (out_dir / "lm_head_int8.bin").write_bytes(head_bytes)
    head_root = hashlib.sha256(head_bytes).digest()
    del lm_f32, w_i8, scales_f16
    if q_dev == "cuda":
        torch.cuda.empty_cache()

    layout = compute_q4_leaf_layout(hidden_size, inter_size, group_size=group_size)
    stride_bytes = layout["stride_bytes"]
    payload_bytes = layout["payload_bytes"]
    pad_tail = b"\x00" * (stride_bytes - payload_bytes)
    leaf_hashes: List[bytes] = []
    backbone_sd: Dict[str, torch.Tensor] = {}
    layers_pfx = f"{lp_root}.layers" if lp_root else "layers"

    with open(out_dir / "cas_experts_q4.bin", "wb") as cas_f:
        for l_idx in range(num_layers):
            lp = f"{layers_pfx}.{l_idx}"
            if l_idx in moe_map:
                gu_t, dn_t = extract_layer_experts_gu_dn(
                    fetcher, lp, moe_map[l_idx], num_experts, hidden_size, inter_size,
                    active_keys, used_special, q_dev,
                )
                gu_qw, gu_qs, gu_qb = quantize_torch_to_q4_bytes(gu_t, group_size=group_size)
                del gu_t
                dn_qw, dn_qs, dn_qb = quantize_torch_to_q4_bytes(dn_t, group_size=group_size)
                del dn_t
                for e_idx in range(num_experts):
                    raw_leaf = (
                        gu_qw[e_idx].tobytes() + gu_qs[e_idx].tobytes() + gu_qb[e_idx].tobytes()
                        + dn_qw[e_idx].tobytes() + dn_qs[e_idx].tobytes() + dn_qb[e_idx].tobytes()
                        + pad_tail
                    )
                    leaf_hashes.append(hashlib.sha256(raw_leaf).digest())
                    cas_f.write(raw_leaf)

            layer_pfx = f"{lp}."
            for k in sorted(active_keys):
                if k.startswith(layer_pfx) and k not in used_special and not k.endswith("_scale_inv"):
                    t_cpu = pop_tensor_dequant(fetcher, k, active_keys, used_special, device="cpu")
                    ck = canonicalize_hf_key(k, lp_root)
                    backbone_sd[ck] = _cast_backbone_tensor(ck, t_cpu)

            if q_dev == "cuda":
                torch.cuda.empty_cache()
            sys.stdout.write(f"[Deimos-HF] Layer {l_idx + 1}/{num_layers} ({len(leaf_hashes)}/{num_moe_layers * num_experts} Q4 leaves)\n")
            sys.stdout.flush()

    for k in sorted(active_keys - used_special):
        if k.endswith("_scale_inv"):
            continue
        t_cpu = pop_tensor_dequant(fetcher, k, active_keys, used_special, device="cpu")
        ck = canonicalize_hf_key(k, lp_root)
        backbone_sd[ck] = _cast_backbone_tensor(ck, t_cpu)

    bb_path = out_dir / "hf_backbone.safetensors"
    st_save_torch(backbone_sd, str(bb_path))
    backbone_sd.clear()
    gc.collect()
    with open(bb_path, "rb") as f_bb:
        core_root = hashlib.file_digest(f_bb, "sha256").digest()
    shutil.rmtree(tmp_dl, ignore_errors=True)

    t_cfg.pop("quantization_config", None)
    t_cfg.pop("quantization", None)
    t_cfg["num_experts"] = num_experts
    if "moe_intermediate_size" not in t_cfg:
        t_cfg["intermediate_size"] = inter_size
    (out_dir / "config.json").write_text(json.dumps(t_cfg, indent=2))
    (out_dir / "hf_source_config.json").write_text(json.dumps(raw_cfg, indent=2))

    model_root_hex, expert_root_hex = build_deimidx2_file(
        idx_path=out_dir / "merkle_tree.idx",
        num_layers=num_moe_layers,
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
                "store_kind": "hf_hybrid_q4",
                "version": 2,
                "source_model": str(repo_id_or_dir),
                "quant_bits": 4,
                "quant_group_size": group_size,
                "num_layers": num_moe_layers,
                "num_total_layers": num_layers,
                "moe_layer_ids": moe_layer_ids,
                "moe_containers": moe_containers,
                "num_experts_per_layer": num_experts,
                "total_experts": num_moe_layers * num_experts,
                "expert_stride_bytes": stride_bytes,
                "expert_payload_bytes": payload_bytes,
                "orig_safetensors_bytes": orig_bytes,
                "model_merkle_root": model_root_hex,
                "expert_tree_root": expert_root_hex,
                "conversion_time_sec": round(time.perf_counter() - t0, 2),
            },
            indent=2,
        )
    )
    return out_dir
