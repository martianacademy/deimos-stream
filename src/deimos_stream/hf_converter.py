#!/usr/bin/env python3
"""
DeimosStream HuggingFace Hybrid-MoE -> `DEIMIDX2` Q4 Converter (`src/deimos_stream/hf_converter.py`)
====================================================================================================
Converts modern 2026 HuggingFace Hybrid & Multimodal Sparse MoE checkpoints (such as
`Qwen/Qwen3.6-35B-A3B`, `Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3-Next-80B-A3B-Instruct`, and
any `AutoModelForCausalLM` with 3D `gate_up_proj`/`down_proj` or 2D per-expert weights) into:
  - `cas_experts_q4.bin` + `merkle_tree.idx` (`DEIMIDX2` 16 KB page-aligned Q4 Merkle leaves)
  - `hf_backbone.safetensors` (FP16 non-expert text layers, stripping vision/MTP overhead)
  - `embed_tokens.bin` (zero-VRAM `mmap` FP16 row table)
  - `lm_head_int8.bin` (tiled INT8 output head saving ~50% head VRAM + ~840 MiB temp spikes)

Supports reference-counted shard streaming from HuggingFace Hub (`delete_downloaded_shards=True`)
so 67+ GiB checkpoints convert cleanly inside small cloud `/tmp` disks with GPU-accelerated Q4!
"""

from __future__ import annotations

from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import re
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
from deimos_stream.torch_arena import HAS_TORCH

if HAS_TORCH:
    from safetensors.torch import save_file as st_save_torch
    import torch
else:
    torch = None  # type: ignore[assignment]


def quantize_torch_to_q4_bytes(
    weight_t: "torch.Tensor", group_size: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU-accelerated 4-bit affine group-64 quantizer producing exact `DEIMIDX2` / MLX layout:
      `(qw_u32, scales_f16, biases_f16)` on CPU NumPy.
    """
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
                repo_id=self.repo_id,
                filename=shard_name,
                local_dir=str(self.cache_dir),
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


def _fetch_metadata_files(repo_or_dir: str, out_dir: Path, tmp_dir: Path) -> Tuple[Dict[str, object], Dict[str, str], int]:
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
        src = p.resolve()
        for mf in meta_files:
            if (src / mf).exists():
                shutil.copy2(src / mf, out_dir / mf)
        raw_cfg = json.loads((src / "config.json").read_text())
        idx_p = src / "model.safetensors.index.json"
        if idx_p.exists():
            idx_d = json.loads(idx_p.read_text())
            tot_b = int(idx_d.get("metadata", {}).get("total_size", 0))
            return raw_cfg, dict(idx_d["weight_map"]), tot_b
        single = src / "model.safetensors"
        with safe_open(str(single), framework="pt", device="cpu") as sf:
            wm = {k: "model.safetensors" for k in sf.keys()}
        return raw_cfg, wm, single.stat().st_size

    from huggingface_hub import hf_hub_download

    cfg_path = Path(hf_hub_download(repo_id=repo_or_dir, filename="config.json", local_dir=str(tmp_dir)))
    shutil.copy2(cfg_path, out_dir / "config.json")
    for mf in meta_files[1:]:
        try:
            mp = Path(hf_hub_download(repo_id=repo_or_dir, filename=mf, local_dir=str(tmp_dir)))
            shutil.copy2(mp, out_dir / mf)
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


def convert_hf_hybrid_moe_to_deimos(
    repo_id_or_dir: str,
    output_store_dir: Path,
    group_size: int = 64,
    delete_downloaded_shards: bool = True,
) -> Path:
    """
    Converts any HuggingFace Hybrid/Multimodal Sparse MoE (e.g. `Qwen/Qwen3.6-35B-A3B`)
    into a `DEIMIDX2` Q4 Merkle Store ready for `DeimosHFHybridMoEEngine`.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for convert_hf_hybrid_moe_to_deimos.")
    t0 = time.perf_counter()
    out_dir = Path(output_store_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dl = out_dir / "_dl_cache"
    compile_deimos_dylib(force=False)

    raw_cfg, weight_map, orig_bytes = _fetch_metadata_files(repo_id_or_dir, out_dir, tmp_dl)
    t_cfg = dict(raw_cfg.get("text_config", raw_cfg))  # type: ignore[arg-type]

    num_layers = int(t_cfg["num_hidden_layers"])
    hidden_size = int(t_cfg["hidden_size"])
    inter_size = int(t_cfg.get("moe_intermediate_size") or t_cfg["intermediate_size"])
    num_experts = int(
        t_cfg.get("num_experts") or t_cfg.get("num_local_experts") or t_cfg["n_routed_experts"]
    )
    vocab_size = int(t_cfg["vocab_size"])
    q_dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Classify checkpoint keys (ignoring vision tower & MTP heads)
    ignore_re = re.compile(r"^(model\.visual\.|visual\.|mtp\.)")
    active_keys = {k for k in weight_map if not ignore_re.match(k)}
    has_lm_prefix = any(k.startswith("model.language_model.") for k in active_keys)
    lp_root = "model.language_model" if has_lm_prefix else "model"

    fetcher = _RefCountedShardFetcher(
        repo_id_or_dir, tmp_dl, active_keys, weight_map, delete_shards=delete_downloaded_shards
    )

    # 1. Extract `embed_tokens.bin` + `lm_head_int8.bin`
    emb_key = f"{lp_root}.embed_tokens.weight"
    emb_t = fetcher.pop_tensor(emb_key, device="cpu").to(torch.float16).contiguous()
    emb_bytes = emb_t.numpy().tobytes()
    (out_dir / "embed_tokens.bin").write_bytes(emb_bytes)
    emb_root = hashlib.sha256(emb_bytes).digest()
    del emb_bytes

    lm_key_cands = ("lm_head.weight", "language_model.lm_head.weight")
    lm_key = next((k for k in lm_key_cands if k in active_keys), None)
    lm_f32 = (fetcher.pop_tensor(lm_key, device=q_dev) if lm_key else emb_t.to(q_dev)).to(torch.float32)
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

    # 2. Stream layers 0..L-1: pack experts -> `cas_experts_q4.bin` & collect FP16 text backbone
    layout = compute_q4_leaf_layout(hidden_size, inter_size, group_size=group_size)
    stride_bytes = layout["stride_bytes"]
    payload_bytes = layout["payload_bytes"]
    pad_tail = b"\x00" * (stride_bytes - payload_bytes)
    leaf_hashes: List[bytes] = []
    backbone_sd: Dict[str, torch.Tensor] = {}

    def _canon_key(k: str) -> str:
        return "model." + k[len("model.language_model.") :] if k.startswith("model.language_model.") else k

    used_special = {emb_key} | ({lm_key} if lm_key else set())

    with open(out_dir / "cas_experts_q4.bin", "wb") as cas_f:
        for l_idx in range(num_layers):
            lp = f"{lp_root}.layers.{l_idx}"
            gu_3d_k = f"{lp}.mlp.experts.gate_up_proj"
            dn_3d_k = f"{lp}.mlp.experts.down_proj"

            if gu_3d_k in active_keys:
                used_special.update((gu_3d_k, dn_3d_k))
                gu_t = fetcher.pop_tensor(gu_3d_k, device=q_dev)
                gu_qw, gu_qs, gu_qb = quantize_torch_to_q4_bytes(gu_t, group_size=group_size)
                del gu_t
                dn_t = fetcher.pop_tensor(dn_3d_k, device=q_dev)
                dn_qw, dn_qs, dn_qb = quantize_torch_to_q4_bytes(dn_t, group_size=group_size)
                del dn_t
            else:
                gu_list, dn_list = [], []
                for e_idx in range(num_experts):
                    gk = f"{lp}.mlp.experts.{e_idx}.gate_proj.weight"
                    uk = f"{lp}.mlp.experts.{e_idx}.up_proj.weight"
                    dk = f"{lp}.mlp.experts.{e_idx}.down_proj.weight"
                    used_special.update((gk, uk, dk))
                    gu_list.append(torch.cat([fetcher.pop_tensor(gk), fetcher.pop_tensor(uk)], dim=0))
                    dn_list.append(fetcher.pop_tensor(dk))
                gu_qw, gu_qs, gu_qb = quantize_torch_to_q4_bytes(
                    torch.stack(gu_list, dim=0).to(q_dev), group_size=group_size
                )
                dn_qw, dn_qs, dn_qb = quantize_torch_to_q4_bytes(
                    torch.stack(dn_list, dim=0).to(q_dev), group_size=group_size
                )
                del gu_list, dn_list

            for e_idx in range(num_experts):
                raw_leaf = (
                    gu_qw[e_idx].tobytes()
                    + gu_qs[e_idx].tobytes()
                    + gu_qb[e_idx].tobytes()
                    + dn_qw[e_idx].tobytes()
                    + dn_qs[e_idx].tobytes()
                    + dn_qb[e_idx].tobytes()
                    + pad_tail
                )
                leaf_hashes.append(hashlib.sha256(raw_leaf).digest())
                cas_f.write(raw_leaf)

            # Collect all non-expert tensors belonging to layer `l_idx` right away so shard refcounts hit 0!
            layer_pfx = f"{lp}."
            for k in sorted(active_keys):
                if k.startswith(layer_pfx) and k not in used_special:
                    used_special.add(k)
                    t_cpu = fetcher.pop_tensor(k, device="cpu")
                    if t_cpu.dtype == torch.bfloat16:
                        t_cpu = t_cpu.to(torch.float16)
                    backbone_sd[_canon_key(k)] = t_cpu.contiguous()

            if q_dev == "cuda":
                torch.cuda.empty_cache()
            sys.stdout.write(
                f"[Deimos-HF] Layer {l_idx + 1}/{num_layers} "
                f"({(l_idx + 1) * num_experts}/{num_layers * num_experts} Q4 Merkle leaves)\n"
            )
            sys.stdout.flush()

    # Remaining non-expert top-level text keys (e.g. `model.norm.weight`)
    for k in sorted(active_keys - used_special):
        t_cpu = fetcher.pop_tensor(k, device="cpu")
        if t_cpu.dtype == torch.bfloat16:
            t_cpu = t_cpu.to(torch.float16)
        backbone_sd[_canon_key(k)] = t_cpu.contiguous()

    bb_path = out_dir / "hf_backbone.safetensors"
    st_save_torch(backbone_sd, str(bb_path))
    backbone_sd.clear()
    gc.collect()
    with open(bb_path, "rb") as f_bb:
        core_root = hashlib.file_digest(f_bb, "sha256").digest()
    shutil.rmtree(tmp_dl, ignore_errors=True)

    # Save normalized text config (`config.json`) with `num_experts` standardized
    t_cfg["num_experts"] = num_experts
    t_cfg["intermediate_size"] = inter_size
    (out_dir / "config.json").write_text(json.dumps(t_cfg, indent=2))
    (out_dir / "hf_source_config.json").write_text(json.dumps(raw_cfg, indent=2))

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
                "store_kind": "hf_hybrid_q4",
                "version": 2,
                "source_model": str(repo_id_or_dir),
                "quant_bits": 4,
                "quant_group_size": group_size,
                "num_layers": num_layers,
                "num_experts_per_layer": num_experts,
                "total_experts": num_layers * num_experts,
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
