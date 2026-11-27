#!/usr/bin/env python3
"""
DeimosStream Native Multi-Token Prediction (`MTP` / `NextN`) Store Extractor
(`src/deimos_stream/mtp_store.py`)
============================================================================
Discovers, dequantizes (`FP8 E4M3` / `UE8M0` / `BF16`), and exports native checkpoint
MTP / NextN heads into an optional `mtp_head.safetensors` + `mtp_manifest.json` pair
without altering `DEIMIDX2` (`cas_experts_q4.bin` / `merkle_tree.idx`) backward compatibility:

  1. `qwen_mtp` (`Qwen3.5-MoE`, `Qwen3.6-35B-A3B`, `Qwen3.8`, `Qwen3-Next`):
     - `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight`, `mtp.fc.weight`,
       `mtp.layers.0.*`, `mtp.norm.weight` (`zero_centered_norm=True`).
  2. `nextn_layer` (`GLM-5.3-Flash`, `GLM-4.5 / 4.7-Flash`, `DeepSeek-V3 / V3.2 / R1`):
     - `{lp_root}.layers.{num_hidden_layers}.{enorm, hnorm, eh_proj, shared_head.norm}`
       (or `nextn.{enorm, hnorm, e_proj+h_proj, shared_head_norm}`) + decoder layer weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Tuple

import numpy as np
from safetensors import safe_open

from deimos_stream.model_adapters import (
    detect_lm_prefix,
    extract_layer_experts_gu_dn,
    is_companion_scale_key,
    pop_tensor_dequant,
)
from deimos_stream.torch_arena import HAS_TORCH

if HAS_TORCH:
    from safetensors.torch import save_file as st_save_torch
    import torch
else:
    torch = None  # type: ignore[assignment]

_ZERO_CENTERED_FAMILIES = {
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_next",
    "qwen4_exp_text",
}

_F32_KEEP_SUFFIXES = ("A_log", "dt_bias", "e_score_correction_bias", "expert_bias")


def _with_companion_scales(keys: Set[str], all_keys: Set[str]) -> Set[str]:
    out = set(keys)
    for k in list(keys):
        for cand in (f"{k}_scale_inv", f"{k}.scale_inv"):
            if cand in all_keys:
                out.add(cand)
        if k.endswith(".weight"):
            stem = k[: -len(".weight")]
            for cand in (f"{stem}.weight_scale_inv", f"{stem}.scale"):
                if cand in all_keys:
                    out.add(cand)
    return out


def discover_mtp_spec(raw_cfg: Dict[str, Any], weight_map: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Inspects `config.json` + `weight_map` and returns MTP metadata if a native head exists."""
    t_cfg = raw_cfg.get("text_config", raw_cfg)
    if not isinstance(t_cfg, dict):
        return None
    mtype = str(t_cfg.get("model_type") or raw_cfg.get("model_type") or "").lower()
    if mtype in {"deepseek_v4", "deepseek_v41"}:
        # DeepSeek-V4 uses 3-stage DSpark Markov head rather than single-layer NextN eh_proj
        return None

    num_layers = int(t_cfg.get("num_hidden_layers", 0) or 0)
    all_keys = set(weight_map.keys())
    zero_centered = mtype in _ZERO_CENTERED_FAMILIES

    for mtp_pfx in ("mtp", "model.mtp", "model.language_model.mtp"):
        fc_k = f"{mtp_pfx}.fc.weight"
        en_k = f"{mtp_pfx}.pre_fc_norm_embedding.weight"
        hn_k = f"{mtp_pfx}.pre_fc_norm_hidden.weight"
        on_k = f"{mtp_pfx}.norm.weight"
        l0_pfx = f"{mtp_pfx}.layers.0."
        if fc_k in all_keys and en_k in all_keys and hn_k in all_keys and on_k in all_keys:
            base = {k for k in all_keys if k.startswith(f"{mtp_pfx}.")}
            # Exclude higher MTP layers if any (`layers.1+`) to keep footprint minimal
            base = {
                k
                for k in base
                if not (k.startswith(f"{mtp_pfx}.layers.") and not k.startswith(l0_pfx))
            }
            return {
                "style": "qwen_mtp",
                "prefix": mtp_pfx,
                "layer_prefix": f"{mtp_pfx}.layers.0",
                "enorm_key": en_k,
                "hnorm_key": hn_k,
                "eh_proj_key": fc_k,
                "e_proj_key": None,
                "h_proj_key": None,
                "head_norm_key": on_k,
                "zero_centered_norm": zero_centered,
                "needed_keys": _with_companion_scales(base, all_keys),
            }

    n_nextn = int(t_cfg.get("num_nextn_predict_layers", 0) or 0)
    if n_nextn <= 0 or num_layers <= 0:
        return None

    core_keys = {k for k in all_keys if f".layers.{num_layers}." not in k}
    lp_root = detect_lm_prefix(core_keys or all_keys)
    lp = f"{lp_root}.layers.{num_layers}" if lp_root else f"layers.{num_layers}"
    lp_dot = f"{lp}."
    layer_keys = {k for k in all_keys if k.startswith(lp_dot)}
    if not layer_keys:
        return None

    en_k = next((k for k in (f"{lp}.enorm.weight", f"{lp}.nextn.enorm.weight") if k in all_keys), None)
    hn_k = next((k for k in (f"{lp}.hnorm.weight", f"{lp}.nextn.hnorm.weight") if k in all_keys), None)
    on_k = next(
        (
            k
            for k in (
                f"{lp}.shared_head.norm.weight",
                f"{lp}.nextn.shared_head_norm.weight",
                f"{lp}.norm.weight",
            )
            if k in all_keys
        ),
        None,
    )
    eh_k = next((k for k in (f"{lp}.eh_proj.weight", f"{lp}.nextn.eh_proj.weight") if k in all_keys), None)
    ep_k = f"{lp}.nextn.e_proj.weight" if f"{lp}.nextn.e_proj.weight" in all_keys else None
    hp_k = f"{lp}.nextn.h_proj.weight" if f"{lp}.nextn.h_proj.weight" in all_keys else None

    if en_k is None or hn_k is None or on_k is None or (eh_k is None and (ep_k is None or hp_k is None)):
        return None

    return {
        "style": "nextn_layer",
        "prefix": lp,
        "layer_prefix": lp,
        "enorm_key": en_k,
        "hnorm_key": hn_k,
        "eh_proj_key": eh_k,
        "e_proj_key": ep_k,
        "h_proj_key": hp_k,
        "head_norm_key": on_k,
        "zero_centered_norm": zero_centered,
        "needed_keys": _with_companion_scales(layer_keys, all_keys),
    }


def build_mtp_tensors_from_fetcher(
    spec: Dict[str, Any],
    raw_cfg: Dict[str, Any],
    fetcher: Any,
    quantize_fn: Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray]],
    group_size: int = 64,
    q_dev: str = "cpu",
) -> Tuple[Dict[str, "torch.Tensor"], Dict[str, Any]]:
    """Extracts and quantizes an MTP head into `(state_dict, mtp_manifest)`."""
    t_cfg = dict(raw_cfg.get("text_config", raw_cfg))
    hidden_size = int(t_cfg["hidden_size"])
    inter_size = int(
        t_cfg.get("moe_intermediate_size") or t_cfg.get("ffn_hidden_size") or t_cfg["intermediate_size"]
    )
    num_experts = int(
        t_cfg.get("num_experts") or t_cfg.get("num_local_experts") or t_cfg.get("n_routed_experts") or 0
    )

    mtp_keys: Set[str] = set(spec["needed_keys"])
    used: Set[str] = set()
    sd: Dict[str, torch.Tensor] = {}

    sd["mtp.enorm.weight"] = pop_tensor_dequant(fetcher, spec["enorm_key"], mtp_keys, used, "cpu").to(torch.float32)
    sd["mtp.hnorm.weight"] = pop_tensor_dequant(fetcher, spec["hnorm_key"], mtp_keys, used, "cpu").to(torch.float32)
    sd["mtp.head_norm.weight"] = pop_tensor_dequant(fetcher, spec["head_norm_key"], mtp_keys, used, "cpu").to(
        torch.float32
    )
    if spec["eh_proj_key"] is not None:
        eh_w = pop_tensor_dequant(fetcher, spec["eh_proj_key"], mtp_keys, used, "cpu").to(torch.float16)
    else:
        e_w = pop_tensor_dequant(fetcher, spec["e_proj_key"], mtp_keys, used, "cpu").to(torch.float16)
        h_w = pop_tensor_dequant(fetcher, spec["h_proj_key"], mtp_keys, used, "cpu").to(torch.float16)
        eh_w = torch.cat([e_w, h_w], dim=1).contiguous()
    sd["mtp.eh_proj.weight"] = eh_w.contiguous()

    lp = str(spec["layer_prefix"])
    moe_container: Optional[str] = None
    if num_experts > 0:
        for attr in ("mlp", "feed_forward", "block_sparse_moe", "ffn"):
            ep = f"{lp}.{attr}.experts."
            if (
                f"{lp}.{attr}.experts.gate_up_proj" in mtp_keys
                or f"{lp}.{attr}.experts.0.gate_proj.weight" in mtp_keys
                or f"{lp}.{attr}.experts.0.w1.weight" in mtp_keys
                or any(k.startswith(ep) for k in mtp_keys)
            ):
                moe_container = attr
                break

    if moe_container is not None:
        gu_t, dn_t = extract_layer_experts_gu_dn(
            fetcher, lp, moe_container, num_experts, hidden_size, inter_size, mtp_keys, used, q_dev
        )
        gu_qw, gu_qs, gu_qb = quantize_fn(gu_t, group_size=group_size, q_dev=q_dev)
        del gu_t
        dn_qw, dn_qs, dn_qb = quantize_fn(dn_t, group_size=group_size, q_dev=q_dev)
        del dn_t
        sd["experts.gu_qw"] = torch.from_numpy(gu_qw.view(np.int32)).contiguous()
        sd["experts.gu_qs"] = torch.from_numpy(gu_qs).view(torch.float16).contiguous()
        sd["experts.gu_qb"] = torch.from_numpy(gu_qb).view(torch.float16).contiguous()
        sd["experts.dn_qw"] = torch.from_numpy(dn_qw.view(np.int32)).contiguous()
        sd["experts.dn_qs"] = torch.from_numpy(dn_qs).view(torch.float16).contiguous()
        sd["experts.dn_qb"] = torch.from_numpy(dn_qb).view(torch.float16).contiguous()

    skip_prefixes = (
        f"{lp}.shared_head.",
        f"{lp}.nextn.",
        f"{lp}.embed_tokens.",
        f"{lp}.nextn_hc_head.",
    )
    lp_dot = f"{lp}."
    attn_type = "full_attention"
    for k in sorted(mtp_keys):
        if k in used or is_companion_scale_key(k, mtp_keys) or not k.startswith(lp_dot):
            continue
        if k.startswith(skip_prefixes):
            used.add(k)
            fetcher.pop_tensor(k, device="cpu")
            continue
        sub = k[len(lp_dot) :]
        if sub.startswith("linear_attn."):
            attn_type = "linear_attention"
        elif ".indexer." in sub:
            attn_type = "indexed_attention"
        t_cpu = pop_tensor_dequant(fetcher, k, mtp_keys, used, device="cpu")
        if t_cpu.dtype == torch.bfloat16 or (
            t_cpu.dtype == torch.float32 and not sub.endswith(_F32_KEEP_SUFFIXES)
        ):
            t_cpu = t_cpu.to(torch.float16)
        sd[f"model.layers.0.{sub}"] = t_cpu.contiguous()

    manifest = {
        "format": "DEIMOS_MTP_V1",
        "style": spec["style"],
        "zero_centered_norm": bool(spec["zero_centered_norm"]),
        "has_routed_moe": moe_container is not None,
        "moe_container": moe_container or "mlp",
        "num_experts": num_experts if moe_container is not None else 0,
        "hidden_size": hidden_size,
        "intermediate_size": inter_size,
        "quant_group_size": group_size,
        "layer_type": attn_type,
    }
    return sd, manifest


def export_mtp_head_to_store(
    spec: Optional[Dict[str, Any]],
    raw_cfg: Dict[str, Any],
    fetcher: Any,
    out_dir: Path,
    quantize_fn: Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray]],
    group_size: int = 64,
    q_dev: str = "cpu",
) -> Optional[Dict[str, Any]]:
    """Writes `mtp_head.safetensors` & `mtp_manifest.json` into `out_dir` if `spec` is present."""
    if spec is None:
        return None
    sd, manifest = build_mtp_tensors_from_fetcher(
        spec, raw_cfg, fetcher, quantize_fn=quantize_fn, group_size=group_size, q_dev=q_dev
    )
    st_save_torch(sd, str(out_dir / "mtp_head.safetensors"))
    (out_dir / "mtp_manifest.json").write_text(json.dumps(manifest, indent=2))
    sd.clear()
    return manifest


class _LocalSafetensorsMTPFetcher:
    """Lightweight read-only tensor loader for extracting MTP weights from raw `.safetensors` dirs."""

    def __init__(self, checkpoint_dir: Path, weight_map: Dict[str, str]):
        self.checkpoint_dir = checkpoint_dir
        self.weight_map = weight_map
        self._handles: Dict[str, Any] = {}

    def pop_tensor(self, key: str, device: str = "cpu") -> "torch.Tensor":
        sn = self.weight_map[key]
        sf = self._handles.get(sn)
        if sf is None:
            sf = safe_open(str(self.checkpoint_dir / sn), framework="pt", device=device)
            self._handles[sn] = sf
        return sf.get_tensor(key)

    def close(self) -> None:
        self._handles.clear()
