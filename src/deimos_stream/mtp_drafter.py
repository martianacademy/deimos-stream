#!/usr/bin/env python3
"""
DeimosStream Native PyTorch / CUDA Multi-Token Prediction (`MTP` / `NextN`) Drafter
(`src/deimos_stream/mtp_drafter.py`)
===================================================================================
Loads checkpoint-native MTP / NextN heads (`mtp_head.safetensors` or raw `.safetensors` shards)
and executes stateful multi-token drafting across PyTorch/CUDA and Apple Silicon `MLX`
(delegating MLX execution to `mtp_mlx.py`).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from deimos_stream.model_adapters import bind_hf_moe_layer, extract_swiglu_params
from deimos_stream.mtp_mlx import HAS_MLX, MLXNativeMTPDrafter
from deimos_stream.mtp_store import (
    _LocalSafetensorsMTPFetcher,
    build_mtp_tensors_from_fetcher,
    discover_mtp_spec,
)
from deimos_stream.speculative import (
    _caches_have_recurrent_state,
    _prepare_croppable_hf_cache,
    _restore_recurrent_and_trim,
    _snapshot_recurrent_state,
    _trim_kv_only,
)
from deimos_stream.torch_arena import HAS_TORCH
from deimos_stream.triton_q4 import run_paged_q4_swiglu

if HAS_TORCH:
    from safetensors.torch import load_file as st_load_torch
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _NNModule = nn.Module
else:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _NNModule = object  # type: ignore[assignment,misc]


def _torch_rms_norm(
    x: "torch.Tensor", weight_f32: "torch.Tensor", eps: float, zero_centered: bool
) -> "torch.Tensor":
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    scale = (1.0 + weight_f32) if zero_centered else weight_f32
    return (normed * scale).to(x.dtype)


def _dequant_q4_selected_torch(
    qw_i32: "torch.Tensor", qs_f16: "torch.Tensor", qb_f16: "torch.Tensor", group_size: int = 64
) -> "torch.Tensor":
    """Unpacks selected `[M, out_dim, in_dim // 8]` Q4 expert rows to `float32` on CPU/MPS/CUDA."""
    m_cnt, out_d, packed_in = qw_i32.shape
    in_d = packed_in * 8
    raw_u8 = qw_i32.contiguous().view(torch.uint8).reshape(m_cnt, out_d, in_d // 2)
    lo = (raw_u8 & 0x0F).to(torch.float32)
    hi = ((raw_u8 >> 4) & 0x0F).to(torch.float32)
    q = torch.stack([lo, hi], dim=-1).reshape(m_cnt, out_d, in_d // group_size, group_size)
    return (q * qs_f16.float().unsqueeze(-1) + qb_f16.float().unsqueeze(-1)).reshape(m_cnt, out_d, in_d)


class DeimosMTPQ4Experts(_NNModule):  # type: ignore[misc,valid-type]
    """Compact 1-layer Group-64 Q4 MoE block for PyTorch / CUDA MTP heads."""

    def __init__(
        self,
        sd: Dict[str, "torch.Tensor"],
        device: "torch.device",
        dtype: "torch.dtype",
        group_size: int,
        swiglu_params: tuple[float, float, float],
    ):
        super().__init__()
        self.group_size = int(group_size)
        self.dtype = dtype
        self.swiglu_limit, self.swiglu_alpha, self.swiglu_up_offset = swiglu_params
        tgt = device if device.type == "cuda" else torch.device("cpu")
        self.gu_w = sd.pop("experts.gu_qw").to(tgt).contiguous()
        self.gu_s = sd.pop("experts.gu_qs").to(tgt).contiguous()
        self.gu_b = sd.pop("experts.gu_qb").to(tgt).contiguous()
        self.dn_w = sd.pop("experts.dn_qw").to(tgt).contiguous()
        self.dn_s = sd.pop("experts.dn_qs").to(tgt).contiguous()
        self.dn_b = sd.pop("experts.dn_qb").to(tgt).contiguous()

    def forward(
        self,
        hidden_states: "torch.Tensor",
        top_k_index: "torch.Tensor",
        top_k_weights: "torch.Tensor",
        shared_fn: Optional[Any] = None,
    ) -> "torch.Tensor":
        x_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
        y_sh = shared_fn() if callable(shared_fn) else None

        if x_2d.device.type == "cuda":
            y_exp = run_paged_q4_swiglu(
                x_2d.to(torch.float16),
                self.gu_w,
                self.gu_s,
                self.gu_b,
                self.dn_w,
                self.dn_s,
                self.dn_b,
                top_k_index.to(torch.int32),
                top_k_weights.to(torch.float16),
                group_size=self.group_size,
                swiglu_limit=self.swiglu_limit,
                swiglu_alpha=self.swiglu_alpha,
                swiglu_up_offset=self.swiglu_up_offset,
            ).to(x_2d.dtype)
        else:
            idx_cpu = top_k_index.detach().to("cpu", dtype=torch.long)
            w_cpu = top_k_weights.detach().to("cpu", dtype=torch.float32)
            x_cpu = x_2d.detach().to("cpu", dtype=torch.float32)
            n_tok, k_sel = idx_cpu.shape
            flat_idx = idx_cpu.reshape(-1)
            gu_f = _dequant_q4_selected_torch(
                self.gu_w[flat_idx], self.gu_s[flat_idx], self.gu_b[flat_idx], self.group_size
            ).reshape(n_tok, k_sel, -1, x_cpu.shape[-1])
            dn_f = _dequant_q4_selected_torch(
                self.dn_w[flat_idx], self.dn_s[flat_idx], self.dn_b[flat_idx], self.group_size
            ).reshape(n_tok, k_sel, x_cpu.shape[-1], -1)
            gu_out = torch.einsum("td,tkod->tko", x_cpu, gu_f)
            gate, up = gu_out.chunk(2, dim=-1)
            if self.swiglu_limit > 0.0:
                gate = gate.clamp(max=self.swiglu_limit)
                up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            act = (gate * torch.sigmoid(self.swiglu_alpha * gate)) * (up + self.swiglu_up_offset)
            exp_out = torch.einsum("tki,tkdi->tkd", act, dn_f)
            y_exp = (exp_out * w_cpu.unsqueeze(-1)).sum(dim=1).to(device=x_2d.device, dtype=x_2d.dtype)

        out = y_exp + y_sh if y_sh is not None else y_exp
        return out.reshape_as(hidden_states)


class TorchNativeMTPDrafter:
    """Stateful checkpoint-native MTP drafter for `DeimosHFHybridMoEEngine` & `DeimosFP8SafetensorsEngine`."""

    is_stateful_mtp: bool = True

    def __init__(self, eng: Any, sd: Dict[str, "torch.Tensor"], manifest: Dict[str, Any]):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import convert_and_load_state_dict_in_model
        from transformers.modeling_utils import LoadStateDictConfig

        self.eng = eng
        self.device: torch.device = eng.device
        self.dtype: torch.dtype = eng.compute_dtype
        self.zero_centered = bool(manifest.get("zero_centered_norm", False))
        self.eps = float(eng.config.get("rms_norm_eps", eng.config.get("layer_norm_epsilon", 1e-6)))

        self.enorm_w = sd.pop("mtp.enorm.weight").to(device=self.device, dtype=torch.float32)
        self.hnorm_w = sd.pop("mtp.hnorm.weight").to(device=self.device, dtype=torch.float32)
        self.head_norm_w = sd.pop("mtp.head_norm.weight").to(device=self.device, dtype=torch.float32)
        self.eh_proj_w = sd.pop("mtp.eh_proj.weight").to(device=self.device, dtype=self.dtype)

        src_cfg_p = Path(eng.store_dir) / "hf_source_config.json"
        cfg_src = src_cfg_p if src_cfg_p.exists() else Path(eng.store_dir)
        hf_cfg = copy.deepcopy(AutoConfig.from_pretrained(str(cfg_src), trust_remote_code=True))
        for c_obj in (hf_cfg, getattr(hf_cfg, "text_config", None)):
            if c_obj is not None and hasattr(c_obj, "quantization_config"):
                delattr(c_obj, "quantization_config")

        t_cfg = getattr(hf_cfg, "text_config", hf_cfg)
        has_moe = bool(manifest.get("has_routed_moe", True))
        l_type = str(manifest.get("layer_type", "full_attention"))
        t_cfg.num_hidden_layers = 1
        if hasattr(t_cfg, "first_k_dense_replace"):
            t_cfg.first_k_dense_replace = 0 if has_moe else 1
        if hasattr(t_cfg, "num_dense_layers"):
            t_cfg.num_dense_layers = 0 if has_moe else 1
        if hasattr(t_cfg, "mlp_only_layers"):
            t_cfg.mlp_only_layers = []
        if hasattr(t_cfg, "moe_layers"):
            t_cfg.moe_layers = [0] if has_moe else []
        if hasattr(t_cfg, "layer_types"):
            t_cfg.layer_types = [l_type]
        if hasattr(t_cfg, "mlp_layer_types"):
            t_cfg.mlp_layer_types = ["sparse" if has_moe else "dense"]
        if hasattr(t_cfg, "indexer_types") and isinstance(t_cfg.indexer_types, list) and t_cfg.indexer_types:
            t_cfg.indexer_types = [t_cfg.indexer_types[-1]]

        m_cls = AutoModelForCausalLM._model_mapping.get(type(t_cfg), None)
        with torch.device("meta"):
            if m_cls is not None:
                t_cfg._attn_implementation = "sdpa" if getattr(m_cls, "_supports_sdpa", False) else "eager"
                t_cfg.dtype = self.dtype
                shell = AutoModelForCausalLM.from_config(t_cfg, dtype=self.dtype, trust_remote_code=True)
            else:
                v_cls = AutoModelForImageTextToText._model_mapping.get(type(hf_cfg), None)
                hf_cfg.text_config._attn_implementation = (
                    "sdpa" if getattr(v_cls, "_supports_sdpa", False) else "eager"
                )
                hf_cfg.text_config.dtype = self.dtype
                hf_cfg.dtype = self.dtype
                shell = AutoModelForImageTextToText.from_config(
                    hf_cfg, dtype=self.dtype, trust_remote_code=True
                )
                if hasattr(shell.model, "visual"):
                    shell.model.visual = None
                if hasattr(shell.model, "vision_tower"):
                    shell.model.vision_tower = None
                t_cfg = hf_cfg.text_config

        inner = getattr(shell.model, "language_model", shell.model)
        inner.embed_tokens = nn.Identity()
        for norm_attr in ("norm", "embedding_norm", "final_layernorm", "ln_f"):
            if hasattr(inner, norm_attr):
                setattr(inner, norm_attr, nn.Identity())
        shell.lm_head = nn.Identity()

        if has_moe:
            q4_exp = DeimosMTPQ4Experts(
                sd,
                device=self.device,
                dtype=self.dtype,
                group_size=int(manifest.get("quant_group_size", 64)),
                swiglu_params=extract_swiglu_params(eng.config),
            )
            bind_hf_moe_layer(inner.layers[0], str(manifest.get("moe_container", "mlp")), q4_exp)

        is_vlm = hasattr(shell.model, "language_model")
        load_sd = {
            (f"model.language_model.{k[len('model.'):]}" if (is_vlm and k.startswith("model.")) else k): v.to(
                self.device
            )
            for k, v in sd.items()
        }
        sd.clear()
        convert_and_load_state_dict_in_model(
            shell,
            load_sd,
            LoadStateDictConfig(
                device_map={"": self.device},
                dtype=self.dtype,
                weight_mapping=get_model_conversion_mapping(shell),
            ),
        )
        load_sd.clear()
        for _, parent_mod in list(inner.named_modules()):
            for child_name, child_mod in list(parent_mod.named_children()):
                if any(b.device.type == "meta" for b in child_mod.buffers(recurse=False)):
                    setattr(parent_mod, child_name, type(child_mod)(config=t_cfg).to(self.device))

        inner.eval()
        self.inner = inner
        self._caches: List[Any] = [None]
        self._pending_h: Optional[torch.Tensor] = None

    def _fuse_eh(self, tok_ids: Sequence[int], h_seq: "torch.Tensor") -> "torch.Tensor":
        t_in = torch.tensor([list(tok_ids)], dtype=torch.long, device=self.device)
        emb = self.eng._mmap_emb(t_in).to(device=self.device, dtype=self.dtype)
        h_in = h_seq.to(device=self.device, dtype=self.dtype)
        e_n = _torch_rms_norm(emb, self.enorm_w, self.eps, self.zero_centered)
        h_n = _torch_rms_norm(h_in, self.hnorm_w, self.eps, self.zero_centered)
        return F.linear(torch.cat([e_n, h_n], dim=-1), self.eh_proj_w)

    def _run_layer(self, z0: "torch.Tensor") -> "torch.Tensor":
        out = self.inner(inputs_embeds=z0, cache=self._caches[0], use_cache=True)
        self._caches[0] = out["cache"] if "cache" in out else out.past_key_values
        if not _caches_have_recurrent_state(self._caches):
            _prepare_croppable_hf_cache(self._caches)
        return _torch_rms_norm(out.last_hidden_state, self.head_norm_w, self.eps, self.zero_centered)

    @torch.inference_mode()
    def on_prefill(self, prompt_ids: Sequence[int], prefill_hidden: Optional["torch.Tensor"]) -> None:
        self._caches = [None]
        self._pending_h = None
        if prefill_hidden is None or len(prompt_ids) == 0:
            return
        h_3d = prefill_hidden.reshape(1, -1, self.eh_proj_w.shape[0])
        if len(prompt_ids) >= 2 and h_3d.shape[1] == len(prompt_ids):
            self._run_layer(self._fuse_eh(prompt_ids[1:], h_3d[:, :-1, :]))
        self._pending_h = h_3d[:, -1:, :].detach().clone()

    @torch.inference_mode()
    def draft_step(self, cur_tok: int, limit: int) -> List[int]:
        if limit <= 0 or self._pending_h is None:
            return []
        h_cur = self._run_layer(self._fuse_eh([int(cur_tok)], self._pending_h))
        logits_1 = self.eng.model.lm_head(h_cur)[:, -1, :]
        d1 = int(torch.argmax(logits_1, dim=-1).item())
        drafts = [d1]
        if limit <= 1:
            return drafts

        has_rec = _caches_have_recurrent_state(self._caches)
        snap = _snapshot_recurrent_state(self._caches) if has_rec else None
        extra = 0
        prev_tok = d1
        prev_h = h_cur
        for _ in range(1, int(limit)):
            prev_h = self._run_layer(self._fuse_eh([prev_tok], prev_h))
            extra += 1
            lk = self.eng.model.lm_head(prev_h)[:, -1, :]
            prev_tok = int(torch.argmax(lk, dim=-1).item())
            drafts.append(prev_tok)

        if extra > 0:
            if has_rec:
                _restore_recurrent_and_trim(self._caches, snap, extra)
            else:
                _trim_kv_only(self._caches, extra)
        return drafts

    @torch.inference_mode()
    def on_target_step(
        self,
        cur_tok: int,
        accepted_drafts: Sequence[int],
        target_hidden_seq: Optional["torch.Tensor"],
        draft_ran: bool,
    ) -> None:
        if target_hidden_seq is None:
            self._pending_h = None
            return
        if not draft_ran and self._pending_h is not None:
            self._run_layer(self._fuse_eh([int(cur_tok)], self._pending_h))
        h_3d = target_hidden_seq.reshape(1, -1, self.eh_proj_w.shape[0])
        n_acc = len(accepted_drafts)
        if n_acc > 0 and h_3d.shape[1] >= n_acc + 1:
            self._run_layer(self._fuse_eh(accepted_drafts, h_3d[:, :n_acc, :]))
        idx_last = min(n_acc, h_3d.shape[1] - 1)
        self._pending_h = h_3d[:, idx_last : idx_last + 1, :].detach().clone()

    def __call__(self, cur_tok: int, limit: int) -> List[int]:
        return self.draft_step(cur_tok, limit)


def attach_mtp_drafter_if_available(eng: Any) -> Optional[Any]:
    """Lazily loads `mtp_head.safetensors` (or raw `.safetensors` MTP shards) onto `eng.mtp_drafter`."""
    existing = getattr(eng, "mtp_drafter", None)
    if existing is not None:
        return existing
    if getattr(eng, "_mtp_probed", False):
        return None
    eng._mtp_probed = True

    store_dir: Optional[Path] = getattr(eng, "store_dir", None)
    if store_dir is None:
        return None
    mtp_p = Path(store_dir) / "mtp_head.safetensors"
    man_p = Path(store_dir) / "mtp_manifest.json"
    backend = str(getattr(eng, "backend_name", ""))

    if mtp_p.exists() and man_p.exists():
        manifest = json.loads(man_p.read_text())
        if backend == "mlx_hybrid" and HAS_MLX:
            drafter = MLXNativeMTPDrafter(eng, mtp_p, manifest)
            eng.mtp_drafter = drafter
            return drafter
        if HAS_TORCH and hasattr(eng, "model") and hasattr(eng, "_mmap_emb"):
            sd = st_load_torch(str(mtp_p), device="cpu")
            drafter_t = TorchNativeMTPDrafter(eng, sd, manifest)
            eng.mtp_drafter = drafter_t
            return drafter_t

    idx_obj = getattr(eng, "index", None)
    if HAS_TORCH and idx_obj is not None and hasattr(idx_obj, "raw_cfg") and hasattr(eng, "_mmap_emb"):
        from deimos_stream.hf_converter import quantize_torch_to_q4_bytes
        from safetensors import safe_open

        wm_p = Path(store_dir) / "model.safetensors.index.json"
        if wm_p.exists():
            wm = json.loads(wm_p.read_text()).get("weight_map", {})
        else:
            shards = sorted(Path(store_dir).glob("*.safetensors"))
            if len(shards) != 1:
                return None
            with safe_open(str(shards[0]), framework="pt", device="cpu") as sf:
                wm = {k: shards[0].name for k in sf.keys()}
        spec = discover_mtp_spec(idx_obj.raw_cfg, wm)
        if spec is not None and all((Path(store_dir) / wm[k]).exists() for k in spec["needed_keys"]):
            fetcher = _LocalSafetensorsMTPFetcher(Path(store_dir), wm)
            try:
                q_dev = (
                    "cuda"
                    if getattr(eng, "device", None) is not None and eng.device.type == "cuda"
                    else "cpu"
                )
                sd_raw, man_raw = build_mtp_tensors_from_fetcher(
                    spec, idx_obj.raw_cfg, fetcher, quantize_fn=quantize_torch_to_q4_bytes, q_dev=q_dev
                )
            finally:
                fetcher.close()
            drafter_fp8 = TorchNativeMTPDrafter(eng, sd_raw, man_raw)
            eng.mtp_drafter = drafter_fp8
            return drafter_fp8

    return None
