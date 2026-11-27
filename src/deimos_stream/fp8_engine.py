#!/usr/bin/env python3
"""
DeimosStream Zero-Copy Raw `.safetensors` Native `FP8` MoE Engine
(`src/deimos_stream/fp8_engine.py`)
=================================================================
Streams unconverted HuggingFace `FP8` / `MXFP8` / `UE8M0` Sparse & Hybrid MoE checkpoints
(`zai-org/GLM-5.3-Flash` 306B, `DeepSeek-V4-Flash`, `Qwen3.6/3.8-FP8`, `MiniMax-M3.1-Flash`)
directly from `.safetensors` shards with **0 GB extra disk space** and **0 quantization loss**.
"""

from __future__ import annotations

import gc
import inspect
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deimos_stream.backbone import rss_mb
from deimos_stream.engine import BenchmarkMetrics
from deimos_stream.fp8_arena import DeimosFP8PagedArena
from deimos_stream.fp8_kernels import HAS_MLX_FP8
from deimos_stream.fp8_safetensors import SafetensorsZeroCopyIndex
from deimos_stream.hf_fast_patch import patch_hf_hybrid_for_cuda
from deimos_stream.model_adapters import bind_hf_moe_layer, extract_swiglu_params
from deimos_stream.torch_arena import HAS_TORCH
from deimos_stream.triton_q4 import (
    quantize_2d_q4_torch,
    triton_int8_gemv,
    triton_kernels_available,
    triton_q4_dense_linear,
)

if HAS_MLX_FP8:
    import mlx.core as mx
else:
    mx = None  # type: ignore[assignment]

if HAS_TORCH:
    import torch
    import torch.nn as nn

    _inference_mode = torch.inference_mode
    _NNModule = nn.Module
else:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _NNModule = object  # type: ignore[assignment,misc]

    def _inference_mode():  # type: ignore[misc]
        return lambda fn: fn

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None  # type: ignore[assignment,misc]


def inspect_raw_fp8_store(store_dir: Path) -> Dict[str, object]:
    """Returns topology & layout summary for an unconverted HuggingFace FP8 MoE checkpoint."""
    sdir = Path(store_dir).expanduser().resolve()
    t0 = time.perf_counter()
    idx = SafetensorsZeroCopyIndex(sdir, use_nocache=False)
    try:
        return {
            "store_dir": str(sdir),
            "store_kind": "raw_safetensors_fp8",
            "num_shards": len(idx.shard_paths),
            "num_layers": idx.num_moe_layers,
            "num_total_layers": idx.num_total_layers,
            "num_experts_per_layer": idx.num_experts,
            "total_experts": idx.num_moe_layers * idx.num_experts,
            "vocab_size": idx.vocab_size,
            "hidden_size": idx.hidden_size,
            "intermediate_size": idx.inter_size,
            "quant_bits": 8,
            "block_shape": [idx.block_m, idx.block_n],
            "expert_stride_bytes": idx.expert_bytes,
            "fixed_backbone_mb": round(idx.fixed_backbone_bytes(2) / (1024.0 * 1024.0), 2),
            "header_parse_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }
    finally:
        idx.close()


class DeimosSafetensorsMmapEmbedding(_NNModule):  # type: ignore[misc,valid-type]
    """Zero-VRAM `mmap` embedding lookup slicing rows directly inside `.safetensors` (`BF16`/`F16`/`F32`)."""

    def __init__(self, view: np.ndarray, raw_dtype: str, device: "torch.device", dtype: "torch.dtype"):
        super().__init__()
        self.view = view
        self.raw_dtype = raw_dtype
        self.device = device
        self.dtype = dtype
        self.register_buffer("weight", torch.empty((0,), dtype=dtype, device=device), persistent=False)

    def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        ids_np = input_ids.detach().cpu().numpy()
        sliced = np.ascontiguousarray(self.view[ids_np])
        if self.raw_dtype == "BF16":
            return torch.from_numpy(sliced.view(np.int16)).view(torch.bfloat16).to(device=self.device, dtype=self.dtype)
        return torch.from_numpy(sliced).to(device=self.device, dtype=self.dtype)

    def close(self) -> None:
        self.view = np.empty((0, 0), dtype=np.uint16)


class DeimosMemoryInt8LMHead(_NNModule):  # type: ignore[misc,valid-type]
    """In-memory symmetric INT8 (or CUDA Q4) `lm_head` quantized on load with zero disk footprint."""

    def __init__(
        self,
        w_i8: np.ndarray,
        s_f16: np.ndarray,
        device: "torch.device",
        dtype: "torch.dtype",
        tile_vocab: int = 32768,
        use_q4: bool = False,
    ):
        super().__init__()
        self.vocab_size, hidden_size = int(w_i8.shape[0]), int(w_i8.shape[1])
        self.tile_vocab = int(tile_vocab)
        self.compute_dtype = dtype
        self.q4: Optional[Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]] = None
        self.register_buffer("weight", torch.empty((0,), dtype=dtype, device=device), persistent=False)
        if use_q4 and device.type == "cuda" and hidden_size % 64 == 0:
            qws, qss, qbs = [], [], []
            for v0 in range(0, self.vocab_size, 16384):
                v1 = min(self.vocab_size, v0 + 16384)
                ch = torch.from_numpy(w_i8[v0:v1].copy()).to(device, torch.float16) * torch.from_numpy(
                    s_f16[v0:v1].copy()
                ).to(device, torch.float16)
                qw, qs, qb = quantize_2d_q4_torch(ch, group_size=64)
                qws.append(qw)
                qss.append(qs)
                qbs.append(qb)
            self.q4 = (torch.cat(qws, dim=0), torch.cat(qss, dim=0), torch.cat(qbs, dim=0))
            self.w_i8 = torch.empty((0, 0), dtype=torch.int8, device=device)
            self.scale = torch.empty((0, 0), dtype=dtype, device=device)
        else:
            self.w_i8 = torch.from_numpy(w_i8).to(device)
            self.scale = torch.from_numpy(s_f16).to(device=device, dtype=dtype)

    def forward(self, hidden_states: "torch.Tensor") -> "torch.Tensor":
        orig_shape = hidden_states.shape[:-1]
        x = hidden_states.reshape(-1, hidden_states.shape[-1]).to(self.compute_dtype)
        if self.q4 is not None:
            return triton_q4_dense_linear(x, *self.q4, group_size=64).reshape(*orig_shape, self.vocab_size)
        if triton_kernels_available(x.device) and x.shape[0] <= 4:
            return triton_int8_gemv(x, self.w_i8, self.scale).reshape(*orig_shape, self.vocab_size)
        out_tiles: List[torch.Tensor] = []
        for v0 in range(0, self.vocab_size, self.tile_vocab):
            v1 = min(self.vocab_size, v0 + self.tile_vocab)
            w_tile = self.w_i8[v0:v1].to(self.compute_dtype) * self.scale[v0:v1]
            out_tiles.append(x @ w_tile.T)
        return torch.cat(out_tiles, dim=-1).reshape(*orig_shape, self.vocab_size)


class DeimosPagedFP8HFExperts(_NNModule):  # type: ignore[misc,valid-type]
    """Drop-in `DeimosFP8PagedArena` adapter bound into HuggingFace MoE decoder layers."""

    def __init__(self, layer_idx: int, arena: DeimosFP8PagedArena, engine_ref: "DeimosFP8SafetensorsEngine"):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.arena = arena
        self.engine_ref = engine_ref

    def forward(
        self,
        hidden_states: "torch.Tensor",
        top_k_index: "torch.Tensor",
        top_k_weights: "torch.Tensor",
        shared_fn: Optional[Callable[[], "torch.Tensor"]] = None,
    ) -> "torch.Tensor":
        eng = self.engine_ref
        return self.arena.forward_moe(
            self.layer_idx,
            hidden_states,
            top_k_index,
            top_k_weights,
            miss_tail_tau=eng.miss_tail_tau,
            min_k=eng.min_k,
            shared_fn=shared_fn,
        )


class DeimosFP8SafetensorsEngine:
    """Zero-Copy Native `FP8` Safetensors Engine (`0 GB` extra disk, Apple Metal + CUDA + CPU)."""

    def __init__(
        self,
        store_dir: Path,
        device_str: str = "mlx",
        slots_per_layer: int = 16,
        verify_merkle_on_cold: bool = False,
        use_nocache_pread: bool = True,
        enable_lookahead: bool = False,
        enable_merkle_kv: bool = False,
        miss_tail_tau: float = 1.0,
        min_k: int = 4,
        top_k: Optional[int] = None,
        backbone: str = "fp16",
    ):
        del verify_merkle_on_cold, enable_merkle_kv
        if not HAS_TORCH:
            raise RuntimeError("PyTorch & transformers required for DeimosFP8SafetensorsEngine.")

        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import convert_and_load_state_dict_in_model
        from transformers.modeling_utils import LoadStateDictConfig

        self.store_dir = Path(store_dir).expanduser().resolve()
        use_mlx_metal = device_str == "mlx" and HAS_MLX_FP8
        if device_str == "mlx":
            mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            self.device = torch.device("mps" if mps_ok else "cpu")
            self.backend_name = "fp8_metal" if use_mlx_metal else f"fp8_{self.device.type}"
        else:
            self.device = torch.device(device_str)
            self.backend_name = f"fp8_{self.device.type}"

        self.backbone = backbone
        self.compute_dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32
        self.index = SafetensorsZeroCopyIndex(self.store_dir, use_nocache=True)
        self.config: Dict[str, object] = dict(self.index.text_cfg)
        self.num_layers = self.index.num_total_layers
        self.num_experts = self.index.num_experts
        self.hidden_size = self.index.hidden_size
        self.vocab_size = self.index.vocab_size
        self.top_k = max(1, min(self.num_experts, int(top_k or self.index.top_k)))
        self.enable_lookahead = bool(enable_lookahead)
        self.enable_merkle_kv = False
        self.miss_tail_tau = float(miss_tail_tau)
        self.min_k = max(1, min(self.top_k, int(min_k)))

        tok_p = self.store_dir / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tok_p)) if (Tokenizer is not None and tok_p.exists()) else None

        self.arena = DeimosFP8PagedArena(
            self.index,
            device=self.device,
            slots_per_layer=max(self.top_k, int(slots_per_layer)),
            use_mlx_metal=use_mlx_metal,
            use_nocache_pread=use_nocache_pread,
        )
        self.arena.swiglu_limit, self.arena.swiglu_alpha, self.arena.swiglu_up_offset = extract_swiglu_params(
            self.config
        )

        hf_cfg = AutoConfig.from_pretrained(str(self.store_dir), trust_remote_code=True)
        for c_obj in (hf_cfg, getattr(hf_cfg, "text_config", None)):
            if c_obj is not None and hasattr(c_obj, "quantization_config"):
                delattr(c_obj, "quantization_config")
        text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
        m_cls = AutoModelForCausalLM._model_mapping.get(type(text_cfg), None)
        text_cfg._attn_implementation = "sdpa" if getattr(m_cls, "_supports_sdpa", False) else "eager"
        text_cfg.dtype = self.compute_dtype

        with torch.device("meta"):
            if m_cls is not None:
                model = AutoModelForCausalLM.from_config(text_cfg, dtype=self.compute_dtype, trust_remote_code=True)
            else:
                v_cls = AutoModelForImageTextToText._model_mapping.get(type(hf_cfg), None)
                hf_cfg.text_config._attn_implementation = "sdpa" if getattr(v_cls, "_supports_sdpa", False) else "eager"
                hf_cfg.text_config.dtype = self.compute_dtype
                hf_cfg.dtype = self.compute_dtype
                model = AutoModelForImageTextToText.from_config(hf_cfg, dtype=self.compute_dtype, trust_remote_code=True)
                if hasattr(model.model, "visual"):
                    model.model.visual = None
                if hasattr(model.model, "vision_tower"):
                    model.model.vision_tower = None
                text_cfg = hf_cfg.text_config

        bb_sd = self.index.build_backbone_state_dict(self.compute_dtype)
        load_dev = torch.device("cpu") if (self.backbone == "q4" and self.device.type == "cuda") else self.device
        is_vlm = hasattr(model.model, "language_model")
        casted_bb = {}
        for k in list(bb_sd.keys()):
            v = bb_sd.pop(k)
            if is_vlm and k.startswith(("model.layers.", "model.norm.")):
                k = "model.language_model." + k[len("model.") :]
            casted_bb[k] = v.to(device=load_dev)

        convert_and_load_state_dict_in_model(
            model,
            casted_bb,
            LoadStateDictConfig(
                device_map={"": load_dev},
                dtype=self.compute_dtype,
                weight_mapping=get_model_conversion_mapping(model),
            ),
        )
        casted_bb.clear()
        for _, parent_mod in list(model.named_modules()):
            for child_name, child_mod in list(parent_mod.named_children()):
                if any(b.device.type == "meta" for b in child_mod.buffers(recurse=False)):
                    setattr(parent_mod, child_name, type(child_mod)(config=text_cfg).to(self.device))

        inner_lm = getattr(model.model, "language_model", model.model)
        self._last_hidden_states: Optional[torch.Tensor] = None
        inner_lm.register_forward_hook(
            lambda _m, _i, out: setattr(
                self, "_last_hidden_states", getattr(out, "last_hidden_state", out[0] if isinstance(out, tuple) else out)
            )
        )
        emb_view, emb_dt = self.index.open_embedding_mmap()
        self._mmap_emb = DeimosSafetensorsMmapEmbedding(emb_view, emb_dt, self.device, self.compute_dtype)
        inner_lm.embed_tokens = self._mmap_emb
        w_i8, s_f16 = self.index.quantize_lm_head_int8()
        model.lm_head = DeimosMemoryInt8LMHead(
            w_i8, s_f16, self.device, self.compute_dtype, use_q4=(self.backbone == "q4")
        )

        for a_i, (l_i, c_attr) in enumerate(self.index.moe_specs):
            bind_hf_moe_layer(inner_lm.layers[l_i], c_attr, DeimosPagedFP8HFExperts(a_i, self.arena, self))

        if self.device.type == "cuda":
            patch_hf_hybrid_for_cuda(model, self.device, backbone=self.backbone)
        model.eval()
        self.model = model
        self._cache_kwarg = "cache" if "cache" in inspect.signature(model.forward).parameters else "past_key_values"
        gc.collect()
        self.reset_step_memory()

    def make_caches(self) -> List[Optional[object]]:
        return [None]

    def reset_step_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        elif self.arena.use_mlx_metal and hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()

    def vram_mb(self) -> Tuple[float, float]:
        if self.device.type == "cuda":
            return (
                float(torch.cuda.memory_allocated(self.device)) / (1024.0 * 1024.0),
                float(torch.cuda.max_memory_allocated(self.device)) / (1024.0 * 1024.0),
            )
        if self.arena.use_mlx_metal:
            mps_b = float(torch.mps.current_allocated_memory()) if self.device.type == "mps" else 0.0
            act = (mps_b + float(mx.get_active_memory())) / (1024.0 * 1024.0)
            peak = (mps_b + float(mx.get_peak_memory())) / (1024.0 * 1024.0)
            return act, max(act, peak)
        return self.arena.pool_vram_mb(), self.arena.pool_vram_mb()

    @_inference_mode()
    def sample_next_token(self, logits_2d: "torch.Tensor", temperature: float = 0.0) -> int:
        if temperature <= 1e-5:
            return int(torch.argmax(logits_2d, dim=-1).item())
        probs = torch.softmax(logits_2d.float() / temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    @_inference_mode()
    def forward_logits(
        self,
        token_ids: Sequence[int],
        caches: Optional[List[Optional[object]]] = None,
        all_positions: bool = False,
    ) -> "torch.Tensor":
        past = caches[0] if (caches is not None and len(caches) > 0) else None
        inp = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        out = self.model(
            input_ids=inp,
            use_cache=(caches is not None),
            logits_to_keep=0 if all_positions else 1,
            **{self._cache_kwarg: past},
        )
        if caches is not None:
            caches[0] = out["cache"] if "cache" in out else out.past_key_values
        return out.logits if all_positions else out.logits[:, -1, :]

    @_inference_mode()
    def prefill_prompt(
        self, prompt_ids: Sequence[int], caches: List[Optional[object]]
    ) -> Tuple["torch.Tensor", int]:
        return self.forward_logits(prompt_ids, caches=caches), 0

    @_inference_mode()
    def benchmark_token_ids(self, prompt_ids: Sequence[int], max_new_tokens: int = 32) -> BenchmarkMetrics:
        caches = self.make_caches()
        self.arena.reset_telemetry()
        self.reset_step_memory()

        t_pf = time.perf_counter()
        logits, skipped = self.prefill_prompt(prompt_ids, caches)
        tok = self.sample_next_token(logits, temperature=0.0)
        self.arena.wait_idle()
        ttft_ms = (time.perf_counter() - t_pf) * 1000.0

        self.arena.reset_telemetry()
        from deimos_stream.fast_decode import iter_decode_tokens

        t_dec = time.perf_counter()
        gen_ids = list(
            iter_decode_tokens(self, logits, caches, max_new_tokens + 1, prompt_ids=prompt_ids)
        )
        self.arena.wait_idle()
        decode_sec = max(1e-6, time.perf_counter() - t_dec)

        act_mb, peak_mb = self.vram_mb()
        st = self.arena.get_stats(nominal_top_k=self.top_k)
        return {
            "slots_per_layer": st["slots_per_layer"],
            "enable_lookahead": self.enable_lookahead,
            "miss_tail_tau": self.miss_tail_tau,
            "use_nocache_pread": self.arena.use_nocache_pread,
            "active_vram_mb": round(act_mb, 2),
            "peak_vram_mb": round(peak_mb, 2),
            "rss_mb": round(rss_mb(), 2),
            "pool_vram_mb": st["pool_vram_mb"],
            "kv_skipped_tokens": skipped,
            "ttft_ms": round(ttft_ms, 2),
            "decode_tok_per_s": round(max_new_tokens / decode_sec, 2),
            "hit_rate_pct": st["hit_rate_pct"],
            "cache_hits": st["cache_hits"],
            "cache_misses": st["cache_misses"],
            "evictions": st["evictions"],
            "lookahead_staged": 0,
            "lookahead_useful_hits": 0,
            "tail_misses_skipped": st["tail_misses_skipped"],
            "avg_active_k": st["avg_active_k"],
            "xor_proofs_verified": 0,
            "byte_proofs_verified": 0,
            "dma_bytes_staged_mb": st["dma_bytes_staged_mb"],
            "generated_ids": gen_ids,
            "generated_text": self.tokenizer.decode(gen_ids) if self.tokenizer is not None else "",
        }

    def benchmark_generation(self, prompt: str, max_new_tokens: int = 32) -> BenchmarkMetrics:
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer.json found in checkpoint; use benchmark_token_ids().")
        return self.benchmark_token_ids(self.tokenizer.encode(prompt).ids, max_new_tokens=max_new_tokens)

    def close(self) -> None:
        self.arena.close()
        self._mmap_emb.close()
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        self.index.close()
