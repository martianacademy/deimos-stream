#!/usr/bin/env python3
"""
DeimosStream Universal Apple Silicon `MLX` Sparse-MoE Engine (`src/deimos_stream/mlx_universal.py`)
==================================================================================================
Executes any `DEIMIDX2` `hf_hybrid_q4` store (`DeepSeek-V2`, `DeepSeek-V3/R1/V3.2`, `GLM-4.5`,
`Moonlight/Kimi-K2`, `Qwen2/3-MoE`, `Llama-4`, `Mixtral`, `MiniMax`, `Jamba`, `ERNIE-4.5-MoE`)
natively on Apple Silicon Metal GPUs by:
  1. Resolving the native `mlx_lm.models.<arch>` model skeleton lazily (`0` bytes for unpaged experts).
  2. Replacing every routed `SwitchGLU` submodule in `moe_layer_ids` with `DeimosMLXSwitchGLU`
     backed by our zero-copy Unified-Memory `PagedExpertArena` (`mx.gather_qmm`).
  3. Applying `model.sanitize(...)` on `hf_backbone.safetensors` (preserving `MLA` `kv_b_proj`
     splitting, `YaRN` RoPE, `Mamba2` SSM states, and `NoAux` sigmoid router biases).
  4. Binding zero-VRAM `mmap` embeddings (`DeimosMLXMmapEmbedding`) and quantized Metal `lm_head`
     (`DeimosMLXInt8LMHead`).
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deimos_stream.arena import ArenaStats, DeimosMerkleBridge, PagedExpertArena
from deimos_stream.backbone import rss_mb
from deimos_stream.engine import HAS_MLX, BenchmarkMetrics
from deimos_stream.mlx_hybrid import (
    DeimosMLXInt8LMHead,
    DeimosMLXMmapEmbedding,
    _get_active_and_peak_mb,
)

if HAS_MLX:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.switch_layers import SwitchGLU
    from mlx_lm.utils import _get_classes as mlx_get_classes

    _NNModule = nn.Module
else:
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    make_prompt_cache = None  # type: ignore[assignment]
    SwitchGLU = object  # type: ignore[assignment,misc]
    mlx_get_classes = None  # type: ignore[assignment]
    _NNModule = object  # type: ignore[assignment,misc]

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None  # type: ignore[assignment,misc]


class DeimosMLXSwitchGLU(_NNModule):  # type: ignore[misc,valid-type]
    """Drop-in `PagedExpertArena` (`mx.gather_qmm`) replacement for `mlx_lm` `SwitchGLU`."""

    def __init__(
        self,
        arena_idx: int,
        arena: PagedExpertArena,
        engine_ref: "DeimosMLXUniversalMoEEngine",
        custom_act: Any = None,
    ):
        super().__init__()
        self.arena_idx = int(arena_idx)
        self.arena = arena
        self.engine_ref = engine_ref
        self.custom_act = custom_act

    def __call__(self, x: "mx.array", indices: "mx.array") -> "mx.array":
        orig_prefix = x.shape[:-1]
        D = int(x.shape[-1])
        K = int(indices.shape[-1])
        x_2d = x.reshape(-1, D).astype(mx.float16)
        ind_2d = indices.reshape(-1, K)
        N = int(x_2d.shape[0])
        a_i = self.arena_idx
        inter = self.arena.intermediate_size
        max_c = max(1, self.arena.slots_per_layer // max(1, K))
        ones_sc = np.ones((min(N, max_c), K), dtype=np.float32)

        mx.eval(ind_2d)
        ind_np = np.asarray(ind_2d, dtype=np.int32)
        outs: List[mx.array] = []
        for s0 in range(0, N, max_c):
            s1 = min(N, s0 + max_c)
            sc_slice = ones_sc if (s1 - s0) == ones_sc.shape[0] else np.ones((s1 - s0, K), dtype=np.float32)
            phys_np, _ = self.arena.resolve_and_prefetch(
                a_i, ind_np[s0:s1], sc_slice, miss_tail_tau=1.0
            )
            phys = mx.array(phys_np)
            gu = mx.gather_qmm(
                x_2d[s0:s1][:, None, None, :],
                self.arena.gu_w[a_i], self.arena.gu_s[a_i], self.arena.gu_b[a_i],
                rhs_indices=phys, transpose=True, group_size=self.arena.group_size, bits=4,
            )
            if self.custom_act is not None:
                act = self.custom_act(gu[..., inter:], gu[..., :inter]).astype(mx.float16)
            else:
                act = nn.silu(gu[..., :inter]) * gu[..., inter:]
            dn = mx.gather_qmm(
                act,
                self.arena.dn_w[a_i], self.arena.dn_s[a_i], self.arena.dn_b[a_i],
                rhs_indices=phys, transpose=True, group_size=self.arena.group_size, bits=4,
            ).squeeze(-2)
            mx.eval(dn)
            outs.append(dn)
        res = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        return res.reshape(*orig_prefix, K, D).astype(x.dtype)


def _replace_layer_switch_glu(
    module: Any, arena_idx: int, arena: PagedExpertArena, eng: "DeimosMLXUniversalMoEEngine"
) -> bool:
    """Recursively replaces the routed `SwitchGLU` child inside a single decoder layer."""
    for attr_k, child in list(module.children().items()):
        if isinstance(child, SwitchGLU):
            c_act = getattr(child, "activation", None)
            if c_act is not None and type(c_act).__name__ == "SwiGLU":
                c_act = None
            module[attr_k] = DeimosMLXSwitchGLU(arena_idx, arena, eng, custom_act=c_act)
            return True
        if isinstance(child, nn.Module) and _replace_layer_switch_glu(child, arena_idx, arena, eng):
            return True
    return False


class DeimosMLXUniversalMoEEngine:
    """Native Apple Silicon `MLX` engine for universal `hf_hybrid_q4` Sparse & Hybrid MoE stores."""

    backend_name: str = "mlx_universal"
    accepts_mx_tokens: bool = True

    def __init__(
        self,
        store_dir: Path,
        slots_per_layer: int = 16,
        verify_merkle_on_cold: bool = True,
        use_nocache_pread: bool = False,
        enable_lookahead: bool = True,
        enable_merkle_kv: bool = False,
        miss_tail_tau: float = 1.0,
        min_k: int = 4,
        top_k: Optional[int] = None,
        backbone: str = "fp16",
    ):
        if not HAS_MLX:
            raise RuntimeError("DeimosMLXUniversalMoEEngine requires Apple Silicon with `mlx` & `mlx-lm`.")

        self.store_dir = Path(store_dir).expanduser().resolve()
        self.backbone: str = backbone
        self.config: Dict[str, Any] = json.loads((self.store_dir / "config.json").read_text())
        self.manifest: Dict[str, Any] = json.loads((self.store_dir / "merkle_manifest.json").read_text())

        self.num_layers = int(self.config["num_hidden_layers"])
        self.moe_layer_ids: List[int] = list(self.manifest.get("moe_layer_ids", range(self.num_layers)))
        self.num_experts = int(self.config["num_experts"])
        self.hidden_size = int(self.config["hidden_size"])
        self.vocab_size = int(self.config["vocab_size"])
        default_k = int(self.config.get("num_experts_per_tok") or self.config.get("num_experts_per_token") or 8)
        self.top_k = max(1, min(self.num_experts, int(top_k or default_k)))
        self.enable_lookahead = bool(enable_lookahead)
        self.enable_merkle_kv = False
        self.miss_tail_tau = float(miss_tail_tau)
        self.min_k = max(1, min(self.top_k, int(min_k)))

        tok_p = self.store_dir / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tok_p)) if (Tokenizer is not None and tok_p.exists()) else None

        self.bridge = DeimosMerkleBridge(
            self.store_dir, cas_filename="cas_experts_q4.bin", enable_nocache_fd=True
        )
        self.arena = PagedExpertArena(
            self.bridge,
            slots_per_layer=max(self.top_k, int(slots_per_layer)),
            verify_merkle_on_cold=verify_merkle_on_cold,
            use_nocache_pread=use_nocache_pread,
        )

        mtype = str(self.config.get("model_type", ""))
        cfg_for_mlx = dict(self.config)
        rp = cfg_for_mlx.get("rope_parameters") or {}
        if isinstance(rp, dict):
            cfg_for_mlx.setdefault("rope_theta", rp.get("rope_theta", 10000.0))
            cfg_for_mlx.setdefault("partial_rotary_factor", rp.get("partial_rotary_factor", 1.0))
            if "rope_scaling" not in cfg_for_mlx:
                cfg_for_mlx["rope_scaling"] = rp if rp.get("rope_type", "default") != "default" else None
        cfg_for_mlx.setdefault("rope_theta", 10000.0)
        cfg_for_mlx.setdefault("rope_scaling", None)
        cfg_for_mlx.setdefault("partial_rotary_factor", 1.0)
        cfg_for_mlx.setdefault("attention_chunk_size", cfg_for_mlx.get("attention_chunk_size") or 8192)
        cfg_for_mlx.setdefault("model_max_length", cfg_for_mlx.get("max_position_embeddings", 32768))
        cfg_for_mlx["tie_word_embeddings"] = False
        if cfg_for_mlx.get("rotary_dim") and cfg_for_mlx.get("head_dim"):
            cfg_for_mlx["rotary_dim"] = min(int(cfg_for_mlx["rotary_dim"]), int(cfg_for_mlx["head_dim"]))
        if mtype == "kimi_linear" and "linear_attn_config" not in cfg_for_mlx:
            ltypes = cfg_for_mlx.get("layer_types") or []
            cfg_for_mlx["linear_attn_config"] = {
                "num_heads": int(cfg_for_mlx.get("linear_num_heads", cfg_for_mlx["num_attention_heads"])),
                "head_dim": int(cfg_for_mlx.get("linear_head_dim") or cfg_for_mlx.get("linear_head_k_dim", 128)),
                "short_conv_kernel_size": int(cfg_for_mlx.get("linear_conv_kernel_dim", 4)),
                "kda_layers": [i + 1 for i, t in enumerate(ltypes) if t != "full_attention"],
            }
        if mtype in {"llama4_text", "llama4"}:
            from mlx_lm.models import llama4 as mlx_llama4

            m_args = mlx_llama4.ModelArgs(text_config=cfg_for_mlx, model_type="llama4")
            model: Any = mlx_llama4.Model(m_args)
        elif mtype in {"minimax_m3_vl", "minimax_m3_vl_text"}:
            from mlx_lm.models import minimax_m3_vl as mlx_mm3

            m_args = mlx_mm3.ModelArgs.from_dict({"text_config": cfg_for_mlx, "model_type": "minimax_m3_vl"})
            model = mlx_mm3.Model(m_args)
        else:
            model_cls, args_cls = mlx_get_classes(cfg_for_mlx)
            m_args = args_cls.from_dict(cfg_for_mlx)
            model = model_cls(m_args)

        lm_wrap = getattr(model, "language_model", model)
        inner = getattr(lm_wrap, "model", lm_wrap)
        layers_seq = getattr(model, "layers", getattr(lm_wrap, "layers", getattr(inner, "layers", None)))
        for a_idx, l_idx in enumerate(self.moe_layer_ids):
            if not _replace_layer_switch_glu(layers_seq[l_idx], a_idx, self.arena, self):
                raise RuntimeError(f"Could not locate MLX SwitchGLU on layer {l_idx} (model_type={mtype!r}).")

        self._mmap_emb = DeimosMLXMmapEmbedding(
            self.store_dir / "embed_tokens.bin", self.vocab_size, self.hidden_size
        )
        if hasattr(inner, "embed_tokens"):
            inner.embed_tokens = self._mmap_emb

        lm_bits = 4 if (self.backbone == "q4" and self.hidden_size % 64 == 0) else 8
        self.lm_head = DeimosMLXInt8LMHead(
            self.store_dir / "lm_head_int8.bin",
            self.vocab_size, self.hidden_size, bits=lm_bits, group_size=64,
        )
        self._mmap_emb.as_linear = self.lm_head  # type: ignore[attr-defined]
        if hasattr(lm_wrap, "lm_head"):
            lm_wrap.lm_head = self.lm_head
        if hasattr(model, "args") and hasattr(model.args, "tie_word_embeddings"):
            model.args.tie_word_embeddings = False

        raw_bb: Dict[str, mx.array] = mx.load(str(self.store_dir / "hf_backbone.safetensors"))
        if mtype in {"minimax_m3_vl", "minimax_m3_vl_text"}:
            moe_set = set(self.moe_layer_ids)
            remapped: Dict[str, mx.array] = {}
            for k, v in raw_bb.items():
                nk = f"language_model.{k}" if k.startswith("model.") else k
                for li in moe_set:
                    nk = nk.replace(f".layers.{li}.mlp.", f".layers.{li}.block_sparse_moe.")
                nk = nk.replace(".block_sparse_moe.gate.e_score_correction_bias", ".block_sparse_moe.e_score_correction_bias")
                if nk.endswith(".gate_up_proj.weight") and ".shared_experts." not in nk:
                    g_part, u_part = mx.split(v, 2, axis=0)
                    stem = nk[: -len(".gate_up_proj.weight")]
                    remapped[f"{stem}.gate_proj.weight"] = mx.contiguous(g_part)
                    remapped[f"{stem}.up_proj.weight"] = mx.contiguous(u_part)
                else:
                    remapped[nk] = v
            raw_bb = remapped
        elif mtype == "kimi_linear":
            remapped_k: Dict[str, mx.array] = {}
            for k, v in raw_bb.items():
                nk = k.replace(".self_attn.forget_gate.", ".self_attn.")
                nk = nk.replace(".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias")
                if nk.endswith(".self_attn.conv1d.weight") and int(v.shape[0]) % 3 == 0:
                    kd = int(v.shape[0]) // 3
                    stem = nk[: -len(".conv1d.weight")]
                    remapped_k[f"{stem}.q_conv1d.weight"] = mx.contiguous(v[:kd])
                    remapped_k[f"{stem}.k_conv1d.weight"] = mx.contiguous(v[kd : 2 * kd])
                    remapped_k[f"{stem}.v_conv1d.weight"] = mx.contiguous(v[2 * kd :])
                else:
                    remapped_k[nk] = v
            raw_bb = remapped_k

        if hasattr(model, "sanitize") and mtype not in {"llama4_text", "llama4"}:
            raw_bb = model.sanitize(raw_bb)
        clean_bb: Dict[str, mx.array] = {}
        f32_keep = ("e_score_correction_bias", "expert_bias", "A_log", "dt_bias")
        for k, v in raw_bb.items():
            if k.startswith(("model.embed_tokens.", "embed_tokens.", "lm_head.", "language_model.lm_head.")):
                continue
            if mtype in {"llama4_text", "llama4"} and k.startswith("model."):
                k = f"language_model.{k}"
            if v.dtype in (mx.float32, mx.bfloat16) and not k.endswith(f32_keep):
                v = v.astype(mx.float16)
            clean_bb[k] = v
        raw_bb.clear()

        model.load_weights(list(clean_bb.items()), strict=False)
        clean_bb.clear()

        if self.backbone == "q4":
            def _q_pred(path: str, mod: Any) -> bool:
                low = path.lower()
                if any(s in low for s in ("gate", "router", "lm_head", "embed", "norm")):
                    if not low.endswith(("gate_proj", "up_proj", "down_proj")):
                        return False
                return isinstance(mod, nn.Linear) and mod.weight.shape[-1] >= 128 and mod.weight.shape[-1] % 64 == 0

            nn.quantize(model, group_size=64, bits=4, class_predicate=_q_pred)

        model.eval()
        mx.eval(model.parameters())
        self.model = model
        gc.collect()
        mx.clear_cache()
        self.reset_step_memory()

    def make_caches(self) -> List[Any]:
        return list(make_prompt_cache(self.model))

    def reset_step_memory(self) -> None:
        reset_fn = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
        reset_fn()

    def vram_mb(self) -> Tuple[float, float]:
        return _get_active_and_peak_mb()

    def sample_next_token(self, logits_2d: "mx.array", temperature: float = 0.0) -> int:
        if temperature <= 1e-5:
            t = mx.argmax(logits_2d, axis=-1)
        else:
            t = mx.random.categorical(logits_2d.astype(mx.float32) * (1.0 / temperature))
        mx.eval(t)
        return int(t.item())

    def forward_logits(
        self,
        token_ids: Any,
        caches: Optional[List[Any]] = None,
        all_positions: bool = False,
    ) -> "mx.array":
        inp = (
            token_ids.reshape(1, -1)
            if (mx is not None and isinstance(token_ids, mx.array))
            else mx.array([list(token_ids)], dtype=mx.int32)
        )
        logits = self.model(inp, cache=caches)
        out = logits if all_positions else logits[:, -1, :]
        mx.eval(out)
        return out

    def prefill_prompt(self, prompt_ids: Sequence[int], caches: List[Any]) -> Tuple["mx.array", int]:
        return self.forward_logits(prompt_ids, caches=caches), 0

    def benchmark_token_ids(self, prompt_ids: Sequence[int], max_new_tokens: int = 32) -> BenchmarkMetrics:
        caches = self.make_caches()
        self.arena.reset_telemetry()
        self.reset_step_memory()

        t_pf = time.perf_counter()
        logits, skipped = self.prefill_prompt(prompt_ids, caches)
        tok = self.sample_next_token(logits)
        ttft_ms = (time.perf_counter() - t_pf) * 1000.0

        self.arena.reset_telemetry()
        gen_ids = [tok]
        t_dec = time.perf_counter()
        for _ in range(max_new_tokens):
            logits = self.forward_logits([gen_ids[-1]], caches=caches)
            gen_ids.append(self.sample_next_token(logits))
        decode_sec = max(1e-6, time.perf_counter() - t_dec)
        self.arena.wait_idle()

        act_mb, peak_mb = self.vram_mb()
        st: ArenaStats = self.arena.get_stats(nominal_top_k=self.top_k)
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
            "lookahead_staged": st["lookahead_staged"],
            "lookahead_useful_hits": st["lookahead_useful_hits"],
            "tail_misses_skipped": st["tail_misses_skipped"],
            "avg_active_k": st["avg_active_k"],
            "xor_proofs_verified": st["xor_proofs_verified"],
            "byte_proofs_verified": st["byte_proofs_verified"],
            "dma_bytes_staged_mb": st["dma_bytes_staged_mb"],
            "generated_ids": gen_ids,
            "generated_text": self.tokenizer.decode(gen_ids) if self.tokenizer is not None else "",
        }

    def benchmark_generation(self, prompt: str, max_new_tokens: int = 32) -> BenchmarkMetrics:
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer.json found in store; use benchmark_token_ids().")
        return self.benchmark_token_ids(self.tokenizer.encode(prompt).ids, max_new_tokens=max_new_tokens)

    def close(self) -> None:
        self.arena.close()
        self._mmap_emb.close()
        del self.model
        gc.collect()
        mx.clear_cache()
        self.bridge.close()
