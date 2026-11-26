#!/usr/bin/env python3
"""
DeimosStream Native Apple Silicon `MLX` Hybrid-MoE Engine (`src/deimos_stream/mlx_hybrid.py`)
=============================================================================================
Runs 2026 HuggingFace Hybrid Sparse MoE stores (`store_kind == "hf_hybrid_q4"`, e.g.
`Qwen/Qwen3.6-35B-A3B` — 35B params, 66.97 GiB BF16, 40 GatedDeltaNet + GQA layers × 256
experts = 10,240 experts) natively on Apple Silicon Metal GPUs using the EXACT SAME universal
`DEIMIDX2` Q4 store as the NVIDIA CUDA Triton backend:
  1. Uses `mlx_lm.models.qwen3_5.TextModel` for native Metal `gated_delta_update` (linear
     attention) + `Qwen3NextAttention` (GQA + per-head QK-Norm) + `ArraysCache`/`KVCache`.
  2. Replaces `SwitchGLU` BEFORE evaluation (`0` bytes allocated for unpaged experts) with
     `DeimosMLXPagedSparseMoeBlock` backed by zero-copy Unified-Memory `PagedExpertArena`
     (`mx.gather_qmm`), Zero-GIL C++17 Cross-Layer Residual Lookahead, and Miss-Only Dynamic-K.
  3. Replaces `embed_tokens` with zero-VRAM CPU `mmap` lookup (`DeimosMLXMmapEmbedding`) and
     `lm_head` with vocab-tiled INT8 `DeimosMLXInt8LMHead` (`485 MiB` for 248K vocab).
"""

from __future__ import annotations

import gc
import json
import mmap
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deimos_stream.arena import ArenaStats, DeimosMerkleBridge, PagedExpertArena
from deimos_stream.backbone import rss_mb
from deimos_stream.engine import HAS_MLX, BenchmarkMetrics

if HAS_MLX:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import TextModel as Qwen35TextModel
    from mlx_lm.models.qwen3_5 import TextModelArgs as Qwen35TextModelArgs
    from mlx_lm.models.qwen3_next import Qwen3NextMLP

    _NNModule = nn.Module
else:
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    Qwen35TextModel = None  # type: ignore[assignment,misc]
    Qwen35TextModelArgs = None  # type: ignore[assignment,misc]
    Qwen3NextMLP = None  # type: ignore[assignment,misc]
    _NNModule = object  # type: ignore[assignment,misc]

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None  # type: ignore[assignment,misc]


def _get_active_and_peak_mb() -> Tuple[float, float]:
    act_fn = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
    peak_fn = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    return float(act_fn()) / (1024.0 * 1024.0), float(peak_fn()) / (1024.0 * 1024.0)


def _swiglu_split(gu: "mx.array") -> "mx.array":
    gate, up = mx.split(gu, 2, axis=-1)
    return nn.silu(gate) * up


class DeimosMLXMmapEmbedding(_NNModule):  # type: ignore[misc,valid-type]
    """Zero-VRAM `mmap` token embedding table slicing rows directly from `embed_tokens.bin`."""

    def __init__(self, emb_path: Path, vocab_size: int, hidden_size: int):
        super().__init__()
        self._file = open(emb_path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self.view = np.frombuffer(
            self._mmap, dtype=np.float16, count=vocab_size * hidden_size
        ).reshape(vocab_size, hidden_size)

    def __call__(self, input_ids: "mx.array") -> "mx.array":
        ids_np = np.asarray(input_ids, dtype=np.int64)
        sliced = np.ascontiguousarray(self.view[ids_np])
        return mx.array(sliced)

    def close(self) -> None:
        self.view = np.empty((0, 0), dtype=np.float16)
        self._mmap.close()
        self._file.close()


class DeimosMLXInt8LMHead(_NNModule):  # type: ignore[misc,valid-type]
    """Vocab-tiled INT8 `lm_head` over `lm_head_int8.bin` matching PyTorch `DeimosInt8LMHead`."""

    def __init__(
        self, lm_path: Path, vocab_size: int, hidden_size: int, tile_vocab: int = 32768
    ):
        super().__init__()
        w_nbytes = vocab_size * hidden_size
        with open(lm_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                w_i8 = np.frombuffer(mm, dtype=np.int8, count=w_nbytes).reshape(
                    vocab_size, hidden_size
                ).copy()
                s_f16 = np.frombuffer(
                    mm, dtype=np.float16, count=vocab_size, offset=w_nbytes
                ).reshape(vocab_size, 1).copy()
            finally:
                mm.close()
        self.w_i8 = mx.array(w_i8)
        self.scale = mx.array(s_f16)
        self.vocab_size = int(vocab_size)
        self.tile_vocab = int(tile_vocab)
        mx.eval(self.w_i8, self.scale)

    def __call__(self, x: "mx.array") -> "mx.array":
        orig_shape = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1]).astype(mx.float16)
        tiles: List[mx.array] = []
        for v0 in range(0, self.vocab_size, self.tile_vocab):
            v1 = min(self.vocab_size, v0 + self.tile_vocab)
            w_tile = self.w_i8[v0:v1].astype(mx.float16) * self.scale[v0:v1]
            tiles.append(x2 @ w_tile.T)
        return mx.concatenate(tiles, axis=-1).reshape(*orig_shape, self.vocab_size)


class DeimosMLXPagedSparseMoeBlock(_NNModule):  # type: ignore[misc,valid-type]
    """
    Drop-in zero-copy `PagedExpertArena` (`mx.gather_qmm`) replacement for `Qwen3NextSparseMoeBlock`.
    Preserves exact `gate`, `shared_expert`, and `shared_expert_gate` parameter paths for `load_weights`.
    """

    def __init__(
        self,
        args: Any,
        layer_idx: int,
        arena: PagedExpertArena,
        engine_ref: "DeimosMLXHybridMoEEngine",
    ):
        super().__init__()
        dim = int(args.hidden_size)
        num_experts = int(args.num_experts)
        shared_inter = int(args.shared_expert_intermediate_size)

        self.layer_idx = int(layer_idx)
        self.norm_topk_prob = bool(args.norm_topk_prob)
        self.num_experts = num_experts
        self.top_k = int(engine_ref.top_k)
        self.arena = arena
        self.engine_ref = engine_ref

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.shared_expert = Qwen3NextMLP(dim, shared_inter)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)
        self.next_norm: Optional[Any] = None
        self.next_gate: Optional[Any] = None

    def _run_q4_chunk(
        self, x_chunk: "mx.array", phys_np: np.ndarray, sc_chunk: "mx.array"
    ) -> "mx.array":
        l_i = self.layer_idx
        phys_mx = mx.array(phys_np)
        x_exp = mx.expand_dims(x_chunk, (-2, -3))
        gu_out = mx.gather_qmm(
            x_exp,
            self.arena.gu_w[l_i],
            self.arena.gu_s[l_i],
            self.arena.gu_b[l_i],
            rhs_indices=phys_mx,
            transpose=True,
            group_size=self.arena.group_size,
            bits=4,
        )
        act = _swiglu_split(gu_out)
        dn_out = mx.gather_qmm(
            act,
            self.arena.dn_w[l_i],
            self.arena.dn_s[l_i],
            self.arena.dn_b[l_i],
            rhs_indices=phys_mx,
            transpose=True,
            group_size=self.arena.group_size,
            bits=4,
        )
        return (dn_out.squeeze(-2) * sc_chunk[..., None]).sum(axis=-2)

    def _route_flat(self, x_flat: "mx.array") -> Tuple["mx.array", "mx.array"]:
        logits = self.gate(x_flat)
        probs = mx.softmax(logits, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(probs, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(probs, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.maximum(scores.sum(axis=-1, keepdims=True), mx.array(1e-12))
        return inds.astype(mx.int32), scores.astype(x_flat.dtype)

    def __call__(self, x: "mx.array") -> "mx.array":
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        N = B * L
        eng = self.engine_ref

        inds, scores = self._route_flat(x_flat)
        shared_y = mx.sigmoid(self.shared_expert_gate(x_flat)) * self.shared_expert(x_flat)

        if self.arena.is_fully_pinned:
            mx.eval(inds, scores)
            y = self._run_q4_chunk(x_flat, np.asarray(inds, dtype=np.int32), scores)
            return (y + shared_y).reshape(B, L, D)

        is_single = N == 1
        has_la = (
            eng.enable_lookahead
            and is_single
            and (self.layer_idx + 1 < eng.num_layers)
            and self.next_gate is not None
        )
        pred_np: Optional[np.ndarray] = None
        if has_la:
            la_in = self.next_norm(x_flat) if self.next_norm is not None else x_flat
            la_probs = mx.softmax(self.next_gate(la_in), axis=-1, precise=True)
            la_inds = mx.argpartition(la_probs, kth=-self.top_k, axis=-1)[..., -self.top_k :].astype(mx.int32)
            mx.eval(inds, scores, la_inds)
            pred_np = np.asarray(la_inds, dtype=np.int32)
        else:
            mx.eval(inds, scores)

        ind_np = np.asarray(inds, dtype=np.int32)
        sc_np = np.asarray(scores.astype(mx.float32))
        max_chunk = min(4, max(1, self.arena.slots_per_layer // max(1, self.top_k)))

        if N <= max_chunk:
            phys_np, eff_sc_np = self.arena.resolve_and_prefetch(
                self.layer_idx,
                ind_np,
                sc_np,
                miss_tail_tau=eng.miss_tail_tau if is_single else 1.0,
                min_k=eng.min_k,
                next_layer_idx=(self.layer_idx + 1) if has_la else -1,
                next_pred_ids_np=pred_np,
            )
            eff_sc = (
                scores
                if (not is_single or eng.miss_tail_tau >= 0.9999)
                else mx.array(eff_sc_np).astype(x_flat.dtype)
            )
            y = self._run_q4_chunk(x_flat, phys_np, eff_sc)
            out = (y + shared_y).reshape(B, L, D)
            # Ensure Metal finishes reading unified arena[l] before any subsequent token evicts slots
            mx.eval(out)
            return out

        outs: List[mx.array] = []
        for s0 in range(0, N, max_chunk):
            s1 = min(N, s0 + max_chunk)
            phys_np, _ = self.arena.resolve_and_prefetch(
                self.layer_idx, ind_np[s0:s1], sc_np[s0:s1], miss_tail_tau=1.0
            )
            ch_y = self._run_q4_chunk(x_flat[s0:s1], phys_np, scores[s0:s1])
            mx.eval(ch_y)
            outs.append(ch_y)
        return (mx.concatenate(outs, axis=0) + shared_y).reshape(B, L, D)


class DeimosMLXHybridMoEEngine:
    """Native Apple Silicon `MLX` Hybrid-MoE Engine backed by zero-copy `DEIMIDX2` + `PagedExpertArena`."""

    def __init__(
        self,
        store_dir: Path,
        slots_per_layer: int = 16,
        verify_merkle_on_cold: bool = True,
        use_nocache_pread: bool = True,
        enable_lookahead: bool = True,
        enable_merkle_kv: bool = False,
        miss_tail_tau: float = 1.0,
        min_k: int = 4,
        top_k: Optional[int] = None,
    ):
        if not HAS_MLX:
            raise RuntimeError("Apple Silicon MLX (`mlx` + `mlx-lm`) required for DeimosMLXHybridMoEEngine.")

        self.store_dir = Path(store_dir).expanduser().resolve()
        self.backend_name: str = "mlx_hybrid"
        self.config: Dict[str, object] = json.loads((self.store_dir / "config.json").read_text())
        self.manifest: Dict[str, object] = json.loads(
            (self.store_dir / "merkle_manifest.json").read_text()
        )
        self.num_layers = int(self.config["num_hidden_layers"])  # type: ignore[arg-type]
        self.num_experts = int(self.config["num_experts"])  # type: ignore[arg-type]
        self.hidden_size = int(self.config["hidden_size"])  # type: ignore[arg-type]
        self.vocab_size = int(self.config["vocab_size"])  # type: ignore[arg-type]
        default_k = int(self.config.get("num_experts_per_tok", 8))  # type: ignore[arg-type]
        self.top_k = max(1, min(self.num_experts, int(top_k or default_k)))
        self.enable_lookahead = bool(enable_lookahead)
        self.enable_merkle_kv = False  # Hybrid GatedDeltaNet uses ArraysCache + KVCache
        self.miss_tail_tau = float(miss_tail_tau)
        self.min_k = max(1, min(self.top_k, int(min_k)))

        tok_p = self.store_dir / "tokenizer.json"
        self.tokenizer = (
            Tokenizer.from_file(str(tok_p)) if (Tokenizer is not None and tok_p.exists()) else None
        )

        try:
            set_lim = getattr(mx, "set_cache_limit", None) or getattr(mx.metal, "set_cache_limit", None)
            if set_lim is not None:
                set_lim(64 * 1024 * 1024)
        except Exception:
            pass

        self.bridge = DeimosMerkleBridge(
            self.store_dir, cas_filename="cas_experts_q4.bin", enable_nocache_fd=True
        )
        self.arena = PagedExpertArena(
            self.bridge,
            slots_per_layer=max(self.top_k, int(slots_per_layer)),
            verify_merkle_on_cold=verify_merkle_on_cold,
            use_nocache_pread=use_nocache_pread,
        )

        # Build lazy MLX Qwen3.5/3.6 TextModel skeleton and swap unpaged modules BEFORE mx.eval()
        t_args = Qwen35TextModelArgs.from_dict(dict(self.config))
        model = Qwen35TextModel(t_args)

        self._mmap_emb = DeimosMLXMmapEmbedding(
            self.store_dir / "embed_tokens.bin", self.vocab_size, self.hidden_size
        )
        model.model.embed_tokens = self._mmap_emb
        self._int8_head = DeimosMLXInt8LMHead(
            self.store_dir / "lm_head_int8.bin", self.vocab_size, self.hidden_size
        )
        model.lm_head = self._int8_head

        paged_mlps: List[DeimosMLXPagedSparseMoeBlock] = []
        for l_i in range(self.num_layers):
            p_mlp = DeimosMLXPagedSparseMoeBlock(t_args, l_i, self.arena, self)
            model.model.layers[l_i].mlp = p_mlp
            paged_mlps.append(p_mlp)

        raw_bb = mx.load(str(self.store_dir / "hf_backbone.safetensors"))
        san_bb = model.sanitize(dict(raw_bb))  # type: ignore[arg-type]
        raw_bb.clear()  # type: ignore[union-attr]
        # Cast non-A_log weights cleanly to float16 on Apple Silicon Metal
        cast_bb = {
            k: (v.astype(mx.float16) if (v.dtype == mx.bfloat16 or (v.dtype == mx.float32 and not k.endswith("A_log"))) else v)
            for k, v in san_bb.items()
        }
        model.load_weights(list(cast_bb.items()), strict=False)
        mx.eval(model.parameters())
        cast_bb.clear()
        san_bb.clear()

        for l_i in range(self.num_layers - 1):
            nxt = model.model.layers[l_i + 1]
            paged_mlps[l_i].next_norm = nxt.post_attention_layernorm
            paged_mlps[l_i].next_gate = nxt.mlp.gate

        self.model = model
        gc.collect()
        self.reset_step_memory()

    def make_caches(self) -> List[Any]:
        return self.model.make_cache()

    def reset_step_memory(self) -> None:
        mx.clear_cache()
        mx.reset_peak_memory()

    def vram_mb(self) -> Tuple[float, float]:
        return _get_active_and_peak_mb()

    def sample_next_token(self, logits_2d: "mx.array", temperature: float = 0.0) -> int:
        if temperature <= 1e-5:
            tok = mx.argmax(logits_2d, axis=-1)
        else:
            tok = mx.random.categorical(logits_2d.astype(mx.float32) / temperature)
        mx.eval(tok)
        return int(tok.item())

    def forward_logits(
        self,
        token_ids: Sequence[int],
        caches: Optional[List[Any]] = None,
        all_positions: bool = False,
    ) -> "mx.array":
        inp = mx.array([list(token_ids)], dtype=mx.int32)
        h = self.model.model(inp, cache=caches)
        if not all_positions:
            h = h[:, -1:, :]
        logits = self.model.lm_head(h)
        mx.eval(logits)
        return logits if all_positions else logits[:, -1, :]

    def prefill_prompt(
        self, prompt_ids: Sequence[int], caches: List[Any]
    ) -> Tuple["mx.array", int]:
        return self.forward_logits(prompt_ids, caches=caches, all_positions=False), 0

    def benchmark_token_ids(
        self, prompt_ids: Sequence[int], max_new_tokens: int = 32
    ) -> BenchmarkMetrics:
        caches = self.make_caches()
        self.arena.reset_telemetry()
        self.reset_step_memory()

        t_pf = time.perf_counter()
        logits, skipped = self.prefill_prompt(prompt_ids, caches)
        tok = self.sample_next_token(logits, temperature=0.0)
        ttft_ms = (time.perf_counter() - t_pf) * 1000.0

        self.arena.reset_telemetry()
        gen_ids = [tok]
        t_dec = time.perf_counter()
        for _ in range(max_new_tokens):
            logits = self.forward_logits([gen_ids[-1]], caches=caches)
            gen_ids.append(self.sample_next_token(logits, temperature=0.0))
        decode_sec = max(1e-6, time.perf_counter() - t_dec)
        self.arena.wait_idle()

        act_mb, peak_mb = self.vram_mb()
        st: ArenaStats = self.arena.get_stats(nominal_top_k=self.top_k)
        decoded_text = self.tokenizer.decode(gen_ids) if self.tokenizer is not None else ""
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
            "generated_text": decoded_text,
        }

    def benchmark_generation(self, prompt: str, max_new_tokens: int = 32) -> BenchmarkMetrics:
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer.json found in store; use benchmark_token_ids().")
        return self.benchmark_token_ids(
            self.tokenizer.encode(prompt).ids, max_new_tokens=max_new_tokens
        )

    def close(self) -> None:
        self.arena.close()
        self._mmap_emb.close()
        del self.model
        gc.collect()
        mx.clear_cache()
        self.bridge.close()
