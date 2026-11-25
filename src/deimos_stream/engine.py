#!/usr/bin/env python3
"""
DeimosStream Universal Paged Q4 MoE Inference Engine (`src/deimos_stream/engine.py`)
====================================================================================
Orchestrates:
  - `PagedExpertArena` (`mx.gather_qmm` zero-copy hits + Unified-Memory DMA staging)
  - Zero-GIL C++17 Residual Lookahead Async Prefetch (`W_gate^(l+1) @ RMSNorm(h^(l))`)
  - Cache-Aware Miss-Only Nucleus Dynamic-K (`miss_tail_tau`, default `1.0` bit-exact)
  - `MerkleKVPrefixCache` (`DEIMIDX2` Robin-Hood 16-token KV prefix cache)
"""

from __future__ import annotations

from functools import partial
import gc
import json
import math
import mmap
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
import numpy as np

from deimos_stream.arena import ArenaStats, DeimosMerkleBridge, PagedExpertArena
from deimos_stream.backbone import load_backbone, mlx_mem_mb, rss_mb
from deimos_stream.kv_cache import MerkleKVPrefixCache
from deimos_stream.merkle_builder import convert_fp16_cas_store_to_q4

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None  # type: ignore[assignment,misc]


@mx.compile
def _compiled_swiglu_split(gu_proj: mx.array) -> mx.array:
    gate, up = mx.split(gu_proj, 2, axis=-1)
    return nn.silu(gate) * up


@partial(mx.compile, shapeless=True)
def _compiled_moe_agg(expert_out: mx.array, scores: mx.array) -> mx.array:
    return (expert_out * scores[..., None]).sum(axis=-2)


def _make_compiled_router(top_k: int, norm_topk_prob: bool):
    @mx.compile
    def _route(router_logits: mx.array) -> Tuple[mx.array, mx.array]:
        probs = mx.softmax(router_logits, axis=-1, precise=True)
        inds = mx.argpartition(-probs, kth=top_k - 1, axis=-1)[..., :top_k]
        sc = mx.take_along_axis(probs, inds, axis=-1)
        if norm_topk_prob:
            sc = sc / mx.maximum(mx.sum(sc, axis=-1, keepdims=True), 1e-12)
        return inds, sc

    return _route


class BenchmarkMetrics(TypedDict):
    slots_per_layer: int
    enable_lookahead: bool
    miss_tail_tau: float
    use_nocache_pread: bool
    active_vram_mb: float
    peak_vram_mb: float
    rss_mb: float
    pool_vram_mb: float
    kv_skipped_tokens: int
    ttft_ms: float
    decode_tok_per_s: float
    hit_rate_pct: float
    cache_hits: int
    cache_misses: int
    evictions: int
    lookahead_staged: int
    lookahead_useful_hits: int
    tail_misses_skipped: int
    avg_active_k: float
    xor_proofs_verified: int
    byte_proofs_verified: int
    dma_bytes_staged_mb: float
    generated_ids: List[int]
    generated_text: str


class DeimosPagedMoEEngine:
    """Universal DeimosStream Paged Q4 Sparse MoE Inference Engine."""

    def __init__(
        self,
        store_dir: Path,
        slots_per_layer: int = 16,
        verify_merkle_on_cold: bool = True,
        use_nocache_pread: bool = False,
        enable_lookahead: bool = True,
        enable_merkle_kv: bool = True,
        miss_tail_tau: float = 1.0,
        min_k: int = 4,
        top_k: Optional[int] = None,
    ):
        raw_dir = Path(store_dir).expanduser().resolve()
        self.store_dir: Path = (
            raw_dir
            if (raw_dir / "cas_experts_q4.bin").exists()
            else convert_fp16_cas_store_to_q4(raw_dir)
        )
        self.config: Dict[str, object] = json.loads((self.store_dir / "config.json").read_text())
        self.manifest: Dict[str, object] = json.loads(
            (self.store_dir / "merkle_manifest.json").read_text()
        )
        self.num_layers: int = int(self.config["num_hidden_layers"])  # type: ignore[arg-type]
        self.num_experts: int = int(self.config["num_experts"])  # type: ignore[arg-type]
        self.hidden_size: int = int(self.config["hidden_size"])  # type: ignore[arg-type]
        self.intermediate_size: int = int(self.config["intermediate_size"])  # type: ignore[arg-type]
        self.num_heads: int = int(self.config["num_attention_heads"])  # type: ignore[arg-type]
        self.num_kv_heads: int = int(self.config.get("num_key_value_heads", self.num_heads))  # type: ignore[arg-type]
        self.head_dim: int = int(self.config.get("head_dim", self.hidden_size // self.num_heads))  # type: ignore[arg-type]
        self.rms_eps: float = float(self.config["rms_norm_eps"])  # type: ignore[arg-type]
        self.rope_theta: float = float(self.config.get("rope_theta", 10000.0))  # type: ignore[arg-type]
        self.vocab_size: int = int(self.config["vocab_size"])  # type: ignore[arg-type]
        self.norm_topk_prob: bool = bool(self.config.get("norm_topk_prob", False))
        default_k = int(self.config["num_experts_per_tok"])  # type: ignore[arg-type]
        self.top_k: int = max(1, min(self.num_experts, int(top_k or default_k)))
        self.enable_lookahead: bool = bool(enable_lookahead)
        self.enable_merkle_kv: bool = bool(enable_merkle_kv)
        self.miss_tail_tau: float = float(miss_tail_tau)
        self.min_k: int = max(1, min(self.top_k, int(min_k)))

        tok_path = self.store_dir / "tokenizer.json"
        self.tokenizer = (
            Tokenizer.from_file(str(tok_path))
            if (Tokenizer is not None and tok_path.exists())
            else None
        )

        try:
            set_lim = getattr(mx, "set_cache_limit", None) or getattr(
                mx.metal, "set_cache_limit", None
            )
            if set_lim is not None:
                set_lim(32 * 1024 * 1024)
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
        self.kv_cache = MerkleKVPrefixCache(
            self.store_dir,
            self.bridge,
            num_layers=self.num_layers,
            seed_root_hex=str(self.manifest["model_merkle_root"]),
        )

        self._emb_file = open(self.store_dir / "embed_tokens.bin", "rb")
        self._emb_mmap = mmap.mmap(self._emb_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.emb_np_view: np.ndarray = np.frombuffer(
            self._emb_mmap, dtype=np.float16, count=self.vocab_size * self.hidden_size
        ).reshape(self.vocab_size, self.hidden_size)

        (
            self.in_norms,
            self.post_norms,
            self.attns,
            self.routers,
            self.shared_mlps,
            self.final_norm,
            (self.lm_head_qw, self.lm_head_qs, self.lm_head_qb),
        ) = load_backbone(
            self.store_dir,
            self.num_layers,
            self.num_experts,
            self.hidden_size,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.vocab_size,
            self.rms_eps,
            self.rope_theta,
        )
        self.route_fn = _make_compiled_router(self.top_k, self.norm_topk_prob)
        mx.clear_cache()
        mx.reset_peak_memory()

    def _gather_moe_q4(
        self, l_idx: int, x_flat: mx.array, phys_idx: mx.array, scores: mx.array
    ) -> mx.array:
        x_exp = mx.expand_dims(x_flat, (-2, -3))
        gu_out = mx.gather_qmm(
            x_exp,
            self.arena.gu_w[l_idx],
            self.arena.gu_s[l_idx],
            self.arena.gu_b[l_idx],
            rhs_indices=phys_idx,
            transpose=True,
            group_size=self.arena.group_size,
            bits=4,
        )
        act = _compiled_swiglu_split(gu_out)
        dn_out = mx.gather_qmm(
            act,
            self.arena.dn_w[l_idx],
            self.arena.dn_s[l_idx],
            self.arena.dn_b[l_idx],
            rhs_indices=phys_idx,
            transpose=True,
            group_size=self.arena.group_size,
            bits=4,
        )
        y = _compiled_moe_agg(dn_out.squeeze(-2), scores)
        sh = self.shared_mlps[l_idx]
        if sh is not None:
            sg, su, sd, sgate = sh
            sh_out = (nn.silu(x_flat @ sg.T) * (x_flat @ su.T)) @ sd.T
            if sgate is not None:
                sh_out = sh_out * mx.sigmoid(x_flat @ sgate.T)
            y = y + sh_out
        return y

    def forward_logits(
        self,
        token_ids: Sequence[int],
        caches: Optional[List[KVCache]] = None,
        all_positions: bool = False,
    ) -> mx.array:
        emb_np = self.emb_np_view[list(token_ids)][None, :, :]
        h = mx.array(emb_np)
        mask = create_attention_mask(h, caches[0] if caches is not None else None)
        is_single_tok = len(token_ids) == 1
        max_toks_per_chunk = max(1, self.arena.slots_per_layer // self.top_k)

        for l_idx in range(self.num_layers):
            c = caches[l_idx] if caches is not None else None
            h_post_attn = h + self.attns[l_idx](self.in_norms[l_idx](h), mask, c)
            normed_post = self.post_norms[l_idx](h_post_attn)
            B, L, D = normed_post.shape
            x_flat = normed_post.reshape(-1, D)
            indices, scores = self.route_fn(x_flat @ self.routers[l_idx].T)

            if self.arena.is_fully_pinned:
                y_moe = self._gather_moe_q4(l_idx, x_flat, mx.stop_gradient(indices), scores)
                h = h_post_attn + y_moe.reshape(B, L, D)
                continue

            has_la = self.enable_lookahead and is_single_tok and (l_idx + 1 < self.num_layers)
            la_ind: Optional[mx.array] = None
            if has_la:
                la_norm = self.post_norms[l_idx + 1](h_post_attn).reshape(-1, D)
                la_ind, _ = self.route_fn(la_norm @ self.routers[l_idx + 1].T)
                mx.eval(x_flat, indices, scores, la_ind)
            else:
                mx.eval(x_flat, indices, scores)

            ind_np = np.asarray(indices, dtype=np.int32)
            sc_np = np.asarray(scores, dtype=np.float32)
            pred_np = np.asarray(la_ind, dtype=np.int32) if la_ind is not None else None

            if ind_np.shape[0] <= max_toks_per_chunk:
                phys_np, eff_sc_np = self.arena.resolve_and_prefetch(
                    l_idx,
                    ind_np,
                    sc_np,
                    miss_tail_tau=self.miss_tail_tau if is_single_tok else 1.0,
                    min_k=self.min_k,
                    next_layer_idx=(l_idx + 1) if has_la else -1,
                    next_pred_ids_np=pred_np,
                )
                phys_mx = mx.array(phys_np, dtype=mx.int32)
                eff_sc_mx = (
                    scores
                    if (not is_single_tok or self.miss_tail_tau >= 0.9999)
                    else mx.array(eff_sc_np.astype(np.float16))
                )
                y_moe = self._gather_moe_q4(l_idx, x_flat, phys_mx, eff_sc_mx)
                h = h_post_attn + y_moe.reshape(B, L, D)
            else:
                chunks_out: List[mx.array] = []
                for start in range(0, ind_np.shape[0], max_toks_per_chunk):
                    end = min(ind_np.shape[0], start + max_toks_per_chunk)
                    phys_np, _ = self.arena.resolve_and_prefetch(
                        l_idx, ind_np[start:end], sc_np[start:end], miss_tail_tau=1.0
                    )
                    c_out = self._gather_moe_q4(
                        l_idx,
                        x_flat[start:end],
                        mx.array(phys_np, dtype=mx.int32),
                        scores[start:end],
                    )
                    mx.eval(c_out)
                    chunks_out.append(c_out)
                h = h_post_attn + mx.concatenate(chunks_out, axis=0).reshape(B, L, D)

        h_out = self.final_norm(h if all_positions else h[:, -1:, :])
        if not self.arena.is_fully_pinned:
            mx.eval(h_out)
        logits = mx.quantized_matmul(
            h_out,
            self.lm_head_qw,
            self.lm_head_qs,
            self.lm_head_qb,
            transpose=True,
            group_size=64,
            bits=8,
        )
        return logits if all_positions else logits.squeeze(1)

    def prefill_prompt(
        self, prompt_ids: Sequence[int], caches: List[KVCache]
    ) -> Tuple[mx.array, int]:
        """Executes prompt prefill using `DEIMIDX2` Merkle KV-Prefix Cache when enabled."""
        skipped = 0
        if self.enable_merkle_kv and len(prompt_ids) >= 17:
            skipped = self.kv_cache.match_longest_prefix(prompt_ids, caches)
        logits = self.forward_logits(prompt_ids[skipped:], caches=caches)
        mx.eval(logits)
        if self.enable_merkle_kv and len(prompt_ids) >= 17:
            self.kv_cache.save_prefix_blocks(prompt_ids, caches)
        return logits, skipped

    def benchmark_token_ids(
        self, prompt_ids: Sequence[int], max_new_tokens: int = 32
    ) -> BenchmarkMetrics:
        """Runs deterministic greedy prefill + decode over `prompt_ids` and collects telemetry."""
        caches = [KVCache() for _ in range(self.num_layers)]
        self.arena.reset_telemetry()
        mx.clear_cache()
        mx.reset_peak_memory()

        t_pf = time.perf_counter()
        logits, skipped = self.prefill_prompt(prompt_ids, caches)
        tok = mx.argmax(logits, axis=-1)
        mx.eval(tok)
        ttft_ms = (time.perf_counter() - t_pf) * 1000.0

        self.arena.reset_telemetry()
        gen_ids = [int(tok.item())]
        t_dec = time.perf_counter()
        for _ in range(max_new_tokens):
            logits = self.forward_logits([gen_ids[-1]], caches=caches)
            tok = mx.argmax(logits, axis=-1)
            mx.eval(tok)
            gen_ids.append(int(tok.item()))
        decode_sec = max(1e-6, time.perf_counter() - t_dec)
        self.arena.wait_idle()

        act_mb, peak_mb = mlx_mem_mb()
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

    def compute_perplexity(
        self, texts: Sequence[str], single_token_mode: bool = False
    ) -> float:
        if self.tokenizer is None:
            raise RuntimeError("No tokenizer.json found in store for perplexity evaluation.")
        total_nll = 0.0
        total_tokens = 0
        for txt in texts:
            ids = self.tokenizer.encode(txt).ids
            if len(ids) < 2:
                continue
            if not single_token_mode:
                logits = self.forward_logits(ids[:-1], caches=None, all_positions=True).astype(
                    mx.float32
                )
            else:
                caches = [KVCache() for _ in range(self.num_layers)]
                step_logits = [
                    self.forward_logits([tid], caches=caches).astype(mx.float32)
                    for tid in ids[:-1]
                ]
                logits = mx.stack(step_logits, axis=1)
            targets = mx.array(ids[1:], dtype=mx.int32)[None, :]
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            chosen = mx.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
            mx.eval(chosen)
            total_nll += float(-mx.sum(chosen).item())
            total_tokens += len(ids) - 1
        return math.exp(total_nll / max(1, total_tokens))

    def close(self) -> None:
        self.arena.close()
        self.in_norms.clear()
        self.post_norms.clear()
        self.attns.clear()
        self.routers.clear()
        self.shared_mlps.clear()
        self.kv_cache.ram_lru.clear()
        self.emb_np_view = np.empty((0, 0), dtype=np.float16)
        gc.collect()
        mx.clear_cache()
        self._emb_mmap.close()
        self._emb_file.close()
        self.bridge.close()
