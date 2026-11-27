#!/usr/bin/env python3
"""
DeimosStream Universal PyTorch (`CUDA` / `CPU`) Paged MoE Engine (`src/deimos_stream/torch_engine.py`)
=====================================================================================================
Runs `DEIMIDX2` 4-bit Merkle-Paged Sparse MoE checkpoints on NVIDIA CUDA GPUs (and CPU CI) with:
  - `TorchUniversalAttention` (FlashAttention-2 / cuDNN SDPA + MHA/GQA + QK-Norm)
  - `TorchPagedExpertArena` (`pin_memory=True` async PCIe DMA + ABA-safe `slot_gen` sync)
  - Cross-Layer Residual Lookahead + Miss-Only Nucleus Dynamic-K (`miss_tail_tau`)
  - `TorchMerkleKVPrefixCache` (`DEIMIDX2` Robin-Hood 16-token KV prefix cache)
"""

from __future__ import annotations

import gc
import json
import math
import mmap
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from deimos_stream.arena import ArenaStats, DeimosMerkleBridge
from deimos_stream.backbone import rss_mb
from deimos_stream.engine import BenchmarkMetrics
from deimos_stream.torch_arena import HAS_TORCH, TorchPagedExpertArena
from deimos_stream.torch_backbone import load_torch_backbone, rms_norm
from deimos_stream.torch_kv_cache import TorchKVCache, TorchMerkleKVPrefixCache
from deimos_stream.triton_q4 import (
    quantize_2d_q4_torch,
    run_paged_q4_swiglu,
    triton_int8_gemv,
    triton_kernels_available,
    triton_q4_dense_linear,
)

if HAS_TORCH:
    import torch
    import torch.nn.functional as F

    _inference_mode = torch.inference_mode
else:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

    def _inference_mode():  # type: ignore[misc]
        return lambda fn: fn

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None  # type: ignore[assignment,misc]


class TorchDeimosPagedMoEEngine:
    """Universal PyTorch (`cuda` / `cpu`) Paged Q4 Sparse MoE Inference Engine."""

    def __init__(
        self,
        store_dir: Path,
        device_str: str = "cuda",
        slots_per_layer: int = 16,
        verify_merkle_on_cold: bool = True,
        use_nocache_pread: bool = False,
        enable_lookahead: bool = True,
        enable_merkle_kv: bool = True,
        miss_tail_tau: float = 1.0,
        min_k: int = 4,
        top_k: Optional[int] = None,
        backbone: str = "fp16",
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not installed (`pip install 'deimos-stream[cuda]'`).")
        self.store_dir: Path = Path(store_dir).expanduser().resolve()
        if not (self.store_dir / "cas_experts_q4.bin").exists():
            raise FileNotFoundError(f"cas_experts_q4.bin not found in {self.store_dir}")

        self.device = torch.device(device_str)
        if self.device.type == "cuda":
            torch.set_num_threads(1)
        self.backend_name: str = self.device.type
        self.backbone: str = backbone
        self.compute_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

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

        self.bridge = DeimosMerkleBridge(
            self.store_dir, cas_filename="cas_experts_q4.bin", enable_nocache_fd=True
        )
        self.arena = TorchPagedExpertArena(
            self.bridge,
            device=self.device,
            slots_per_layer=max(self.top_k, int(slots_per_layer)),
            verify_merkle_on_cold=verify_merkle_on_cold,
            use_nocache_pread=use_nocache_pread,
        )
        self.kv_cache = TorchMerkleKVPrefixCache(
            self.store_dir,
            self.bridge,
            num_layers=self.num_layers,
            seed_root_hex=str(self.manifest["model_merkle_root"]),
            device=self.device,
            compute_dtype=self.compute_dtype,
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
            (self.lm_head_i8, self.lm_head_scale),
        ) = load_torch_backbone(
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
            self.device,
            self.compute_dtype,
        )
        self.lm_head_q4: Optional[Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]] = None
        if self.backbone == "q4" and self.hidden_size % 64 == 0:
            for att in self.attns:
                att.quantize_weights(64)
            qws, qss, qbs = [], [], []
            for v0 in range(0, self.vocab_size, 16384):
                v1 = min(self.vocab_size, v0 + 16384)
                chunk_f16 = self.lm_head_i8[v0:v1].to(torch.float16) * self.lm_head_scale[v0:v1].to(torch.float16)
                qw, qs, qb = quantize_2d_q4_torch(chunk_f16, group_size=64)
                qws.append(qw)
                qss.append(qs)
                qbs.append(qb)
                del chunk_f16
            self.lm_head_q4 = (torch.cat(qws, dim=0), torch.cat(qss, dim=0), torch.cat(qbs, dim=0))
            self.lm_head_i8 = torch.empty((0, 0), dtype=torch.int8, device=self.device)
            self.lm_head_scale = torch.empty((0, 0), dtype=self.compute_dtype, device=self.device)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self.arena.warm_host_cache()
        self.reset_step_memory()

    def _route(self, router_logits: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        probs = torch.softmax(router_logits.float(), dim=-1)
        sc, inds = torch.topk(probs, k=self.top_k, dim=-1)
        if self.norm_topk_prob:
            sc = sc / torch.clamp(sc.sum(dim=-1, keepdim=True), min=1e-12)
        return inds.to(torch.int32), sc.to(self.compute_dtype)

    def _gather_moe_q4(
        self, l_idx: int, x_flat: "torch.Tensor", phys_idx: "torch.Tensor", scores: "torch.Tensor"
    ) -> "torch.Tensor":
        flat_slots = phys_idx.reshape(-1).to(torch.long)
        y = run_paged_q4_swiglu(
            x_flat,
            self.arena.gu_w[l_idx],
            self.arena.gu_s[l_idx],
            self.arena.gu_b[l_idx],
            self.arena.dn_w[l_idx],
            self.arena.dn_s[l_idx],
            self.arena.dn_b[l_idx],
            flat_slots,
            scores,
            group_size=self.arena.group_size,
        )

        sh = self.shared_mlps[l_idx]
        if sh is not None:
            sg, su, sd, sgate = sh
            sh_out = (F.silu(x_flat @ sg.T) * (x_flat @ su.T)) @ sd.T
            if sgate is not None:
                sh_out = sh_out * torch.sigmoid(x_flat @ sgate.T)
            y = y + sh_out
        return y

    @_inference_mode()
    def forward_logits(
        self,
        token_ids: Sequence[int],
        caches: Optional[List[TorchKVCache]] = None,
        all_positions: bool = False,
    ) -> "torch.Tensor":
        emb_np = self.emb_np_view[list(token_ids)][None, :, :].copy()
        h = torch.from_numpy(emb_np).to(self.device, self.compute_dtype)
        is_single_tok = len(token_ids) == 1
        max_toks_per_chunk = max(1, self.arena.slots_per_layer // self.top_k)

        for l_idx in range(self.num_layers):
            c = caches[l_idx] if caches is not None else None
            h_post_attn = h + self.attns[l_idx](
                rms_norm(h, self.in_norms[l_idx], self.rms_eps), c
            )
            normed_post = rms_norm(h_post_attn, self.post_norms[l_idx], self.rms_eps)
            B, L, D = normed_post.shape
            x_flat = normed_post.view(-1, D)
            indices, scores = self._route(x_flat @ self.routers[l_idx].T)

            if self.arena.is_cuda:
                sh = self.shared_mlps[l_idx]

                def _sh_fn(sh_t=sh, xf=x_flat):  # type: ignore[no-untyped-def]
                    if sh_t is None:
                        return None
                    sg, su, sd, sgate = sh_t
                    sh_out = (F.silu(xf @ sg.T) * (xf @ su.T)) @ sd.T
                    if sgate is not None:
                        sh_out = sh_out * torch.sigmoid(xf @ sgate.T)
                    return sh_out

                y_moe = self.arena.forward_moe(
                    l_idx,
                    x_flat,
                    indices,
                    scores,
                    miss_tail_tau=self.miss_tail_tau if is_single_tok else 1.0,
                    min_k=self.min_k,
                    shared_fn=_sh_fn if sh is not None else None,
                )
                h = h_post_attn + y_moe.view(B, L, D)
                continue

            if self.arena.is_fully_pinned:
                h = h_post_attn + self._gather_moe_q4(l_idx, x_flat, indices, scores).view(B, L, D)
                continue

            has_la = self.enable_lookahead and is_single_tok and (l_idx + 1 < self.num_layers)
            pred_np: Optional[np.ndarray] = None
            if has_la:
                la_norm = rms_norm(h_post_attn, self.post_norms[l_idx + 1], self.rms_eps).view(-1, D)
                la_ind, _ = self._route(la_norm @ self.routers[l_idx + 1].T)
                pred_np = la_ind.cpu().numpy()

            ind_np = indices.cpu().numpy()
            sc_np = scores.float().cpu().numpy()

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
                phys_t = torch.from_numpy(phys_np).to(self.device)
                eff_sc_t = (
                    scores
                    if (not is_single_tok or self.miss_tail_tau >= 0.9999)
                    else torch.from_numpy(eff_sc_np).to(self.device, self.compute_dtype)
                )
                h = h_post_attn + self._gather_moe_q4(l_idx, x_flat, phys_t, eff_sc_t).view(B, L, D)
            else:
                chunks: List[torch.Tensor] = []
                for start in range(0, ind_np.shape[0], max_toks_per_chunk):
                    end = min(ind_np.shape[0], start + max_toks_per_chunk)
                    phys_np, _ = self.arena.resolve_and_prefetch(
                        l_idx, ind_np[start:end], sc_np[start:end], miss_tail_tau=1.0
                    )
                    chunks.append(
                        self._gather_moe_q4(
                            l_idx,
                            x_flat[start:end],
                            torch.from_numpy(phys_np).to(self.device),
                            scores[start:end],
                        )
                    )
                h = h_post_attn + torch.cat(chunks, dim=0).view(B, L, D)

        h_out = rms_norm(
            h if all_positions else h[:, -1:, :], self.final_norm, self.rms_eps
        )
        if self.lm_head_q4 is not None:
            logits = triton_q4_dense_linear(h_out, *self.lm_head_q4, group_size=64)
        elif triton_kernels_available(self.device) and h_out.shape[1] <= 4:
            logits = triton_int8_gemv(
                h_out.reshape(-1, self.hidden_size), self.lm_head_i8, self.lm_head_scale
            ).view(h_out.shape[0], h_out.shape[1], self.vocab_size)
        else:
            lm_w = self.lm_head_i8.to(self.compute_dtype) * self.lm_head_scale
            logits = h_out @ lm_w.T
        return logits if all_positions else logits.squeeze(1)

    def make_caches(self) -> List[TorchKVCache]:
        return [TorchKVCache() for _ in range(self.num_layers)]

    def reset_step_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def vram_mb(self) -> Tuple[float, float]:
        if self.device.type == "cuda":
            return (
                float(torch.cuda.memory_allocated(self.device)) / (1024.0 * 1024.0),
                float(torch.cuda.max_memory_allocated(self.device)) / (1024.0 * 1024.0),
            )
        return self.arena.pool_vram_mb(), self.arena.pool_vram_mb()

    @_inference_mode()
    def sample_next_token(self, logits_2d: "torch.Tensor", temperature: float = 0.0) -> int:
        if temperature <= 1e-5:
            return int(torch.argmax(logits_2d, dim=-1).item())
        probs = torch.softmax(logits_2d.float() / temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    @_inference_mode()
    def prefill_prompt(
        self, prompt_ids: Sequence[int], caches: List[TorchKVCache]
    ) -> Tuple["torch.Tensor", int]:
        skipped = 0
        if self.enable_merkle_kv and len(prompt_ids) >= 17:
            skipped = self.kv_cache.match_longest_prefix(prompt_ids, caches)
        logits = self.forward_logits(prompt_ids[skipped:], caches=caches)
        if self.enable_merkle_kv and len(prompt_ids) >= 17:
            self.kv_cache.save_prefix_blocks(prompt_ids, caches)
        return logits, skipped

    @_inference_mode()
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
        from deimos_stream.fast_decode import iter_decode_tokens

        t_dec = time.perf_counter()
        gen_ids = list(
            iter_decode_tokens(self, logits, caches, max_new_tokens + 1, prompt_ids=prompt_ids)
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
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

    @_inference_mode()
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
                logits = self.forward_logits(ids[:-1], caches=None, all_positions=True).float()
            else:
                caches = self.make_caches()
                steps = [self.forward_logits([tid], caches=caches).float() for tid in ids[:-1]]
                logits = torch.stack(steps, dim=1)
            targets = torch.tensor(ids[1:], dtype=torch.long, device=self.device)[None, :]
            log_probs = F.log_softmax(logits, dim=-1)
            chosen = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
            total_nll += float(-chosen.sum().item())
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
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._emb_mmap.close()
        self._emb_file.close()
        self.bridge.close()
