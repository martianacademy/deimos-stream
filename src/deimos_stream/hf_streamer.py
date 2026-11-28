#!/usr/bin/env python3
"""
DeimosStream Zero-RAM `meta`-Skeleton HuggingFace Hybrid-MoE Engine
(`src/deimos_stream/hf_streamer.py`)
===================================================================
Runs massive 2026 Hybrid Sparse MoE models (`Qwen/Qwen3.6-35B-A3B` — 35B params, 67 GiB BF16,
40 GatedDeltaNet + GQA layers × 256 experts = 10,240 experts) inside ~3.5–4.6 GiB CUDA VRAM:
  1. Instantiates `AutoModelForCausalLM` on `torch.device("meta")` (`0` bytes host/GPU RAM).
  2. Replaces all `layers[l].mlp.experts` with `DeimosPagedHFExperts` backed by C++17 `DEIMIDX2`
     Merkle proofs + `TorchPagedExpertArena` (`pin_memory=True` async CUDA stream DMA +
     Cross-Layer Residual Lookahead + Cache-Aware Miss-Only Nucleus Dynamic-K).
  3. Replaces `embed_tokens` with zero-VRAM CPU `mmap` lookup (`DeimosMmapEmbedding`, saving ~1 GiB)
     and `lm_head` with vocab-tiled INT8 `DeimosInt8LMHead` (saving ~1.3 GiB of VRAM + spikes).
"""

from __future__ import annotations

import gc
import inspect
import json
import mmap
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from deimos_stream.arena import ArenaStats, DeimosMerkleBridge
from deimos_stream.backbone import rss_mb
from deimos_stream.engine import BenchmarkMetrics
from deimos_stream.hf_cuda_graph import DeimosHFCudaGraphDecode, can_use_hf_cuda_graph
from deimos_stream.hf_fast_patch import patch_hf_hybrid_for_cuda
from deimos_stream.model_adapters import (
    bind_hf_moe_layer,
    extract_swiglu_params,
    load_hf_meta_skeleton,
)
from deimos_stream.torch_arena import HAS_TORCH, TorchPagedExpertArena
from deimos_stream.triton_q4 import (
    quantize_2d_q4_torch,
    run_paged_q4_swiglu,
    triton_int8_gemv,
    triton_kernels_available,
    triton_q4_dense_linear,
)

if HAS_TORCH:
    from safetensors.torch import load_file as st_load_torch
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _inference_mode = torch.inference_mode
    _NNModule = nn.Module
else:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _NNModule = object  # type: ignore[assignment,misc]

    def _inference_mode():  # type: ignore[misc]
        return lambda fn: fn

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None  # type: ignore[assignment,misc]


class DeimosMmapEmbedding(_NNModule):  # type: ignore[misc,valid-type]
    """Zero-VRAM `mmap` token embedding table slicing rows directly from `embed_tokens.bin`."""

    def __init__(
        self,
        emb_path: Path,
        vocab_size: int,
        hidden_size: int,
        device: "torch.device",
        dtype: "torch.dtype",
    ):
        super().__init__()
        self._file = open(emb_path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self.view = np.frombuffer(
            self._mmap, dtype=np.float16, count=vocab_size * hidden_size
        ).reshape(vocab_size, hidden_size)
        self.device = device
        self.dtype = dtype
        self.register_buffer("weight", torch.empty((0,), dtype=dtype, device=device), persistent=False)

    def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        ids_np = input_ids.detach().cpu().numpy()
        sliced = np.ascontiguousarray(self.view[ids_np])
        return torch.from_numpy(sliced).to(device=self.device, dtype=self.dtype)

    def close(self) -> None:
        self.view = np.empty((0, 0), dtype=np.float16)
        self._mmap.close()
        self._file.close()


class DeimosInt8LMHead(_NNModule):  # type: ignore[misc,valid-type]
    """Vocab-tiled INT8/Q4 `lm_head` eliminating 970 MiB FP16 temporary dequantization spikes."""

    def __init__(
        self,
        lm_path: Path,
        vocab_size: int,
        hidden_size: int,
        device: "torch.device",
        dtype: "torch.dtype",
        tile_vocab: int = 32768,
        use_q4: bool = False,
    ):
        super().__init__()
        w_nbytes = vocab_size * hidden_size
        self.vocab_size = int(vocab_size)
        self.tile_vocab = int(tile_vocab)
        self.compute_dtype = dtype
        self.q4: Optional[Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]] = None

        with open(lm_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                w_i8 = np.frombuffer(mm, dtype=np.int8, count=w_nbytes).reshape(vocab_size, hidden_size)
                s_f16 = np.frombuffer(mm, dtype=np.float16, count=vocab_size, offset=w_nbytes).reshape(
                    vocab_size, 1
                )
                if use_q4 and hidden_size % 64 == 0:
                    qws, qss, qbs = [], [], []
                    for v0 in range(0, vocab_size, 16384):
                        v1 = min(vocab_size, v0 + 16384)
                        ch = torch.from_numpy(w_i8[v0:v1].copy()).to(device, torch.float16) * torch.from_numpy(
                            s_f16[v0:v1].copy()
                        ).to(device, torch.float16)
                        qw, qs, qb = quantize_2d_q4_torch(ch, group_size=64)
                        qws.append(qw)
                        qss.append(qs)
                        qbs.append(qb)
                        del ch
                    self.q4 = (torch.cat(qws, dim=0), torch.cat(qss, dim=0), torch.cat(qbs, dim=0))
                    self.w_i8 = torch.empty((0, 0), dtype=torch.int8, device=device)
                    self.scale = torch.empty((0, 0), dtype=dtype, device=device)
                else:
                    self.w_i8 = torch.from_numpy(w_i8.copy()).to(device)
                    self.scale = torch.from_numpy(s_f16.copy()).to(device=device, dtype=dtype)
                del w_i8, s_f16
            finally:
                mm.close()

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


class DeimosPagedHFExperts(_NNModule):  # type: ignore[misc,valid-type]
    """Drop-in `DEIMIDX2` Hybrid Paged Q4 replacement for `Qwen3_5MoeExperts` & HF MoE blocks."""

    def __init__(
        self,
        layer_idx: int,
        arena: TorchPagedExpertArena,
        engine_ref: "DeimosHFHybridMoEEngine",
    ):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.arena = arena
        self.engine_ref = engine_ref
        self.next_norm: Optional[nn.Module] = None
        self.next_gate: Optional[nn.Module] = None

    def forward(
        self,
        hidden_states: "torch.Tensor",
        top_k_index: "torch.Tensor",
        top_k_weights: "torch.Tensor",
        shared_fn: Optional[object] = None,
    ) -> "torch.Tensor":
        eng = self.engine_ref
        la_fn = None
        if (
            not self.arena.is_cuda
            and eng.enable_lookahead
            and hidden_states.shape[0] == 1
            and (self.layer_idx + 1 < self.arena.num_layers)
            and self.next_gate is not None
        ):
            def _la() -> "torch.Tensor":
                la_in = self.next_norm(hidden_states) if self.next_norm is not None else hidden_states
                la_out = self.next_gate(la_in)  # type: ignore[misc]
                if isinstance(la_out, tuple):
                    return la_out[2] if len(la_out) >= 3 else la_out[1]
                return torch.topk(la_out, k=eng.top_k, dim=-1)[1]
            la_fn = _la

        return self.arena.forward_moe(
            self.layer_idx,
            hidden_states,
            top_k_index,
            top_k_weights,
            miss_tail_tau=eng.miss_tail_tau,
            min_k=eng.min_k,
            shared_fn=shared_fn,  # type: ignore[arg-type]
            la_ids_fn=la_fn,
        )


class DeimosHFHybridMoEEngine:
    """Zero-RAM `meta`-skeleton HuggingFace Hybrid-MoE Engine backed by `DEIMIDX2` + `TorchPagedExpertArena`."""

    def __init__(
        self,
        store_dir: Path,
        device_str: str = "cuda",
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
        if not HAS_TORCH:
            raise RuntimeError("PyTorch & transformers required for DeimosHFHybridMoEEngine.")

        self.store_dir = Path(store_dir).expanduser().resolve()
        self.device = torch.device(device_str)
        if self.device.type == "cuda":
            torch.set_num_threads(1)
        self.backend_name: str = f"hf_{self.device.type}"
        self.backbone: str = backbone
        self.compute_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.config: Dict[str, object] = json.loads((self.store_dir / "config.json").read_text())
        self.manifest: Dict[str, object] = json.loads(
            (self.store_dir / "merkle_manifest.json").read_text()
        )
        self.num_layers = int(self.config["num_hidden_layers"])  # type: ignore[arg-type]
        self.moe_layer_ids: List[int] = [
            int(x) for x in self.manifest.get("moe_layer_ids", list(range(self.num_layers)))  # type: ignore[union-attr]
        ]
        self.moe_containers: List[str] = [
            str(x) for x in self.manifest.get("moe_containers", ["mlp"] * len(self.moe_layer_ids))  # type: ignore[union-attr]
        ]
        self.num_experts = int(self.config["num_experts"])  # type: ignore[arg-type]
        self.hidden_size = int(self.config["hidden_size"])  # type: ignore[arg-type]
        self.vocab_size = int(self.config["vocab_size"])  # type: ignore[arg-type]
        default_k = int(
            self.config.get("num_experts_per_tok") or self.config.get("num_experts_per_token") or 8  # type: ignore[arg-type]
        )
        self.top_k = max(1, min(self.num_experts, int(top_k or default_k)))
        self.enable_lookahead = bool(enable_lookahead)
        self.enable_merkle_kv = False  # Hybrid recurrent GatedDeltaNet uses HF DynamicCache
        self.miss_tail_tau = float(miss_tail_tau)
        self.min_k = max(1, min(self.top_k, int(min_k)))

        tok_p = self.store_dir / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tok_p)) if (Tokenizer is not None and tok_p.exists()) else None
        self.bridge = DeimosMerkleBridge(self.store_dir, cas_filename="cas_experts_q4.bin", enable_nocache_fd=True)
        self.arena = TorchPagedExpertArena(
            self.bridge, device=self.device, slots_per_layer=max(self.top_k, int(slots_per_layer)),
            verify_merkle_on_cold=verify_merkle_on_cold, use_nocache_pread=use_nocache_pread,
        )
        self.arena.swiglu_limit, self.arena.swiglu_alpha, self.arena.swiglu_up_offset = extract_swiglu_params(
            self.config
        )

        model, _, bb_path = load_hf_meta_skeleton(
            self.store_dir, self.device, self.compute_dtype, self.backbone
        )
        self._cache_kwarg = "cache" if "cache" in inspect.signature(model.forward).parameters else "past_key_values"
        gc.collect()

        inner_lm = getattr(model.model, "language_model", model.model)
        self._last_hidden_states: Optional[torch.Tensor] = None
        inner_lm.register_forward_hook(
            lambda _m, _i, out: setattr(
                self, "_last_hidden_states", getattr(out, "last_hidden_state", out[0] if isinstance(out, tuple) else out)
            )
        )
        self._mmap_emb = DeimosMmapEmbedding(
            self.store_dir / "embed_tokens.bin", self.vocab_size, self.hidden_size, self.device, self.compute_dtype
        )
        inner_lm.embed_tokens = self._mmap_emb
        # The lm_head stays INT8 for every backbone: the fused `__dp4a` INT8 GEMV is what the CUDA
        # Graph head segment captures. `memory_budget.plan_slots` plans this INT8 size for hybrid
        # stores on cuda/cpu, so keep the two in sync if this ever switches to `use_q4=True`.
        model.lm_head = DeimosInt8LMHead(
            self.store_dir / "lm_head_int8.bin", self.vocab_size, self.hidden_size,
            self.device, self.compute_dtype, use_q4=False,
        )

        paged_mods: List[DeimosPagedHFExperts] = []
        router_mods: List[Optional[nn.Module]] = []
        for a_i, l_i in enumerate(self.moe_layer_ids):
            pmod = DeimosPagedHFExperts(a_i, self.arena, self)
            c_attr = self.moe_containers[a_i] if a_i < len(self.moe_containers) else "mlp"
            router_mods.append(bind_hf_moe_layer(inner_lm.layers[l_i], c_attr, pmod))
            paged_mods.append(pmod)

        for a_i in range(len(paged_mods) - 1):
            if self.moe_layer_ids[a_i + 1] == self.moe_layer_ids[a_i] + 1 and router_mods[a_i + 1] is not None:
                nxt = inner_lm.layers[self.moe_layer_ids[a_i + 1]]
                paged_mods[a_i].next_norm = getattr(nxt, "post_attention_layernorm", getattr(nxt, "pre_ff_layernorm", None))
                paged_mods[a_i].next_gate = router_mods[a_i + 1]

        patch_hf_hybrid_for_cuda(model, self.device, backbone=self.backbone)
        if self.arena._evict_host_on_stage and hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            for fp in (bb_path, self.store_dir / "lm_head_int8.bin"):
                fd = os.open(fp, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
        model.eval()
        self.model = model
        self.use_cuda_graph: bool = True
        self._graph_runner: Optional[DeimosHFCudaGraphDecode] = (
            DeimosHFCudaGraphDecode(self) if can_use_hf_cuda_graph(self) else None
        )
        self.arena.warm_host_cache()
        self.reset_step_memory()

    def make_caches(self) -> List[Optional[object]]:
        return [None]

    def reset_step_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
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
    def forward_logits(
        self,
        token_ids: Sequence[int],
        caches: Optional[List[Optional[object]]] = None,
        all_positions: bool = False,
    ) -> "torch.Tensor":
        past = caches[0] if (caches is not None and len(caches) > 0) else None
        if (
            self.use_cuda_graph
            and self._graph_runner is not None
            and not all_positions
            and len(token_ids) == 1
            and past is not None
        ):
            g_logits = self._graph_runner.step(int(token_ids[0]), past)
            if g_logits is not None:
                return g_logits
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
        logits = self.forward_logits(prompt_ids, caches=caches)
        if self.use_cuda_graph and self._graph_runner is not None and caches and caches[0] is not None:
            self._graph_runner._sync_from_hf_cache(caches[0])
        return logits, 0

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
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
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

    def close(self) -> None:
        if self._graph_runner is not None:
            self._graph_runner.close()
            self._graph_runner = None
        self.arena.close()
        self._mmap_emb.close()
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.bridge.close()
