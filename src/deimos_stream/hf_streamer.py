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
import math
import mmap
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from deimos_stream.arena import ArenaStats, DeimosMerkleBridge
from deimos_stream.backbone import rss_mb
from deimos_stream.engine import BenchmarkMetrics
from deimos_stream.quant_q4 import unpack_q4_torch
from deimos_stream.torch_arena import HAS_TORCH, TorchPagedExpertArena

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

    def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        ids_np = input_ids.detach().cpu().numpy()
        sliced = np.ascontiguousarray(self.view[ids_np])
        return torch.from_numpy(sliced).to(device=self.device, dtype=self.dtype)

    def close(self) -> None:
        self.view = np.empty((0, 0), dtype=np.float16)
        self._mmap.close()
        self._file.close()


class DeimosInt8LMHead(_NNModule):  # type: ignore[misc,valid-type]
    """Vocab-tiled INT8 `lm_head` eliminating 970 MiB FP16 temporary dequantization spikes."""

    def __init__(
        self,
        lm_path: Path,
        vocab_size: int,
        hidden_size: int,
        device: "torch.device",
        dtype: "torch.dtype",
        tile_vocab: int = 32768,
    ):
        super().__init__()
        w_nbytes = vocab_size * hidden_size
        with open(lm_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                w_i8 = np.frombuffer(mm, dtype=np.int8, count=w_nbytes).reshape(
                    vocab_size, hidden_size
                ).copy()
                s_f16 = np.frombuffer(mm, dtype=np.float16, count=vocab_size, offset=w_nbytes).reshape(
                    vocab_size, 1
                ).copy()
            finally:
                mm.close()
        self.w_i8 = torch.from_numpy(w_i8).to(device)
        self.scale = torch.from_numpy(s_f16).to(device=device, dtype=dtype)
        self.vocab_size = int(vocab_size)
        self.tile_vocab = int(tile_vocab)
        self.compute_dtype = dtype

    def forward(self, hidden_states: "torch.Tensor") -> "torch.Tensor":
        orig_shape = hidden_states.shape[:-1]
        x = hidden_states.reshape(-1, hidden_states.shape[-1]).to(self.compute_dtype)
        out_tiles: List[torch.Tensor] = []
        for v0 in range(0, self.vocab_size, self.tile_vocab):
            v1 = min(self.vocab_size, v0 + self.tile_vocab)
            w_tile = self.w_i8[v0:v1].to(self.compute_dtype) * self.scale[v0:v1]
            out_tiles.append(x @ w_tile.T)
        return torch.cat(out_tiles, dim=-1).reshape(*orig_shape, self.vocab_size)


class DeimosPagedHFExperts(_NNModule):  # type: ignore[misc,valid-type]
    """Drop-in `DEIMIDX2` Paged Q4 replacement for `Qwen3_5MoeExperts` & HF MoE blocks."""

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

    def _run_q4_chunk(
        self, x_chunk: "torch.Tensor", phys_np: np.ndarray, sc_chunk: "torch.Tensor"
    ) -> "torch.Tensor":
        N, D = x_chunk.shape
        K = phys_np.shape[1]
        dt = x_chunk.dtype
        flat_slots = torch.from_numpy(phys_np.reshape(-1)).to(device=x_chunk.device, dtype=torch.long)

        gu_w = unpack_q4_torch(
            self.arena.gu_w[self.layer_idx][flat_slots],
            self.arena.gu_s[self.layer_idx][flat_slots],
            self.arena.gu_b[self.layer_idx][flat_slots],
            group_size=self.arena.group_size,
            out_dtype=dt,
        )
        x_rep = x_chunk.unsqueeze(1).expand(N, K, D).reshape(N * K, D, 1)
        gate, up = torch.bmm(gu_w, x_rep).squeeze(-1).chunk(2, dim=-1)
        act = (F.silu(gate) * up).unsqueeze(-1)

        dn_w = unpack_q4_torch(
            self.arena.dn_w[self.layer_idx][flat_slots],
            self.arena.dn_s[self.layer_idx][flat_slots],
            self.arena.dn_b[self.layer_idx][flat_slots],
            group_size=self.arena.group_size,
            out_dtype=dt,
        )
        dn_out = torch.bmm(dn_w, act).squeeze(-1).view(N, K, D)
        return (dn_out * sc_chunk.to(dt).unsqueeze(-1)).sum(dim=1)

    def forward(
        self,
        hidden_states: "torch.Tensor",
        top_k_index: "torch.Tensor",
        top_k_weights: "torch.Tensor",
    ) -> "torch.Tensor":
        N, _ = hidden_states.shape
        top_k = int(top_k_index.shape[-1])
        eng = self.engine_ref

        if self.arena.is_fully_pinned:
            return self._run_q4_chunk(
                hidden_states, top_k_index.detach().cpu().numpy().astype(np.int32), top_k_weights
            )

        is_single = N == 1
        has_la = (
            eng.enable_lookahead
            and is_single
            and (self.layer_idx + 1 < eng.num_layers)
            and self.next_gate is not None
        )
        pred_np: Optional[np.ndarray] = None
        if has_la:
            with torch.no_grad():
                la_in = self.next_norm(hidden_states) if self.next_norm is not None else hidden_states
                la_out = self.next_gate(la_in)  # type: ignore[misc]
                la_sel = la_out[2] if isinstance(la_out, tuple) and len(la_out) >= 3 else la_out[0]
                pred_np = la_sel.detach().cpu().numpy().astype(np.int32)

        ind_np = top_k_index.detach().cpu().numpy().astype(np.int32)
        sc_np = top_k_weights.detach().float().cpu().numpy()
        max_chunk = min(4, max(1, self.arena.slots_per_layer // max(1, top_k)))

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
            eff_sc_t = (
                top_k_weights
                if (not is_single or eng.miss_tail_tau >= 0.9999)
                else torch.from_numpy(eff_sc_np).to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
            )
            return self._run_q4_chunk(hidden_states, phys_np, eff_sc_t)

        outs: List[torch.Tensor] = []
        for s0 in range(0, N, max_chunk):
            s1 = min(N, s0 + max_chunk)
            phys_np, _ = self.arena.resolve_and_prefetch(
                self.layer_idx, ind_np[s0:s1], sc_np[s0:s1], miss_tail_tau=1.0
            )
            outs.append(self._run_q4_chunk(hidden_states[s0:s1], phys_np, top_k_weights[s0:s1]))
        return torch.cat(outs, dim=0)


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
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch & transformers required for DeimosHFHybridMoEEngine.")
        from transformers import AutoConfig, AutoModelForCausalLM

        self.store_dir = Path(store_dir).expanduser().resolve()
        self.device = torch.device(device_str)
        self.backend_name: str = f"hf_{self.device.type}"
        self.compute_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

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
        self.enable_merkle_kv = False  # Hybrid recurrent GatedDeltaNet uses HF DynamicCache
        self.miss_tail_tau = float(miss_tail_tau)
        self.min_k = max(1, min(self.top_k, int(min_k)))

        tok_p = self.store_dir / "tokenizer.json"
        self.tokenizer = (
            Tokenizer.from_file(str(tok_p)) if (Tokenizer is not None and tok_p.exists()) else None
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

        hf_cfg = AutoConfig.from_pretrained(str(self.store_dir), trust_remote_code=True)
        text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
        text_cfg._attn_implementation = "sdpa"
        text_cfg.dtype = self.compute_dtype

        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(text_cfg, trust_remote_code=True)
        self._cache_kwarg = (
            "cache" if "cache" in inspect.signature(model.forward).parameters else "past_key_values"
        )

        self._mmap_emb = DeimosMmapEmbedding(
            self.store_dir / "embed_tokens.bin",
            self.vocab_size,
            self.hidden_size,
            self.device,
            self.compute_dtype,
        )
        model.model.embed_tokens = self._mmap_emb
        model.lm_head = DeimosInt8LMHead(
            self.store_dir / "lm_head_int8.bin",
            self.vocab_size,
            self.hidden_size,
            self.device,
            self.compute_dtype,
        )
        if hasattr(model.model, "rotary_emb"):
            rot_cls = type(model.model.rotary_emb)
            model.model.rotary_emb = rot_cls(config=text_cfg).to(self.device)

        paged_mods: List[DeimosPagedHFExperts] = []
        for l_i in range(self.num_layers):
            pmod = DeimosPagedHFExperts(l_i, self.arena, self)
            model.model.layers[l_i].mlp.experts = pmod
            paged_mods.append(pmod)

        bb_tensors = st_load_torch(str(self.store_dir / "hf_backbone.safetensors"), device="cpu")
        casted_bb = {
            k: (v.to(device=self.device, dtype=self.compute_dtype) if v.is_floating_point() and v.dtype != torch.float32 else v.to(device=self.device))
            for k, v in bb_tensors.items()
        }
        bb_tensors.clear()
        model.load_state_dict(casted_bb, strict=False, assign=True)
        casted_bb.clear()
        gc.collect()

        for l_i in range(self.num_layers - 1):
            nxt = model.model.layers[l_i + 1]
            paged_mods[l_i].next_norm = getattr(nxt, "post_attention_layernorm", None)
            paged_mods[l_i].next_gate = nxt.mlp.gate

        model.eval()
        self.model = model
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
        inp = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        past = caches[0] if (caches is not None and len(caches) > 0) else None
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
        gen_ids = [tok]
        t_dec = time.perf_counter()
        for _ in range(max_new_tokens):
            logits = self.forward_logits([gen_ids[-1]], caches=caches)
            gen_ids.append(self.sample_next_token(logits, temperature=0.0))
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
        self.arena.close()
        self._mmap_emb.close()
        del self.model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.bridge.close()
