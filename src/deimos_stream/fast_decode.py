#!/usr/bin/env python3
"""
DeimosStream Fast Decode Paths (`src/deimos_stream/fast_decode.py`)
==================================================================
1. `iter_decode_tokens`: the token loop shared by `bench`, `chat` and `serve`.
   * Pinned MLX engines (every expert resident): mlx-lm-style async pipeline. Step t+1 is queued
     with `mx.async_eval` before token t is read back, so the CPU and GPU no longer take turns.
     Measured on OLMoE / M4 Max: Q4 backbone 319-328 -> 371-385 tok/s, with greedy tokens identical.
   * Everything else (paged arena, PyTorch, hybrid engines): the original synchronous loop.
   Both paths have the same semantics. A stop id is never yielded or fed; a token is fed only
   when another step follows; and if the caller stops early, the KV caches end in exactly the
   state the synchronous loop would leave them in (the pipeline trims its one extra queued token).
2. `apply_q4_backbone`: optional `--backbone q4`. It quantizes attention (qkv/o) and the lm_head
   to 4-bit group 64 in memory (the mlx-lm Q4 format). Routers, norms and shared experts stay FP16.
   Measured: 259 -> 337 tok/s pinned, PPL 11.50 -> 11.54 (mlx-lm Q4: 11.74).
"""

from __future__ import annotations

from typing import Any, Collection, Iterator, List, Optional

try:
    import mlx.core as mx
except ImportError:  # PyTorch-only installs
    mx = None  # type: ignore[assignment]


def supports_pipeline(eng: Any) -> bool:
    """True when `eng` accepts `mx.array` tokens and has no paging syncs (all experts pinned)."""
    return bool(
        mx is not None
        and getattr(eng, "accepts_mx_tokens", False)
        and getattr(eng, "pipelined_decode", True)
        and eng.arena.is_fully_pinned
    )


def _sample_lazy(logits_2d: "mx.array", temperature: float) -> "mx.array":
    if temperature <= 1e-5:
        return mx.argmax(logits_2d, axis=-1)
    return mx.random.categorical(logits_2d.astype(mx.float32) / temperature)


def iter_decode_tokens(
    eng: Any,
    logits: Any,
    caches: List[Any],
    max_new_tokens: int,
    temperature: float = 0.0,
    stop_ids: Optional[Collection[int]] = None,
) -> Iterator[int]:
    """Yields up to `max_new_tokens` ids, starting from the prefill `logits`.

    Close the generator (`gen.close()`) as soon as you stop consuming it, so the KV caches
    are trimmed before they are saved or reused.
    """
    stops = stop_ids or ()
    if not supports_pipeline(eng):
        tok = eng.sample_next_token(logits, temperature=temperature)
        for step in range(max_new_tokens):
            if tok in stops:
                return
            yield tok
            if step + 1 < max_new_tokens:
                logits = eng.forward_logits([tok], caches=caches)
                tok = eng.sample_next_token(logits, temperature=temperature)
        return

    y = _sample_lazy(logits, temperature)
    mx.async_eval(y)
    fed_current = False
    try:
        for step in range(max_new_tokens):
            has_next = step + 1 < max_new_tokens
            nxt = None
            if has_next:
                nxt = _sample_lazy(eng.forward_logits(y, caches=caches), temperature)
                mx.async_eval(nxt)
            tok = int(y.item())
            fed_current = has_next
            if tok in stops:
                return
            yield tok
            fed_current = False
            y = nxt  # type: ignore[assignment]
    finally:
        if fed_current:  # the token was queued into the caches but never accepted
            for c in caches:
                c.trim(1)


def apply_q4_backbone(eng: Any, group_size: int = 64) -> None:
    """Quantizes a `DeimosPagedMoEEngine`'s attention projections and lm_head to Q4 in place."""
    for att in eng.attns:
        att.quantize_weights(group_size=group_size, bits=4)
    head = mx.dequantize(
        eng.lm_head_qw, eng.lm_head_qs, eng.lm_head_qb, group_size=64, bits=eng.lm_head_bits
    )
    eng.lm_head_qw, eng.lm_head_qs, eng.lm_head_qb = mx.quantize(head, group_size=group_size, bits=4)
    mx.eval(eng.lm_head_qw, eng.lm_head_qs, eng.lm_head_qb)
    eng.lm_head_bits = 4
    eng.lm_head_group_size = group_size
    eng.backbone = "q4"
    del head
    mx.clear_cache()
