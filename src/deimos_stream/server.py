#!/usr/bin/env python3
"""
DeimosStream OpenAI-Compatible FastAPI Gateway (`src/deimos_stream/server.py`)
==============================================================================
Exposes thread-safe HTTP & Server-Sent Events (SSE) endpoints across `MLX`, `CUDA`, and `CPU`:
  - `GET  /health`               — Live backend, VRAM, RSS, Arena hit-rate & `DEIMIDX2` telemetry
  - `GET  /v1/models`            — OpenAI-compatible model listing
  - `POST /v1/chat/completions`  — Streaming (`text/event-stream`) & non-streaming completions
  - `POST /v1/deimos/config`     — Live hot-update for `slots_per_layer`, `miss_tail_tau`, `enable_lookahead`
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
import threading
import time
from typing import AsyncIterator, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from deimos_stream import __version__
from deimos_stream.backbone import rss_mb
from deimos_stream.cli import inspect_store
from deimos_stream.detokenizer import DecodeCapableTokenizer, IncrementalStreamingDetokenizer
from deimos_stream.factory import AnyDeimosEngine, create_paged_moe_engine, resolve_backend
from deimos_stream.fast_decode import iter_decode_tokens
from deimos_stream.memory_budget import SlotsArg
from deimos_stream.schemas import (
    ChatChoice,
    ChatChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionUsage,
    DeimosConfigUpdateRequest,
    DeimosRequestTelemetry,
    FallbackByteTokenizer,
    check_stop_buffer as _check_stop_buffer,
    format_chat_messages,
    normalize_stops,
)


def create_deimos_app(
    store_dir: Path,
    backend: str = "auto",
    slots_per_layer: SlotsArg = "auto",
    miss_tail_tau: float = 0.85,
    enable_lookahead: bool = True,
    enable_merkle_kv: bool = True,
    use_nocache_pread: bool = True,
    verify_merkle_on_cold: bool = True,
    model_name: Optional[str] = None,
    max_memory_gb: Optional[float] = None,
    backbone: str = "fp16",
) -> FastAPI:
    """Creates a thread-safe OpenAI-compatible FastAPI app backed by `mlx`, `cuda`, or `cpu`."""
    resolved_dir = Path(store_dir).expanduser().resolve()
    if not (resolved_dir / "merkle_tree.idx").exists():
        raise FileNotFoundError(f"merkle_tree.idx not found in {resolved_dir}")

    default_model_id = model_name or resolved_dir.name
    lock = threading.Lock()
    state: Dict[str, Optional[AnyDeimosEngine]] = {"engine": None}
    active_backend = resolve_backend(backend)

    if active_backend is not None:
        state["engine"] = create_paged_moe_engine(
            store_dir=resolved_dir,
            backend=active_backend,
            slots_per_layer=slots_per_layer,
            max_memory_gb=max_memory_gb,
            backbone=backbone,
            verify_merkle_on_cold=verify_merkle_on_cold,
            use_nocache_pread=use_nocache_pread,
            enable_lookahead=enable_lookahead,
            enable_merkle_kv=enable_merkle_kv,
            miss_tail_tau=miss_tail_tau,
        )

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            with lock:
                eng = state["engine"]
                if eng is not None:
                    eng.close()
                    state["engine"] = None

    app = FastAPI(
        title="DeimosStream OpenAI API Gateway",
        description="Universal Low-VRAM Sparse MoE & LLM Merkle-Paged Streaming Engine (MLX • CUDA • C++17)",
        version=__version__,
        lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _stop_ids_for(eng: AnyDeimosEngine) -> Set[int]:
        if eng.tokenizer is None:
            return set()
        eos_cfg = eng.config.get("eos_token_id", 50279)
        sids: Set[int] = {int(x) for x in eos_cfg} if isinstance(eos_cfg, list) else {int(eos_cfg), 0, 50278, 50279}
        for sp_tok in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
            tid = eng.tokenizer.token_to_id(sp_tok)
            if tid is not None:
                sids.add(int(tid))
        return sids

    def _encode_messages(
        eng: AnyDeimosEngine, messages: Sequence[ChatMessage]
    ) -> Tuple[List[int], DecodeCapableTokenizer]:
        is_chatml = eng.tokenizer is not None and eng.tokenizer.token_to_id("<|im_start|>") is not None
        has_think = is_chatml and eng.tokenizer is not None and eng.tokenizer.token_to_id("</think>") is not None
        prompt_str = format_chat_messages(messages, use_chatml=is_chatml, append_empty_think=has_think)
        if eng.tokenizer is not None:
            return eng.tokenizer.encode(prompt_str).ids, eng.tokenizer
        fb = FallbackByteTokenizer(eng.vocab_size)
        return fb.encode_text(prompt_str), fb

    @app.get("/health")
    def health() -> Dict[str, object]:
        with lock:
            eng = state["engine"]
            if eng is None:
                info = inspect_store(resolved_dir, verify_all_leaves=False)
                return {"status": "inspect_only", "backend": None, "store": info}
            act_mb, peak_mb = eng.vram_mb()
            st = eng.arena.get_stats(nominal_top_k=eng.top_k)
            return {
                "status": "healthy",
                "backend": eng.backend_name,
                "model": default_model_id,
                "store_dir": str(eng.store_dir),
                "model_merkle_root": str(eng.manifest["model_merkle_root"]),
                "num_layers": eng.num_layers,
                "num_experts": eng.num_experts,
                "slots_per_layer": st["slots_per_layer"],
                "miss_tail_tau": eng.miss_tail_tau,
                "enable_lookahead": eng.enable_lookahead,
                "active_vram_mb": round(act_mb, 2),
                "peak_vram_mb": round(peak_mb, 2),
                "rss_mb": round(rss_mb(), 2),
                "arena": st,
            }

    @app.get("/v1/models")
    def list_models() -> Dict[str, object]:
        created_ts = int(resolved_dir.stat().st_mtime)
        return {
            "object": "list",
            "data": [
                {
                    "id": default_model_id,
                    "object": "model",
                    "created": created_ts,
                    "owned_by": "martianacademy",
                }
            ],
        }

    @app.post("/v1/deimos/config")
    def update_config(req: DeimosConfigUpdateRequest) -> Dict[str, object]:
        with lock:
            eng = state["engine"]
            if eng is None:
                raise HTTPException(status_code=503, detail="No MLX or PyTorch execution runtime available.")
            if (
                req.slots_per_layer is not None
                and (req.slots_per_layer == "auto" or req.slots_per_layer != eng.arena.slots_per_layer)
            ):
                la = req.enable_lookahead if req.enable_lookahead is not None else eng.enable_lookahead
                tau = req.miss_tail_tau if req.miss_tail_tau is not None else eng.miss_tail_tau
                kv = eng.enable_merkle_kv
                nc = eng.arena.use_nocache_pread
                b_name = eng.backend_name
                eng.close()
                eng = create_paged_moe_engine(
                    store_dir=resolved_dir,
                    backend=b_name,
                    slots_per_layer=req.slots_per_layer,
                    max_memory_gb=max_memory_gb,
                    backbone=backbone,
                    use_nocache_pread=nc,
                    enable_lookahead=la,
                    enable_merkle_kv=kv,
                    miss_tail_tau=tau,
                )
                state["engine"] = eng
            else:
                if req.miss_tail_tau is not None:
                    eng.miss_tail_tau = float(req.miss_tail_tau)
                if req.enable_lookahead is not None:
                    eng.enable_lookahead = bool(req.enable_lookahead)
            return {
                "status": "ok",
                "backend": eng.backend_name,
                "slots_per_layer": eng.arena.slots_per_layer,
                "miss_tail_tau": eng.miss_tail_tau,
                "enable_lookahead": eng.enable_lookahead,
            }

    def _generate_events(
        req: ChatCompletionRequest,
    ) -> Iterator[Tuple[str, Optional[str], Optional[DeimosRequestTelemetry], int, int]]:
        with lock:
            eng = state["engine"]
            if eng is None:
                raise HTTPException(status_code=503, detail="No MLX or PyTorch execution runtime available.")
            saved_tau, saved_la = eng.miss_tail_tau, eng.enable_lookahead
            if req.miss_tail_tau is not None:
                eng.miss_tail_tau = float(req.miss_tail_tau)
            if req.enable_lookahead is not None:
                eng.enable_lookahead = bool(req.enable_lookahead)
            try:
                prompt_ids, tok_impl = _encode_messages(eng, req.messages)
                max_new = int(req.max_completion_tokens or req.max_tokens or 256)
                stops = normalize_stops(req.stop)
                stop_ids = _stop_ids_for(eng)
                caches = eng.make_caches()
                detok = IncrementalStreamingDetokenizer(tok_impl)

                eng.arena.reset_telemetry()
                eng.reset_step_memory()

                t_pf = time.perf_counter()
                logits, skipped = eng.prefill_prompt(prompt_ids, caches)  # type: ignore[arg-type]
                ttft_ms = (time.perf_counter() - t_pf) * 1000.0

                eng.arena.reset_telemetry()
                gen_ids: List[int] = []
                pending_buf = ""
                finish_reason = "length"
                t_dec = time.perf_counter()

                gen = iter_decode_tokens(eng, logits, caches, max_new, req.temperature, stop_ids)
                try:
                    for tok_id in gen:
                        gen_ids.append(tok_id)
                        pending_buf += detok.add_token(tok_id)
                        safe_txt, pending_buf, hit_stop = _check_stop_buffer(pending_buf, stops)
                        if safe_txt:
                            yield ("delta", safe_txt, None, 0, 0)
                        if hit_stop:
                            finish_reason = "stop"
                            break
                finally:
                    gen.close()
                if finish_reason != "stop" and len(gen_ids) < max_new:
                    finish_reason = "stop"  # the generator ended on a stop token id

                if pending_buf and finish_reason != "stop":
                    yield ("delta", pending_buf, None, 0, 0)
                elif pending_buf and finish_reason == "stop":
                    safe_tail, _, hit_end = _check_stop_buffer(pending_buf, stops)
                    final_tail = safe_tail if hit_end else pending_buf
                    if final_tail:
                        yield ("delta", final_tail, None, 0, 0)

                decode_sec = max(1e-6, time.perf_counter() - t_dec)
                eng.arena.wait_idle()

                if eng.enable_merkle_kv and gen_ids:
                    full_seq = list(prompt_ids) + gen_ids
                    if len(full_seq) >= 17 and caches[0].offset >= 16:
                        eng.kv_cache.save_prefix_blocks(full_seq[: caches[0].offset + 1], caches)  # type: ignore[arg-type]

                act_mb, peak_mb = eng.vram_mb()
                st = eng.arena.get_stats(nominal_top_k=eng.top_k)
                tps = round(max(1, len(gen_ids) - 1) / decode_sec, 2) if len(gen_ids) > 1 else 0.0
                tel = DeimosRequestTelemetry(
                    backend=eng.backend_name,
                    ttft_ms=round(ttft_ms, 2),
                    decode_tok_per_s=tps,
                    active_vram_mb=round(act_mb, 2),
                    peak_vram_mb=round(peak_mb, 2),
                    rss_mb=round(rss_mb(), 2),
                    pool_vram_mb=st["pool_vram_mb"],
                    slots_per_layer=st["slots_per_layer"],
                    hit_rate_pct=st["hit_rate_pct"],
                    sync_misses=st["cache_misses"],
                    lookahead_useful_hits=st["lookahead_useful_hits"],
                    tail_misses_skipped=st["tail_misses_skipped"],
                    avg_active_k=st["avg_active_k"],
                    kv_skipped_tokens=skipped,
                    xor_proofs=st["xor_proofs_verified"],
                    byte_proofs=st["byte_proofs_verified"],
                )
                yield ("done", finish_reason, tel, len(prompt_ids), len(gen_ids))
            finally:
                eng.miss_tail_tau = saved_tau
                eng.enable_lookahead = saved_la

    @app.post("/v1/chat/completions", response_model=None)
    def chat_completions(
        req: ChatCompletionRequest,
    ) -> Union[ChatCompletionResponse, StreamingResponse]:
        if state["engine"] is None:
            raise HTTPException(status_code=503, detail="No MLX or PyTorch execution runtime available.")
        req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created_ts = int(time.time())
        model_id = req.model or default_model_id

        if not req.stream:
            text_parts: List[str] = []
            fin_reason = "stop"
            final_tel: Optional[DeimosRequestTelemetry] = None
            p_toks, c_toks = 0, 0
            for kind, payload, tel, pn, cn in _generate_events(req):
                if kind == "delta" and payload:
                    text_parts.append(payload)
                elif kind == "done":
                    fin_reason = payload or "stop"
                    final_tel = tel
                    p_toks, c_toks = pn, cn
            assert final_tel is not None
            return ChatCompletionResponse(
                id=req_id,
                created=created_ts,
                model=model_id,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatChoiceMessage(role="assistant", content="".join(text_parts)),
                        finish_reason=fin_reason,
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=p_toks,
                    completion_tokens=c_toks,
                    total_tokens=p_toks + c_toks,
                ),
                deimos_telemetry=final_tel,
            )

        def _sse_stream() -> Iterator[str]:
            role_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"
            for kind, payload, tel, pn, cn in _generate_events(req):
                if kind == "delta" and payload:
                    c_obj = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": payload}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(c_obj)}\n\n"
                elif kind == "done" and tel is not None:
                    end_obj = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": payload}],
                        "usage": {
                            "prompt_tokens": pn,
                            "completion_tokens": cn,
                            "total_tokens": pn + cn,
                        },
                        "deimos_telemetry": tel.model_dump(),
                    }
                    yield f"data: {json.dumps(end_obj)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse_stream(), media_type="text/event-stream")

    return app
