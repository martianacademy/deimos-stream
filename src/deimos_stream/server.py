#!/usr/bin/env python3
"""
DeimosStream OpenAI-Compatible FastAPI Gateway (`src/deimos_stream/server.py`)
==============================================================================
Exposes thread-safe HTTP & Server-Sent Events (SSE) endpoints over `DeimosPagedMoEEngine`:
  - `GET  /health`               — Live VRAM, RSS, Arena hit-rate & `DEIMIDX2` Merkle telemetry
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
from typing import AsyncIterator, Dict, Iterator, List, Literal, Optional, Sequence, Set, Tuple, Union
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from deimos_stream import __version__
from deimos_stream.backbone import mlx_mem_mb, rss_mb
from deimos_stream.cli import inspect_store
from deimos_stream.detokenizer import DecodeCapableTokenizer, IncrementalStreamingDetokenizer
from deimos_stream.engine import HAS_MLX, DeimosPagedMoEEngine

if HAS_MLX:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
else:
    mx = None  # type: ignore[assignment]
    KVCache = object  # type: ignore[assignment,misc]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage] = Field(..., min_length=1)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    miss_tail_tau: Optional[float] = Field(default=None, ge=0.5, le=1.0)
    enable_lookahead: Optional[bool] = None


class DeimosConfigUpdateRequest(BaseModel):
    slots_per_layer: Optional[int] = Field(default=None, ge=1, le=512)
    miss_tail_tau: Optional[float] = Field(default=None, ge=0.5, le=1.0)
    enable_lookahead: Optional[bool] = None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class DeimosRequestTelemetry(BaseModel):
    ttft_ms: float
    decode_tok_per_s: float
    active_vram_mb: float
    peak_vram_mb: float
    rss_mb: float
    pool_vram_mb: float
    slots_per_layer: int
    hit_rate_pct: float
    sync_misses: int
    lookahead_useful_hits: int
    tail_misses_skipped: int
    avg_active_k: float
    kv_skipped_tokens: int
    xor_proofs: int
    byte_proofs: int


class ChatChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatChoiceMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: CompletionUsage
    deimos_telemetry: DeimosRequestTelemetry


class _FallbackByteTokenizer:
    """Deterministic fallback tokenizer used only when a synthetic test store omits `tokenizer.json`."""

    def __init__(self, vocab_size: int):
        self.vocab_size: int = max(8, int(vocab_size))

    def encode_text(self, text: str) -> List[int]:
        raw = text.encode("utf-8")
        return [(b % (self.vocab_size - 2)) + 2 for b in raw] or [2]

    def decode(self, ids: List[int]) -> str:
        return " ".join(f"tok{int(i)}" for i in ids)


def _format_messages(messages: Sequence[ChatMessage]) -> str:
    parts: List[str] = []
    for m in messages:
        role = "system" if m.role == "developer" else m.role
        if role == "assistant":
            parts.append(f"<|assistant|>\n{m.content.strip()}<|endoftext|>\n")
        else:
            parts.append(f"<|{role}|>\n{m.content.strip()}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def _normalize_stops(stop: Optional[Union[str, List[str]]]) -> List[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    return [s for s in stop if s]


def _check_stop_buffer(buf: str, stops: Sequence[str]) -> Tuple[str, str, bool]:
    """Splits `buf` into `(safe_to_emit, holdback, matched_stop)` without leaking stop prefixes."""
    if not stops or not buf:
        return buf, "", False
    earliest: Optional[int] = None
    for s in stops:
        idx = buf.find(s)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is not None:
        return buf[:earliest], "", True

    hold = 0
    for s in stops:
        max_k = min(len(buf), len(s) - 1)
        for k in range(max_k, 0, -1):
            if buf.endswith(s[:k]) and k > hold:
                hold = k
                break
    if hold > 0:
        return buf[:-hold], buf[-hold:], False
    return buf, "", False


def create_deimos_app(
    store_dir: Path,
    slots_per_layer: int = 16,
    miss_tail_tau: float = 0.85,
    enable_lookahead: bool = True,
    enable_merkle_kv: bool = True,
    use_nocache_pread: bool = True,
    model_name: Optional[str] = None,
) -> FastAPI:
    """Creates a thread-safe OpenAI-compatible FastAPI app backed by `DeimosPagedMoEEngine`."""
    resolved_dir = Path(store_dir).expanduser().resolve()
    if not (resolved_dir / "merkle_tree.idx").exists():
        raise FileNotFoundError(f"merkle_tree.idx not found in {resolved_dir}")

    default_model_id = model_name or resolved_dir.name
    lock = threading.Lock()
    state: Dict[str, Optional[DeimosPagedMoEEngine]] = {"engine": None}

    if HAS_MLX:
        state["engine"] = DeimosPagedMoEEngine(
            resolved_dir,
            slots_per_layer=slots_per_layer,
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
        description="Universal Low-VRAM Sparse MoE & LLM Merkle-Paged Streaming Engine",
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

    def _stop_ids_for(eng: DeimosPagedMoEEngine) -> Set[int]:
        if eng.tokenizer is None:
            return set()
        eos_cfg = eng.config.get("eos_token_id", 50279)
        if isinstance(eos_cfg, list):
            return {int(x) for x in eos_cfg}
        return {int(eos_cfg), 0, 50278, 50279}

    def _encode_messages(
        eng: DeimosPagedMoEEngine, messages: Sequence[ChatMessage]
    ) -> Tuple[List[int], DecodeCapableTokenizer]:
        prompt_str = _format_messages(messages)
        if eng.tokenizer is not None:
            return eng.tokenizer.encode(prompt_str).ids, eng.tokenizer
        fb = _FallbackByteTokenizer(eng.vocab_size)
        return fb.encode_text(prompt_str), fb

    @app.get("/health")
    def health() -> Dict[str, object]:
        with lock:
            eng = state["engine"]
            if eng is None:
                info = inspect_store(resolved_dir, verify_all_leaves=False)
                return {"status": "inspect_only", "mlx_available": False, "store": info}
            act_mb, peak_mb = mlx_mem_mb()
            st = eng.arena.get_stats(nominal_top_k=eng.top_k)
            return {
                "status": "healthy",
                "mlx_available": True,
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
                raise HTTPException(status_code=503, detail="MLX Metal runtime unavailable on host.")
            if (
                req.slots_per_layer is not None
                and req.slots_per_layer != eng.arena.slots_per_layer
            ):
                la = req.enable_lookahead if req.enable_lookahead is not None else eng.enable_lookahead
                tau = req.miss_tail_tau if req.miss_tail_tau is not None else eng.miss_tail_tau
                kv = eng.enable_merkle_kv
                nc = eng.arena.use_nocache_pread
                eng.close()
                eng = DeimosPagedMoEEngine(
                    resolved_dir,
                    slots_per_layer=req.slots_per_layer,
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
                "slots_per_layer": eng.arena.slots_per_layer,
                "miss_tail_tau": eng.miss_tail_tau,
                "enable_lookahead": eng.enable_lookahead,
            }

    def _generate_events(
        req: ChatCompletionRequest, req_id: str, created_ts: int, model_id: str
    ) -> Iterator[Tuple[str, Optional[str], Optional[DeimosRequestTelemetry], int, int]]:
        """Locked generator yielding `("delta", text, None, 0, 0)` then `("done", reason, tel, p_n, c_n)`."""
        with lock:
            eng = state["engine"]
            if eng is None:
                raise HTTPException(status_code=503, detail="MLX Metal runtime unavailable on host.")
            saved_tau, saved_la = eng.miss_tail_tau, eng.enable_lookahead
            if req.miss_tail_tau is not None:
                eng.miss_tail_tau = float(req.miss_tail_tau)
            if req.enable_lookahead is not None:
                eng.enable_lookahead = bool(req.enable_lookahead)
            try:
                prompt_ids, tok_impl = _encode_messages(eng, req.messages)
                max_new = int(req.max_completion_tokens or req.max_tokens or 256)
                stops = _normalize_stops(req.stop)
                stop_ids = _stop_ids_for(eng)
                caches = [KVCache() for _ in range(eng.num_layers)]
                detok = IncrementalStreamingDetokenizer(tok_impl)

                eng.arena.reset_telemetry()
                mx.clear_cache()
                mx.reset_peak_memory()

                def _sample(logits_2d: mx.array) -> mx.array:
                    if req.temperature <= 1e-5:
                        return mx.argmax(logits_2d, axis=-1)
                    return mx.random.categorical(logits_2d.astype(mx.float32) / req.temperature)

                t_pf = time.perf_counter()
                logits, skipped = eng.prefill_prompt(prompt_ids, caches)
                tok_mx = _sample(logits)
                mx.eval(tok_mx)
                ttft_ms = (time.perf_counter() - t_pf) * 1000.0

                eng.arena.reset_telemetry()
                gen_ids: List[int] = []
                pending_buf = ""
                finish_reason = "length"
                t_dec = time.perf_counter()

                for step in range(max_new):
                    tok_id = int(tok_mx.item())
                    if tok_id in stop_ids:
                        finish_reason = "stop"
                        break
                    gen_ids.append(tok_id)
                    pending_buf += detok.add_token(tok_id)
                    safe_txt, pending_buf, hit_stop = _check_stop_buffer(pending_buf, stops)
                    if safe_txt:
                        yield ("delta", safe_txt, None, 0, 0)
                    if hit_stop:
                        finish_reason = "stop"
                        break
                    if step + 1 < max_new:
                        logits = eng.forward_logits([tok_id], caches=caches)
                        tok_mx = _sample(logits)
                        mx.eval(tok_mx)

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
                        eng.kv_cache.save_prefix_blocks(full_seq[: caches[0].offset + 1], caches)

                act_mb, peak_mb = mlx_mem_mb()
                st = eng.arena.get_stats(nominal_top_k=eng.top_k)
                tps = round(max(1, len(gen_ids) - 1) / decode_sec, 2) if len(gen_ids) > 1 else 0.0
                tel = DeimosRequestTelemetry(
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
            raise HTTPException(status_code=503, detail="MLX Metal runtime unavailable on host.")
        req_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created_ts = int(time.time())
        model_id = req.model or default_model_id

        if not req.stream:
            text_parts: List[str] = []
            fin_reason = "stop"
            final_tel: Optional[DeimosRequestTelemetry] = None
            p_toks, c_toks = 0, 0
            for kind, payload, tel, pn, cn in _generate_events(req, req_id, created_ts, model_id):
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
            for kind, payload, tel, pn, cn in _generate_events(req, req_id, created_ts, model_id):
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
