#!/usr/bin/env python3
"""
DeimosStream OpenAI API Validation Schemas & Stop-Buffer Helpers (`src/deimos_stream/schemas.py`)
=================================================================================================
"""

from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field


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
    slots_per_layer: Optional[Union[Literal["auto"], Annotated[int, Field(ge=1, le=512)]]] = None
    miss_tail_tau: Optional[float] = Field(default=None, ge=0.5, le=1.0)
    enable_lookahead: Optional[bool] = None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class DeimosRequestTelemetry(BaseModel):
    backend: str
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


class FallbackByteTokenizer:
    """Deterministic fallback tokenizer used only when a synthetic test store omits `tokenizer.json`."""

    def __init__(self, vocab_size: int):
        self.vocab_size: int = max(8, int(vocab_size))

    def encode_text(self, text: str) -> List[int]:
        raw = text.encode("utf-8")
        return [(b % (self.vocab_size - 2)) + 2 for b in raw] or [2]

    def decode(self, ids: List[int]) -> str:
        return " ".join(f"tok{int(i)}" for i in ids)


def format_chat_messages(messages: Sequence[ChatMessage], use_chatml: bool = False, append_empty_think: bool = False) -> str:
    parts: List[str] = []
    if use_chatml:
        for m in messages:
            role = "system" if m.role == "developer" else m.role
            parts.append(f"<|im_start|>{role}\n{m.content.strip()}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n" if append_empty_think else "<|im_start|>assistant\n")
        return "".join(parts)
    for m in messages:
        role = "system" if m.role == "developer" else m.role
        if role == "assistant":
            parts.append(f"<|assistant|>\n{m.content.strip()}<|endoftext|>\n")
        else:
            parts.append(f"<|{role}|>\n{m.content.strip()}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def normalize_stops(stop: Optional[Union[str, List[str]]]) -> List[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    return [s for s in stop if s]


def check_stop_buffer(buf: str, stops: Sequence[str]) -> Tuple[str, str, bool]:
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
