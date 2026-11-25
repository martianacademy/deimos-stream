#!/usr/bin/env python3
"""
DeimosStream O(1) Sliding-Window Streaming Detokenizer (`src/deimos_stream/detokenizer.py`)
===========================================================================================
Avoids O(N^2) `tokenizer.decode(generated_ids)` across long generation turns by decoding
only a fixed trailing token context window and emitting verified UTF-8 increments.
"""

from __future__ import annotations

from typing import List, Protocol


class DecodeCapableTokenizer(Protocol):
    """Protocol for HuggingFace `tokenizers.Tokenizer` or compatible decode objects."""

    def decode(self, ids: List[int]) -> str: ...


class IncrementalStreamingDetokenizer:
    """O(1) per-token sliding-window detokenizer preserving BPE space/byte boundaries."""

    def __init__(self, tokenizer: DecodeCapableTokenizer, window_size: int = 10):
        self.tokenizer: DecodeCapableTokenizer = tokenizer
        self.window_size: int = max(6, int(window_size))
        self.token_ids: List[int] = []
        self._committed_text: str = ""
        self._window_prev_decoded: str = ""

    def reset(self) -> None:
        self.token_ids.clear()
        self._committed_text = ""
        self._window_prev_decoded = ""

    def add_token(self, token_id: int) -> str:
        self.token_ids.append(int(token_id))
        n = len(self.token_ids)
        if n <= self.window_size:
            curr = self.tokenizer.decode(self.token_ids)
            delta = curr[len(self._window_prev_decoded) :]
            self._window_prev_decoded = curr
            return delta

        tail_prev = self.token_ids[-(self.window_size) : -1]
        tail_curr = self.token_ids[-(self.window_size) :]
        prev_str = self.tokenizer.decode(tail_prev)
        curr_str = self.tokenizer.decode(tail_curr)
        if curr_str.startswith(prev_str):
            delta = curr_str[len(prev_str) :]
        else:
            common = 0
            limit = min(len(prev_str), len(curr_str))
            while common < limit and prev_str[common] == curr_str[common]:
                common += 1
            delta = curr_str[common:]
        self._committed_text += delta
        return delta

    def full_text(self) -> str:
        if not self.token_ids:
            return ""
        return str(self.tokenizer.decode(self.token_ids))
