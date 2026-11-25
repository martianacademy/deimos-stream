#!/usr/bin/env python3
"""
OpenAI Gateway (`server.py`) & Wheel Packaging Suite (`tests/test_server_and_packaging.py`)
===========================================================================================
Verifies:
  1. Cross-platform `/health`, `/v1/models`, and stop-sequence holdback logic (`_check_stop_buffer`).
  2. Standalone MLX `POST /v1/chat/completions` (`stream=False` & `stream=True` SSE) exact parity,
     stop-string truncation, and live `POST /v1/deimos/config` arena hot-reconfiguration.
  3. Real-hardware 6.92B `OLMoE-1B-7B` OpenAI SSE streaming + multi-turn Merkle KV-prefix reuse.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import List
import unittest

from fastapi.testclient import TestClient
import numpy as np

from deimos_stream.engine import HAS_MLX
from deimos_stream.merkle_builder import build_deimidx2_file, compute_q4_leaf_layout
from deimos_stream.server import _check_stop_buffer, create_deimos_app
from deimos_stream.torch_arena import HAS_TORCH
from tests.test_merkle_and_arena import build_synthetic_moe_store, find_real_olmoe_store


def _parse_sse_events(raw_body: str) -> List[dict]:
    events: List[dict] = []
    for line in raw_body.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: ") :].strip()
            if payload == "[DONE]":
                events.append({"done_marker": True})
            else:
                events.append(json.loads(payload))
    return events


class TestDeimosServerAndPackaging(unittest.TestCase):
    def test_1_stop_buffer_and_cross_platform_health_models(self) -> None:
        safe, hold, hit = _check_stop_buffer("Hello world<|en", ["<|endoftext|>"])
        self.assertEqual(safe, "Hello world")
        self.assertEqual(hold, "<|en")
        self.assertFalse(hit)

        safe2, hold2, hit2 = _check_stop_buffer(hold + "doftext|>trailing", ["<|endoftext|>"])
        self.assertEqual(safe2, "")
        self.assertEqual(hold2, "")
        self.assertTrue(hit2)

        if not HAS_MLX and not HAS_TORCH:
            with tempfile.TemporaryDirectory() as tmp:
                tdir = Path(tmp)
                layout = compute_q4_leaf_layout(128, 64, group_size=64)
                stride = layout["stride_bytes"]
                blk = np.zeros((stride,), dtype=np.uint8).tobytes()
                (tdir / "cas_experts_q4.bin").write_bytes(blk * 4)
                build_deimidx2_file(
                    tdir / "merkle_tree.idx",
                    2,
                    2,
                    256,
                    128,
                    64,
                    4,
                    64,
                    stride,
                    layout["payload_bytes"],
                    [hashlib.sha256(blk).digest()] * 4,
                )
                app = create_deimos_app(tdir, model_name="deimos-ci-model")
                with TestClient(app) as client:
                    h = client.get("/health").json()
                    self.assertEqual(h["status"], "inspect_only")
                    m = client.get("/v1/models").json()
                    self.assertEqual(m["data"][0]["id"], "deimos-ci-model")
                    r = client.post(
                        "/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "hi"}]},
                    )
                    self.assertEqual(r.status_code, 503)

    @unittest.skipUnless(HAS_MLX, "MLX Metal GPU runtime requires macOS arm64")
    def test_2_openai_json_and_sse_streaming_parity_and_hot_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = build_synthetic_moe_store(
                Path(tmp), num_layers=2, num_experts=12, top_k=4, with_shared_expert=True
            )
            app = create_deimos_app(
                store_dir, slots_per_layer=4, miss_tail_tau=1.0, model_name="synth-moe-q4"
            )
            with TestClient(app) as client:
                h = client.get("/health").json()
                self.assertEqual(h["status"], "healthy")
                self.assertEqual(h["slots_per_layer"], 4)

                models = client.get("/v1/models").json()
                self.assertEqual(models["data"][0]["id"], "synth-moe-q4")

                req_body = {
                    "model": "synth-moe-q4",
                    "max_tokens": 10,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": "Verify Merkle paging parity."}],
                }
                r_json = client.post("/v1/chat/completions", json={**req_body, "stream": False})
                self.assertEqual(r_json.status_code, 200)
                body_json = r_json.json()
                json_text = body_json["choices"][0]["message"]["content"]
                self.assertEqual(body_json["usage"]["completion_tokens"], 10)

                r_sse = client.post("/v1/chat/completions", json={**req_body, "stream": True})
                self.assertEqual(r_sse.status_code, 200)
                self.assertIn("text/event-stream", r_sse.headers["content-type"])
                events = _parse_sse_events(r_sse.text)
                self.assertTrue(events[-1].get("done_marker"))
                streamed_text = "".join(
                    ev["choices"][0]["delta"].get("content", "")
                    for ev in events
                    if "choices" in ev
                )
                self.assertEqual(streamed_text, json_text)
                self.assertIn("deimos_telemetry", events[-2])

                r_cfg = client.post(
                    "/v1/deimos/config", json={"slots_per_layer": 8, "miss_tail_tau": 0.85}
                )
                self.assertEqual(r_cfg.status_code, 200)
                self.assertEqual(r_cfg.json()["slots_per_layer"], 8)

    @unittest.skipUnless(HAS_MLX, "MLX Metal GPU runtime requires macOS arm64")
    def test_3_real_olmoe_openai_sse_and_multiturn_kv_reuse_if_present(self) -> None:
        olmoe_dir = find_real_olmoe_store()
        if olmoe_dir is None:
            self.skipTest("Local 6.92B OLMoE Q4 store not present (CI environment)")

        app = create_deimos_app(
            olmoe_dir, slots_per_layer=16, miss_tail_tau=0.85, model_name="olmoe-1b-7b-q4"
        )
        with TestClient(app) as client:
            msgs = [
                {
                    "role": "user",
                    "content": "Explain virtual memory page tables in one short sentence.",
                }
            ]
            r1 = client.post(
                "/v1/chat/completions",
                json={"model": "olmoe-1b-7b-q4", "messages": msgs, "max_tokens": 24, "stream": True},
            )
            self.assertEqual(r1.status_code, 200)
            ev1 = _parse_sse_events(r1.text)
            reply1 = "".join(
                e["choices"][0]["delta"].get("content", "") for e in ev1 if "choices" in e
            ).strip()
            self.assertGreater(len(reply1), 10)

            msgs.append({"role": "assistant", "content": reply1})
            msgs.append({"role": "user", "content": "Why does an LFU cache help?"})
            r2 = client.post(
                "/v1/chat/completions",
                json={"model": "olmoe-1b-7b-q4", "messages": msgs, "max_tokens": 16, "stream": False},
            )
            self.assertEqual(r2.status_code, 200)
            tel2 = r2.json()["deimos_telemetry"]
            self.assertGreaterEqual(int(tel2["kv_skipped_tokens"]), 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
