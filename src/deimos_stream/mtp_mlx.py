#!/usr/bin/env python3
"""
DeimosStream Native Apple Silicon `MLX` Multi-Token Prediction (`MTP`) Drafter
(`src/deimos_stream/mtp_mlx.py`)
==============================================================================
Executes checkpoint-native MTP heads (`mtp_head.safetensors`) on Apple Silicon Metal
using `mx.gather_qmm` for 4-bit Group-64 Q4 MTP experts and zero-copy shared `embed_tokens`
and `lm_head` from `DeimosMLXHybridMoEEngine`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from deimos_stream.speculative import _trim_kv_only

try:
    import mlx.core as mx
    import mlx.nn as mx_nn

    HAS_MLX = True
except ImportError:
    mx = None  # type: ignore[assignment]
    mx_nn = None  # type: ignore[assignment]
    HAS_MLX = False


class MLXNativeMTPDrafter:
    """Native Apple Silicon `MLX` MTP drafter for `DeimosMLXHybridMoEEngine` (`Qwen3.5/3.6`)."""

    is_stateful_mtp: bool = True

    def __init__(self, eng: Any, mtp_path: Path, manifest: Dict[str, Any]):
        from mlx_lm.models.qwen3_5 import TextModel as Qwen35TextModel
        from mlx_lm.models.qwen3_5 import TextModelArgs as Qwen35TextModelArgs
        from mlx_lm.models.qwen3_next import Qwen3NextMLP

        self.eng = eng
        self.zero_centered = bool(manifest.get("zero_centered_norm", True))
        self.eps = float(eng.config.get("rms_norm_eps", 1e-6))
        self.group_size = int(manifest.get("quant_group_size", 64))
        self.has_moe = bool(manifest.get("has_routed_moe", True))

        raw_w: Dict[str, Any] = dict(mx.load(str(mtp_path)))  # type: ignore[arg-type]
        self.enorm_w = raw_w.pop("mtp.enorm.weight").astype(mx.float32)
        self.hnorm_w = raw_w.pop("mtp.hnorm.weight").astype(mx.float32)
        self.head_norm_w = raw_w.pop("mtp.head_norm.weight").astype(mx.float32)
        self.eh_proj_w = raw_w.pop("mtp.eh_proj.weight").astype(mx.float16)

        if self.has_moe:
            self.gu_w = raw_w.pop("experts.gu_qw").view(mx.uint32)
            self.gu_s = raw_w.pop("experts.gu_qs").astype(mx.float16)
            self.gu_b = raw_w.pop("experts.gu_qb").astype(mx.float16)
            self.dn_w = raw_w.pop("experts.dn_qw").view(mx.uint32)
            self.dn_s = raw_w.pop("experts.dn_qs").astype(mx.float16)
            self.dn_b = raw_w.pop("experts.dn_qb").astype(mx.float16)

        m_cfg = dict(eng.config)
        m_cfg["num_hidden_layers"] = 1
        m_cfg["full_attention_interval"] = 1
        m_cfg["layer_types"] = ["full_attention"]
        t_args = Qwen35TextModelArgs.from_dict(m_cfg)
        t_model = Qwen35TextModel(t_args)
        t_model.model.ssm_idx = None
        t_model.model.fa_idx = 0
        t_model.model.embed_tokens = mx_nn.Identity()
        t_model.model.norm = mx_nn.Identity()

        if self.has_moe:
            drafter_ref = self

            class _MLXMTPMoE(mx_nn.Module):
                def __init__(self, args: Any):
                    super().__init__()
                    self.top_k = args.num_experts_per_tok
                    self.norm_topk_prob = args.norm_topk_prob
                    self.gate = mx_nn.Linear(args.hidden_size, args.num_experts, bias=False)
                    self.shared_expert = Qwen3NextMLP(
                        int(args.hidden_size), int(args.shared_expert_intermediate_size)
                    )
                    self.shared_expert_gate = mx_nn.Linear(args.hidden_size, 1, bias=False)

                def __call__(self, x: "mx.array") -> "mx.array":
                    orig = x.shape
                    x2 = x.reshape(-1, orig[-1])
                    gates = mx.softmax(self.gate(x2).astype(mx.float32), axis=-1)
                    inds = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k :]
                    scores = mx.take_along_axis(gates, inds, axis=-1)
                    if self.norm_topk_prob:
                        scores = scores / mx.maximum(mx.sum(scores, axis=-1, keepdims=True), 1e-20)
                    scores = scores.astype(x2.dtype)
                    gu = mx.gather_qmm(
                        x2[:, None, None, :],
                        drafter_ref.gu_w,
                        drafter_ref.gu_s,
                        drafter_ref.gu_b,
                        rhs_indices=inds.astype(mx.uint32),
                        transpose=True,
                        group_size=drafter_ref.group_size,
                        bits=4,
                    )
                    g_part, u_part = mx.split(gu, 2, axis=-1)
                    act = mx_nn.silu(g_part) * u_part
                    dn = mx.gather_qmm(
                        act,
                        drafter_ref.dn_w,
                        drafter_ref.dn_s,
                        drafter_ref.dn_b,
                        rhs_indices=inds.astype(mx.uint32),
                        transpose=True,
                        group_size=drafter_ref.group_size,
                        bits=4,
                    ).squeeze(2)
                    y_exp = mx.sum(dn * scores[..., None], axis=1)
                    y_sh = mx.sigmoid(self.shared_expert_gate(x2)) * self.shared_expert(x2)
                    return (y_exp + y_sh).reshape(orig)

            t_model.model.layers[0].mlp = _MLXMTPMoE(t_args)

        san = t_model.sanitize(raw_w)
        raw_w.clear()
        cast_w = {
            k: (v.astype(mx.float16) if v.dtype in (mx.bfloat16, mx.float32) and not k.endswith("A_log") else v)
            for k, v in san.items()
        }
        t_model.load_weights(list(cast_w.items()), strict=False)
        mx.eval(t_model.parameters(), self.enorm_w, self.hnorm_w, self.head_norm_w, self.eh_proj_w)
        self.t_model = t_model
        self._caches: List[Any] = t_model.make_cache()
        self._pending_h: Optional[mx.array] = None

    def _rms(self, x: "mx.array", w: "mx.array") -> "mx.array":
        scale = (1.0 + w) if self.zero_centered else w
        return mx.fast.rms_norm(x.astype(mx.float32), scale, self.eps).astype(mx.float16)

    def _fuse_eh(self, tok_ids: Sequence[int], h_seq: "mx.array") -> "mx.array":
        inp = mx.array([list(tok_ids)], dtype=mx.int32)
        emb = self.eng._mmap_emb(inp).astype(mx.float16)
        cat = mx.concatenate(
            [self._rms(emb, self.enorm_w), self._rms(h_seq.astype(mx.float16), self.hnorm_w)], axis=-1
        )
        return cat @ self.eh_proj_w.T

    def _run_layer(self, z0: "mx.array") -> "mx.array":
        h_raw = self.t_model.model(z0, cache=self._caches, input_embeddings=z0)
        return self._rms(h_raw, self.head_norm_w)

    def on_prefill(self, prompt_ids: Sequence[int], prefill_hidden: Optional["mx.array"]) -> None:
        self._caches = self.t_model.make_cache()
        self._pending_h = None
        if prefill_hidden is None or len(prompt_ids) == 0:
            return
        h_3d = prefill_hidden.reshape(1, -1, int(self.eh_proj_w.shape[0]))
        if len(prompt_ids) >= 2 and int(h_3d.shape[1]) == len(prompt_ids):
            h_pf = self._run_layer(self._fuse_eh(prompt_ids[1:], h_3d[:, :-1, :]))
            mx.eval(h_pf)
        self._pending_h = h_3d[:, -1:, :]
        mx.eval(self._pending_h)

    def draft_step(self, cur_tok: int, limit: int) -> List[int]:
        if limit <= 0 or self._pending_h is None:
            return []
        h_cur = self._run_layer(self._fuse_eh([int(cur_tok)], self._pending_h))
        l1 = self.eng.model.lm_head(h_cur)[:, -1, :]
        d1_mx = mx.argmax(l1, axis=-1)
        mx.eval(d1_mx, h_cur)
        d1 = int(d1_mx.item())
        drafts = [d1]
        if limit <= 1:
            return drafts

        extra = 0
        prev_tok = d1
        prev_h = h_cur
        for _ in range(1, int(limit)):
            prev_h = self._run_layer(self._fuse_eh([prev_tok], prev_h))
            extra += 1
            lk = self.eng.model.lm_head(prev_h)[:, -1, :]
            dk_mx = mx.argmax(lk, axis=-1)
            mx.eval(dk_mx, prev_h)
            prev_tok = int(dk_mx.item())
            drafts.append(prev_tok)

        if extra > 0:
            _trim_kv_only(self._caches, extra)
        return drafts

    def on_target_step(
        self,
        cur_tok: int,
        accepted_drafts: Sequence[int],
        target_hidden_seq: Optional["mx.array"],
        draft_ran: bool,
    ) -> None:
        if target_hidden_seq is None:
            self._pending_h = None
            return
        if not draft_ran and self._pending_h is not None:
            h0 = self._run_layer(self._fuse_eh([int(cur_tok)], self._pending_h))
            mx.eval(h0)
        h_3d = target_hidden_seq.reshape(1, -1, int(self.eh_proj_w.shape[0]))
        n_acc = len(accepted_drafts)
        if n_acc > 0 and int(h_3d.shape[1]) >= n_acc + 1:
            h_acc = self._run_layer(self._fuse_eh(accepted_drafts, h_3d[:, :n_acc, :]))
            mx.eval(h_acc)
        idx_last = min(n_acc, int(h_3d.shape[1]) - 1)
        self._pending_h = h_3d[:, idx_last : idx_last + 1, :]
        mx.eval(self._pending_h)

    def __call__(self, cur_tok: int, limit: int) -> List[int]:
        return self.draft_step(cur_tok, limit)
