#!/usr/bin/env python3
"""
DeimosStream CLI & Interactive Streaming Runner (`src/deimos_stream/cli.py`)
============================================================================
Subcommands:
  - `deimos-stream convert --model-dir <safetensors_dir> --output-dir <store_dir>`
  - `deimos-stream chat --store-dir <store_dir> --slots 16 --tau 0.85`
  - `deimos-stream bench --store-dir <store_dir> --slots 8,16,64 --tokens 64`
  - `deimos-stream inspect --store-dir <store_dir>`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

try:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
except ImportError:
    mx = None  # type: ignore[assignment]
    KVCache = object  # type: ignore[assignment,misc]

from deimos_stream.arena import DeimosMerkleBridge
from deimos_stream.backbone import mlx_mem_mb, rss_mb
from deimos_stream.converter import convert_safetensors_moe_to_deimos_q4
from deimos_stream.detokenizer import IncrementalStreamingDetokenizer
from deimos_stream.engine import DeimosPagedMoEEngine


class DeimosChatSession:
    """Interactive and scripted streaming session over `DeimosPagedMoEEngine`."""

    def __init__(
        self,
        store_dir: Path,
        slots_per_layer: int = 16,
        miss_tail_tau: float = 0.85,
        enable_lookahead: bool = True,
        enable_merkle_kv: bool = True,
        use_nocache_pread: bool = True,
        temperature: float = 0.0,
    ):
        self.store_dir: Path = Path(store_dir).expanduser().resolve()
        self.temperature: float = max(0.0, float(temperature))
        self.engine = DeimosPagedMoEEngine(
            self.store_dir,
            slots_per_layer=slots_per_layer,
            use_nocache_pread=use_nocache_pread,
            enable_lookahead=enable_lookahead,
            enable_merkle_kv=enable_merkle_kv,
            miss_tail_tau=miss_tail_tau,
        )
        eos_cfg = self.engine.config.get("eos_token_id", 50279)
        if isinstance(eos_cfg, list):
            self.stop_ids: Set[int] = {int(x) for x in eos_cfg}
        else:
            self.stop_ids = {int(eos_cfg), 0, 50278, 50279}
        self.history: List[Tuple[str, str]] = []

    def format_chat_prompt(self, user_text: str) -> List[int]:
        if self.engine.tokenizer is None:
            raise RuntimeError("No tokenizer.json present in Merkle store.")
        chunks: List[str] = []
        for u_prev, a_prev in self.history:
            chunks.append(
                f"<|user|>\n{u_prev.strip()}\n<|assistant|>\n{a_prev.strip()}<|endoftext|>\n"
            )
        chunks.append(f"<|user|>\n{user_text.strip()}\n<|assistant|>\n")
        return self.engine.tokenizer.encode("".join(chunks)).ids

    def _sample_token(self, logits_2d: mx.array) -> mx.array:
        if self.temperature <= 1e-5:
            return mx.argmax(logits_2d, axis=-1)
        scaled = logits_2d.astype(mx.float32) / self.temperature
        return mx.random.categorical(scaled)

    def stream_turn(
        self, user_prompt: str, max_new_tokens: int = 256, print_live: bool = True
    ) -> Dict[str, object]:
        if self.engine.tokenizer is None:
            raise RuntimeError("No tokenizer.json present in Merkle store.")
        prompt_ids = self.format_chat_prompt(user_prompt)
        caches = [KVCache() for _ in range(self.engine.num_layers)]
        detok = IncrementalStreamingDetokenizer(self.engine.tokenizer)

        self.engine.arena.reset_telemetry()
        mx.clear_cache()
        mx.reset_peak_memory()

        t_pf = time.perf_counter()
        logits, skipped = self.engine.prefill_prompt(prompt_ids, caches)
        tok_mx = self._sample_token(logits)
        mx.eval(tok_mx)
        ttft_ms = (time.perf_counter() - t_pf) * 1000.0

        self.engine.arena.reset_telemetry()
        gen_ids: List[int] = []
        t_dec = time.perf_counter()

        for step in range(max_new_tokens):
            tok_id = int(tok_mx.item())
            if tok_id in self.stop_ids:
                break
            gen_ids.append(tok_id)
            delta = detok.add_token(tok_id)
            if print_live and delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
            if step + 1 < max_new_tokens:
                logits = self.engine.forward_logits([tok_id], caches=caches)
                tok_mx = self._sample_token(logits)
                mx.eval(tok_mx)

        decode_sec = max(1e-6, time.perf_counter() - t_dec)
        self.engine.arena.wait_idle()
        if print_live:
            sys.stdout.write("\n")
            sys.stdout.flush()

        full_reply = detok.full_text().strip()
        self.history.append((user_prompt, full_reply))

        if self.engine.enable_merkle_kv and gen_ids:
            full_seq = list(prompt_ids) + gen_ids
            if len(full_seq) >= 17 and caches[0].offset >= 16:
                self.engine.kv_cache.save_prefix_blocks(full_seq[: caches[0].offset + 1], caches)

        act_mb, peak_mb = mlx_mem_mb()
        cur_rss = rss_mb()
        st = self.engine.arena.get_stats(nominal_top_k=self.engine.top_k)
        dec_count = max(1, len(gen_ids) - 1)
        tps = round(dec_count / decode_sec, 2) if len(gen_ids) > 1 else 0.0

        telemetry: Dict[str, object] = {
            "text": full_reply,
            "num_prompt_tokens": len(prompt_ids),
            "kv_skipped_tokens": skipped,
            "num_generated_tokens": len(gen_ids),
            "ttft_ms": round(ttft_ms, 2),
            "decode_tok_per_s": tps,
            "active_vram_mb": round(act_mb, 2),
            "peak_vram_mb": round(peak_mb, 2),
            "rss_mb": round(cur_rss, 2),
            "pool_vram_mb": st["pool_vram_mb"],
            "slots_per_layer": st["slots_per_layer"],
            "hit_rate_pct": st["hit_rate_pct"],
            "sync_misses": st["cache_misses"],
            "lookahead_useful_hits": st["lookahead_useful_hits"],
            "tail_misses_skipped": st["tail_misses_skipped"],
            "avg_active_k": st["avg_active_k"],
            "xor_proofs": st["xor_proofs_verified"],
            "byte_proofs": st["byte_proofs_verified"],
        }
        if print_live:
            self._print_telemetry(telemetry)
        return telemetry

    def _print_telemetry(self, t: Dict[str, object]) -> None:
        print("─" * 78)
        print(
            f"DEIMOS-STREAM PAGED-MERKLE TELEMETRY "
            f"(Slots: {t['slots_per_layer']}/{self.engine.num_experts}/layer | "
            f"LA: {'ON' if self.engine.enable_lookahead else 'OFF'} | tau={self.engine.miss_tail_tau:.2f})"
        )
        print("─" * 78)
        print(
            f"  Active / Peak GPU VRAM : {t['active_vram_mb']} MiB / {t['peak_vram_mb']} MiB "
            f"(Pool: {t['pool_vram_mb']} MiB)"
        )
        print(f"  Process RSS            : {t['rss_mb']} MiB")
        print(
            f"  Prefill TTFT           : {t['ttft_ms']} ms "
            f"(Prompt: {t['num_prompt_tokens']} toks, KV Skipped: {t['kv_skipped_tokens']} toks)"
        )
        print(
            f"  Decode Throughput      : {t['decode_tok_per_s']} tok/s "
            f"({t['num_generated_tokens']} toks, Avg K={t['avg_active_k']})"
        )
        print(
            f"  Expert Cache Hit Rate  : {t['hit_rate_pct']}% "
            f"(Sync Misses: {t['sync_misses']} | LA Hits: {t['lookahead_useful_hits']} | "
            f"Skipped Tail: {t['tail_misses_skipped']})"
        )
        print(
            f"  DEIMIDX2 Merkle Proofs : {t['xor_proofs']} XOR proofs | "
            f"{t['byte_proofs']} Cold SHA-256 Leaf Proofs"
        )
        print("─" * 78)

    def run_interactive(self, max_new_tokens: int = 256) -> None:
        print(
            f"\n[DeimosStream Ready] Store: {self.engine.store_dir} | "
            f"Slots/Layer: {self.engine.arena.slots_per_layer}/{self.engine.num_experts}\n"
            "Commands: /slots <8|16|32|64>, /tau <0.5..1.0>, /lookahead <on|off>, /clear, exit\n"
        )
        while True:
            try:
                user_in = input("\nYou > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_in:
                continue
            if user_in.lower() in {"exit", "quit", "/exit"}:
                break
            if user_in == "/clear":
                self.history.clear()
                print("[History cleared]")
                continue
            if user_in.startswith("/tau "):
                self.engine.miss_tail_tau = max(0.5, min(1.0, float(user_in.split()[1])))
                print(f"[Updated miss_tail_tau = {self.engine.miss_tail_tau:.2f}]")
                continue
            if user_in.startswith("/lookahead "):
                self.engine.enable_lookahead = user_in.split()[1].lower() in {"on", "1", "true"}
                print(f"[Updated enable_lookahead = {self.engine.enable_lookahead}]")
                continue
            if user_in.startswith("/slots "):
                new_slots = int(user_in.split()[1])
                la = self.engine.enable_lookahead
                tau = self.engine.miss_tail_tau
                kv = self.engine.enable_merkle_kv
                nc = self.engine.arena.use_nocache_pread
                self.engine.close()
                self.engine = DeimosPagedMoEEngine(
                    self.store_dir,
                    slots_per_layer=new_slots,
                    use_nocache_pread=nc,
                    enable_lookahead=la,
                    enable_merkle_kv=kv,
                    miss_tail_tau=tau,
                )
                print(
                    f"[Rebuilt PagedExpertArena with {self.engine.arena.slots_per_layer} slots/layer]"
                )
                continue

            sys.stdout.write("\nDeimosStream > ")
            sys.stdout.flush()
            self.stream_turn(user_in, max_new_tokens=max_new_tokens, print_live=True)

    def close(self) -> None:
        self.engine.close()


def inspect_store(store_dir: Path, verify_all_leaves: bool = False) -> Dict[str, object]:
    """Verifies `DEIMIDX2` Tier-1 Merkle tree (and optionally all cold SHA-256 leaves)."""
    sdir = Path(store_dir).expanduser().resolve()
    t0 = time.perf_counter()
    bridge = DeimosMerkleBridge(sdir, enable_nocache_fd=False)
    try:
        h = bridge.header
        tier1_ms = (time.perf_counter() - t0) * 1000.0
        if verify_all_leaves:
            for l_i in range(int(h.num_layers)):
                for e_i in range(int(h.num_experts)):
                    bridge.verify_leaf(l_i, e_i, verify_bytes=True)
        st = bridge.get_stats()
        info: Dict[str, object] = {
            "store_dir": str(sdir),
            "magic": bytes(h.magic).decode("ascii", errors="replace"),
            "version": int(h.version),
            "num_layers": int(h.num_layers),
            "num_experts_per_layer": int(h.num_experts),
            "total_experts": int(h.num_layers) * int(h.num_experts),
            "tree_leaf_cap": int(h.tree_leaf_cap),
            "vocab_size": int(h.vocab_size),
            "hidden_size": int(h.hidden_size),
            "intermediate_size": int(h.intermediate_size),
            "quant_bits": int(h.quant_bits),
            "quant_group_size": int(h.quant_group_size),
            "expert_stride_bytes": int(h.expert_stride_bytes),
            "kv_table_slots": int(h.kv_table_slots),
            "model_merkle_root": bytes(h.base_model_root).hex(),
            "expert_tree_root": bytes(h.expert_tree_root).hex(),
            "tier1_verify_ms": round(tier1_ms, 3),
            "xor_proofs_verified": st["xor_proofs_verified"],
            "byte_proofs_verified": st["byte_proofs_verified"],
        }
        return info
    finally:
        bridge.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="deimos-stream",
        description="DeimosStream — Low-VRAM Sparse MoE & LLM Merkle-Paged Streaming Engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_conv = sub.add_parser("convert", help="Convert .safetensors Sparse MoE to DEIMIDX2 Q4 store")
    p_conv.add_argument("--model-dir", type=Path, required=True, help="Source .safetensors folder")
    p_conv.add_argument("--output-dir", type=Path, required=True, help="Output DEIMIDX2 store")
    p_conv.add_argument("--group-size", type=int, default=64)

    p_chat = sub.add_parser("chat", help="Launch interactive or single-turn low-VRAM MoE chat")
    p_chat.add_argument("--store-dir", type=Path, required=True, help="DEIMIDX2 store folder")
    p_chat.add_argument("--slots", type=int, default=16, help="Q4 expert slots per layer in VRAM")
    p_chat.add_argument("--tau", type=float, default=0.85, help="Miss-only Nucleus Dynamic-K tau")
    p_chat.add_argument("--no-lookahead", action="store_true")
    p_chat.add_argument("--no-kv-cache", action="store_true")
    p_chat.add_argument("--mmap", action="store_true", help="Use OS mmap instead of F_NOCACHE pread")
    p_chat.add_argument("--temp", type=float, default=0.0)
    p_chat.add_argument("--max-tokens", type=int, default=256)
    p_chat.add_argument("--prompt", type=str, default=None)

    p_bench = sub.add_parser("bench", help="Benchmark VRAM, hit rate, TTFT, and decode tok/s")
    p_bench.add_argument("--store-dir", type=Path, required=True)
    p_bench.add_argument("--slots", type=int, default=16)
    p_bench.add_argument("--tau", type=float, default=0.85)
    p_bench.add_argument("--tokens", type=int, default=32)
    p_bench.add_argument(
        "--prompt",
        type=str,
        default="<|user|>\nExplain how Merkle trees verify paged neural weights.\n<|assistant|>\n",
    )

    p_insp = sub.add_parser("inspect", help="Verify DEIMIDX2 Merkle index & print topology")
    p_insp.add_argument("--store-dir", type=Path, required=True)
    p_insp.add_argument("--verify-leaves", action="store_true", help="Verify all SHA-256 leaves")

    args = parser.parse_args(argv)

    if args.command == "convert":
        out = convert_safetensors_moe_to_deimos_q4(
            args.model_dir, args.output_dir, group_size=args.group_size
        )
        print(json.dumps({"status": "ok", "output_store_dir": str(out)}, indent=2))
        return 0

    if args.command == "inspect":
        info = inspect_store(args.store_dir, verify_all_leaves=args.verify_leaves)
        print(json.dumps(info, indent=2))
        return 0

    if args.command == "bench":
        eng = DeimosPagedMoEEngine(
            args.store_dir,
            slots_per_layer=args.slots,
            use_nocache_pread=True,
            enable_lookahead=True,
            miss_tail_tau=args.tau,
        )
        try:
            if eng.tokenizer is not None:
                metrics = eng.benchmark_generation(args.prompt, max_new_tokens=args.tokens)
            else:
                metrics = eng.benchmark_token_ids([1, 2, 3, 4, 5, 6, 7, 8], max_new_tokens=args.tokens)
            print(json.dumps(metrics, indent=2))
        finally:
            eng.close()
        return 0

    if args.command == "chat":
        sess = DeimosChatSession(
            store_dir=args.store_dir,
            slots_per_layer=args.slots,
            miss_tail_tau=args.tau,
            enable_lookahead=not args.no_lookahead,
            enable_merkle_kv=not args.no_kv_cache,
            use_nocache_pread=not args.mmap,
            temperature=args.temp,
        )
        try:
            if args.prompt is not None:
                sys.stdout.write("DeimosStream > ")
                sys.stdout.flush()
                sess.stream_turn(args.prompt, max_new_tokens=args.max_tokens, print_live=True)
            else:
                sess.run_interactive(max_new_tokens=args.max_tokens)
        finally:
            sess.close()
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
