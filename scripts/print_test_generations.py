#!/usr/bin/env python
"""
print_test_generations.py

Feed the first N processed test samples through a model (base + optional LoRA)
and print the freshly generated completions to stdout. Useful for manual
spot-checking without writing CSVs.
"""

from __future__ import annotations

import argparse
from typing import List

import torch

from src.backends.hf_infer import (
    build_eval_inputs,
    load_model_and_tokenizer,
    infer_chat_batch,
)


def chunked(indices: List[int], size: int):
    for i in range(0, len(indices), size):
        yield indices[i : i + size]


def main() -> None:
    ap = argparse.ArgumentParser(description="Print fresh model outputs for the first N test samples.")
    ap.add_argument("--test-path", type=str, required=True, help="Path to test JSONL (e.g., artifacts/test/test_banks.jsonl)")
    ap.add_argument("--base-model", type=str, required=True, help="Local path or HF repo ID for the base model")
    ap.add_argument("--lora-path", type=str, default=None, help="Optional LoRA checkpoint directory")
    ap.add_argument("--limit", type=int, default=1, help="Number of samples to run (default: 1)")
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--torch-dtype", type=str, default="bfloat16", help="torch dtype string, e.g., bfloat16/float16")
    ap.add_argument("--force-think", dest="force_think", action="store_true")
    ap.add_argument("--no-force-think", dest="force_think", action="store_false")
    ap.set_defaults(force_think=True)
    args = ap.parse_args()

    chats, y_true, quarters, ids, holding_ts, _permnos, _dates = build_eval_inputs(args.test_path)
    if len(chats) == 0:
        raise SystemExit(f"No valid samples found in {args.test_path}")

    limit = min(max(args.limit, 1), len(chats))
    chats = chats[:limit]
    y_true = y_true[:limit]
    quarters = quarters[:limit]
    ids = ids[:limit]
    holding_ts = holding_ts[:limit]

    tokenizer, model = load_model_and_tokenizer(
        args.base_model,
        args.lora_path,
        torch_dtype=args.torch_dtype,
    )
    model.eval()

    indices = list(range(len(chats)))
    for batch in chunked(indices, max(1, args.batch_size)):
        batch_msgs = [chats[i] for i in batch]
        completions = infer_chat_batch(
            tokenizer,
            model,
            batch_msgs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            force_think=args.force_think,
        )
        for i, completion in zip(batch, completions):
            print("=" * 80)
            print(f"[sample #{ids[i]} | quarter={quarters[i]} | holding_t={holding_ts[i]} | y_true={y_true[i]}]")
            print(completion.strip())
            print("=" * 80)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
