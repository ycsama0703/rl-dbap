#!/usr/bin/env python
"""
inspect_eval_outputs.py

Utility to run inference on a test JSONL and inspect raw model outputs vs. the parsed y_pred.
This helps diagnose issues such as every prediction collapsing to the same value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import torch

from src.backends.hf_infer import (
    build_eval_inputs,
    load_model_and_tokenizer,
    infer_chat_batch,
    extract_pred,
)


def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect raw outputs from Qwen on a test dataset (JSONL)."
    )
    ap.add_argument(
        "--test-path",
        type=str,
        required=True,
        help="Path to test JSONL (same as --test_path for run_eval).",
    )
    ap.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model name or local path.",
    )
    ap.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Optional LoRA path (same as run_eval --lora_path).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference.",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generation length (matches run_eval default 48, can be larger to capture full output).",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default deterministic).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit number of samples processed.",
    )
    ap.add_argument(
        "--out-jsonl",
        type=str,
        default=None,
        help="Optional path to save raw outputs as JSONL.",
    )
    args = ap.parse_args()

    print("[inspect] loading data...")
    chat_inputs, y_true, quarters, ids = build_eval_inputs(args.test_path)
    total = len(chat_inputs)
    if args.limit is not None:
        total = min(total, args.limit)
        chat_inputs = chat_inputs[:total]
        y_true = y_true[:total]
        quarters = quarters[:total]
        ids = ids[:total]
    print(f"[inspect] samples: {len(chat_inputs)}")

    print("[inspect] loading model...")
    tokenizer, model = load_model_and_tokenizer(
        args.base_model, args.lora_path, torch_dtype="bfloat16"
    )
    model.eval()

    raw_outputs: List[str] = []
    preds: List[float | None] = []

    print("[inspect] running inference...")
    for batch in chunked(list(range(len(chat_inputs))), args.batch_size):
        batch_msgs = [chat_inputs[i] for i in batch]
        outs = infer_chat_batch(
            tokenizer,
            model,
            batch_msgs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        raw_outputs.extend(outs)
        preds.extend(extract_pred(t) for t in outs)

    records: List[Dict[str, Any]] = []
    fail_count = 0
    ones_count = 0

    for i, (idx, quarter, yt, raw, pred) in enumerate(
        zip(ids, quarters, y_true, raw_outputs, preds)
    ):
        parsed = pred
        if parsed is None:
            fail_count += 1
        elif parsed == 1.0:
            ones_count += 1
        records.append(
            {
                "id": int(idx),
                "quarter": quarter,
                "y_true": yt,
                "raw_output": raw,
                "parsed_pred": parsed,
            }
        )

    print(f"[inspect] total samples: {len(records)}")
    print(f"[inspect] parsed_pred == None: {fail_count}")
    print(f"[inspect] parsed_pred == 1.0: {ones_count}")

    # show a few examples
    print("\n=== sample outputs ===")
    for rec in records[:5]:
        print(
            f"id={rec['id']} quarter={rec['quarter']} y_true={rec['y_true']} parsed={rec['parsed_pred']}"
        )
        print(rec["raw_output"])
        print("-" * 40)

    if args.out_jsonl:
        out_path = Path(args.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fout:
            for rec in records:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[inspect] saved raw outputs to {out_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

