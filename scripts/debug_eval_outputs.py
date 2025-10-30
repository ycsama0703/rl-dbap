#!/usr/bin/env python
"""
debug_eval_outputs.py

Run inference on the test set and dump raw outputs together with parsed
predictions and absolute errors to a CSV for manual inspection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any

import torch

from src.backends.hf_infer import (
    build_eval_inputs,
    load_model_and_tokenizer,
    infer_chat_batch,
    extract_pred,
)


def chunked(indices: List[int], size: int):
    for i in range(0, len(indices), size):
        yield indices[i : i + size]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dump raw model outputs with parsed predictions and absolute errors."
    )
    ap.add_argument("--test-path", type=str, required=True)
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora-path", type=str, default=None, help="LoRA checkpoint directory")
    ap.add_argument("--out-csv", type=str, default="outputs/debug_eval_outputs.csv")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--torch-dtype", type=str, default="bfloat16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force-think", dest="force_think", action="store_true")
    ap.add_argument("--no-force-think", dest="force_think", action="store_false")
    ap.set_defaults(force_think=False)
    args = ap.parse_args()

    chats, y_true, quarters, ids, holding_ts = build_eval_inputs(args.test_path)
    if args.limit is not None:
        limit = min(args.limit, len(chats))
        chats = chats[:limit]
        y_true = y_true[:limit]
        quarters = quarters[:limit]
        ids = ids[:limit]
        holding_ts = holding_ts[:limit]

    tokenizer, model = load_model_and_tokenizer(
        args.base_model, args.lora_path, torch_dtype=args.torch_dtype
    )
    model.eval()

    raw_outputs: List[str] = []
    preds: List[float | None] = []

    indices = list(range(len(chats)))
    for batch in chunked(indices, args.batch_size):
        batch_msgs = [chats[i] for i in batch]
        completions = infer_chat_batch(
            tokenizer,
            model,
            batch_msgs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            force_think=args.force_think,
        )
        for idx, completion in zip(batch, completions):
            raw_outputs.append(completion)
            preds.append(extract_pred(completion, holding_ts[idx]))

    rows: List[Dict[str, Any]] = []
    for idx, quarter, yt, ht, raw, pred in zip(ids, quarters, y_true, holding_ts, raw_outputs, preds):
        entry: Dict[str, Any] = {
            "id": idx,
            "quarter": quarter,
            "holding_t": ht,
            "y_true": yt,
            "raw_output": raw,
            "parsed_pred": pred,
            "true_delta": None,
            "pred_delta": None,
            "abs_error": None,
        }
        if ht is not None and yt is not None:
            entry["true_delta"] = yt - ht
        if pred is not None and ht is not None:
            entry["pred_delta"] = pred - ht
        if pred is not None and yt is not None:
            entry["abs_error"] = abs(pred - yt)
        rows.append(entry)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "quarter",
                "holding_t",
                "y_true",
                "raw_output",
                "parsed_pred",
                "true_delta",
                "pred_delta",
                "abs_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[debug] Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
