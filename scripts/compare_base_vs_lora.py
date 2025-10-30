#!/usr/bin/env python
"""
compare_base_vs_lora.py

Generate side-by-side outputs for a subset of test prompts using the
vanilla base model and a GRPO-finetuned LoRA checkpoint. Results are
stored in a CSV with raw completions, parsed deltas, abs predictions,
and extracted <think> segments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.backends.hf_infer import (
    build_eval_inputs,
    load_model_and_tokenizer,
    infer_chat_batch,
)

ANSWER_JSON = re.compile(r'\{.*\}', re.S)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.IGNORECASE)
DELTA_RE = re.compile(r'"holding_delta"\s*:\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE)
TP1_RE = re.compile(r'"holding_tp1"\s*:\s*(-?\d+(?:\.\d+)?)', re.IGNORECASE)


def parse_completion(raw: str, holding_t: Optional[float]) -> Dict[str, Any]:
    """Extract delta prediction, absolute prediction, and think segment from raw text."""
    think_match = THINK_RE.search(raw or "")
    think_text = think_match.group(1).strip() if think_match else ""

    delta = None
    tp1 = None

    # try to parse JSON object if present
    try:
        json_match = ANSWER_JSON.search(raw or "")
        if json_match:
            obj = json.loads(json_match.group(0))
            if isinstance(obj, dict):
                if obj.get("holding_delta") is not None:
                    delta = float(obj["holding_delta"])
                elif obj.get("holding_tp1") is not None:
                    tp1 = float(obj["holding_tp1"])
    except Exception:
        pass

    if delta is None:
        m_delta = DELTA_RE.search(raw or "")
        if m_delta:
            try:
                delta = float(m_delta.group(1))
            except Exception:
                delta = None

    if tp1 is None:
        m_tp1 = TP1_RE.search(raw or "")
        if m_tp1:
            try:
                tp1 = float(m_tp1.group(1))
            except Exception:
                tp1 = None

    abs_pred = None
    if delta is not None and holding_t is not None and math.isfinite(holding_t):
        abs_pred = holding_t + delta
    elif tp1 is not None:
        abs_pred = tp1

    return {
        "raw": raw,
        "think": think_text,
        "delta": delta,
        "tp1": tp1,
        "abs_pred": abs_pred,
    }


def generate_outputs(model_id: str, lora_path: Optional[str], chats: List[List[Dict[str, str]]],
                     max_new_tokens: int, temperature: float, torch_dtype: str,
                     force_think: bool) -> List[str]:
    tokenizer, model = load_model_and_tokenizer(model_id, lora_path, torch_dtype)
    model.eval()
    return infer_chat_batch(
        tokenizer,
        model,
        chats,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        force_think=force_think,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare base vs LoRA model outputs on a subset of prompts.")
    ap.add_argument("--test-path", type=str, required=True)
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B-Instruct")
    ap.add_argument("--lora-path", type=str, required=True, help="LoRA checkpoint directory")
    ap.add_argument("--out-csv", type=str, default="outputs/base_vs_lora.csv")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--torch-dtype", type=str, default="bfloat16")
    ap.add_argument("--force-think", dest="force_think", action="store_true",
                    help="Force generations to begin with <think>.")
    ap.add_argument("--no-force-think", dest="force_think", action="store_false",
                    help="Disable forced <think> prefix (default).")
    ap.set_defaults(force_think=False)
    args = ap.parse_args()

    chat_inputs, y_true, quarters, ids, holding_ts = build_eval_inputs(args.test_path)
    limit = min(args.limit, len(chat_inputs))
    subset_idxs = list(range(limit))

    # Generate outputs
    base_completions = generate_outputs(
        args.base_model,
        None,
        [chat_inputs[i] for i in subset_idxs],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        torch_dtype=args.torch_dtype,
        force_think=args.force_think,
    )
    lora_completions = generate_outputs(
        args.base_model,
        args.lora_path,
        [chat_inputs[i] for i in subset_idxs],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        torch_dtype=args.torch_dtype,
        force_think=args.force_think,
    )

    rows = []
    for idx, base_raw, lora_raw in zip(subset_idxs, base_completions, lora_completions):
        holding_t = holding_ts[idx]
        base_info = parse_completion(base_raw, holding_t)
        lora_info = parse_completion(lora_raw, holding_t)

        rows.append({
            "id": ids[idx],
            "quarter": quarters[idx],
            "holding_t": holding_t,
            "y_true": y_true[idx],
            "base_raw": base_info["raw"],
            "base_think": base_info["think"],
            "base_delta": base_info["delta"],
            "base_tp1": base_info["tp1"],
            "base_abs_pred": base_info["abs_pred"],
            "lora_raw": lora_info["raw"],
            "lora_think": lora_info["think"],
            "lora_delta": lora_info["delta"],
            "lora_tp1": lora_info["tp1"],
            "lora_abs_pred": lora_info["abs_pred"],
        })

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[compare] saved comparison for {len(rows)} samples to {out_path}")


if __name__ == "__main__":
    main()
