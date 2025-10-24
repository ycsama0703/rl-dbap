#!/usr/bin/env python
"""
Run evaluation for base / SFT / GRPO models on the same test set.

Example:
    python scripts/run_eval_suite.py \
        --test artifacts/sft/test.jsonl \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --sft-lora outputs/sft_qwen2.5_7b \
        --grpo-lora output/grpo_qwen2.5_7b/checkpoint-1000 \
        --out-dir artifacts/eval_suite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None

from src.cli.run_eval import run_one, compare


def _norm_path(p: Optional[str]) -> Optional[str]:
    if not p or p.lower() == "none":
        return None
    return str(Path(p).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate base/SFT/GRPO models on a common test set.")
    parser.add_argument("--test", type=str, default="artifacts/sft/test.jsonl", help="Path to test jsonl.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft-lora", type=str, default="")
    parser.add_argument("--grpo-lora", type=str, default="")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to store evaluation outputs.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    test_path = Path(args.test)
    if not test_path.exists():
        sys.exit(f"[eval-suite] test set not found: {test_path}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    stages = [
        ("base", None),
        ("sft", _norm_path(args.sft_lora)),
        ("grpo", _norm_path(args.grpo_lora)),
    ]

    prev_pred_csv = None
    iterator = tqdm(stages, desc="eval stages") if tqdm else stages
    for stage, lora_path in iterator:
        if stage != "base" and not lora_path:
            print(f"[eval-suite] skip {stage}: no LoRA path provided.")
            continue
        if lora_path and not Path(lora_path).exists():
            print(f"[eval-suite] skip {stage}: path missing -> {lora_path}")
            continue
        target_dir = out_root / stage
        print(f"[eval-suite] running evaluation for {stage} -> {target_dir}")
        run_one(
            model_id=args.base_model,
            lora_path=lora_path,
            test_path=str(test_path),
            out_dir=str(target_dir),
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            torch_dtype=args.torch_dtype,
        )
        pred_csv = target_dir / "pred_detail.csv"
        if prev_pred_csv is not None and pred_csv.exists():
            compare(str(prev_pred_csv), str(pred_csv), str(target_dir))
        prev_pred_csv = pred_csv if pred_csv.exists() else prev_pred_csv


if __name__ == "__main__":
    main()
