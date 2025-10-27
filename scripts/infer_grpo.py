#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick inference script for GRPO LoRA checkpoints.

Example:
    python scripts/infer_grpo.py \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --checkpoint outputs/grpo_qwen2.5_7b/v2-20251021-091539/checkpoint-1000 \
        --prompt_file examples/my_prompt.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.backends.hf_infer import (
    load_model_and_tokenizer,
    infer_chat_batch,
    extract_pred,
)


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    data = sys.stdin.read()
    if not data.strip():
        raise ValueError(
            "No prompt provided. Use --prompt, --prompt_file, or pipe text via stdin."
        )
    return data


def main():
    parser = argparse.ArgumentParser(description="Run inference with a GRPO LoRA checkpoint.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model to load (HF repo or local path).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the GRPO LoRA checkpoint directory (e.g., outputs/.../checkpoint-1000).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt text (if provided overrides --prompt_file).",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="",
        help="Path to a text file containing the prompt.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a quantitative portfolio manager. Respond with valid JSON only.",
        help="System message.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Torch dtype when loading the base model.",
    )
    args = parser.parse_args()

    prompt = read_prompt(args).strip()
    if not prompt:
        raise ValueError("Prompt is empty after stripping whitespace.")

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": prompt},
    ]

    tokenizer, model = load_model_and_tokenizer(args.base_model, args.checkpoint, args.torch_dtype)
    generations = infer_chat_batch(
        tokenizer,
        model,
        [messages],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    output = generations[0]
    print("=== Raw Model Output ===")
    print(output)
    parsed = extract_pred(output)
    if parsed is not None:
        print("\n=== Parsed holding_tp1 ===")
        print(parsed)
    else:
        print("\n(No numeric holding_tp1 found in output.)")


if __name__ == "__main__":
    main()
