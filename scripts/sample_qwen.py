#!/usr/bin/env python
"""
sample_qwen.py

Quick script to sample multiple completions from Qwen/Qwen2.5-7B-Instruct
for a given prompt. It prints every sample to stdout and optionally saves
them to a JSONL file.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompt(path: pathlib.Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def prepare_inputs(tokenizer, system: str, user_prompt: str):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(chat, return_tensors="pt")
    return chat, enc


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample multiple completions from Qwen2.5-7B-Instruct.")
    ap.add_argument("--prompt-file", type=pathlib.Path, required=True,
                    help="Path to a text file containing the user prompt (without the system message).")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                    help="Model ID to load from Hugging Face Hub.")
    ap.add_argument("--system", type=str,
                    default="You are a quantitative portfolio manager. Respond with valid JSON only.",
                    help="System prompt to prepend.")
    ap.add_argument("--num-samples", type=int, default=10,
                    help="Number of outputs to sample (default: 10).")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="Maximum tokens to generate per sample (default: 256).")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="Sampling temperature (default: 0.8).")
    ap.add_argument("--top-p", type=float, default=0.9,
                    help="Top-p nucleus sampling (default: 0.9).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility.")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="Optional path to save outputs as JSONL.")
    ap.add_argument("--raw-output", action="store_true",
                    help="If set, print/store the full decoded text without removing the prompt prefix.")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[sample_qwen] loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    user_prompt = load_prompt(args.prompt_file)
    chat_template, encoded = prepare_inputs(tokenizer, args.system, user_prompt)
    encoded = encoded.to(model.device)

    outputs: List[str] = []
    print(f"[sample_qwen] sampling {args.num_samples} completions...")
    for i in range(args.num_samples):
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        if args.raw_output:
            completion = text.strip()
        else:
            completion = text[len(chat_template):].strip()
        outputs.append(completion)
        print(f"\n=== sample {i + 1} ===")
        print(completion)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fout:
            for completion in outputs:
                fout.write(json.dumps({"output": completion}, ensure_ascii=False) + "\n")
        print(f"\n[sample_qwen] saved outputs to {args.out}")


if __name__ == "__main__":
    main()
