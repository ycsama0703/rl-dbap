#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract user prompts from evaluation JSONL and emit lightweight files for inference.

Usage example:
    python scripts/export_infer_prompts.py \
        --in artifacts/sft/test.jsonl \
        --out-dir artifacts/test \
        --stem test

This will create:
    artifacts/test/test_prompts_base.jsonl
    artifacts/test/test_prompts_grpo.jsonl

Each line contains: {"id": <int>, "system": "...", "prompt": "..."}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_messages(fp: Path):
    with fp.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = record.get("messages", [])
            if not messages:
                continue
            system_msg = next((m for m in messages if m.get("role") == "system"), None)
            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            if not (system_msg and user_msg):
                continue
            system_text = (system_msg.get("content") or "").strip()
            user_text = (user_msg.get("content") or "").strip()
            if not user_text:
                continue
            yield idx, system_text, user_text


def write_jsonl(out_path: Path, rows):
    with out_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Convert eval JSONL with messages into inference-friendly prompts.")
    parser.add_argument("--in", dest="inp", required=True, type=str, help="Input JSONL produced for eval (with messages list).")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to store the exported prompts.")
    parser.add_argument("--stem", type=str, default="test", help="Stem name for generated files (default: test).")
    args = parser.parse_args()

    src = Path(args.inp)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, system_text, user_text in load_messages(src):
        rows.append({"id": idx, "system": system_text, "prompt": user_text})

    if not rows:
        raise ValueError("No valid prompts extracted; check input file format.")

    base_path = out_dir / f"{args.stem}_prompts_base.jsonl"
    grpo_path = out_dir / f"{args.stem}_prompts_grpo.jsonl"

    write_jsonl(base_path, rows)
    write_jsonl(grpo_path, rows)

    print(f"Exported {len(rows)} prompts -> {base_path}")
    print(f"Exported {len(rows)} prompts -> {grpo_path}")


if __name__ == "__main__":
    main()
