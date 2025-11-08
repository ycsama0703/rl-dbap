#!/usr/bin/env python3
"""
subsample_jsonl.py

Utility to subsample a JSONL file (without replacement) and write the result to a new file.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Subsample a JSONL file to N entries.")
    ap.add_argument("--input", required=True, help="Path to source JSONL.")
    ap.add_argument("--output", required=True, help="Path to write the subsampled JSONL.")
    ap.add_argument("--limit", type=int, required=True, help="Number of entries to keep.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    ap.add_argument(
        "--keep-order",
        action="store_true",
        help="If set, keep the original order of the sampled items (after selection).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    inp = Path(args.input)
    outp = Path(args.output)
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")
    with inp.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.limit > len(rows):
        raise SystemExit(f"--limit {args.limit} exceeds total rows {len(rows)} (input: {inp})")

    rng = random.Random(args.seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    picked_idx = sorted(indices[: args.limit]) if args.keep_order else indices[: args.limit]
    picked = [rows[i] for i in picked_idx]

    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for rec in picked:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[subsample] Wrote {len(picked)} rows -> {outp}")


if __name__ == "__main__":
    main()
