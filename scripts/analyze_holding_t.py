#!/usr/bin/env python
"""
analyze_holding_t.py

Quick script to inspect statistics (min, max, mean, std, quantiles) of `holding_t`
in a prompts JSONL file (e.g., artifacts/prompts_hist_sft/banks.jsonl).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import math


def load_holding_t(path: Path) -> List[float]:
    values: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ht = rec.get("holding_t")
                if ht is None:
                    continue
                val = float(ht)
                if math.isfinite(val):
                    values.append(val)
            except Exception:
                continue
    return values


def describe(values: List[float]) -> dict:
    if not values:
        return {}
    import numpy as np

    arr = np.array(values, dtype=float)
    stats = {
        "count": arr.size,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Show statistics of holding_t in a prompts JSONL.")
    ap.add_argument("--file", type=str, required=True,
                    help="Path to prompts JSONL (e.g., artifacts/prompts_hist_sft/banks.jsonl).")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(path)

    values = load_holding_t(path)
    if not values:
        print("No holding_t values found.")
        return

    stats = describe(values)
    print(f"File: {path}")
    print(f"Samples: {stats.pop('count')}")
    for key, val in stats.items():
        print(f"{key}: {val:.6f}")


if __name__ == "__main__":
    main()

