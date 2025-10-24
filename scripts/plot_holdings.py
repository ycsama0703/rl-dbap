#!/usr/bin/env python
"""
Quick visualization for holding_t / label_tp1 (next-period holdings).

Usage:
    python scripts/plot_holdings.py --data artifacts/samples/grpo/banks.jsonl --out plots/holdings.png

If --out is omitted, the plot will be shown interactively.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def collect_values(fp: Path) -> tuple[np.ndarray, np.ndarray]:
    cur_vals = []
    next_vals = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key, bucket in (
                ("holding_t", cur_vals),
                ("label_tp1", next_vals),
            ):
                val = obj.get(key)
                if val is None:
                    continue
                try:
                    bucket.append(float(val))
                except (TypeError, ValueError):
                    continue
    return np.array(cur_vals, dtype=np.float32), np.array(next_vals, dtype=np.float32)


def _percentile_bins(values: np.ndarray, num_bins: int = 100) -> Iterable[float]:
    pct = np.linspace(0, 100, num_bins + 1)
    return np.unique(np.percentile(values, pct))


def plot_histograms(cur: np.ndarray, nxt: np.ndarray, out_path: Path | None = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    def _hist(ax, data: np.ndarray, title: str, bins: Iterable[float], log: bool = False):
        if data.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            return
        ax.hist(data, bins=bins, color="#4c72b0", alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel("holding value")
        ax.set_ylabel("count")
        if log:
            ax.set_yscale("log")

    # Raw histograms with symmetric fixed-width bins
    for idx, (vals, label) in enumerate(((cur, "holding_t"), (nxt, "label_tp1"))):
        if vals.size:
            span = np.percentile(vals, [0.5, 99.5])
            bins = np.linspace(span[0], span[1], 60)
        else:
            bins = 10
        _hist(axes[idx], vals, f"{label} histogram (trimmed 0.5-99.5%)", bins)

    # Percentile-based bins (robust to extremes) with log y-scale
    for idx, (vals, label) in enumerate(((cur, "holding_t"), (nxt, "label_tp1")), start=2):
        bins = _percentile_bins(vals) if vals.size else 10
        _hist(axes[idx], vals, f"{label} histogram (percentile bins, log count)", bins, log=True)

    fig.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize holding distributions from GRPO dataset JSONL.")
    parser.add_argument("--data", type=Path, required=True, help="Path to GRPO dataset jsonl.")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to save plot image.")
    args = parser.parse_args()

    cur, nxt = collect_values(args.data)
    print(f"Loaded {cur.size} current holdings, {nxt.size} next-period holdings.")
    if cur.size:
        print(f"holding_t mean={cur.mean():.4f} std={cur.std():.4f} min={cur.min():.4f} max={cur.max():.4f}")
    if nxt.size:
        print(f"label_tp1 mean={nxt.mean():.4f} std={nxt.std():.4f} min={nxt.min():.4f} max={nxt.max():.4f}")
    plot_histograms(cur, nxt, args.out)


if __name__ == "__main__":
    main()
