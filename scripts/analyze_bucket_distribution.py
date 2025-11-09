#!/usr/bin/env python3
"""
analyze_bucket_distribution.py

Produce bucketed counts/percentages for y_true and model predictions (parsed_pred).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_EDGES = [-float("inf"), -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, float("inf")]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Bucket y_true and parsed_pred into ranges with counts and percentages."
    )
    ap.add_argument(
        "--truth-csv",
        required=True,
        help="CSV containing y_true column.",
    )
    ap.add_argument(
        "--run",
        nargs=2,
        metavar=("NAME", "CSV"),
        action="append",
        required=True,
        help="Add model prediction CSVs: --run base outputs/debug_eval_base.csv ...",
    )
    ap.add_argument(
        "--edges",
        nargs="+",
        type=float,
        help="Bucket edges (e.g., --edges -0.2 -0.1 -0.05 0.05 0.1 0.2). "
             "Always includes +/- inf endpoints.",
    )
    ap.add_argument(
        "--out-csv",
        help="Optional path to save the bucket table.",
    )
    return ap.parse_args()


def bucket_series(series: pd.Series, edges: List[float]) -> pd.Series:
    bins = [-float("inf")] + edges + [float("inf")]
    labels = []
    for i in range(len(bins) - 1):
        left, right = bins[i], bins[i + 1]
        if left == -float("inf"):
            label = f"<= {right:.2f}"
        elif right == float("inf"):
            label = f">= {left:.2f}"
        else:
            label = f"({left:.2f}, {right:.2f}]"
        labels.append(label)
    cats = pd.cut(series, bins=bins, labels=labels, include_lowest=True, right=True)
    return cats


def summarize(series: pd.Series, edges: List[float], label: str) -> pd.DataFrame:
    cats = bucket_series(series, edges)
    counts = cats.value_counts().reindex(cats.cat.categories, fill_value=0)
    total = counts.sum()
    perc = counts / total * 100
    return pd.DataFrame({
        "interval": counts.index,
        label: [f"{c} ({p:.1f}%)" for c, p in zip(counts, perc)],
    })


def main() -> None:
    args = parse_args()
    edges = args.edges or [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]

    truth_df = pd.read_csv(Path(args.truth_csv))
    if "y_true" not in truth_df.columns:
        raise SystemExit("truth CSV missing y_true column")
    truth_series = pd.to_numeric(truth_df["y_true"], errors="coerce").dropna()
    table = summarize(truth_series, edges, "y_true")

    for name, csv_path in args.run:
        df = pd.read_csv(Path(csv_path))
        if "parsed_pred" not in df.columns:
            raise SystemExit(f"{csv_path} missing parsed_pred column")
        series = pd.to_numeric(df["parsed_pred"], errors="coerce").dropna()
        table = table.merge(
            summarize(series, edges, f"{name}_pred"),
            on="interval",
            how="left",
        )

    print(table.to_string(index=False))
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out_csv, index=False)
        print(f"[bucket] saved table to {args.out_csv}")


if __name__ == "__main__":
    main()
