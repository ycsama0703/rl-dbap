#!/usr/bin/env python3
"""
analyze_output_distributions.py

Compute descriptive statistics for multiple debug_eval_outputs CSVs
(e.g., base / SFT / GRPO). Useful for比较不同模型预测、误差分布。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


COLUMNS = [
    "holding_t",
    "y_true",
    "parsed_pred",
    "true_tp1",
    "pred_tp1",
    "abs_error",
    "abs_tp1_error",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute per-column distribution stats for multiple debug_eval_outputs CSVs."
    )
    ap.add_argument(
        "--run",
        nargs=2,
        metavar=("NAME", "CSV"),
        action="append",
        required=True,
        help="Add a run to compare, e.g. --run base outputs/debug_eval_outputs_base.csv",
    )
    ap.add_argument(
        "--truth-csv",
        help="Optional separate CSV for y_true statistics (if omitted, y_true is read from each run).",
    )
ap.add_argument(
    "--columns",
    nargs="+",
    default=COLUMNS,
    help=f"Columns to summarize (default: {', '.join(COLUMNS)})",
)
    ap.add_argument(
        "--out-csv",
        help="Optional path to save the long-form statistics table.",
    )
    return ap.parse_args()


def summarize_series(series: pd.Series) -> dict:
    data = pd.to_numeric(series, errors="coerce").dropna()
    if data.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(data.count()),
        "mean": float(data.mean()),
        "std": float(data.std(ddof=0)),
        "min": float(data.min()),
        "p25": float(data.quantile(0.25)),
        "median": float(data.quantile(0.5)),
        "p75": float(data.quantile(0.75)),
        "max": float(data.max()),
    }


def analyze_run(name: str, csv_path: Path, columns: List[str]) -> List[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"{name}: CSV not found -> {csv_path}")
    df = pd.read_csv(csv_path)
    rows: List[dict] = []
    for col in columns:
        if col not in df.columns:
            print(f"[warn] {name}: column '{col}' missing, skipping.")
            continue
        stats = summarize_series(df[col])
        stats.update({"model": name, "column": col})
        rows.append(stats)
    return rows


def main() -> None:
    args = parse_args()
stats_rows: List[dict] = []
# Determine which columns to compute per run (y_true handled separately if truth_csv provided)
run_columns = list(args.columns)
if args.truth_csv:
    stats_rows.extend(analyze_run("truth", Path(args.truth_csv), ["y_true"]))
    run_columns = [col for col in args.columns if col != "y_true"]

for name, csv_path in args.run:
    stats_rows.extend(analyze_run(name, Path(csv_path), run_columns))

    if not stats_rows:
        raise SystemExit("No statistics computed (check column names / CSV paths).")

    stats_df = pd.DataFrame(stats_rows)
    stats_df = stats_df[
        ["model", "column", "count", "mean", "std", "min", "p25", "median", "p75", "max"]
    ]

    # Nicely formatted pivot for stdout
    pivot = stats_df.pivot(index="column", columns="model", values="mean")
    print("=== Mean values by column ===")
    print(pivot)
    print("\n=== Detailed stats ===")
    print(stats_df.to_string(index=False))

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(out_path, index=False)
        print(f"[info] saved stats to {out_path}")


if __name__ == "__main__":
    main()
