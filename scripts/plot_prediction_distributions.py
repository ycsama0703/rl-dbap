#!/usr/bin/env python3
"""
plot_prediction_distributions.py

Create comparison plots (histogram + KDE) for y_true and models' parsed_pred distributions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Visualize y_true vs model predictions (parsed_pred) distributions."
    )
    ap.add_argument(
        "--truth-csv",
        required=True,
        help="CSV containing y_true column (e.g., any debug_eval_outputs file).",
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
        "--bins",
        type=int,
        default=60,
        help="Number of histogram bins (default: 60).",
    )
    ap.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        help="Optional x-axis limits, e.g., --xlim -1 1",
    )
    ap.add_argument(
        "--out",
        default="outputs/prediction_distributions.png",
        help="Output plot path (default: outputs/prediction_distributions.png).",
    )
    return ap.parse_args()


def load_series(csv_path: Path, column: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"{csv_path} missing column {column}")
    return pd.to_numeric(df[column], errors="coerce").dropna()


def main() -> None:
    args = parse_args()
    truth = load_series(Path(args.truth_csv), "y_true")

    plt.figure(figsize=(10, 6))
    sns.histplot(truth, bins=args.bins, color="black", label="truth (y_true)", stat="density", kde=True, alpha=0.35)

    for name, csv_path in args.run:
        preds = load_series(Path(csv_path), "parsed_pred")
        sns.kdeplot(preds, label=f"{name} parsed_pred")
        sns.histplot(preds, bins=args.bins, alpha=0.2, stat="density")

    plt.title("Holding Log Delta Distributions")
    plt.xlabel("value")
    plt.ylabel("density")
    if args.xlim:
        plt.xlim(args.xlim)
    plt.legend()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[plot] saved distribution figure to {out_path}")


if __name__ == "__main__":
    main()
