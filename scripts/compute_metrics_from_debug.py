#!/usr/bin/env python
"""
compute_metrics_from_debug.py

Given a CSV produced by scripts/debug_eval_outputs.py, compute the same metrics as run_eval.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import basic_regression, topk


def compute_metrics(path: Path, out_csv: Optional[Path] = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalise column names
    if "parsed_pred" not in df.columns or "y_true" not in df.columns:
        raise ValueError(f"{path} does not contain required columns 'parsed_pred' and 'y_true'")

    df["y_pred"] = pd.to_numeric(df["parsed_pred"], errors="coerce")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    if "holding_t" in df.columns:
        df["holding_t"] = pd.to_numeric(df["holding_t"], errors="coerce")
    if "quarter" not in df.columns:
        df["quarter"] = "NA"
    df["quarter"] = df["quarter"].fillna("NA")

    total = len(df)
    valid = df[df["y_pred"].notna()].copy()
    coverage = 100.0 * len(valid) / total if total else np.nan

    valid = valid.set_index("id", drop=False) if "id" in valid.columns else valid

    mae, rmse, r2, smape, ic, ric = basic_regression(valid)
    rec, pre, ndcg = topk(valid, "quarter", k=50)

    metrics_df = pd.DataFrame(
        [
            {
                "coverage%": coverage,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "sMAPE%": smape,
                "IC": ic,
                "RankIC": ric,
                "Recall@50": rec,
                "Precision@50": pre,
                "NDCG@50": ndcg,
            }
        ]
    )

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(out_csv, index=False)

    return metrics_df


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute metrics from debug_eval_outputs CSV.")
    ap.add_argument("--debug-csv", required=True, help="Path to CSV produced by debug_eval_outputs.py")
    ap.add_argument("--out-csv", help="Optional path to write metrics CSV")
    args = ap.parse_args()

    metrics = compute_metrics(Path(args.debug_csv), Path(args.out_csv) if args.out_csv else None)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
