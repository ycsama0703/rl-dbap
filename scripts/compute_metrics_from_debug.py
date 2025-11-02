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
    if "true_tp1" in df.columns:
        df["true_tp1"] = pd.to_numeric(df["true_tp1"], errors="coerce")
    if "pred_tp1" in df.columns:
        df["pred_tp1"] = pd.to_numeric(df["pred_tp1"], errors="coerce")
    if "holding_t" in df.columns:
        df["holding_t"] = pd.to_numeric(df["holding_t"], errors="coerce")
    if "quarter" not in df.columns:
        df["quarter"] = "NA"
    df["quarter"] = df["quarter"].fillna("NA")

    total = len(df)
    valid = df[df["y_pred"].notna()].copy()
    coverage = 100.0 * len(valid) / total if total else np.nan

    valid = valid.set_index("id", drop=False) if "id" in valid.columns else valid

    mae_log, rmse_log, r2_log, smape_log, ic_log, ric_log = basic_regression(valid)
    rec_log, pre_log, ndcg_log = topk(valid, "quarter", k=50)

    has_tp1_cols = {"true_tp1", "pred_tp1"}.issubset(valid.columns)
    if has_tp1_cols:
        valid_abs = valid.dropna(subset=["true_tp1", "pred_tp1"]).copy()
    else:
        valid_abs = pd.DataFrame()
    if has_tp1_cols and not valid_abs.empty:
        subset_cols = ["true_tp1", "pred_tp1", "quarter"]
        if "id" in valid_abs.columns:
            subset_cols.append("id")
        valid_abs_metrics = valid_abs[subset_cols].rename(
            columns={"true_tp1": "y_true", "pred_tp1": "y_pred"}
        )
        mae_tp1, rmse_tp1, r2_tp1, smape_tp1, ic_tp1, ric_tp1 = basic_regression(valid_abs_metrics)
        rec_tp1, pre_tp1, ndcg_tp1 = topk(valid_abs_metrics, "quarter", k=50)
    else:
        mae_tp1 = rmse_tp1 = r2_tp1 = smape_tp1 = ic_tp1 = ric_tp1 = np.nan
        rec_tp1 = pre_tp1 = ndcg_tp1 = np.nan

    metrics_df = pd.DataFrame(
        [
            {
                "coverage%": coverage,
                "MAE_log": mae_log,
                "RMSE_log": rmse_log,
                "R2_log": r2_log,
                "sMAPE_log%": smape_log,
                "IC_log": ic_log,
                "RankIC_log": ric_log,
                "Recall@50_log": rec_log,
                "Precision@50_log": pre_log,
                "NDCG@50_log": ndcg_log,
                "MAE_tp1": mae_tp1,
                "RMSE_tp1": rmse_tp1,
                "R2_tp1": r2_tp1,
                "sMAPE_tp1%": smape_tp1,
                "IC_tp1": ic_tp1,
                "RankIC_tp1": ric_tp1,
                "Recall@50_tp1": rec_tp1,
                "Precision@50_tp1": pre_tp1,
                "NDCG@50_tp1": ndcg_tp1,
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
