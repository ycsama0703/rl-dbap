#!/usr/bin/env python
"""
compute_metrics_from_debug.py

Given a CSV produced by scripts/debug_eval_outputs.py, compute the same metrics as run_eval,
with updated logic:
- Metrics include all rows from which "holding_log_delta" can be parsed.
- coverage_valid% still reflects structural validity.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional
import re

import numpy as np
import pandas as pd
from src.evaluation.metrics import basic_regression, topk


def _trim_by_quantile(df: pd.DataFrame, value_col: str, quantile: float) -> pd.DataFrame:
    if quantile is None or value_col not in df.columns:
        return df
    if not (0.0 < quantile <= 1.0):
        raise ValueError(f"error quantile must be in (0,1], got {quantile}")
    series = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if series.empty:
        return df
    threshold = series.quantile(quantile)
    return df[df[value_col] <= threshold].copy()


def _detect_holding_log_delta(raw_output: str) -> Optional[float]:
    """
    Try to extract holding_log_delta value from the raw_output JSON snippet.
    Example: <answer>{"holding_log_delta": 0.05}</answer>
    """
    if not isinstance(raw_output, str):
        return None
    match = re.search(r'"holding_log_delta"\s*:\s*(-?\d+(?:\.\d+)?)', raw_output)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    return None


def compute_metrics(
    path: Path,
    out_csv: Optional[Path] = None,
    error_quantile: Optional[float] = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 新增列：能否解析出holding_log_delta
    df["parsed_holding_log_delta"] = df["raw_output"].apply(_detect_holding_log_delta)
    df["is_parsable"] = df["parsed_holding_log_delta"].notna()

    # coverage_valid%：结构合规的比例（原逻辑不变，如果存在is_valid列）
    if "is_valid" in df.columns:
        valid_coverage = 100.0 * df["is_valid"].mean()
    else:
        valid_coverage = np.nan

    # coverage_filtered%：能解析出holding_log_delta的比例
    total = len(df)
    filtered_coverage = 100.0 * df["is_parsable"].sum() / total if total else np.nan

    # 用能解析出的样本计算指标
    df_valid = df[df["is_parsable"]].copy()

    if "parsed_pred" not in df_valid.columns and "parsed_holding_log_delta" in df_valid.columns:
        df_valid["parsed_pred"] = df_valid["parsed_holding_log_delta"]

    if "parsed_pred" not in df_valid.columns or "y_true" not in df_valid.columns:
        raise ValueError("Required columns parsed_pred and y_true missing.")

    df_valid["y_pred"] = pd.to_numeric(df_valid["parsed_pred"], errors="coerce")
    df_valid["y_true"] = pd.to_numeric(df_valid["y_true"], errors="coerce")

    df_valid["abs_error"] = (df_valid["y_pred"] - df_valid["y_true"]).abs()

    if error_quantile is not None:
        df_valid = _trim_by_quantile(df_valid, "abs_error", error_quantile)

    df_valid = df_valid.set_index("id", drop=False) if "id" in df_valid.columns else df_valid

    mae_log, rmse_log, r2_log, smape_log, ic_log, ric_log = basic_regression(df_valid)
    rec_log, pre_log, ndcg_log = topk(df_valid, "quarter" if "quarter" in df_valid.columns else None, k=50)

    metrics_df = pd.DataFrame(
        [
            {
                "coverage_filtered%": filtered_coverage,
                "coverage_valid%": valid_coverage,
                "MAE_log": mae_log,
                "RMSE_log": rmse_log,
                "R2_log": r2_log,
                "sMAPE_log%": smape_log,
                "IC_log": ic_log,
                "RankIC_log": ric_log,
                "Recall@50_log": rec_log,
                "Precision@50_log": pre_log,
                "NDCG@50_log": ndcg_log,
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
    ap.add_argument(
        "--error-quantile",
        type=float,
        default=None,
        help="Optional upper quantile threshold for absolute errors (e.g., 0.99 keeps the lowest 99%%).",
    )
    args = ap.parse_args()

    metrics = compute_metrics(
        Path(args.debug_csv),
        Path(args.out_csv) if args.out_csv else None,
        error_quantile=args.error_quantile,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
