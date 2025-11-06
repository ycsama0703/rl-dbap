#!/usr/bin/env python
"""
compute_metrics_from_debug.py

Enhanced version:
- 只要 <answer> 中有合法的 holding_log_delta 数值，就计入指标计算；
- 如果没有 <answer>，不纳入指标计算；
- coverage_filtered% 表示能从 <answer> 提取出数值的比例；
- coverage_valid% 表示结构完全合规（有 <think> + <answer>）的比例。
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from src.evaluation.metrics import basic_regression, topk


# ======== helper: parse holding_log_delta ========
def _extract_holding_log_delta(text: str) -> Optional[float]:
    """Extract holding_log_delta value inside <answer> ... </answer>"""
    if not isinstance(text, str):
        return None
    # 只匹配 <answer> 中的内容
    m = re.search(
        r"<answer>.*?holding_log_delta['\"\s:]*([\-−]?\d+(?:\.\d+)?).*?</answer>",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    try:
        return float(m.group(1).replace("−", "-"))  # 修正 unicode minus
    except Exception:
        return None


# ======== quantile trimming ========
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


# ======== main compute ========
def compute_metrics(
    path: Path,
    out_csv: Optional[Path] = None,
    error_quantile: Optional[float] = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ---- Step 1. 重新解析 <answer> 部分 ----
    df["parsed_pred"] = df["raw_output"].apply(_extract_holding_log_delta)
    df["has_answer"] = df["parsed_pred"].notna()
    df["is_valid"] = df["has_answer"] & df["raw_output"].str.contains("<think>", case=False, na=False)

    total = len(df)
    coverage_filtered = 100.0 * df["has_answer"].sum() / total if total else np.nan
    coverage_valid = 100.0 * df["is_valid"].sum() / total if total else np.nan

    # ---- Step 2. 基本列规范化 ----
    if "y_true" not in df.columns:
        raise ValueError(f"{path} missing required column y_true")

    df["y_pred"] = pd.to_numeric(df["parsed_pred"], errors="coerce")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    if "quarter" not in df.columns:
        df["quarter"] = "NA"

    # ---- Step 3. 只保留能解析出预测值的样本进行指标计算 ----
    valid = df[df["y_pred"].notna()].copy()
    valid["abs_error"] = (valid["y_pred"] - valid["y_true"]).abs()
    if error_quantile is not None:
        valid = _trim_by_quantile(valid, "abs_error", error_quantile)

    # ---- Step 4. 计算指标 ----
    mae, rmse, r2, smape, ic, ric = basic_regression(valid)
    rec, pre, ndcg = topk(valid, "quarter", k=50)

    metrics_df = pd.DataFrame(
        [
            {
                "coverage_filtered%": coverage_filtered,
                "coverage_valid%": coverage_valid,
                "MAE_log": mae,
                "RMSE_log": rmse,
                "R2_log": r2,
                "sMAPE_log%": smape,
                "IC_log": ic,
                "RankIC_log": ric,
                "Recall@50_log": rec,
                "Precision@50_log": pre,
                "NDCG@50_log": ndcg,
            }
        ]
    )

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(out_csv, index=False)

    return metrics_df


# ======== CLI ========
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
