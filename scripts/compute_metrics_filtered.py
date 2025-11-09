#!/usr/bin/env python
"""
compute_metrics_filtered.py

Wrapper around compute_metrics_from_debug.py that first filters out
problematic rows (e.g., holding_t ≈ 0 或 true/pred tp1 极小值), 然后再计算指标。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.compute_metrics_from_debug import compute_metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Filter debug_eval_outputs CSV (去掉 near-zero holding/tp1) and compute metrics."
    )
    ap.add_argument("--debug-csv", required=True, help="原始 debug_eval_outputs CSV")
    ap.add_argument("--out-csv", help="指标输出 CSV 路径（可选）")
    ap.add_argument(
        "--filtered-csv",
        help="过滤后的临时 CSV（默认生成在原文件旁，后缀 *_filtered.csv）",
    )
    ap.add_argument(
        "--holding-eps",
        type=float,
        default=1e-4,
        help="过滤 holding_t 的绝对值阈值（默认 1e-4）",
    )
    ap.add_argument(
        "--tp1-eps",
        type=float,
        default=1e-3,
        help="过滤 true/pred tp1 的绝对值阈值（默认 1e-3）",
    )
    ap.add_argument(
        "--no-filter-pred-tp1",
        action="store_true",
        help="只过滤 true_tp1，不额外过滤 pred_tp1",
    )
    ap.add_argument(
        "--error-quantile",
        type=float,
        default=None,
        help="可选：传给 compute_metrics 的 abs_error 上分位截断（如 0.99）",
    )
    ap.add_argument(
        "--keep-filtered",
        action="store_true",
        help="保留过滤后的 CSV（默认计算完即删除）",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.debug_csv)
    if not src.exists():
        raise SystemExit(f"[error] debug CSV not found: {src}")

    df = pd.read_csv(src)

    for col in ("holding_t", "true_tp1", "pred_tp1"):
        if col not in df.columns:
            raise SystemExit(f"[error] CSV 缺少列 {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mask = pd.Series(True, index=df.index)
    if args.holding_eps is not None and args.holding_eps > 0:
        mask &= df["holding_t"].abs() >= args.holding_eps
    if args.tp1_eps is not None and args.tp1_eps > 0:
        mask &= df["true_tp1"].abs() >= args.tp1_eps
        if not args.no_filter_pred_tp1:
            mask &= df["pred_tp1"].abs() >= args.tp1_eps

    filtered = df[mask].copy()
    removed = len(df) - len(filtered)
    if filtered.empty:
        raise SystemExit("[error] 过滤后没有剩余样本，请调低阈值。")

    filtered_path = (
        Path(args.filtered_csv)
        if args.filtered_csv
        else src.with_name(f"{src.stem}_filtered{src.suffix}")
    )
    filtered.to_csv(filtered_path, index=False)

    print(
        f"[filter] kept {len(filtered)} / {len(df)} rows "
        f"(removed {removed}) -> {filtered_path}"
    )

    metrics = compute_metrics(
        filtered_path,
        Path(args.out_csv) if args.out_csv else None,
        error_quantile=args.error_quantile,
    )
    print(metrics.to_string(index=False))

    if not args.keep_filtered:
        try:
            filtered_path.unlink()
            print(f"[filter] removed temporary file {filtered_path}")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
