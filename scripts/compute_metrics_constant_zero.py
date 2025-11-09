#!/usr/bin/env python3
"""
compute_metrics_constant_zero.py

Takes a debug_eval_outputs CSV and replaces every prediction with zero to see
what the evaluation metrics would look like for a degenerate model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.compute_metrics_from_debug import compute_metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute metrics assuming all predictions are zero."
    )
    ap.add_argument("--debug-csv", required=True, help="Original debug_eval_outputs CSV.")
    ap.add_argument(
        "--out-csv",
        help="Where to save metrics (optional).",
    )
    ap.add_argument(
        "--error-quantile",
        type=float,
        default=None,
        help="Optional abs_error quantile trimming.",
    )
    ap.add_argument(
        "--tmp-csv",
        help="Optional path to store the modified CSV (defaults to *_zero.csv).",
    )
    ap.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep the modified CSV instead of deleting it.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.debug_csv)
    if not src.exists():
        raise SystemExit(f"debug CSV not found: {src}")

    df = pd.read_csv(src)
    for col in ("raw_output", "parsed_pred", "pred_tp1", "abs_error", "abs_tp1_error"):
        if col not in df.columns:
            raise SystemExit(f"CSV missing column '{col}' (needed for override).")

    df["parsed_pred"] = 0.0
    df["raw_output"] = "<answer>{\"holding_log_delta\": 0.0}</answer>"
    df["abs_error"] = (df["y_true"].astype(float) - 0.0).abs()

    if "holding_t" in df.columns:
        try:
            import numpy as np

            df["pred_tp1"] = np.exp(df["parsed_pred"]) * (df["holding_t"] + 1e-6) - 1e-6
            df["abs_tp1_error"] = (df["pred_tp1"] - df["true_tp1"]).abs()
        except Exception:
            df["pred_tp1"] = 0.0
            df["abs_tp1_error"] = None
    else:
        df["pred_tp1"] = 0.0
        df["abs_tp1_error"] = None

    tmp_path = (
        Path(args.tmp_csv)
        if args.tmp_csv
        else src.with_name(f"{src.stem}_zero{src.suffix}")
    )
    df.to_csv(tmp_path, index=False)
    print(f"[zero] wrote modified CSV -> {tmp_path}")

    metrics = compute_metrics(
        tmp_path,
        Path(args.out_csv) if args.out_csv else None,
        error_quantile=args.error_quantile,
    )
    print(metrics.to_string(index=False))

    if not args.keep_tmp:
        try:
            tmp_path.unlink()
            print(f"[zero] removed temporary file {tmp_path}")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
