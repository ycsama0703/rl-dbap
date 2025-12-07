#!/usr/bin/env python
"""
aggregate_predictions_base.py

For base model outputs where raw_output may be messy, extract the first number as holding_tp1,
then aggregate by (permno, date) and compute per-stock metrics (MAE, WAPE, totals).
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd, numpy as np

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_first_number(text: str) -> float | None:
    if not isinstance(text, str):
        return None
    m = NUM_RE.search(text)
    if not m:
        return None
    try:
        val = float(m.group(0))
        if np.isfinite(val):
            return val
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser(description="Aggregate base-model predictions with loose number extraction.")
    ap.add_argument("--debug-csv", required=True, help="CSV from scripts/debug_eval_outputs.py (raw_output required)")
    ap.add_argument("--out-prefix", default="outputs/agg_base", help="Prefix for aggregated outputs")
    ap.add_argument("--ticker-mapping", default="data/ticker_mapping.csv",
                    help="CSV with PERMNO,TICKER,COMNAM columns for labeling outputs")
    args = ap.parse_args()

    df = pd.read_csv(args.debug_csv)
    if df.empty:
        raise SystemExit(f"empty debug csv: {args.debug_csv}")
    if "raw_output" not in df.columns:
        raise SystemExit("debug CSV missing raw_output")
    if "permno" not in df.columns or "date" not in df.columns:
        raise SystemExit("debug CSV missing permno/date; rerun debug_eval_outputs after recent changes.")

    df["pred_tp1"] = df["raw_output"].apply(extract_first_number)
    df = df[df["pred_tp1"].notna()]
    if df.empty:
        raise SystemExit("no predictions parsed from raw_output")

    if "true_tp1" not in df.columns or df["true_tp1"].isna().all():
        raise SystemExit("true_tp1 missing; ensure debug_eval_outputs was run on absolute targets.")

    agg = (
        df.groupby(["permno", "date"], as_index=False)
        .agg(
            pred_tp1_sum=("pred_tp1", "sum"),
            true_tp1_sum=("true_tp1", "sum"),
            samples=("id", "count"),
        )
    )
    agg["abs_err"] = agg["pred_tp1_sum"] - agg["true_tp1_sum"]
    agg["ape"] = (agg["abs_err"].abs() / agg["true_tp1_sum"].abs().clip(lower=1e-6))
    agg["pct_err"] = 100.0 * agg["abs_err"] / agg["true_tp1_sum"].replace(0, np.nan)

    per_stock = (
        agg.groupby("permno")
        .agg(
            mae=("abs_err", lambda x: np.mean(np.abs(x))),
            wape=("abs_err", lambda x: np.sum(np.abs(x)) / np.sum(np.abs(agg.loc[x.index, "true_tp1_sum"]))),
            true_tp1_total=("true_tp1_sum", "sum"),
            pred_tp1_total=("pred_tp1_sum", "sum"),
            samples=("samples", "sum"),
        )
        .reset_index()
        .sort_values("wape")
    )

    # Attach ticker/name labels if available
    try:
        map_df = pd.read_csv(args.ticker_mapping)
        map_last = map_df.sort_index().groupby("PERMNO").tail(1)[["PERMNO", "TICKER", "COMNAM"]]
        agg = agg.merge(map_last, left_on="permno", right_on="PERMNO", how="left").drop(columns=["PERMNO"])
        agg = agg.rename(columns={"TICKER": "ticker", "COMNAM": "name"})
        per_stock = per_stock.merge(map_last, left_on="permno", right_on="PERMNO", how="left").drop(columns=["PERMNO"])
        per_stock = per_stock.rename(columns={"TICKER": "ticker", "COMNAM": "name"})
        agg = agg[["permno", "ticker", "name", "date", "pred_tp1_sum", "true_tp1_sum", "abs_err", "ape", "pct_err", "samples"]]
        per_stock = per_stock[["permno", "ticker", "name", "mae", "wape", "true_tp1_total", "pred_tp1_total", "samples"]]
    except Exception as e:
        print(f"[aggregate-base] mapping load failed or skipped: {e}")

    out_base = Path(args.out_prefix)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    agg_path = out_base.with_suffix(".by_date_permno.csv")
    per_stock_path = out_base.with_suffix(".per_stock.csv")
    agg.to_csv(agg_path, index=False)
    per_stock.to_csv(per_stock_path, index=False)
    print(f"[aggregate-base] wrote {len(agg)} rows to {agg_path}")
    print(f"[aggregate-base] wrote {len(per_stock)} rows to {per_stock_path}")


if __name__ == "__main__":
    main()
