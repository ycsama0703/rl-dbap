#!/usr/bin/env python
"""
aggregate_predictions.py

Take a debug_eval_outputs CSV, aggregate predictions and labels by (date, permno),
and emit per-sample and per-stock metrics for quick inspection.
"""
from __future__ import annotations
import argparse, pandas as pd, numpy as np
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Aggregate holding predictions by date/permno")
    ap.add_argument("--debug-csv", required=True, help="CSV from scripts/debug_eval_outputs.py")
    ap.add_argument("--out-prefix", default="outputs/agg", help="Prefix for aggregated outputs")
    ap.add_argument("--ticker-mapping", default="data/ticker_mapping.csv",
                    help="CSV with PERMNO,TICKER,COMNAM columns for labeling outputs")
    args = ap.parse_args()

    df = pd.read_csv(args.debug_csv)
    if df.empty:
        raise SystemExit(f"empty debug csv: {args.debug_csv}")
    df = df[df["parsed_pred"].notna()]
    if df.empty:
        raise SystemExit("no parsed predictions to aggregate")
    # ensure required columns
    if "permno" not in df.columns or "date" not in df.columns:
        raise SystemExit("debug CSV missing permno/date; rerun debug_eval_outputs after recent changes.")

    # If pred_tp1 is missing but holding_t exists, try reconstruct from parsed_pred (log delta)
    if "pred_tp1" not in df.columns or df["pred_tp1"].isna().all():
        LOG_EPS = 1e-6
        df["pred_tp1"] = np.exp(df["parsed_pred"]) * (df["holding_t"] + LOG_EPS) - LOG_EPS

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
        # reorder for readability
        agg = agg[["permno", "ticker", "name", "date", "pred_tp1_sum", "true_tp1_sum", "abs_err", "ape", "samples"]]
        per_stock = per_stock[
            ["permno", "ticker", "name", "mae", "wape", "true_tp1_total", "pred_tp1_total", "samples"]
        ]
    except Exception as e:
        print(f"[aggregate] mapping load failed or skipped: {e}")

    out_base = Path(args.out_prefix)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    agg_path = out_base.with_suffix(".by_date_permno.csv")
    per_stock_path = out_base.with_suffix(".per_stock.csv")
    agg.to_csv(agg_path, index=False)
    per_stock.to_csv(per_stock_path, index=False)

    print(f"[aggregate] wrote {len(agg)} rows to {agg_path}")
    print(f"[aggregate] wrote {len(per_stock)} rows to {per_stock_path}")


if __name__ == "__main__":
    main()
