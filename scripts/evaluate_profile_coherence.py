#!/usr/bin/env python
"""
Check behavioral coherence: within each (investor_type, profile_k),
regress delta_w (log change) on signals (beta, profit, be, Gat, etc.)
to see if response directions are consistent and differ across profiles.

Example:
python scripts/evaluate_profile_coherence.py \
  --holdings data/parquet_data/mutual_funds.parquet \
  --profiles artifacts/features/mutual_funds_iq_profile.csv \
  --out-prefix artifacts/features/coherence_mutual_funds

Assumptions on holdings columns (override with flags if needed):
  - type (or investor_type)
  - mgrno (or investor_id)
  - quarter (or date parsable to quarter)
  - holding_t (current) and holding_t1 (next)   OR holding and holding_t1
  - signals: beta, profit, be, Gat (optional, missing -> 0)
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--holdings", required=True, help="Holdings panel (csv/parquet).")
    p.add_argument("--profiles", nargs="+", required=True, help="Profile files (csv/parquet).")
    p.add_argument("--out-prefix", required=True, help="Prefix for outputs (csv).")
    p.add_argument("--holding-col", default="holding_t", help="Column for current holding.")
    p.add_argument("--holding-next-col", default="holding_t1", help="Column for next holding.")
    p.add_argument("--price-col", default=None, help="Optional price column; if given we weight by price for missing next.")
    p.add_argument(
        "--signals",
        nargs="+",
        default=["profit", "beta", "be", "Gat"],
        help="Signal columns to regress on.",
    )
    return p.parse_args()


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def _read_many(paths: List[str]) -> pd.DataFrame:
    dfs = [_read(p) for p in paths]
    return pd.concat(dfs, axis=0, ignore_index=True)


def _norm_type(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.replace(" ", "_")


def _to_quarter(col: pd.Series) -> pd.Series:
    try:
        return pd.PeriodIndex(pd.to_datetime(col), freq="Q").astype(str)
    except Exception:
        return pd.PeriodIndex(col, freq="Q").astype(str)


def prepare_data(args):
    hold = _read(args.holdings)
    prof = _read_many(args.profiles)

    typ_col_h = "type" if "type" in hold.columns else "investor_type"
    typ_col_p = "type" if "type" in prof.columns else "investor_type"
    hold["investor_type"] = _norm_type(hold[typ_col_h])
    prof["investor_type"] = _norm_type(prof[typ_col_p])

    key_inv_h = "mgrno" if "mgrno" in hold.columns else "investor_id"
    key_inv_p = "mgrno" if "mgrno" in prof.columns else "investor_id"
    hold = hold.rename(columns={key_inv_h: "investor_id"})
    prof = prof.rename(columns={key_inv_p: "investor_id"})

    if "quarter" in hold.columns:
        hold["quarter_str"] = _to_quarter(hold["quarter"])
    elif "date" in hold.columns:
        hold["quarter_str"] = _to_quarter(hold["date"])
    else:
        raise ValueError("Holdings need quarter or date column.")

    if "quarter" in prof.columns:
        prof["quarter_str"] = _to_quarter(prof["quarter"])
    else:
        raise ValueError("Profiles need quarter column.")

    df = hold.merge(
        prof[["investor_id", "quarter_str", "investor_type", "profile_k"]],
        on=["investor_id", "quarter_str", "investor_type"],
        how="inner",
    )

    # holdings
    h_col = args.holding_col if args.holding_col in df.columns else "holding"
    h1_col = args.holding_next_col if args.holding_next_col in df.columns else "holding_t1"
    if h_col not in df.columns or h1_col not in df.columns:
        raise ValueError("Holdings data must contain current and next holding columns.")

    df["_holding_t"] = df[h_col]
    df["_holding_tp1"] = df[h1_col]

    # optional price for scale
    if args.price_col and args.price_col in df.columns:
        df["_holding_t"] = df["_holding_t"] * df[args.price_col]
        df["_holding_tp1"] = df["_holding_tp1"] * df[args.price_col]

    df["delta_log"] = np.log((df["_holding_tp1"] + 1e-9) / (df["_holding_t"] + 1e-9))

    # signals
    sigs = [c for c in args.signals if c in df.columns]
    for s in sigs:
        df[s] = df[s].fillna(0)

    df = df.dropna(subset=["delta_log", "profile_k", "investor_type"])
    return df, sigs


def run_regressions(df: pd.DataFrame, sigs: List[str]) -> pd.DataFrame:
    rows = []
    for (t, k), g in df.groupby(["investor_type", "profile_k"]):
        if len(g) < len(sigs) + 5:  # too few samples
            continue
        X = g[sigs].to_numpy()
        y = g["delta_log"].to_numpy()
        model = LinearRegression()
        model.fit(X, y)
        for coef, name in zip(model.coef_, sigs):
            rows.append(
                {
                    "investor_type": t,
                    "profile_k": k,
                    "signal": name,
                    "coef": coef,
                    "n_obs": len(g),
                }
            )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    df, sigs = prepare_data(args)
    result = run_regressions(df, sigs)
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    out_path = f"{args.out_prefix}_coefs.csv"
    result.to_csv(out_path, index=False)
    print(result.head())
    print(f"[ok] saved coefficients -> {out_path}")


if __name__ == "__main__":
    main()
