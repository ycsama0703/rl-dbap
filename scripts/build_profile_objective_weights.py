#!/usr/bin/env python
"""
Compute profile-level objective weights using holding_log_delta and market aggregates.

New definitions (per profile_k within investor_type):
  - risk_aversion_score   = |holding_log_delta| / max(vix_qtr_prev_mean, eps)
  - herd_behavior_score   = |holding_log_delta| / max(market_volume_qtr_prev_mean, eps)
                            (market_volume_qtr_prev_mean = exp(ln_vol_qtr_prev_mean) if ln provided)
  - profit_driven_score   = |holding_log_delta| / max(profit, eps)

Weights are the normalized mean scores (with a small floor to avoid degeneracy):
  w_i = max(mean_score_i, min_weight)
  weights = w_i / sum_i w_i

Inputs:
  --holdings : panel parquet/csv with columns including holding, profit, vix_qtr_prev_mean,
               ln_vol_qtr_prev_mean (or volume column) and date/quarter.
               Should include all investor types (no whitelist filtering).
  --profiles : profile assignments with mgrno + quarter + profile_k (per investor_type).
  --out-path : where to write the resulting weights (csv or parquet).

The script joins holdings with profiles on (mgrno, quarter), computes holding_log_delta
within (mgrno, permno) over time, then aggregates scores by investor_type + profile_k.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def to_quarter(col: pd.Series) -> pd.Series:
    """Convert a date/quarter-like column to string quarter 'YYYYQn'."""
    if col.dtype == object and col.str.contains("Q").any():
        return col.astype(str)
    # fallback: parse datetime then to quarter
    dt = pd.to_datetime(col)
    return pd.PeriodIndex(dt, freq="Q").astype(str)


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdings", required=True, help="Panel holdings file (parquet/csv)")
    ap.add_argument(
        "--profiles",
        required=False,
        default=None,
        help="Optional profile assignments with profile_k; if omitted, aggregate per investor_type only",
    )
    ap.add_argument(
        "--group-by-profile",
        action="store_true",
        default=False,
        help="If set, aggregate by investor_type + profile_k; otherwise by investor_type only",
    )
    ap.add_argument("--out-path", required=True, help="Output path (csv or parquet)")
    ap.add_argument("--inv-col", default="mgrno")
    ap.add_argument("--permno-col", default="permno")
    ap.add_argument("--date-col", default="date", help="Date or quarter column")
    ap.add_argument("--quarter-col", default="quarter", help="Quarter column name after conversion")
    ap.add_argument("--type-col", default="type")
    ap.add_argument("--holding-col", default="holding", help="Holding at time t")
    ap.add_argument("--profit-col", default="profit")
    ap.add_argument("--vix-col", default="vix_qtr_prev_mean")
    ap.add_argument("--ln-vol-col", default="ln_vol_qtr_prev_mean")
    ap.add_argument("--vol-col", default=None, help="Optional volume column (if ln-vol not present)")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--min-weight", type=float, default=1e-3)
    args = ap.parse_args()

    hold_path = Path(args.holdings)
    prof_path = Path(args.profiles) if args.profiles else None
    out_path = Path(args.out_path)

    df = load_table(hold_path)
    prof = load_table(prof_path) if prof_path else None

    inv = args.inv_col
    perm = args.permno_col
    date_col = args.date_col
    qcol = args.quarter_col
    typ = args.type_col
    hold_col = args.holding_col
    profit_col = args.profit_col
    vix_col = args.vix_col
    ln_vol_col = args.ln_vol_col
    vol_col = args.vol_col
    eps = args.eps
    min_w = args.min_weight

    # Normalize column names/types
    for c in [inv, perm]:
        if c in df.columns:
            df[c] = df[c].astype(str)
        if prof is not None and c in prof.columns:
            prof[c] = prof[c].astype(str)

    if qcol not in df.columns:
        df[qcol] = to_quarter(df[date_col])
    if prof is not None and qcol not in prof.columns:
        prof[qcol] = to_quarter(prof[date_col])

    # investor type to lower
    def _norm_type(series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip().str.lower().str.replace(" ", "_")

    if typ in df.columns:
        df[typ] = _norm_type(df[typ])
    if prof is not None and typ in prof.columns:
        prof[typ] = _norm_type(prof[typ])

    # compute holding_log_delta within (inv, perm)
    df = df.sort_values([inv, perm, qcol])
    df["holding_prev"] = df.groupby([inv, perm])[hold_col].shift(1)
    df["holding_log_delta"] = np.log((df[hold_col] + eps) / (df["holding_prev"] + eps))

    # Merge profiles (optional)
    merged = df
    if prof is not None and args.group_by_profile:
        prof_use = prof[[inv, qcol, "profile_k", typ]].dropna(subset=["profile_k"]).copy()
        prof_use["profile_k"] = prof_use["profile_k"].astype(int)
        merged = df.merge(prof_use, on=[inv, qcol], how="inner", suffixes=("", "_prof"))
        # Resolve investor type column (prefer merged typ from profiles if present)
        if f"{typ}_prof" in merged.columns:
            merged[typ] = merged[f"{typ}_prof"]
    else:
        # If no profile merge, fabricate a single profile_k=0 to keep downstream code simple
        merged = merged.copy()
        merged["profile_k"] = 0

    # Drop rows missing needed inputs
    needed_cols = [profit_col, "holding_prev", "holding_log_delta"]
    if vix_col in merged.columns:
        needed_cols.append(vix_col)
    else:
        raise ValueError(f"Missing required column: {vix_col}")

    volume_series: Optional[pd.Series] = None
    if ln_vol_col and ln_vol_col in merged.columns:
        volume_series = np.exp(merged[ln_vol_col].astype(float))
    elif vol_col and vol_col in merged.columns:
        volume_series = merged[vol_col].astype(float)
    else:
        raise ValueError("Missing volume proxy: provide --ln-vol-col or --vol-col present in data.")

    merged = merged.dropna(subset=needed_cols)
    merged = merged.loc[merged[profit_col].notna() & volume_series.notna()]
    merged = merged.copy()
    merged["market_volume"] = volume_series

    # Raw scores
    merged["risk_aversion_score"] = merged["holding_log_delta"].abs() / np.maximum(merged[vix_col], eps)
    merged["herd_behavior_score"] = merged["holding_log_delta"].abs() / np.maximum(merged["market_volume"], eps)
    merged["profit_driven_score"] = merged["holding_log_delta"].abs() / np.maximum(merged[profit_col], eps)

    # Winsorize(1%,99%) + scale (no centering) within investor_type for comparability
    for col in ["risk_aversion_score", "herd_behavior_score", "profit_driven_score"]:
        def _clip_scale_series(s: pd.Series) -> pd.Series:
            q1 = s.quantile(0.01)
            q99 = s.quantile(0.99)
            clipped = s.clip(lower=q1, upper=q99)
            std = clipped.std()
            std = std if std and std > eps else 1.0
            return clipped / std  # avoid centering to preserve mean differences

        merged[col] = merged.groupby(typ)[col].transform(_clip_scale_series)

    group_cols = [typ]
    if args.group_by_profile:
        group_cols.append("profile_k")
    else:
        merged["profile_k"] = 0  # ensure column exists but grouping only by type

    grp = merged.groupby(group_cols)
    agg = grp[["risk_aversion_score", "herd_behavior_score", "profit_driven_score"]].mean().reset_index()
    agg["n_obs"] = grp.size().values
    if not args.group_by_profile:
        agg["profile_k"] = 0

    # Apply floor and normalize to weights (shift to be positive)
    for col in ["risk_aversion_score", "herd_behavior_score", "profit_driven_score"]:
        agg[col] = agg[col].fillna(0.0)
    # shift each score column so the minimum is >= min_w
    for col in ["risk_aversion_score", "herd_behavior_score", "profit_driven_score"]:
        min_val = agg[col].min()
        agg[col] = agg[col] - min_val + min_w

    agg["score_sum"] = agg[["risk_aversion_score", "herd_behavior_score", "profit_driven_score"]].sum(axis=1)
    agg["risk_aversion_w"] = agg["risk_aversion_score"] / agg["score_sum"]
    agg["herd_behavior_w"] = agg["herd_behavior_score"] / agg["score_sum"]
    agg["profit_driven_w"] = agg["profit_driven_score"] / agg["score_sum"]

    out_cols = [
        typ,
        "profile_k",
        "risk_aversion_w",
        "herd_behavior_w",
        "profit_driven_w",
        "risk_aversion_score",
        "herd_behavior_score",
        "profit_driven_score",
        "n_obs",
    ]
    out_df = agg[out_cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        out_df.to_csv(out_path, index=False)
    else:
        out_df.to_parquet(out_path, index=False)

    print(f"[ok] wrote {len(out_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
