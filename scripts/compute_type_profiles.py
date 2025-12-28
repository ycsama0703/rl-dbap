#!/usr/bin/env python
"""
Compute one time-invariant profile per investor type using quarterly holding deltas
and aggregated market data (VIX + market volume).

Profile definition (per type):

  risk_aversion =
      mean( |Δlog_holding| / market_vol_q_prev )

  herd_behavior =
      mean( |Δlog_holding| / market_volm_q_prev )

  profit_driven =
      mean( sign(Δlog_holding) * profit )

Then convert raw scores into relative preference weights via:
  (1) robust z-score across types
  (2) row-wise centering
  (3) softmax

Daily → quarterly aggregation:
  For a quarter starting at Q_t (e.g., 2016-04-01), use daily data from the
  previous quarter [Q_{t-1}_start, Q_t_start) and take the mean.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute per-type profiles (risk / herd / profit weights).")
    p.add_argument("--panel-dir", required=True, help="Directory with per-type parquet files.")
    p.add_argument("--vix-path", default="data/VIXCLS.csv")
    p.add_argument("--volume-path", default="data/sp500_market_volume.csv")
    p.add_argument("--out-path", default="artifacts/features/type_profiles.csv")
    p.add_argument("--eps", type=float, default=1e-9)
    return p.parse_args()


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _read_with_date(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    date_key = None
    for key in ["date", "observation_date", "fdate"]:
        if key in cols_lower:
            date_key = cols_lower[key]
            break
    if date_key is None:
        raise ValueError(f"{path} must contain a date column")
    df[date_key] = pd.to_datetime(df[date_key])
    df.columns = [c.lower() for c in df.columns]
    return df.rename(columns={date_key.lower(): "date"})


# ------------------------------------------------------------
# Market data: daily → quarterly (prev quarter aligned)
# ------------------------------------------------------------
def load_market_quarterly(vix_path: Path, vol_path: Path) -> pd.DataFrame:
    vix = _read_with_date(vix_path)
    vol = _read_with_date(vol_path)

    if "vixcls" in vix.columns and "vix" not in vix.columns:
        vix = vix.rename(columns={"vixcls": "vix"})
    if "vix" not in vix.columns:
        raise ValueError("VIX file must contain vix or vixcls")

    if "ln_market_volume" not in vol.columns:
        raise ValueError("Volume file must contain ln_market_volume")

    def to_qstart(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["quarter_start"] = pd.PeriodIndex(df["date"], freq="Q").to_timestamp(how="start")
        return df

    vix = to_qstart(vix)
    vol = to_qstart(vol)

    vix_q = vix.groupby("quarter_start", as_index=False)["vix"].mean()
    vol_q = vol.groupby("quarter_start", as_index=False)["ln_market_volume"].mean()

    # shift forward one quarter: (t-1) → t
    shift = pd.offsets.QuarterBegin(startingMonth=1)
    vix_q["date"] = vix_q["quarter_start"] + shift
    vol_q["date"] = vol_q["quarter_start"] + shift

    market = vix_q.merge(vol_q, on="date", how="outer")
    return market[["date", "vix", "ln_market_volume"]]


# ------------------------------------------------------------
# Panel loading
# ------------------------------------------------------------
def load_panel_frames(panel_dir: Path) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for path in sorted(panel_dir.glob("*.parquet")):
        df = pq.read_table(path).to_pandas()
        if "type" not in df.columns:
            df["type"] = path.stem
        else:
            df["type"] = (
                df["type"]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", "_", regex=True)
            )
        frames.append(df)
    return frames


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()
    eps = args.eps

    # market (quarterly, aligned)
    market = load_market_quarterly(Path(args.vix_path), Path(args.volume_path))

    # panel
    panel = pd.concat(load_panel_frames(Path(args.panel_dir)), ignore_index=True)
    if "date" not in panel.columns:
        raise ValueError("panel must contain date column")
    panel["date"] = pd.to_datetime(panel["date"])

    # Δlog holding
    panel["delta_log"] = np.log(panel["holding_t1"]) - np.log(panel["holding_t"])
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta_log"])

    # merge market
    panel = panel.merge(market, on="date", how="left")
    panel = panel.dropna(subset=["vix", "ln_market_volume"])

    # ratios for risk / herd (magnitude-based)
    abs_delta = panel["delta_log"].abs()
    panel["risk_ratio"] = abs_delta / np.maximum(panel["vix"].abs(), eps)
    panel["herd_ratio"] = abs_delta / np.maximum(panel["ln_market_volume"].abs(), eps)

    # ⭐ PROFIT-DRIVEN (scheme A): directional + signal-aligned
    if "profit" not in panel.columns:
        raise ValueError("panel must contain profit column for profit_driven (scheme A)")
    panel["profit_align"] = np.sign(panel["delta_log"]) * panel["profit"]

    # aggregate by type
    agg = (
        panel.groupby("type")
        .agg(
            risk_raw=("risk_ratio", "mean"),
            herd_raw=("herd_ratio", "mean"),
            prof_raw=("profit_align", "mean"),
            n_obs=("delta_log", "size"),
        )
        .reset_index()
    )

    # robust z-score across types
    def robust_z(x: pd.Series) -> pd.Series:
        med = x.median()
        mad = (x - med).abs().median()
        if mad < eps:
            mad = x.std(ddof=0) if x.std(ddof=0) > eps else 1.0
        return (x - med) / mad

    z_df = pd.DataFrame({
        "risk_z": robust_z(agg["risk_raw"]),
        "herd_z": robust_z(agg["herd_raw"]),
        "prof_z": robust_z(agg["prof_raw"]),
    })

    # ⭐ row-wise centering (critical for preference interpretation)
    z_vals = z_df.to_numpy(dtype=float)
    z_vals = z_vals - z_vals.mean(axis=1, keepdims=True)

    # softmax → preference weights
    exp_vals = np.exp(z_vals)
    weights = exp_vals / np.maximum(exp_vals.sum(axis=1, keepdims=True), eps)

    weights_df = pd.DataFrame(
        weights,
        columns=["risk_aversion", "herd_behavior", "profit_driven"],
    )

    out_df = pd.concat([agg, z_df, weights_df], axis=1)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        out_df.to_csv(out_path, index=False)
    else:
        out_df.to_parquet(out_path, index=False)

    print(f"[ok] wrote {len(out_df)} profiles -> {out_path}")


if __name__ == "__main__":
    main()