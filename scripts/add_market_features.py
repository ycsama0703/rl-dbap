"""
Add market-level quarterly features (VIX, log volume, SPX weight) to a panel parquet.

Rules:
- VIXCLS.csv (daily): average within each quarter, then shift by 1 quarter so t uses t-1 average.
- sp500_market_volume.csv (daily): use ln_market_volume, average by quarter, then shift by 1 quarter.
- sp500_with_weights.parquet (quarterly): merge weight on (permno, quarter_date). If no weight, keep NaN.

Usage:
python scripts/add_market_features.py \
  --in-path data/processed/panel_quarter.parquet/mutual_funds.parquet \
  --out-path data/processed/panel_quarter.parquet/mutual_funds_with_mkt.parquet
"""

import argparse
from pathlib import Path

import pandas as pd


def load_quarter_means_vix(vix_path: Path) -> pd.DataFrame:
    vix = pd.read_csv(vix_path, parse_dates=["observation_date"])
    vix["quarter"] = vix["observation_date"].dt.to_period("Q")
    q_mean = vix.groupby("quarter")["VIXCLS"].mean().sort_index()
    q_mean = q_mean.shift(1)  # use t-1 quarter value
    out = q_mean.reset_index().rename(columns={"VIXCLS": "vix_qtr_prev_mean"})
    out["quarter_date"] = out["quarter"].dt.to_timestamp("Q")
    return out[["quarter_date", "vix_qtr_prev_mean"]]


def load_quarter_means_volume(vol_path: Path) -> pd.DataFrame:
    vol = pd.read_csv(vol_path, parse_dates=["fdate"])
    if "ln_market_volume" not in vol.columns:
        raise ValueError("sp500_market_volume.csv must contain ln_market_volume column.")
    vol["quarter"] = vol["fdate"].dt.to_period("Q")
    q_mean = vol.groupby("quarter")["ln_market_volume"].mean().sort_index()
    q_mean = q_mean.shift(1)  # use t-1 quarter value
    out = q_mean.reset_index().rename(columns={"ln_market_volume": "ln_vol_qtr_prev_mean"})
    out["quarter_date"] = out["quarter"].dt.to_timestamp("Q")
    return out[["quarter_date", "ln_vol_qtr_prev_mean"]]


def load_weights(weights_path: Path) -> pd.DataFrame:
    w = pd.read_parquet(weights_path)
    w = w.rename(columns={"PERMNO": "permno", "weight": "spx_weight"})
    w["permno"] = w["permno"].astype("int64")
    w["quarter_date"] = pd.to_datetime(w["quarter_date"]).dt.normalize()
    return w[["permno", "quarter_date", "spx_weight"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-path", required=True)
    ap.add_argument("--out-path", required=True)
    ap.add_argument("--vix-path", default="data/VIXCLS.csv")
    ap.add_argument("--volume-path", default="data/sp500_market_volume.csv")
    ap.add_argument("--weights-path", default="data/sp500_with_weights.parquet")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    df = pd.read_parquet(in_path)
    if "date" not in df.columns:
        raise ValueError("input parquet must have column 'date'")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["quarter_date"] = df["date"].dt.to_period("Q").dt.to_timestamp("Q")

    vix_q = load_quarter_means_vix(Path(args.vix_path))
    vol_q = load_quarter_means_volume(Path(args.volume_path))
    w_q = load_weights(Path(args.weights_path))

    df = df.merge(vix_q, on="quarter_date", how="left")
    df = df.merge(vol_q, on="quarter_date", how="left")
    df = df.merge(w_q, on=["permno", "quarter_date"], how="left")

    df.to_parquet(out_path, index=False)
    print(f"[ok] wrote {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
