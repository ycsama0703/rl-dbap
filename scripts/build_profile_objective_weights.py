#!/usr/bin/env python
"""
Stage 3: Estimate per-profile objective weights (alpha / risk / tc / te).

Steps:
1) Compute portfolio weights w(j,t,s) from holdings (mv if price exists else holding share).
2) Compute delta_w = w_t - w_{t-1} per (investor, permno, quarter).
3) Define proxies:
   - alpha_proxy: profit (fallback Gat, then be)
   - risk_proxy: beta
   - tc_proxy: 1 / me (size proxy), clipped finite
   - te_proxy: abs(w_{t-1} - spx_weight)
4) For each (investor_type, profile_k), run ridge regression:
   delta_w ~ a*alpha - b*risk - c*tc - d*te (fit without intercept)
   clip coefficients to non-negative, then normalize to sum to 1.
5) Save table profile_objective_weights.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Estimate objective weights per (investor_type, profile_k).")
    ap.add_argument("--holdings", required=True, help="Parquet/CSV with stock-level holdings.")
    ap.add_argument("--profiles", required=True, help="Parquet/CSV with investor_quarter_profile (profile_k).")
    ap.add_argument("--out-path", required=True, help="Output parquet/csv for profile_objective_weights.")
    ap.add_argument("--investor-col", default="mgrno")
    ap.add_argument("--stock-col", default="permno")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--type-col", default="type")
    ap.add_argument("--holding-col", default="holding")
    ap.add_argument("--price-col", default="price")
    ap.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge regularization strength.")
    return ap.parse_args()


def load_table(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def save_table(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def compute_weights(df: pd.DataFrame, inv: str, qtr: str, hold: str, price: str) -> pd.DataFrame:
    df["_mv"] = df[hold] * df[price]
    def _w(g: pd.DataFrame) -> pd.DataFrame:
        if g["_mv"].notna().any():
            denom = g["_mv"].sum()
            g["w"] = g["_mv"] / denom if denom != 0 else 0.0
        else:
            denom = g[hold].sum()
            g["w"] = g[hold] / denom if denom != 0 else 0.0
        return g
    return df.groupby([inv, qtr], group_keys=False).apply(_w)


def add_prev_weight(df_w: pd.DataFrame, inv: str, stock: str, qtr: str) -> pd.DataFrame:
    df_w = df_w.sort_values([inv, stock, qtr])
    df_w["w_prev"] = df_w.groupby([inv, stock])["w"].shift(1)
    return df_w


def pick_alpha_proxy(df: pd.DataFrame) -> pd.Series:
    if "profit" in df.columns:
        return df["profit"]
    if "Gat" in df.columns:
        return df["Gat"]
    if "be" in df.columns:
        return df["be"]
    return pd.Series(np.nan, index=df.index)


def main() -> None:
    args = parse_args()
    inv, stock, date = args.investor_col, args.stock_col, args.date_col
    typ = args.type_col
    hold, price = args.holding_col, args.price_col

    df = load_table(args.holdings)
    prof = load_table(args.profiles)

    # normalize type
    if typ in df.columns:
        df[typ] = (
            df[typ]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
        )
    if typ in prof.columns:
        prof[typ] = (
            prof[typ]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
        )

    # ensure date->quarter string
    df[date] = pd.to_datetime(df[date])
    df["quarter"] = df[date].dt.to_period("Q").astype(str)

    # compute weights and previous weights
    df = compute_weights(df, inv, "quarter", hold, price)
    df = add_prev_weight(df, inv, stock, "quarter")
    df["delta_w"] = df["w"] - df["w_prev"]

    # proxies
    df["alpha_proxy"] = pick_alpha_proxy(df)
    df["risk_proxy"] = df["beta"] if "beta" in df.columns else np.nan
    df["tc_proxy"] = np.where(df.get("me") is not None, 1.0 / df["me"].replace(0, np.nan), np.nan)
    # treat inf as nan
    df.loc[~np.isfinite(df["tc_proxy"]), "tc_proxy"] = np.nan
    if "spx_weight" in df.columns:
        df["te_proxy"] = (df["w_prev"] - df["spx_weight"]).abs()
    else:
        df["te_proxy"] = np.nan

    # merge profile_k onto (inv, quarter)
    # avoid suffix clash if type also exists in holdings
    prof = prof[[inv, "quarter", "profile_k", typ]].drop_duplicates()
    df = df.merge(prof, on=[inv, "quarter"], how="inner", suffixes=("", "_prof"))
    # prefer profile type if duplicated
    if f"{typ}_prof" in df.columns:
        df[typ] = df[f"{typ}_prof"]
        df = df.drop(columns=[c for c in df.columns if c.endswith("_prof")])

    # keep rows with prev weights (need delta)
    df_use = df[df["w_prev"].notna()].copy()
    # fill missing proxies with 0 to avoid dropping all rows
    for col in ["alpha_proxy", "risk_proxy", "tc_proxy", "te_proxy"]:
        df_use[col] = df_use[col].fillna(0.0)
    df_use = df_use[df_use["delta_w"].notna()]

    results = []
    for (tval, pk), g in df_use.groupby([typ, "profile_k"]):
        X = g[["alpha_proxy", "risk_proxy", "tc_proxy", "te_proxy"]].to_numpy(dtype=np.float64)
        y = g["delta_w"].to_numpy(dtype=np.float64)
        if len(y) < 10 or np.allclose(y, 0):
            w = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            model = Ridge(alpha=args.ridge_alpha, fit_intercept=False)
            model.fit(X, y)
            coefs = model.coef_
            # map to positive weights: alpha (+), risk (-), tc (-), te (-)
            alpha_w = max(coefs[0], 0)
            risk_w = max(-coefs[1], 0)
            tc_w = max(-coefs[2], 0)
            te_w = max(-coefs[3], 0)
            w = np.array([alpha_w, risk_w, tc_w, te_w], dtype=np.float64)
        s = w.sum()
        if s > 0:
            w = w / s
        results.append(
            {
                "investor_type": tval,
                "profile_k": pk,
                "alpha_w": w[0],
                "risk_w": w[1],
                "tc_w": w[2],
                "te_w": w[3],
                "n_obs": len(g),
            }
        )

    out_df = pd.DataFrame(results)
    save_table(out_df, args.out_path)
    print(f"[ok] wrote {len(out_df)} rows -> {args.out_path}")


if __name__ == "__main__":
    main()
