#!/usr/bin/env python
"""
Build investor-quarter level feature table from stock-level holdings.

Features per (investor_id, quarter):
  - exp_be, exp_profit, exp_gat, exp_beta, exp_spx (weighted averages)
  - bm_gap: benchmark deviation sum|w - spx_weight|
  - turnover: 0.5 * sum|w_t - w_{t-1}|
  - hhi: sum w^2
  - n_pos: count of positions (w > 0)
Winsorize (per investor_type if available) at 1%/99%, then z-score.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build investor-quarter feature table from holdings parquet.")
    ap.add_argument("--in-path", required=True, help="Input parquet with stock-level holdings.")
    ap.add_argument("--out-path", required=True, help="Output parquet/csv for investor-quarter features.")
    ap.add_argument("--investor-col", default="mgrno")
    ap.add_argument("--stock-col", default="permno")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--type-col", default="type")
    ap.add_argument("--holding-col", default="holding")
    ap.add_argument("--price-col", default="price")
    ap.add_argument("--winsor-alpha", type=float, default=0.01, help="Lower/upper tail winsor level (default 1%).")
    ap.add_argument("--no-zscore", action="store_true", help="Disable z-score after winsor.")
    return ap.parse_args()


def winsorize_and_zscore(df: pd.DataFrame, features: List[str], alpha: float, by: str | None, do_z: bool) -> pd.DataFrame:
    def _proc(group: pd.DataFrame) -> pd.DataFrame:
        for f in features:
            if f not in group:
                continue
            lo, hi = group[f].quantile(alpha), group[f].quantile(1 - alpha)
            group[f"{f}_raw"] = group[f]
            group[f] = group[f].clip(lo, hi)
            if do_z:
                std = group[f].std(ddof=0)
                if std and std > 0:
                    group[f] = (group[f] - group[f].mean()) / std
                else:
                    group[f] = 0.0
        return group

    if by and by in df.columns:
        return df.groupby(by, group_keys=False).apply(_proc)
    return _proc(df)


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.in_path)

    inv, stock, date = args.investor_col, args.stock_col, args.date_col
    hold, price, typ = args.holding_col, args.price_col, args.type_col

    # normalize type
    if args.type_col in df.columns:
        df[args.type_col] = (
            df[args.type_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
        )

    # prepare date/quarter
    df[date] = pd.to_datetime(df[date])
    df["quarter"] = df[date].dt.to_period("Q").astype(str)

    # compute weights
    df["_mv"] = df[hold] * df[price]
    def _weight(group: pd.DataFrame) -> pd.DataFrame:
        use_mv = group["_mv"].notna().any()
        if use_mv:
            denom = group["_mv"].sum()
            group["w"] = group["_mv"] / denom if denom != 0 else 0.0
        else:
            denom = group[hold].sum()
            group["w"] = group[hold] / denom if denom != 0 else 0.0
        return group

    df = df.groupby([inv, "quarter"], group_keys=False).apply(_weight)

    # weighted exposures
    for col in ["be", "profit", "Gat", "beta", "spx_weight"]:
        if col in df.columns:
            df[f"w_{col}"] = df["w"] * df[col]

    # aggregate
    agg_dict = {
        "hhi": ("w", lambda x: (x * x).sum()),
        "n_pos": ("w", lambda x: (x > 0).sum()),
    }
    if "w_be" in df.columns:
        agg_dict["exp_be"] = ("w_be", "sum")
    if "w_profit" in df.columns:
        agg_dict["exp_profit"] = ("w_profit", "sum")
    if "w_Gat" in df.columns:
        agg_dict["exp_gat"] = ("w_Gat", "sum")
    if "w_beta" in df.columns:
        agg_dict["exp_beta"] = ("w_beta", "sum")
    if "w_spx_weight" in df.columns:
        agg_dict["exp_spx"] = ("w_spx_weight", "sum")

    agg = df.groupby([inv, "quarter"], as_index=False).agg(**agg_dict)

    # bm_gap
    if "spx_weight" in df.columns:
        bm = (
            df[df["spx_weight"].notna()]
            .assign(abs_gap=lambda r: (r["w"] - r["spx_weight"]).abs())
            .groupby([inv, "quarter"])["abs_gap"]
            .sum()
            .reset_index(name="bm_gap")
        )
        agg = agg.merge(bm, on=[inv, "quarter"], how="left")

    # turnover: need previous quarter weights aligned by stock
    w_cols = [inv, "quarter", stock, "w"]
    w_df = df[w_cols].copy()
    w_df = w_df.pivot_table(index=[inv, "quarter"], columns=stock, values="w", fill_value=0.0)
    w_df = w_df.sort_index(level=[0, 1])
    turnover = []
    last = {}
    for (i, q), row in w_df.iterrows():
        prev = last.get(i)
        if prev is None:
            turnover.append(((i, q), np.nan))
        else:
            # align columns
            a = row
            b = prev.reindex(a.index, fill_value=0.0)
            tval = 0.5 * (a - b).abs().sum()
            turnover.append(((i, q), tval))
        last[i] = row
    t_df = pd.DataFrame(turnover, columns=["key", "turnover"])
    t_df[[inv, "quarter"]] = pd.DataFrame(t_df["key"].tolist(), index=t_df.index)
    t_df = t_df.drop(columns=["key"])
    agg = agg.merge(t_df, on=[inv, "quarter"], how="left")

    # attach type if exists
    if typ in df.columns:
        type_map = df[[inv, typ]].drop_duplicates()
        agg = agg.merge(type_map, on=inv, how="left")
    else:
        agg[typ] = "unknown"

    # winsor + z-score per type
    feature_cols = ["exp_be", "exp_profit", "exp_gat", "exp_beta", "exp_spx", "bm_gap", "turnover", "hhi", "n_pos"]
    agg = winsorize_and_zscore(agg, feature_cols, args.winsor_alpha, typ, not args.no_zscore)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        agg.to_csv(out_path, index=False)
    else:
        agg.to_parquet(out_path, index=False)
    print(f"[ok] wrote {len(agg)} rows -> {out_path}")


if __name__ == "__main__":
    main()
