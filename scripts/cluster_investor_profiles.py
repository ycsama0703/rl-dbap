#!/usr/bin/env python
"""
Stage 2: cluster investor-level profiles (per investor_type) and broadcast to all quarters.

Input: investor_quarter_features (from build_investor_quarter_features.py)
Output: investor_quarter_profile with profile_k, plus cluster centers JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cluster investor-quarter features into profile_k.")
    ap.add_argument("--in-path", required=True, help="Path to investor_quarter_features parquet/csv.")
    ap.add_argument("--out-path", required=True, help="Output parquet/csv with profile_k.")
    ap.add_argument("--investor-col", default="mgrno")
    ap.add_argument("--type-col", default="type")
    ap.add_argument("--quarter-col", default="quarter")
    ap.add_argument("--n-clusters", type=int, default=4)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument(
        "--features",
        nargs="*",
        default=["exp_be", "exp_profit", "exp_gat", "exp_beta", "bm_gap", "turnover", "hhi", "n_pos"],
        help="Feature columns to cluster on (winsor/zscore 已在 Stage1 完成). 聚类先按投资者取时间均值，再广播到所有季度。",
    )
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


def main() -> None:
    args = parse_args()
    df = load_table(args.in_path)

    inv, typ, qtr = args.investor_col, args.type_col, args.quarter_col
    if typ in df.columns:
        df[typ] = (
            df[typ]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
        )
    feats: List[str] = [f for f in args.features if f in df.columns]
    if len(feats) == 0:
        raise SystemExit("no valid feature columns found in input")

    outputs = []
    centers = {}
    for tval, g in df.groupby(typ, sort=False):
        # Aggregate features per investor across time to get a stable profile
        agg = g.groupby(inv, as_index=False)[feats].mean()
        X = agg[feats].fillna(0.0).to_numpy(dtype=np.float32)
        km = KMeans(n_clusters=args.n_clusters, random_state=args.random_state, n_init="auto")
        labels = km.fit_predict(X)
        centers[str(tval)] = km.cluster_centers_.tolist()
        inv_labels = agg[[inv]].copy()
        inv_labels["profile_k"] = labels
        # broadcast profile_k back to all quarters for that investor
        out_g = g[[inv, typ, qtr]].copy()
        out_g = out_g.merge(inv_labels, on=inv, how="left")
        outputs.append(out_g)

    out_df = pd.concat(outputs, axis=0, ignore_index=True)
    save_table(out_df, args.out_path)
    centers_path = Path(args.out_path).with_suffix(".centers.json")
    centers_path.write_text(json.dumps(centers, indent=2))
    print(f"[ok] wrote {len(out_df)} rows -> {args.out_path}")
    print(f"[ok] centers saved -> {centers_path}")


if __name__ == "__main__":
    main()
