#!/usr/bin/env python
"""
Plot cluster centers for profiles in a 2D PCA space.

Inputs:
- features file (CSV/Parquet) from Stage1 (investor_quarter_features)
- profile file (CSV/Parquet) from Stage2 (with profile_k)

Process:
- merge profile_k onto features
- select feature columns (default: exp_be, exp_profit, exp_gat, exp_beta, bm_gap, turnover, hhi, n_pos)
- run PCA to 2 dimensions on all samples
- compute cluster centroids in PCA space for each (investor_type, profile_k)
- scatter plot centroids, colored by investor_type, marker size proportional to n_obs
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot profile cluster centers in 2D PCA space.")
    ap.add_argument(
        "--features", required=True, nargs="+",
        help="One or more Stage1 features CSV/Parquet files (will be concatenated)."
    )
    ap.add_argument(
        "--profiles", required=True, nargs="+",
        help="One or more Stage2 profile CSV/Parquet files (will be concatenated)."
    )
    ap.add_argument(
        "--features-cols",
        nargs="*",
        default=["exp_be", "exp_profit", "exp_gat", "exp_beta", "bm_gap", "turnover", "hhi", "n_pos"],
        help="Feature columns to use for PCA.",
    )
    ap.add_argument("--investor-col", default="mgrno")
    ap.add_argument("--type-col", default="type")
    ap.add_argument("--quarter-col", default="quarter")
    ap.add_argument("--out", default="artifacts/features/profile_centers_pca.png")
    return ap.parse_args()


def load_table(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def normalize_type(df: pd.DataFrame, type_col: str) -> pd.DataFrame:
    if type_col in df.columns:
        df[type_col] = (
            df[type_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
        )
    return df


def main() -> None:
    args = parse_args()
    inv, typ, qtr = args.investor_col, args.type_col, args.quarter_col
    feat_cols = args.features_cols

    f_frames = [normalize_type(load_table(fp), typ) for fp in args.features]
    p_frames = [normalize_type(load_table(fp), typ) for fp in args.profiles]
    f_df = pd.concat(f_frames, ignore_index=True)
    p_df = pd.concat(p_frames, ignore_index=True)

    # merge profile_k; handle type suffixes
    prof_cols = [inv, qtr, "profile_k"] + ([typ] if typ in p_df.columns else [])
    prof = p_df[prof_cols].drop_duplicates()
    df = f_df.merge(prof, on=[inv, qtr], how="inner", suffixes=("", "_prof"))
    if typ not in df.columns:
        if f"{typ}_prof" in df.columns:
            df[typ] = df[f"{typ}_prof"]
        else:
            df[typ] = "unknown"
    # drop suffixed type columns
    df = df.drop(columns=[c for c in df.columns if c.endswith("_prof")])

    # select features
    used_feats = [c for c in feat_cols if c in df.columns]
    if len(used_feats) == 0:
        raise SystemExit("No valid feature columns found.")

    X = df[used_feats].fillna(0.0).to_numpy(dtype=np.float64)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    df["pca1"], df["pca2"] = coords[:, 0], coords[:, 1]

    # compute centroids
    centroids = (
        df.groupby([typ, "profile_k"])
        .agg(
            n_obs=("pca1", "count"),
            pca1=("pca1", "mean"),
            pca2=("pca2", "mean"),
        )
        .reset_index()
    )

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for tval, g in centroids.groupby(typ):
        ax.scatter(g["pca1"], g["pca2"], s=20 + 2 * np.sqrt(g["n_obs"]), label=tval)
        for _, row in g.iterrows():
            ax.text(row["pca1"], row["pca2"], f'p{int(row["profile_k"])}', fontsize=8, ha="center", va="center")

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Profile centroids in PCA space")
    ax.legend()
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"[ok] saved plot -> {out_path}")


if __name__ == "__main__":
    main()
