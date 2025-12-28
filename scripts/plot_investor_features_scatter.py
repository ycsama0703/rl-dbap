#!/usr/bin/env python
"""
Project investor-quarter feature points to 2D and visualize type separation.

Usage:
  python scripts/plot_investor_features_scatter.py \
    --features artifacts/features/mutual_funds_iq_features.csv \
               artifacts/features/banks_iq_features.csv \
    --out artifacts/features/iq_features_scatter.png \
    --sample-per-type 5000

The script:
  - Reads one or more feature files (csv or parquet).
  - Normalizes investor_type (lowercase with underscores).
  - Optionally subsamples per type.
  - Applies PCA (or UMAP if installed and --use-umap) to 2D.
  - Plots a scatter with color by investor_type.
"""

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Feature files (csv/parquet) from build_investor_quarter_features.py",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output PNG path.",
    )
    p.add_argument(
        "--sample-per-type",
        type=int,
        default=5000,
        help="Max rows to sample per investor_type for plotting.",
    )
    p.add_argument(
        "--use-umap",
        action="store_true",
        help="Use UMAP instead of PCA if umap-learn is installed.",
    )
    p.add_argument(
        "--feature-cols",
        nargs="+",
        default=[
            "exp_be",
            "exp_profit",
            "exp_gat",
            "exp_beta",
            "bm_gap",
            "turnover",
            "hhi",
            "n_pos",
        ],
        help="Feature columns to project.",
    )
    return p.parse_args()


def _normalize_type(s: pd.Series) -> pd.Series:
    return s.str.lower().str.replace(" ", "_")


def _load_frames(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        path = Path(p)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True)


def main():
    args = parse_args()
    df = _load_frames(args.features)
    if "type" not in df.columns and "investor_type" not in df.columns:
        raise ValueError("Expected a 'type' or 'investor_type' column in features.")
    typ_col = "type" if "type" in df.columns else "investor_type"
    df["investor_type"] = _normalize_type(df[typ_col].astype(str))

    # Keep only requested feature columns that exist
    feat_cols = [c for c in args.feature_cols if c in df.columns]
    if len(feat_cols) == 0:
        raise ValueError("No feature columns found in data.")
    df = df.dropna(subset=feat_cols + ["investor_type"])

    # Sample per type
    samples = []
    for t, g in df.groupby("investor_type"):
        if len(g) > args.sample_per_type:
            g = g.sample(args.sample_per_type, random_state=42)
        samples.append(g)
    df = pd.concat(samples, axis=0, ignore_index=True)

    X = df[feat_cols].to_numpy()
    reducer_name = "PCA"
    if args.use_umap:
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42)
            reducer_name = "UMAP"
        except Exception:
            reducer = PCA(n_components=2, random_state=42)
            reducer_name = "PCA (UMAP unavailable)"
    else:
        reducer = PCA(n_components=2, random_state=42)

    coords = reducer.fit_transform(X)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    types = sorted(df["investor_type"].unique())
    cmap = plt.get_cmap("tab20")
    color_map = {t: cmap(i % cmap.N) for i, t in enumerate(types)}

    plt.figure(figsize=(10, 8))
    for t in types:
        g = df[df["investor_type"] == t]
        plt.scatter(
            g["x"],
            g["y"],
            s=8,
            alpha=0.5,
            color=color_map[t],
            label=t,
            rasterized=True,
        )
    plt.legend(markerscale=3, fontsize=9, frameon=False)
    plt.title(f"Investor-quarter features projected to 2D ({reducer_name})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300)
    print(f"[ok] saved scatter -> {args.out}")


if __name__ == "__main__":
    main()
