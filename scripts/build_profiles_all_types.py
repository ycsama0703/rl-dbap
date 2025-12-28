#!/usr/bin/env python
"""
Batch pipeline: per-type features + static profile clustering for all parquet files in a folder.

Steps per type:
  1) build_investor_quarter_features.py -> <out_features_dir>/<type>_iq_features.csv
  2) cluster_investor_profiles.py       -> <out_profiles_dir>/<type>_iq_profile.csv

Profiles are static per investor (profile_k averaged over time and broadcast to all quarters).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run(cmd: List[str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Build features + profiles for all per-type parquet files.")
    ap.add_argument("--in-dir", type=Path, required=True, help="Folder with per-type parquet (e.g., panel_quarter_full.parquet).")
    ap.add_argument("--out-features", type=Path, required=True, help="Output folder for investor-quarter features.")
    ap.add_argument("--out-profiles", type=Path, required=True, help="Output folder for profile_k tables.")
    ap.add_argument("--n-clusters", type=int, default=4, help="KMeans clusters per type (default 4).")
    ap.add_argument("--holding-col", default="holding_t")
    ap.add_argument("--price-col", default="prc")
    ap.add_argument("--type-col", default="type")
    ap.add_argument("--skip", nargs="*", default=[], help="Type names to skip (stem without .parquet).")
    args = ap.parse_args()

    in_dir: Path = args.in_dir
    feats_dir: Path = args.out_features
    prof_dir: Path = args.out_profiles
    feats_dir.mkdir(parents=True, exist_ok=True)
    prof_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(in_dir.glob("*.parquet")):
        stem = path.stem
        if stem in args.skip or stem == "all_investors":
            print(f"[skip] {stem}")
            continue
        feat_out = feats_dir / f"{stem}_iq_features.csv"
        prof_out = prof_dir / f"{stem}_iq_profile.csv"

        # Stage 1: features
        run(
            [
                sys.executable,
                "scripts/build_investor_quarter_features.py",
                "--in-path",
                str(path),
                "--out-path",
                str(feat_out),
                "--holding-col",
                args.holding_col,
                "--price-col",
                args.price_col,
                "--type-col",
                args.type_col,
            ]
        )

        # Stage 2: profiles (static per investor)
        run(
            [
                sys.executable,
                "scripts/cluster_investor_profiles.py",
                "--in-path",
                str(feat_out),
                "--out-path",
                str(prof_out),
                "--n-clusters",
                str(args.n_clusters),
            ]
        )

        print(f"[ok] {stem}: features -> {feat_out}, profiles -> {prof_out}")


if __name__ == "__main__":
    main()
