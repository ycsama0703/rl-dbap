#!/usr/bin/env python
"""
Export investor-level profile assignments (one profile per investor) from quarterly profile tables.

Input: one or more *_iq_profile.(csv|parquet) produced by cluster_investor_profiles.py
Output: CSV/Parquet with unique (type, mgrno, profile_k) rows, plus optional summary counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Deduplicate quarterly profile tables to investor-level assignments.")
    ap.add_argument("--profiles", nargs="+", required=True, help="Profile files (csv/parquet) with profile_k.")
    ap.add_argument("--out-path", required=True, help="Output CSV/Parquet for investor-level profiles.")
    ap.add_argument("--investor-col", default="mgrno")
    ap.add_argument("--type-col", default="type")
    ap.add_argument("--quarter-col", default="quarter")
    args = ap.parse_args()

    inv, typ, qtr = args.investor_col, args.type_col, args.quarter_col
    frames: List[pd.DataFrame] = []

    for path_str in args.profiles:
        path = Path(path_str)
        df = load_table(path)
        if typ in df.columns:
            df[typ] = (
                df[typ]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", "_", regex=True)
            )
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)

    # Ensure single profile per investor by taking the first non-null per (type, inv)
    cols = [c for c in [typ, inv, "profile_k"] if c in df_all.columns]
    inv_level = (
        df_all[cols]
        .dropna(subset=["profile_k"])
        .drop_duplicates(subset=[typ, inv], keep="first")
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        inv_level.to_csv(out_path, index=False)
    else:
        inv_level.to_parquet(out_path, index=False)

    # Summary counts per type/profile
    summary = (
        inv_level.groupby([typ, "profile_k"])
        .size()
        .reset_index(name="n_investors")
        .sort_values([typ, "profile_k"])
    )
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"[ok] wrote investor-level profiles: {len(inv_level):,} rows -> {out_path}")
    print(f"[ok] summary counts -> {summary_path}")


if __name__ == "__main__":
    main()
