#!/usr/bin/env python
"""
Plot profile objective weights for multiple investor types.

Reads one or more CSV/Parquet files with columns:
  investor_type, profile_k, alpha_w, risk_w, tc_w, te_w

Outputs a stacked bar chart comparing weights across types/profiles.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize profile objective weights.")
    ap.add_argument(
        "--inputs", nargs="+", required=True, help="CSV/Parquet files with profile_objective_weights."
    )
    ap.add_argument("--out", default="artifacts/features/profile_weights_compare.png")
    return ap.parse_args()


def load_table(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main() -> None:
    args = parse_args()
    frames: List[pd.DataFrame] = []
    for fp in args.inputs:
        df = load_table(fp)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # build label and tidy format
    df["label"] = df["investor_type"].astype(str) + "_p" + df["profile_k"].astype(str)
    tidy = df.melt(
        id_vars=["label"],
        value_vars=["alpha_w", "risk_w", "tc_w", "te_w"],
        var_name="component",
        value_name="weight",
    )

    # plot stacked bars
    fig, ax = plt.subplots(figsize=(10, 5))
    components = ["alpha_w", "risk_w", "tc_w", "te_w"]
    bottom = pd.Series(0.0, index=df["label"])
    for comp in components:
        vals = df.set_index("label")[comp]
        ax.bar(df["label"], vals, bottom=bottom[df["label"]], label=comp)
        bottom[df["label"]] = bottom[df["label"]].to_numpy() + vals.values

    ax.set_ylabel("weight (normalized)")
    ax.set_title("Profile objective weights by investor_type/profile")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"[ok] saved plot -> {out_path}")


if __name__ == "__main__":
    main()
