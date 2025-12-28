#!/usr/bin/env python
"""
Generate a simple semantic summary for each (investor_type, profile_k) using
feature means and optional objective weights. No LLM needed; uses heuristic text.

Example:
python scripts/generate_profile_semantics.py \
  --features artifacts/features/*_iq_features.csv \
  --profiles artifacts/features/*_iq_profile.csv \
  --weights artifacts/features/*_profile_objective_weights.csv \
  --out artifacts/features/profile_semantics.md
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np


DEF_FEATS = ["exp_be", "exp_profit", "exp_gat", "exp_beta", "bm_gap", "turnover", "hhi", "n_pos"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True, help="Feature tables (csv/parquet).")
    p.add_argument("--profiles", nargs="+", required=True, help="Profile tables with profile_k (csv/parquet).")
    p.add_argument("--weights", nargs="*", default=None, help="Optional objective weight tables (csv/parquet).")
    p.add_argument("--feature-cols", nargs="*", default=DEF_FEATS, help="Feature columns to summarize.")
    p.add_argument("--out", required=True, help="Output markdown path.")
    p.add_argument("--topn", type=int, default=3, help="How many strongest features to list.")
    return p.parse_args()


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def _read_many(paths: List[str]) -> pd.DataFrame:
    dfs = [_read(p) for p in paths]
    return pd.concat(dfs, axis=0, ignore_index=True)


def _norm_type(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.replace(" ", "_")


def _to_quarter_str(col: pd.Series) -> pd.Series:
    try:
        return pd.PeriodIndex(pd.to_datetime(col), freq="Q").astype(str)
    except Exception:
        return pd.PeriodIndex(col, freq="Q").astype(str)


def load_and_merge(feat_paths, prof_paths, feat_cols):
    feats = _read_many(feat_paths)
    profs = _read_many(prof_paths)
    typ_f = "type" if "type" in feats.columns else "investor_type"
    typ_p = "type" if "type" in profs.columns else "investor_type"
    feats["investor_type"] = _norm_type(feats[typ_f])
    profs["investor_type"] = _norm_type(profs[typ_p])

    inv_f = "mgrno" if "mgrno" in feats.columns else "investor_id"
    inv_p = "mgrno" if "mgrno" in profs.columns else "investor_id"
    feats = feats.rename(columns={inv_f: "investor_id"})
    profs = profs.rename(columns={inv_p: "investor_id"})

    if "quarter" in feats.columns:
        feats["quarter_str"] = _to_quarter_str(feats["quarter"])
    elif "date" in feats.columns:
        feats["quarter_str"] = _to_quarter_str(feats["date"])
    else:
        raise ValueError("features need quarter or date")
    if "quarter" in profs.columns:
        profs["quarter_str"] = _to_quarter_str(profs["quarter"])
    else:
        raise ValueError("profiles need quarter")

    use_cols = [c for c in feat_cols if c in feats.columns]
    merged = feats.merge(
        profs[["investor_id", "quarter_str", "investor_type", "profile_k"]],
        on=["investor_id", "quarter_str", "investor_type"],
        how="inner",
    )
    merged = merged.dropna(subset=use_cols + ["profile_k"])
    return merged, use_cols


def attach_weights(df: pd.DataFrame, weights_paths: List[str] | None) -> pd.DataFrame:
    if not weights_paths:
        return df
    wdf = _read_many(weights_paths)
    typ_col = "investor_type" if "investor_type" in wdf.columns else "type"
    wdf["investor_type"] = _norm_type(wdf[typ_col])
    wdf = wdf[["investor_type", "profile_k", "alpha_w", "risk_w", "tc_w", "te_w"]]
    return df.merge(wdf, how="left", on=["investor_type", "profile_k"])


def summarize(df: pd.DataFrame, feat_cols: List[str], topn: int) -> str:
    md_lines = ["# Profile semantics (heuristic)", ""]
    for t, g in df.groupby("investor_type"):
        md_lines.append(f"## {t}")
        # type-level mean/std for z-score
        type_mean = g[feat_cols].mean()
        type_std = g[feat_cols].std().replace(0, np.nan)
        for k, p in g.groupby("profile_k"):
            n = len(p)
            means = p[feat_cols].mean()
            z = (means - type_mean) / type_std
            z = z.replace([np.inf, -np.inf], np.nan)
            # pick top positive/negative z
            z_sorted = z.dropna().sort_values(ascending=False)
            top_pos = z_sorted.head(topn)
            top_neg = z_sorted.tail(topn)
            md_lines.append(f"### profile p{k} (n={n})")
            md_lines.append("- top ↑ features (z-score): " + ", ".join(f"{f}={v:.2f}" for f, v in top_pos.items()))
            md_lines.append("- top ↓ features (z-score): " + ", ".join(f"{f}={v:.2f}" for f, v in top_neg.items()))
            if {"alpha_w", "risk_w", "tc_w", "te_w"}.issubset(p.columns):
                aw = p["alpha_w"].iloc[0]
                rw = p["risk_w"].iloc[0]
                tc = p["tc_w"].iloc[0]
                te = p["te_w"].iloc[0]
                md_lines.append(f"- objective_weights: alpha={aw:.2f}, risk={rw:.2f}, tc={tc:.2f}, te={te:.2f}")
            md_lines.append("")
        md_lines.append("")
    return "\n".join(md_lines)


def main():
    args = parse_args()
    merged, use_cols = load_and_merge(args.features, args.profiles, args.feature_cols)
    merged = attach_weights(merged, args.weights)
    md = summarize(merged, use_cols, args.topn)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
