#!/usr/bin/env python
"""
Evaluate profile identifiability:
1) F-test (ANOVA) for each feature across profile_k groups.
2) Simple classifier accuracy vs. random baseline (1/K).

Example:
python scripts/evaluate_profile_identifiability.py \
  --features artifacts/features/mutual_funds_iq_features.csv \
  --profiles artifacts/features/mutual_funds_iq_profile.csv \
  --out-prefix artifacts/features/identifiability_mutual_funds

You can pass multiple --features/--profiles; they will be concatenated.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True, help="Feature files (csv or parquet).")
    p.add_argument("--profiles", nargs="+", required=True, help="Profile files (csv or parquet).")
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
        help="Feature columns to test.",
    )
    p.add_argument("--sample-per-type", type=int, default=5000, help="Max rows per type for speed.")
    p.add_argument("--out-prefix", required=True, help="Prefix for metrics output (csv + txt).")
    return p.parse_args()


def _read(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        path = Path(p)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True)


def _norm_type(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.replace(" ", "_")


def load_and_merge(feat_paths: List[str], prof_paths: List[str], feat_cols: List[str], sample_per_type: int):
    feats = _read(feat_paths)
    profs = _read(prof_paths)

    # normalize type column name
    typ_col_feat = "type" if "type" in feats.columns else "investor_type"
    typ_col_prof = "type" if "type" in profs.columns else "investor_type"
    feats["investor_type"] = _norm_type(feats[typ_col_feat])
    profs["investor_type"] = _norm_type(profs[typ_col_prof])

    # merge on investor + quarter
    key_inv = "mgrno" if "mgrno" in feats.columns else "investor_id"
    feats = feats.rename(columns={key_inv: "investor_id"})
    profs = profs.rename(columns={key_inv: "investor_id"})

    if "quarter" not in feats.columns or "quarter" not in profs.columns:
        raise ValueError("Both features and profiles need a 'quarter' column.")

    df = feats.merge(
        profs[["investor_id", "quarter", "investor_type", "profile_k"]],
        on=["investor_id", "quarter", "investor_type"],
        how="inner",
    )

    # dropna and keep available columns
    use_cols = [c for c in feat_cols if c in df.columns]
    df = df.dropna(subset=use_cols + ["profile_k", "investor_type"])

    # sample per type
    out = []
    for t, g in df.groupby("investor_type"):
        if len(g) > sample_per_type:
            g = g.sample(sample_per_type, random_state=42)
        out.append(g)
    df = pd.concat(out, axis=0, ignore_index=True)
    return df, use_cols


def compute_anova(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    X = df[cols].to_numpy()
    y = df["profile_k"].to_numpy()
    f_stats, p_vals = f_classif(X, y)
    return pd.DataFrame({"feature": cols, "f_stat": f_stats, "p_value": p_vals})


def evaluate_classifier(df: pd.DataFrame, cols: List[str]) -> Tuple[float, float, int]:
    X = df[cols].to_numpy()
    y = df["profile_k"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    baseline = 1.0 / len(np.unique(y))
    return acc, baseline, len(y_test)


def main():
    args = parse_args()
    df, use_cols = load_and_merge(args.features, args.profiles, args.feature_cols, args.sample_per_type)
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    anova_df = compute_anova(df, use_cols)
    anova_path = f"{args.out_prefix}_anova.csv"
    anova_df.to_csv(anova_path, index=False)

    acc, baseline, n_test = evaluate_classifier(df, use_cols)

    summary = [
        f"rows_used: {len(df)}",
        f"unique_types: {df['investor_type'].nunique()}",
        f"unique_profiles: {df['profile_k'].nunique()}",
        f"classifier_accuracy: {acc:.4f}",
        f"baseline_random: {baseline:.4f}",
        f"n_test: {n_test}",
        f"anova_path: {anova_path}",
    ]
    summary_path = f"{args.out_prefix}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary))

    print("\n".join(summary))
    print(f"[ok] saved ANOVA -> {anova_path}")
    print(f"[ok] saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
