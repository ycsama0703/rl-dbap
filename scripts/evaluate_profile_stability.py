#!/usr/bin/env python
"""
Compute profile transition/stability metrics:
- P(stay) per type
- Transition matrix per type

Example:
python scripts/evaluate_profile_stability.py \
  --profiles artifacts/features/mutual_funds_iq_profile.csv \
              artifacts/features/banks_iq_profile.csv \
  --out-prefix artifacts/features/profile_stability
"""

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", nargs="+", required=True, help="Profile files (csv/parquet).")
    p.add_argument("--out-prefix", required=True, help="Prefix for outputs (csv + txt).")
    return p.parse_args()


def _read(paths):
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


def main():
    args = parse_args()
    df = _read(args.profiles)
    if "quarter" not in df.columns:
        raise ValueError("profiles need a 'quarter' column")
    if "profile_k" not in df.columns:
        raise ValueError("profiles need a 'profile_k' column")
    key_inv = "mgrno" if "mgrno" in df.columns else "investor_id"
    if key_inv not in df.columns:
        raise ValueError("profiles need mgrno/investor_id column")

    typ_col = "type" if "type" in df.columns else "investor_type"
    df["investor_type"] = _norm_type(df[typ_col])
    df = df.rename(columns={key_inv: "investor_id"})

    # ensure quarter sortable
    try:
        df["quarter_idx"] = pd.PeriodIndex(pd.to_datetime(df["quarter"]), freq="Q").astype(int)
    except Exception:
        df["quarter_idx"] = pd.PeriodIndex(df["quarter"], freq="Q").astype(int)

    rows = []
    trans_frames = []
    for t, g in df.groupby("investor_type"):
        g = g.sort_values(["investor_id", "quarter_idx"])
        g["prev_profile"] = g.groupby("investor_id")["profile_k"].shift()
        g["stay"] = (g["profile_k"] == g["prev_profile"]).astype(float)
        # drop first quarter per investor (no prev)
        g2 = g.dropna(subset=["prev_profile"])
        if len(g2) == 0:
            continue
        p_stay = g2["stay"].mean()
        rows.append({"investor_type": t, "p_stay": p_stay, "n_transitions": len(g2)})

        trans = pd.crosstab(g2["prev_profile"], g2["profile_k"], normalize="index")
        trans["investor_type"] = t
        trans_frames.append(trans.reset_index().rename(columns={"prev_profile": "from_profile"}))

    summary = pd.DataFrame(rows)
    trans_df = pd.concat(trans_frames, axis=0, ignore_index=True) if trans_frames else pd.DataFrame()

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    summary_path = f"{args.out_prefix}_summary.csv"
    trans_path = f"{args.out_prefix}_transitions.csv"
    summary.to_csv(summary_path, index=False)
    trans_df.to_csv(trans_path, index=False)

    print(summary)
    print(f"[ok] saved summary -> {summary_path}")
    print(f"[ok] saved transitions -> {trans_path}")


if __name__ == "__main__":
    main()
