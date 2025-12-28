#!/usr/bin/env python
"""
Build per-type parquet slices from data/raw/Data1.parquet for a fixed permno universe,
adding shares and holding_t1 with quarter continuity enforced.

The permno whitelist is the union of three cap buckets below (LARGE/MID/SMALL).
Time coverage is restricted to 2015Q1..2024Q3 to match existing panel_quarter outputs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# Permno whitelist
LARGE_CAP = [10107, 14593, 22752, 22111, 84788]
MID_CAP = [91416, 77606, 40125, 89217, 57817]
SMALL_CAP = [91277, 14273, 13947, 80539, 42585]
PERMNO_WHITELIST = set(LARGE_CAP + MID_CAP + SMALL_CAP)

# Numeric code -> canonical type
NUMERIC_TYPE_MAP: Dict[int, str] = {
    0: "households",
    1: "banks",
    2: "insurance_companies",
    3: "investment_advisors",
    4: "mutual_funds",
    5: "pension_funds",
    6: "other",
}

# Date filter: fdate is quarters since 1960Q1; 2015Q1=220, 2024Q3=258.
FDATE_MIN = 220
FDATE_MAX = 258

OUTPUT_COLS = [
    "type",
    "mgrno",
    "permno",
    "date",
    "holding_t",
    "holding_t1",
    "shares",
    "me",
    "be",
    "profit",
    "Gat",
    "beta",
    "aum",
    "outaum",
    "prc",
]


def fdate_to_timestamp(fdate_series: pd.Series) -> pd.Series:
    """Convert integer quarters since 1960Q1 to quarter-start Timestamp."""
    f = fdate_series.astype("Int64")
    year = 1960 + (f // 4)
    quarter = (f % 4) + 1
    return pd.PeriodIndex(year.astype(int).astype(str) + "Q" + quarter.astype(int).astype(str), freq="Q").to_timestamp()


def load_filtered_dataframe(data_path: Path) -> pd.DataFrame:
    dataset = ds.dataset(data_path, format="parquet")
    filt = (
        ds.field("permno").isin(PERMNO_WHITELIST)
        & (ds.field("fdate") >= FDATE_MIN)
        & (ds.field("fdate") <= FDATE_MAX)
    )
    cols = [
        "fdate",
        "mgrno",
        "permno",
        "type",
        "holding",
        "shares",
        "aum",
        "outaum",
        "be",
        "profit",
        "Gat",
        "beta",
        "LNme",
        "LNbe",
    ]
    tbl = dataset.to_table(columns=cols, filter=filt)
    df = tbl.to_pandas()
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["type"] = df["type"].map(lambda x: NUMERIC_TYPE_MAP.get(int(x)) if pd.notna(x) else None)
    df = df[df["type"].notna()]
    df["date"] = fdate_to_timestamp(df["fdate"])
    df = df.drop(columns=["fdate"])

    # Rename columns
    df = df.rename(columns={"holding": "holding_t"})

    # Compute helpers
    if "shares" in df.columns:
        df["prc"] = df.apply(
            lambda r: r["holding_t"] / r["shares"] if pd.notna(r["holding_t"]) and pd.notna(r["shares"]) and r["shares"] != 0 else math.nan,
            axis=1,
        )
    else:
        df["prc"] = math.nan

    if "LNme" in df.columns:
        df["me"] = df["LNme"].apply(lambda v: math.exp(v) if pd.notna(v) else math.nan)
    else:
        df["me"] = math.nan

    if "LNbe" in df.columns and "be" not in df.columns:
        df["be"] = df["LNbe"].apply(lambda v: math.exp(v) if pd.notna(v) else math.nan)

    # Keep only output cols; drop extra
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = math.nan
    df = df[OUTPUT_COLS]

    # Compute holding_t1 with quarter continuity
    df = df.sort_values(["type", "mgrno", "permno", "date"])
    group_keys = ["type", "mgrno", "permno"]
    df["next_date"] = df.groupby(group_keys, observed=False)["date"].shift(-1)
    df["holding_t1"] = df.groupby(group_keys, observed=False)["holding_t"].shift(-1)
    # enforce consecutive quarters
    mask = df["next_date"].notna()
    if mask.any():
        curr_q = df.loc[mask, "date"].dt.to_period("Q").astype("int64")
        next_q = df.loc[mask, "next_date"].dt.to_period("Q").astype("int64")
        gap = (next_q - curr_q).astype("int64")
        break_idx = gap[gap != 1].index
        if len(break_idx):
            df.loc[break_idx, "holding_t1"] = pd.NA
    df = df.drop(columns=["next_date"])

    return df


def write_per_type(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for t, part in df.groupby("type"):
        out_path = out_dir / f"{t}.parquet"
        part = part.drop(columns=["type"])
        part.to_parquet(out_path, index=False, compression="snappy")
        print(f"[ok] wrote {len(part):,} rows -> {out_path}")


def main():
    data_path = Path("data/raw/Data1.parquet")
    out_dir = Path("artifacts/panel_quarter_cap_filtered.parquet")
    df_raw = load_filtered_dataframe(data_path)
    df = prepare_dataframe(df_raw)
    write_per_type(df, out_dir)


if __name__ == "__main__":
    main()
