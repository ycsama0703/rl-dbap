#!/usr/bin/env python
"""
Convert data/raw/Data1.parquet into per-type parquet slices (2015Q1-2024Q3),
including shares and price, with holding_t1 computed for consecutive quarters.

Output columns:
  type, mgrno, permno, date, holding_t, holding_t1, shares, me, be,
  profit, Gat, beta, aum, outaum, prc
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

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

OUTPUT_COLUMNS = [
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

REQUIRED_COLUMNS = [
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
]

# Keep same time coverage as existing panel: 2015Q1 (fdate=220) to 2024Q3 (fdate=258)
FDATE_MIN = 220
FDATE_MAX = 258


@dataclass
class WriterState:
    path: Path
    writer: "pq.ParquetWriter"


def _normalize_type(raw: object) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        if pd.isna(raw):
            return None
        mapped = NUMERIC_TYPE_MAP.get(int(raw))
        return mapped or str(int(raw))
    s = str(raw).strip()
    if not s:
        return None
    return s.lower().replace(" ", "_")


def _fdate_to_timestamp(fdate: pd.Series) -> pd.Series:
    f = pd.to_numeric(fdate, errors="coerce").astype("Int64")
    year = 1960 + (f // 4)
    quarter = (f % 4) + 1
    return pd.PeriodIndex(year.astype(int).astype(str) + "Q" + quarter.astype(int).astype(str), freq="Q").to_timestamp()


def _prepare_chunk(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["type"] = df["type"].map(_normalize_type)
    df = df[df["type"].notna()]
    df["date"] = _fdate_to_timestamp(df["fdate"])
    df = df[df["date"].notna()]
    df = df.rename(columns={"holding": "holding_t"})

    numeric_cols = ["holding_t", "shares", "aum", "outaum", "be", "profit", "Gat", "beta"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["mgrno"] = pd.to_numeric(df["mgrno"], errors="coerce").astype("Int64")
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["type", "date", "mgrno", "permno"])

    # Compute price from holding/shares if possible
    df["prc"] = df.apply(
        lambda r: (r["holding_t"] / r["shares"]) if pd.notna(r["holding_t"]) and pd.notna(r["shares"]) and r["shares"] != 0 else math.nan,
        axis=1,
    )
    # Derive me from LNme if present
    if "LNme" in df.columns:
        df["me"] = df["LNme"].apply(lambda v: math.exp(v) if pd.notna(v) else math.nan)
    if "me" not in df.columns:
        df["me"] = math.nan
    # If be missing but LNbe present, back out
    if "be" not in df.columns and "LNbe" in df.columns:
        df["be"] = df["LNbe"].apply(lambda v: math.exp(v) if pd.notna(v) else math.nan)

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[OUTPUT_COLUMNS + ["fdate"]]  # keep fdate for filtering
    return df


def _compute_leads(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["type", "mgrno", "permno", "date"])
    group_keys = ["type", "mgrno", "permno"]
    df["__next_date"] = df.groupby(group_keys, observed=False)["date"].shift(-1)
    df["holding_t1"] = df.groupby(group_keys, observed=False)["holding_t"].shift(-1)
    mask = df["__next_date"].notna()
    if mask.any():
        curr_q = df.loc[mask, "date"].dt.to_period("Q").astype("int64")
        next_q = df.loc[mask, "__next_date"].dt.to_period("Q").astype("int64")
        gap = (next_q - curr_q).astype("int64")
        break_idx = gap[gap != 1].index
        if len(break_idx):
            df.loc[break_idx, "holding_t1"] = pd.NA
    df = df.drop(columns="__next_date")
    return df


def _split_emit_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    group_keys = ["type", "mgrno", "permno"]
    last_idx = df.groupby(group_keys, sort=False).tail(1).index
    carry = df.loc[last_idx].copy()
    emit = df.drop(index=last_idx)
    return emit, carry


def iter_parquet_batches(path: Path, columns: Iterable[str], batch_size: int) -> Iterator[pd.DataFrame]:
    dataset = ds.dataset(path, format="parquet")
    filt = (ds.field("fdate") >= FDATE_MIN) & (ds.field("fdate") <= FDATE_MAX)
    scanner = ds.Scanner.from_dataset(dataset, columns=columns, filter=filt, batch_size=batch_size)
    for batch in scanner.to_batches():
        yield batch.to_pandas()


def main():
    ap = argparse.ArgumentParser(description="Convert Data1.parquet into per-type parquet slices (2015-2024) with shares.")
    ap.add_argument("--input", type=Path, default=Path("data/raw/Data1.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/panel_quarter_full.parquet"))
    ap.add_argument("--batch-size", type=int, default=500_000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--compression", type=str, default="snappy")
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    writers: Dict[str, WriterState] = {}
    counts: Dict[str, int] = {}
    carry = pd.DataFrame(columns=OUTPUT_COLUMNS + ["fdate"])
    total_rows = 0

    try:
        for i, raw_df in enumerate(iter_parquet_batches(args.input, REQUIRED_COLUMNS + ["LNme", "LNbe"], args.batch_size), start=1):
            prepared = _prepare_chunk(raw_df)
            if not carry.empty:
                prepared = pd.concat([carry, prepared], ignore_index=True)
            prepared = _compute_leads(prepared)
            emit, carry = _split_emit_frames(prepared)
            for frame in [emit]:
                if frame.empty:
                    continue
                for t, part in frame.groupby("type", sort=False):
                    target = out_dir / f"{t}.parquet"
                    part = part.drop(columns=["fdate"])
                    counts[t] = counts.get(t, 0) + len(part)
                    table = pa.Table.from_pandas(part, preserve_index=False)
                    state = writers.get(t)
                    if state is None:
                        if target.exists() and not args.overwrite:
                            raise FileExistsError(f"Output exists: {target} (use --overwrite)")
                        if target.exists():
                            target.unlink()
                        writer = pq.ParquetWriter(target, table.schema, compression=args.compression)
                        writers[t] = WriterState(path=target, writer=writer)
                        state = writers[t]
                    state.writer.write_table(table)
            total_rows += len(raw_df)
            print(f"[convert-data1] processed batch {i:,}, cumulative source rows={total_rows:,}", flush=True)

        if not carry.empty:
            carry = _compute_leads(carry)
            for t, part in carry.groupby("type", sort=False):
                target = out_dir / f"{t}.parquet"
                part = part.drop(columns=["fdate"])
                counts[t] = counts.get(t, 0) + len(part)
                table = pa.Table.from_pandas(part, preserve_index=False)
                state = writers.get(t)
                if state is None:
                    if target.exists() and not args.overwrite:
                        raise FileExistsError(f"Output exists: {target} (use --overwrite)")
                    if target.exists():
                        target.unlink()
                    writer = pq.ParquetWriter(target, table.schema, compression=args.compression)
                    writers[t] = WriterState(path=target, writer=writer)
                    state = writers[t]
                state.writer.write_table(table)
    finally:
        for st in writers.values():
            try:
                st.writer.close()
            except Exception:
                pass

    total_written = sum(counts.values())
    print(f"[convert-data1] done. total source rows (filtered) ~{total_rows:,}, written rows={total_written:,}")
    for t, n in sorted(counts.items()):
        print(f"  - {t}: {n:,}")
    print(f"[convert-data1] output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
