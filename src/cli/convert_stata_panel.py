from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import pandas as pd


TYPE_MAP = {
    "banks": "banks",
    "bank": "banks",
    "Banks": "banks",
    "Households": "households",
    "households": "households",
    "Investment advisors": "investment_advisors",
    "investment advisors": "investment_advisors",
    "Investment Advisors": "investment_advisors",
    "Mutual funds": "mutual_funds",
    "mutual funds": "mutual_funds",
    "Insurance companies": "insurance_companies",
    "insurance companies": "insurance_companies",
    "Pension funds": "pension_funds",
    "pension funds": "pension_funds",
    "Other": "other",
    "other": "other",
}

# Some sources encode type as small integers. Map the observed codes here; fall back to stringified code.
NUMERIC_TYPE_MAP = {
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
    "me",
    "be",
    "profit",
    "Gat",
    "beta",
    "aum",
    "outaum",
    "prc",
]

REQUIRED_SOURCE_COLUMNS = [
    "mgrno",
    "permno",
    "type",
    "fdate",
    "holding",
    "me",
    "be",
    "profit",
    "Gat",
    "beta",
    "aum",
    "outaum",
    "prc",
]


@dataclass
class WriterState:
    path: Path
    writer: "pq.ParquetWriter"


def _maybe_import_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard only
        raise SystemExit(
            "pyarrow is required for parquet output. Install it via `pip install pyarrow` "
            "before running this conversion script."
        ) from exc
    return pa, pq


def _normalize_type(raw: object) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        if pd.isna(raw):
            return None
        # Handle known numeric encodings
        mapped_num = NUMERIC_TYPE_MAP.get(int(raw))
        if mapped_num:
            return mapped_num
        return str(int(raw))
    s = str(raw).strip()
    if not s:
        return None
    mapped = TYPE_MAP.get(s)
    if mapped:
        return mapped
    mapped = TYPE_MAP.get(s.lower())
    if mapped:
        return mapped
    s_norm = (
        s.lower()
        .replace("&", " and ")
        .replace("-", " ")
        .replace("/", " ")
        .replace(",", " ")
        .replace(".", " ")
    )
    s_norm = "_".join(part for part in s_norm.split() if part)
    return s_norm or None


def _prepare_chunk(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty:
        # ensure canonical column names
        df = df.rename(columns={"fdate": "date", "holding": "holding_t"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["type"] = df["type"].map(_normalize_type)
        numeric_cols = ["holding_t", "me", "be", "profit", "Gat", "beta", "aum", "outaum", "prc"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["mgrno"] = pd.to_numeric(df["mgrno"], errors="coerce").astype("Int64")
        df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
        df = df[df["type"].notna()]
        df = df[df["date"].notna()]
        df = df[df["mgrno"].notna() & df["permno"].notna()]
        keep_cols = [c for c in OUTPUT_COLUMNS if c in df.columns or c == "holding_t1"]
        if "holding_t1" not in keep_cols:
            keep_cols.insert(keep_cols.index("holding_t") + 1, "holding_t1")
        df = df.reindex(columns=keep_cols, fill_value=pd.NA)
    return df


def _compute_leads(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["type", "mgrno", "permno", "date"])
    group_keys = ["type", "mgrno", "permno"]
    # Explicit observed=False to keep current behavior and silence future warning
    df["__next_date"] = df.groupby(group_keys, observed=False)["date"].shift(-1)
    df["holding_t1"] = df.groupby(group_keys, observed=False)["holding_t"].shift(-1)

    # Compute quarter gap safely, avoiding NaT -> Int casting issues
    mask = df["__next_date"].notna() & df["date"].notna()
    gap = pd.Series(pd.NA, index=df.index, dtype="Int64")
    if mask.any():
        next_qnum = df.loc[mask, "__next_date"].dt.to_period("Q").astype("int64")
        curr_qnum = df.loc[mask, "date"].dt.to_period("Q").astype("int64")
        diff = (next_qnum - curr_qnum).astype("int64")
        gap.loc[mask] = diff.values
    df.loc[gap != 1, "holding_t1"] = pd.NA
    return df


def _split_emit_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (emit_now, carry_forward)."""
    if df.empty:
        return df, df
    group_keys = ["type", "mgrno", "permno"]
    last_idx = df.groupby(group_keys, sort=False).tail(1).index
    carry = df.loc[last_idx].copy()
    emit = df.drop(index=last_idx)
    emit = emit.drop(columns="__next_date", errors="ignore")
    carry = carry.drop(columns="__next_date", errors="ignore")
    return emit, carry


def _write_frames(
    frames: Iterable[pd.DataFrame],
    *,
    writers: Dict[str, WriterState],
    pa_module,
    pq_module,
    out_dir: Path,
    overwrite: bool,
    compression: str,
    counts: Dict[str, int],
    include_all: bool,
) -> None:
    pa = pa_module
    pq = pq_module

    def write_part(target_key: str, part: pd.DataFrame) -> None:
        if part.empty:
            return
        target = out_dir / f"{target_key}.parquet"
        part = part[OUTPUT_COLUMNS]
        counts[target_key] = counts.get(target_key, 0) + len(part)
        table = pa.Table.from_pandas(part, preserve_index=False)
        state = writers.get(target_key)
        if state is None:
            if target.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"Output file exists ({target}). "
                        "Use --overwrite to replace or write to a different directory."
                    )
                target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(target, table.schema, compression=compression)
            writers[target_key] = WriterState(path=target, writer=writer)
            state = writers[target_key]
        state.writer.write_table(table)

    for frame in frames:
        if frame.empty:
            continue
        for t, part in frame.groupby("type", sort=False):
            write_part(t, part)
        if include_all:
            write_part("all_investors", frame)
    # no explicit return


def _close_writers(writers: Dict[str, WriterState]) -> None:
    for state in writers.values():
        try:
            state.writer.close()
        except Exception:
            pass


def iter_stata_chunks(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    reader = pd.read_stata(
        path,
        columns=REQUIRED_SOURCE_COLUMNS,
        iterator=True,
        chunksize=chunksize,
        convert_categoricals=True,
    )
    for chunk in reader:
        yield chunk


def main():
    ap = argparse.ArgumentParser(
        description="Convert a large Stata panel (Data_2020_2024.dta) into per-type parquet slices compatible with rl-dbap."
    )
    ap.add_argument("--input", type=Path, default=Path("data/Data_2020_2024.dta"), help="Path to the Stata .dta file")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/panel_quarter_2020_2024.parquet"),
        help="Directory where per-type parquet files will be written",
    )
    ap.add_argument("--chunksize", type=int, default=250_000, help="Number of rows to process per chunk")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing parquet files in the output directory",
    )
    ap.add_argument(
        "--compression",
        type=str,
        default="snappy",
        help="Parquet compression codec (default: snappy)",
    )
    args = ap.parse_args()

    pa, pq = _maybe_import_pyarrow()

    if not args.input.exists():
        raise FileNotFoundError(f"Input Stata file not found: {args.input}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    writers: Dict[str, WriterState] = {}
    counts: Dict[str, int] = {}
    carry = pd.DataFrame(columns=OUTPUT_COLUMNS)
    total_rows = 0

    try:
        for i, raw_chunk in enumerate(iter_stata_chunks(args.input, args.chunksize), start=1):
            prepared = _prepare_chunk(raw_chunk)
            if carry is not None and not carry.empty:
                prepared = pd.concat([carry, prepared], ignore_index=True)
            prepared = _compute_leads(prepared)
            emit, carry = _split_emit_frames(prepared)
            _write_frames(
                [emit],
                writers=writers,
                pa_module=pa,
                pq_module=pq,
                out_dir=out_dir,
                overwrite=args.overwrite,
                compression=args.compression,
                counts=counts,
                include_all=True,
            )
            total_rows += len(raw_chunk)
            print(f"[convert-stata] processed chunk {i:,}, cumulative rows={total_rows:,}", flush=True)

        if carry is not None and not carry.empty:
            carry = _compute_leads(carry)
            _write_frames(
                [carry.drop(columns="__next_date", errors="ignore")],
                writers=writers,
                pa_module=pa,
                pq_module=pq,
                out_dir=out_dir,
                overwrite=args.overwrite,
                compression=args.compression,
                counts=counts,
                include_all=True,
            )
    finally:
        _close_writers(writers)

    total_written = sum(counts.values())
    print(f"[convert-stata] done. total source rows={total_rows:,}, written rows={total_written:,}")
    for t, n in sorted(counts.items()):
        print(f"  - {t}: {n:,}")
    print(f"[convert-stata] output directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
