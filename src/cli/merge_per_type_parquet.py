from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

import pandas as pd


def _maybe_import_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.compute as pc  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard only
        raise SystemExit(
            "pyarrow is required to merge parquet files. Install it via `pip install pyarrow`."
        ) from exc
    return pa, pc, pq


def _min_max_date(tbl) -> Tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if "date" not in {n for n in tbl.schema.names} or tbl.num_rows == 0:
        return None, None
    s = tbl.column("date").to_pandas()
    dates = pd.to_datetime(s, errors="coerce", utc=True).dropna()
    if getattr(dates.dtype, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    if dates.empty:
        return None, None
    return dates.min(), dates.max()


def _common_files(dir_a: Path, dir_b: Path, include_all: bool) -> List[str]:
    files_a = {p.name for p in dir_a.glob("*.parquet")}
    files_b = {p.name for p in dir_b.glob("*.parquet")}
    common = sorted(files_a & files_b)
    if not include_all:
        common = [n for n in common if n != "all_investors.parquet"]
    return common


def _canonical_columns() -> List[str]:
    try:
        from src.cli.convert_stata_panel import OUTPUT_COLUMNS  # type: ignore
        return list(OUTPUT_COLUMNS)
    except Exception:
        # Fallback to known order if import fails
        return [
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


def _build_canonical_table(pa, pc, tbl, canon_cols: List[str]):
    # Map alternative column names to canonical ones
    alt_to_canon: Dict[str, str] = {
        "factor1": "me",
        "factor2": "be",
        "factor3": "profit",
        "factor4": "Gat",
        "factor5": "beta",
    }

    # Target Arrow types per canonical column
    t_str = pa.string()
    t_i64 = pa.int64()
    t_f64 = pa.float64()
    t_ts = pa.timestamp("ns")
    type_map: Dict[str, object] = {
        "type": t_str,
        "mgrno": t_i64,
        "permno": t_i64,
        "date": t_ts,
        "holding_t": t_f64,
        "holding_t1": t_f64,
        "me": t_f64,
        "be": t_f64,
        "profit": t_f64,
        "Gat": t_f64,
        "beta": t_f64,
        "aum": t_f64,
        "outaum": t_f64,
        "prc": t_f64,
    }

    names = set(tbl.schema.names)

    def get_array(name: str) -> Optional[object]:
        if name in names:
            return tbl[name]
        # try alternative names
        for alt, canon in alt_to_canon.items():
            if canon == name and alt in names:
                return tbl[alt]
        return None

    def cast_array(arr, target_type):
        try:
            return pc.cast(arr, target_type)
        except Exception:
            # Fallback: via pandas conversion
            s = arr.to_pandas()
            if pa.types.is_timestamp(target_type):
                s = pd.to_datetime(s, errors="coerce")
            return pa.array(s, type=target_type)

    arrays: Dict[str, object] = {}
    for col in canon_cols:
        arr = get_array(col)
        target_type = type_map[col]
        if arr is None:
            arrays[col] = pa.nulls(tbl.num_rows, type=target_type)
        else:
            # If the field is a ChunkedArray or Array
            arrays[col] = cast_array(arr, target_type)

    # Build table with canonical order and types
    fields = [pa.field(c, type_map[c]) for c in canon_cols]
    schema = pa.schema(fields)
    cols = [arrays[c] for c in canon_cols]
    return pa.Table.from_arrays(cols, schema=schema)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge per-type parquet files from two folders (non-overlapping time ranges)")
    ap.add_argument("--dir-a", type=Path, required=True, help="First folder (earlier range)")
    ap.add_argument("--dir-b", type=Path, required=True, help="Second folder (later range)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output folder for merged per-type files")
    ap.add_argument("--include-all-investors", action="store_true", help="Include all_investors.parquet in merge")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    ap.add_argument("--validate", action="store_true", help="Validate that time ranges do not overlap before merging")
    ap.add_argument("--compression", type=str, default="snappy", help="Parquet compression codec (default: snappy)")
    args = ap.parse_args()

    pa, pc, pq = _maybe_import_pyarrow()

    dir_a: Path = args.dir_a
    dir_b: Path = args.dir_b
    out_dir: Path = args.out_dir

    if not dir_a.exists() or not dir_b.exists():
        raise SystemExit("Both --dir-a and --dir-b must exist")
    out_dir.mkdir(parents=True, exist_ok=True)

    common = _common_files(dir_a, dir_b, include_all=args.include_all_investors)
    if not common:
        raise SystemExit("No common *.parquet files found between the two folders")

    print(f"Merging per-type parquet files:\n  A: {dir_a.resolve()}\n  B: {dir_b.resolve()}\n  -> {out_dir.resolve()}\n")
    merged_rows_total = 0

    canon_cols = _canonical_columns()

    for name in common:
        path_a = dir_a / name
        path_b = dir_b / name
        out_path = out_dir / name

        tbl_a_raw = pq.read_table(path_a)
        tbl_b_raw = pq.read_table(path_b)

        # Harmonize both sides to canonical schema
        tbl_a = _build_canonical_table(pa, pc, tbl_a_raw, canon_cols)
        tbl_b = _build_canonical_table(pa, pc, tbl_b_raw, canon_cols)

        # Optionally validate time ranges do not overlap
        if args.validate:
            a_min, a_max = _min_max_date(tbl_a)
            b_min, b_max = _min_max_date(tbl_b)
            if a_min is not None and a_max is not None and b_min is not None and b_max is not None:
                # Overlap if ranges intersect
                latest_start = max(a_min, b_min)
                earliest_end = min(a_max, b_max)
                if latest_start <= earliest_end:
                    raise SystemExit(
                        f"Overlap detected for {name}: A[{a_min}..{a_max}] vs B[{b_min}..{b_max}]"
                    )

        # At this point schemas should match (canonical)
        if tbl_a.schema != tbl_b.schema:
            # As a safety net, attempt a final cast
            try:
                tbl_b = tbl_b.cast(tbl_a.schema)
            except Exception as e:
                raise SystemExit(f"Schema mismatch for {name} even after harmonization: {e}")

        if out_path.exists():
            if not args.overwrite:
                raise SystemExit(f"Output exists: {out_path}. Use --overwrite to replace.")
            out_path.unlink()

        writer = pq.ParquetWriter(out_path, tbl_a.schema, compression=args.compression)
        try:
            writer.write_table(tbl_a)
            writer.write_table(tbl_b)
        finally:
            writer.close()

        n_rows = tbl_a.num_rows + tbl_b.num_rows
        merged_rows_total += n_rows
        print(f"  - {name}: {tbl_a.num_rows} + {tbl_b.num_rows} -> {n_rows}")

    print(f"\nDone. Total merged rows: {merged_rows_total}")


if __name__ == "__main__":
    main()
