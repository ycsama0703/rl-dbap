from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def _maybe_import_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.compute as pc  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard only
        raise SystemExit(
            "pyarrow is required. Install it via `pip install pyarrow`."
        ) from exc
    return pa, pc, pq


def _common_files(src_dir: Path, include_all_investors: bool) -> List[Path]:
    files = sorted(p for p in src_dir.glob("*.parquet"))
    if not include_all_investors:
        files = [p for p in files if p.name != "all_investors.parquet"]
    return files


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    # Accept YYYY, YYYY-MM, YYYY-MM-DD
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise SystemExit(f"Invalid date format: {s}. Use YYYY or YYYY-MM or YYYY-MM-DD.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter per-type parquet files in a folder by date range (inclusive)")
    ap.add_argument("--src-dir", type=Path, required=True, help="Input folder containing per-type parquet files")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output folder for filtered files")
    ap.add_argument("--start", type=str, required=False, help="Start date (inclusive): YYYY[-MM[-DD]]")
    ap.add_argument("--end", type=str, required=False, help="End date (inclusive): YYYY[-MM[-DD]]")
    ap.add_argument("--include-all-investors", action="store_true", help="Include all_investors.parquet")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    ap.add_argument("--compression", type=str, default="snappy", help="Parquet compression codec (default: snappy)")
    args = ap.parse_args()

    pa, pc, pq = _maybe_import_pyarrow()

    src_dir: Path = args.src_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        raise SystemExit(f"--src-dir not found: {src_dir}")

    start_dt = _parse_date(args.start)
    end_dt = _parse_date(args.end)
    if start_dt is None and end_dt is None:
        raise SystemExit("Provide at least one of --start or --end")

    files = _common_files(src_dir, include_all_investors=args.include_all_investors)
    if not files:
        raise SystemExit("No *.parquet files found in --src-dir")

    print(
        f"Filtering per-type files in:\n  {src_dir.resolve()}\n  -> {out_dir.resolve()}\n  Range: [{start_dt or '-inf'} .. {end_dt or '+inf'}] (inclusive)\n"
    )

    total_out_rows = 0

    for path in files:
        tbl = pq.read_table(path)
        if "date" not in {n for n in tbl.schema.names}:
            # Pass-through if no date column
            filtered = tbl
        else:
            col = tbl.column("date")
            cond = None
            # Build scalars matching column type (e.g., timestamp[ns])
            if start_dt is not None:
                start_scalar = pa.scalar(start_dt, type=col.type)
                ge = pc.greater_equal(col, start_scalar)
                cond = ge if cond is None else pc.and_(cond, ge)
            if end_dt is not None:
                end_scalar = pa.scalar(end_dt, type=col.type)
                le = pc.less_equal(col, end_scalar)
                cond = le if cond is None else pc.and_(cond, le)
            filtered = tbl if cond is None else tbl.filter(cond)

        out_path = out_dir / path.name
        if out_path.exists():
            if not args.overwrite:
                raise SystemExit(f"Output exists: {out_path}. Use --overwrite to replace.")
            out_path.unlink()

        writer = pq.ParquetWriter(out_path, filtered.schema, compression=args.compression)
        try:
            writer.write_table(filtered)
        finally:
            writer.close()

        print(f"  - {path.name}: {tbl.num_rows} -> {filtered.num_rows}")
        total_out_rows += filtered.num_rows

    print(f"\nDone. Total rows written: {total_out_rows}")


if __name__ == "__main__":
    main()

