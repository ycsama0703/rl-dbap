from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


@dataclass
class Range:
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    count: int

    def overlaps(self, other: "Range") -> bool:
        if self.start is None or self.end is None:
            return False
        if other.start is None or other.end is None:
            return False
        latest_start = max(self.start, other.start)
        earliest_end = min(self.end, other.end)
        return latest_start <= earliest_end


def _read_date_range(pq_path: Path) -> Range:
    if not pq_path.exists():
        return Range(None, None, 0)
    # Try pyarrow first for environments without pandas parquet engine configured
    try:
        import pyarrow.parquet as pq  # type: ignore

        tbl = pq.read_table(pq_path, columns=["date"])
        if tbl.num_rows == 0 or "date" not in {n for n in tbl.schema.names}:
            return Range(None, None, tbl.num_rows)
        s = tbl.column("date").to_pandas()
        dates = pd.to_datetime(s, errors="coerce", utc=True).dropna()
        # Normalize to tz-naive for safe comparisons
        if getattr(dates.dtype, "tz", None) is not None:
            dates = dates.dt.tz_localize(None)
        if dates.empty:
            return Range(None, None, len(s))
        return Range(dates.min(), dates.max(), len(s))
    except ModuleNotFoundError:
        # Fall back to pandas read_parquet if pyarrow isn't present
        df = pd.read_parquet(pq_path, columns=["date"])  # may still require an engine
        if "date" not in df.columns or df.empty:
            return Range(None, None, len(df))
        dates = pd.to_datetime(df["date"], errors="coerce", utc=True).dropna()
        if getattr(dates.dtype, "tz", None) is not None:
            dates = dates.dt.tz_localize(None)
        if dates.empty:
            return Range(None, None, len(df))
        return Range(dates.min(), dates.max(), len(df))


def main() -> None:
    ap = argparse.ArgumentParser(description="Check per-type parquet time range overlap between two folders")
    ap.add_argument("--dir-a", type=Path, required=True, help="First folder containing per-type parquet files")
    ap.add_argument("--dir-b", type=Path, required=True, help="Second folder containing per-type parquet files")
    ap.add_argument("--include-all-investors", action="store_true", help="Include all_investors.parquet in checks")
    args = ap.parse_args()

    dir_a: Path = args.dir_a
    dir_b: Path = args.dir_b
    if not dir_a.exists() or not dir_b.exists():
        raise SystemExit("Both --dir-a and --dir-b must exist")

    # Discover common type files
    files_a = {p.name for p in dir_a.glob("*.parquet")}
    files_b = {p.name for p in dir_b.glob("*.parquet")}
    common = sorted(files_a & files_b)
    if not args.include_all_investors:
        common = [n for n in common if n != "all_investors.parquet"]

    if not common:
        raise SystemExit("No common *.parquet files found between the two folders")

    results: Dict[str, Dict[str, Range]] = {}
    any_overlap = False

    print(f"Comparing time ranges between:\n  A: {dir_a.resolve()}\n  B: {dir_b.resolve()}\n")
    print("type/file, A_start, A_end, A_count, B_start, B_end, B_count, overlap")

    for name in common:
        rng_a = _read_date_range(dir_a / name)
        rng_b = _read_date_range(dir_b / name)
        overlap = rng_a.overlaps(rng_b)
        any_overlap = any_overlap or overlap
        results[name] = {"A": rng_a, "B": rng_b}
        print(
            f"{name[:-8] if name.endswith('.parquet') else name}, "
            f"{rng_a.start}, {rng_a.end}, {rng_a.count}, "
            f"{rng_b.start}, {rng_b.end}, {rng_b.count}, "
            f"{overlap}"
        )

    print("\nSummary:")
    if any_overlap:
        print("- Overlap detected for at least one type. Review before merging.")
    else:
        print("- No overlaps detected across common types.")

    # Optional: show gaps
    for name, d in results.items():
        ra, rb = d["A"], d["B"]
        if ra.start is None or rb.start is None:
            continue
        # Determine ordering
        if ra.end is not None and rb.start is not None and ra.end < rb.start:
            gap_days = (rb.start - ra.end).days
            print(f"  * {name[:-8]}: A precedes B by {gap_days} days (A_end={ra.end}, B_start={rb.start})")
        elif rb.end is not None and ra.start is not None and rb.end < ra.start:
            gap_days = (ra.start - rb.end).days
            print(f"  * {name[:-8]}: B precedes A by {gap_days} days (B_end={rb.end}, A_start={ra.start})")


if __name__ == "__main__":
    main()
