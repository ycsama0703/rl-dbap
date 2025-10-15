from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Set

import pandas as pd

from src.cli.map_ticker_names import load_mapping


def collect_permnos_from_parquet(dir_or_file: Path) -> pd.Series:
    paths = [dir_or_file] if dir_or_file.is_file() else sorted(dir_or_file.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no parquet files found under: {dir_or_file}")
    vals = []
    for fp in paths:
        try:
            try:
                df = pd.read_parquet(fp, columns=["permno"], engine="pyarrow")
            except Exception:
                df = pd.read_parquet(fp, columns=["permno"])  # fallback
            if "permno" not in df.columns:
                continue
            vals.append(df["permno"])  # type: ignore
        except Exception:
            continue
    if not vals:
        return pd.Series(dtype="Int64")
    s = pd.concat(vals, ignore_index=True)
    s = pd.to_numeric(s, errors="coerce").astype("Int64")
    s = s.dropna()
    return s


def collect_permnos_from_prompts(dir_or_file: Path) -> pd.Series:
    files = [dir_or_file] if dir_or_file.is_file() else sorted(dir_or_file.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no jsonl files found under: {dir_or_file}")
    permnos = []
    pat = re.compile(r"^Ticker:\s*(\d+)", flags=re.MULTILINE)
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    v = rec.get("permno")
                    if v is not None:
                        try:
                            permnos.append(int(v))
                            continue
                        except Exception:
                            pass
                    prompt = rec.get("prompt") or rec.get("query") or ""
                    m = pat.search(prompt)
                    if m:
                        try:
                            permnos.append(int(m.group(1)))
                        except Exception:
                            pass
        except Exception:
            continue
    if not permnos:
        return pd.Series(dtype="Int64")
    s = pd.Series(permnos, dtype="Int64")
    s = s.dropna()
    return s


def main():
    ap = argparse.ArgumentParser(description="Check coverage of PERMNO->company mapping against data")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--parquet-dir", type=str, help="Directory or file with parquet containing 'permno'")
    src.add_argument("--prompts-dir", type=str, help="Directory or file with prompts jsonl")
    ap.add_argument("--mapping", type=str, default="data/ticker_mapping.csv",
                    help="CSV path with columns PERMNO, COMNAM, TICKER")
    ap.add_argument("--export-missing", type=str, default=None,
                    help="Optional CSV to save missing PERMNOs with counts")
    args = ap.parse_args()

    mapping = load_mapping(Path(args.mapping))
    mapped_permnos: Set[int] = set(mapping.keys())

    if args.parquet_dir:
        src_series = collect_permnos_from_parquet(Path(args.parquet_dir))
    else:
        src_series = collect_permnos_from_prompts(Path(args.prompts_dir))

    if src_series.empty:
        print("no permno found in source")
        return

    total_unique = int(src_series.dropna().astype(int).nunique())
    src_unique = set(src_series.dropna().astype(int).unique().tolist())
    missing = sorted(src_unique - mapped_permnos)
    covered = total_unique - len(missing)

    print(f"Total unique PERMNO in source: {total_unique}")
    print(f"Mapped (found in mapping):     {covered}")
    print(f"Unmapped (missing in mapping): {len(missing)}")
    cov_pct = 0.0 if total_unique == 0 else 100.0 * covered / total_unique
    print(f"Coverage: {cov_pct:.2f}%")

    if len(missing) > 0:
        vc = src_series.dropna().astype(int).value_counts()
        df_miss = pd.DataFrame({
            "permno": missing,
            "count": [int(vc.get(p, 0)) for p in missing],
        }).sort_values(["count", "permno"], ascending=[False, True])
        print("Top 20 missing PERMNO by frequency:")
        print(df_miss.head(20).to_string(index=False))
        if args.export_missing:
            outp = Path(args.export_missing)
            outp.parent.mkdir(parents=True, exist_ok=True)
            df_miss.to_csv(outp, index=False)
            print(f"saved missing list -> {outp}")


if __name__ == "__main__":
    main()

