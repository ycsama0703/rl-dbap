from __future__ import annotations
import argparse, json, time
from pathlib import Path
import pandas as pd
import numpy as np

from src.prompts.sampler import build_continuous_windows, stratified_sample_windows
from src.prompts.builder import PromptRow, build_history_prompt

try:
    # Reuse mapping utilities if available
    from src.cli.map_ticker_names import load_mapping, replace_ticker_line  # type: ignore
except Exception:
    load_mapping = None  # type: ignore
    replace_ticker_line = None  # type: ignore


def _row_from_dfrow(r: pd.Series) -> PromptRow:
    return PromptRow(
        mgrno=r.get("mgrno"),
        permno=r.get("permno"),
        investor_type=(r.get("type") or "Unknown"),
        holding_t=r.get("holding_t"),
        holding_t1=r.get("holding_t1"),
        me=r.get("me") or r.get("factor1"),
        be=r.get("be") or r.get("factor2"),
        profit=r.get("profit") or r.get("factor3"),
        Gat=r.get("Gat") or r.get("factor4"),
        beta=r.get("beta") or r.get("factor5"),
        aum=r.get("aum"),
        outaum=r.get("outaum"),
        prc=r.get("prc"),
        date=str(r.get("date")) if "date" in r.index else None,
    )


def _normalize_prompt_text(s: str) -> str:
    """Fix any legacy encoding artifacts to match prior prompt style exactly.
    - Replace broken '≥' and arrows in instruction examples.
    """
    try:
        s = s.replace("holding_delta �?-holding_t", "holding_delta ≥ -holding_t")
        s = s.replace("me�?(", "me↓ (")
        s = s.replace("profit�?(", "profit↑ (")
        s = s.replace(" �?moderate", " → moderate")
    except Exception:
        pass
    return s


REQUIRED_COLS = [
    "type", "mgrno", "permno", "date", "holding_t", "holding_t1",
    "me", "be", "profit", "Gat", "beta", "aum", "outaum", "prc"
]


def _read_min_columns(fp: Path) -> pd.DataFrame:
    # Try to read a subset of columns to save memory; fallback to full read if not supported
    try:
        # prefer pyarrow if available for speed; fall back silently
        try:
            return pd.read_parquet(fp, columns=[c for c in REQUIRED_COLS if True], engine="pyarrow")
        except Exception:
            return pd.read_parquet(fp, columns=[c for c in REQUIRED_COLS if True])
    except Exception:
        return pd.read_parquet(fp)


def build_for_file(
    in_file: Path,
    out_file: Path,
    per_type_limit: int,
    time_bins: int,
    cap_per_pair: int,
    seed: int,
    *,
    date_start: str | None = None,
    date_end: str | None = None,
    head: int | None = None,
    progress_every: int = 1000,
    use_tqdm: bool = False,
    mapping: dict | None = None,
    exclude_zero_holding_t: bool = False,
    include_permnos: set[int] | None = None,
    take_all: bool = False,
    limit_override: int | None = None,
):
    t0 = time.perf_counter()
    print(f"[history-prompts] reading: {in_file}", flush=True)
    df = _read_min_columns(in_file)
    # Optional date filtering to shrink data for speed
    if date_start or date_end:
        try:
            if "date" in df.columns:
                ds = pd.to_datetime(date_start) if date_start else None
                de = pd.to_datetime(date_end) if date_end else None
                if ds is not None:
                    df = df[df["date"] >= ds]
                if de is not None:
                    df = df[df["date"] <= de]
        except Exception:
            pass
    if "type" not in df.columns:
        df = df.copy(); df["type"] = in_file.stem
    if include_permnos:
        try:
            df = df[df["permno"].isin(include_permnos)]
        except Exception:
            pass
    if head is not None and head > 0:
        # Keep head after sorting for deterministic subset
        df = df.sort_values(["type", "mgrno", "permno", "date"]).head(head)
    # create windows and sample
    print(f"[history-prompts] building windows...", flush=True)
    windows = build_continuous_windows(df, use_tqdm=use_tqdm, progress_every=progress_every, label=in_file.stem)
    print(
        f"[history-prompts] {in_file.stem}: rows={len(df):,} "
        f"groups={df.groupby(['type','mgrno','permno'], sort=False, observed=False).ngroups} "
        f"windows={len(windows):,}",
        flush=True,
    )
    def _nonzero_window(ws):
        if not exclude_zero_holding_t:
            return ws
        kept = []
        for w in ws:
            try:
                v = df.loc[w.idx_t]["holding_t"]
                if pd.isna(v):
                    continue
                if float(v) == 0.0:
                    continue
                kept.append(w)
            except Exception:
                continue
        return kept

    if take_all:
        picked = _nonzero_window(windows)
        total_before = len(picked)
        if limit_override is not None:
            lim = max(0, int(limit_override))
            picked = picked[:lim]
        print(
            f"[history-prompts] {in_file.stem}: selected {len(picked):,} / {total_before:,} windows (no sampling)",
            flush=True,
        )
    else:
        picked = stratified_sample_windows(
            df,
            windows,
            per_type_limit=per_type_limit,
            time_bins=time_bins,
            cap_per_pair=cap_per_pair,
            seed=seed,
            exclude_zero_holding_t=exclude_zero_holding_t,
        )
        if limit_override is not None and limit_override >= 0:
            lim = max(0, int(limit_override))
            if lim < len(picked):
                picked = picked[:lim]
        print(f"[history-prompts] {in_file.stem}: sampled={len(picked):,} (limit={per_type_limit})", flush=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    total = len(picked)
    pbar = None
    _use_tqdm = False
    if use_tqdm:
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=total, desc=f"write {in_file.stem}")
            _use_tqdm = True
        except Exception:
            _use_tqdm = False
    with out_file.open("w", encoding="utf-8") as f:
        for i, w in enumerate(picked, 1):
            rows = [df.loc[idx] for idx in (w.idx_tm3, w.idx_tm2, w.idx_tm1, w.idx_t)]
            hist_rows = [_row_from_dfrow(r) for r in rows]
            prompt, extras = build_history_prompt(hist_rows, hide_date=True, target="delta", strict_contract=True)
            prompt = _normalize_prompt_text(prompt)
            # Automatically map Ticker line to include company name if mapping provided
            if mapping is not None and replace_ticker_line is not None:
                permno_val = extras.get("permno")
                try:
                    permno_int = int(permno_val) if permno_val is not None else None
                except Exception:
                    permno_int = None
                name = ticker = None
                if permno_int is not None and permno_int in mapping:
                    # mapping value shape (name, ticker)
                    val = mapping[permno_int]
                    if isinstance(val, (list, tuple)) and len(val) >= 2:
                        name, ticker = val[0], val[1]
                prompt = replace_ticker_line(prompt, permno_int, name, ticker, mode="append")
                if ticker:
                    extras["ticker"] = ticker
                if name:
                    extras["company"] = name
            rec = {"prompt": prompt, **extras}
            # Ensure numpy scalars are converted to Python types for JSON serialization
            def _json_default(o):
                if isinstance(o, (np.integer,)):
                    return int(o)
                if isinstance(o, (np.floating,)):
                    return float(o)
                if isinstance(o, (np.bool_,)):
                    return bool(o)
                return str(o)

            f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
            n += 1
            if _use_tqdm and pbar is not None:
                pbar.update(1)
            elif progress_every and (i % progress_every == 0 or i == total):
                print(f"[history-prompts] {in_file.stem}: wrote {i}/{total}", flush=True)
    if pbar is not None:
        try:
            pbar.close()
        except Exception:
            pass
    t1 = time.perf_counter()
    print(f"[history-prompts] {in_file.stem}: done in {t1 - t0:.1f}s -> {out_file}", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser(description="Build history-style prompts (t-3..t) with per-type sampling")
    ap.add_argument("--in-dir", type=str, default="data/processed/panel_quarter.parquet",
                    help="Input directory with per-investor-type parquet files")
    ap.add_argument("--out-dir", type=str, default="artifacts/prompts_hist",
                    help="Output directory for per-type JSONL files")
    ap.add_argument("--include-types", type=str, default="",
                    help="Comma-separated file stems to include (e.g., 'banks,mutual_funds'). Empty = all.")
    ap.add_argument("--exclude-types", type=str, default="",
                    help="Comma-separated file stems to exclude.")
    ap.add_argument("--per-type-limit", type=int, default=1000)
    ap.add_argument("--time-bins", type=int, default=10)
    ap.add_argument("--cap-per-pair", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    # Speed/visibility knobs
    ap.add_argument("--max-files", type=int, default=None, help="Process at most N parquet files")
    ap.add_argument("--date-start", type=str, default=None, help="Filter rows with date >= this (e.g., 2015-01-01)")
    ap.add_argument("--date-end", type=str, default=None, help="Filter rows with date <= this (e.g., 2020-12-31)")
    ap.add_argument("--head", type=int, default=None, help="Use only first N rows after sorting (debug speed)")
    ap.add_argument("--progress-every", type=int, default=2000, help="Print progress every N writes if no tqdm")
    ap.add_argument("--use-tqdm", action="store_true", help="Use tqdm progress bars if installed")
    ap.add_argument("--exclude-zero-holding-t", action="store_true", help="Exclude windows where t-time holding_t == 0")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    inc = {s.strip() for s in args.include_types.split(',') if s.strip()}
    exc = {s.strip() for s in args.exclude_types.split(',') if s.strip()}
    total = 0
    # Load ticker mapping automatically if available
    mp = None
    try:
        if load_mapping is not None:
            default_map = Path("data/ticker_mapping.csv")
            if default_map.exists():
                mp = load_mapping(default_map)
                print(f"[history-prompts] loaded mapping: {len(mp)} PERMNO -> (name,ticker)")
            else:
                print("[history-prompts] mapping file not found: data/ticker_mapping.csv (skipping name mapping)")
        else:
            print("[history-prompts] mapping utilities unavailable (skipping name mapping)")
    except Exception as e:
        print(f"[history-prompts] failed to load mapping: {e}")

    for i, fp in enumerate(sorted(in_dir.glob("*.parquet")), 1):
        stem = fp.stem
        if inc and stem not in inc:
            continue
        if exc and stem in exc:
            continue
        outp = out_dir / f"{fp.stem}.jsonl"
        n = build_for_file(
            fp,
            outp,
            args.per_type_limit,
            args.time_bins,
            args.cap_per_pair,
            args.seed,
            date_start=args.date_start,
            date_end=args.date_end,
            head=args.head,
            progress_every=args.progress_every,
            use_tqdm=args.use_tqdm,
            mapping=mp,
            exclude_zero_holding_t=args.exclude_zero_holding_t,
        )
        print(f"[history-prompts] {fp.stem}: wrote {n} -> {outp}")
        total += n
        if args.max_files is not None and i >= args.max_files:
            print(f"[history-prompts] reached max-files={args.max_files}, stopping.")
            break
    print(f"[history-prompts] total: {total}")


if __name__ == "__main__":
    main()
