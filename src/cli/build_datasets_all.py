from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

from src.cli.build_history_prompts import build_for_file
from src.cli.build_type_datasets import _convert_prompts_to_sft, _convert_prompts_to_grpo

try:
    from src.cli.map_ticker_names import load_mapping  # type: ignore
except Exception:
    load_mapping = None  # type: ignore


DEFAULT_TYPES = [
    "banks",
    "households",
    "insurance_companies",
    "investment_advisors",
    "mutual_funds",
    "other",
    "pension_funds",
]


def discover_types(in_dir: Path) -> List[str]:
    names = []
    for p in sorted(in_dir.glob("*.parquet")):
        if p.name == "all_investors.parquet":
            continue
        names.append(p.stem)
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description="One-click build SFT/GRPO/TEST datasets for multiple firm types")
    ap.add_argument("--in-dir", type=Path, default=Path("data/processed/panel_quarter.parquet"))
    ap.add_argument("--types", type=str, default="",
                    help="Comma-separated type names; empty = auto-discover from --in-dir")
    ap.add_argument("--per-type-limit", type=int, default=2000)
    # Date windows per your latest spec: TEST=last 2y, GRPO grows by 1y
    ap.add_argument("--sft-end", type=str, default="2018-12-31")
    ap.add_argument("--grpo-start", type=str, default="2019-01-01")
    ap.add_argument("--grpo-end", type=str, default="2022-12-31")
    ap.add_argument("--test-start", type=str, default="2023-01-01")
    ap.add_argument("--exclude-zero-holding-t", action="store_true", default=True)
    ap.add_argument("--include-zero-holding-t", dest="exclude_zero_holding_t", action="store_false")
    ap.add_argument("--time-bins", type=int, default=10)
    ap.add_argument("--cap-per-pair", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--head", type=int, default=None, help="Use only first N rows after sorting (debug)")
    ap.add_argument("--progress-every", type=int, default=50000)
    ap.add_argument("--use-tqdm", action="store_true")
    ap.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    args = ap.parse_args()

    in_dir: Path = args.in_dir
    if not in_dir.exists():
        raise FileNotFoundError(f"input directory not found: {in_dir}")

    # Resolve list of types
    types: List[str]
    if args.types.strip():
        types = [s.strip() for s in args.types.split(",") if s.strip()]
    else:
        types = discover_types(in_dir)
        if not types:
            # fallback to default order if directory is empty
            types = list(DEFAULT_TYPES)

    # Prepare outputs
    root = args.artifacts_root
    out_prompts_sft = root / "prompts_hist_sft"
    out_prompts_grpo = root / "prompts_hist_grpo"
    out_prompts_test = root / "prompts_hist_test"
    out_prompts_sft.mkdir(parents=True, exist_ok=True)
    out_prompts_grpo.mkdir(parents=True, exist_ok=True)
    out_prompts_test.mkdir(parents=True, exist_ok=True)
    out_sft = root / "sft"; out_sft.mkdir(parents=True, exist_ok=True)
    out_grpo = root / "grpo"; out_grpo.mkdir(parents=True, exist_ok=True)
    out_test = root / "test"; out_test.mkdir(parents=True, exist_ok=True)

    # Optional mapping
    mapping = None
    try:
        if load_mapping is not None:
            mp_path = Path("data/ticker_mapping.csv")
            if mp_path.exists():
                mapping = load_mapping(mp_path)
                print(f"[build-all] loaded mapping: {len(mapping)} entries")
    except Exception as e:
        print(f"[build-all] mapping load failed: {e}")

    total_prompts = {"sft": 0, "grpo": 0, "test": 0}
    total_chat = {"sft": 0, "grpo": 0, "test": 0}

    # Build prompts and convert for each type
    for t in types:
        in_file = in_dir / f"{t}.parquet"
        if not in_file.exists():
            print(f"[build-all] skip type={t}: {in_file} not found")
            continue

        # 1) prompts for SFT, GRPO, TEST
        ph_sft = out_prompts_sft / f"{t}.jsonl"
        ph_grpo = out_prompts_grpo / f"{t}.jsonl"
        ph_test = out_prompts_test / f"{t}.jsonl"

        print(f"[build-all] {t}: build SFT prompts (..{args.sft_end})")
        n_sft = build_for_file(
            in_file,
            ph_sft,
            args.per_type_limit,
            args.time_bins,
            args.cap_per_pair,
            args.seed,
            date_start=None,
            date_end=args.sft_end,
            head=args.head,
            progress_every=args.progress_every,
            use_tqdm=args.use_tqdm,
            mapping=mapping,
            exclude_zero_holding_t=args.exclude_zero_holding_t,
        )
        total_prompts["sft"] += n_sft

        print(f"[build-all] {t}: build GRPO prompts ({args.grpo_start}..{args.grpo_end})")
        n_grpo = build_for_file(
            in_file,
            ph_grpo,
            args.per_type_limit,
            args.time_bins,
            args.cap_per_pair,
            args.seed,
            date_start=args.grpo_start,
            date_end=args.grpo_end,
            head=args.head,
            progress_every=args.progress_every,
            use_tqdm=args.use_tqdm,
            mapping=mapping,
            exclude_zero_holding_t=args.exclude_zero_holding_t,
        )
        total_prompts["grpo"] += n_grpo

        print(f"[build-all] {t}: build TEST prompts (.. from {args.test_start})")
        n_test = build_for_file(
            in_file,
            ph_test,
            args.per_type_limit,
            args.time_bins,
            args.cap_per_pair,
            args.seed,
            date_start=args.test_start,
            date_end=None,
            head=args.head,
            progress_every=args.progress_every,
            use_tqdm=args.use_tqdm,
            mapping=mapping,
            exclude_zero_holding_t=args.exclude_zero_holding_t,
        )
        total_prompts["test"] += n_test

        # 2) Convert prompts -> datasets
        sft_out = out_sft / f"sft_train_{t}.jsonl"
        print(f"[build-all] {t}: convert SFT prompts -> {sft_out}")
        total_chat["sft"] += _convert_prompts_to_sft(
            ph_sft,
            sft_out,
            system="You are a quantitative portfolio manager.",
            with_think=True,
            contract_mode="delta",
            decimals=2,
            think_template="",
        )

        grpo_out = out_grpo / f"grpo_{t}.jsonl"
        print(f"[build-all] {t}: convert GRPO prompts -> {grpo_out}")
        total_chat["grpo"] += _convert_prompts_to_grpo(
            ph_grpo,
            grpo_out,
            system="You are a quantitative portfolio manager. Respond with valid JSON only.",
            no_think_example=False,
        )

        test_out = out_test / f"test_{t}.jsonl"
        print(f"[build-all] {t}: convert TEST prompts -> {test_out}")
        total_chat["test"] += _convert_prompts_to_sft(
            ph_test,
            test_out,
            system="You are a quantitative portfolio manager.",
            with_think=False,
            contract_mode="absolute",
            decimals=2,
            think_template="",
        )

    # Summary
    print("\n[build-all] summary")
    print(f"  prompts: SFT={total_prompts['sft']}  GRPO={total_prompts['grpo']}  TEST={total_prompts['test']}")
    print(f"  chat:    SFT={total_chat['sft']}  GRPO={total_chat['grpo']}  TEST={total_chat['test']}")
    print(f"  out:     sft={out_sft}  grpo={out_grpo}  test={out_test}")


if __name__ == "__main__":
    main()
