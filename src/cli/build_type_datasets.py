from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Optional
import pandas as pd

from src.cli.build_history_prompts import build_for_file

try:
    from src.cli.map_ticker_names import load_mapping  # type: ignore
except Exception:
    load_mapping = None  # type: ignore


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _convert_prompts_to_sft(inp: Path, outp: Path, *, system: str, with_think: bool = False, limit: Optional[int] = None) -> int:
    from src.cli.prompts_to_sft import _resolve_label  # reuse label logic
    _ensure_dir(outp)
    n = 0
    with inp.open("r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as fout:
        for line in f:
            rec = json.loads(line)
            prompt = rec.get("prompt") or rec.get("query")
            if not prompt:
                continue
            label_tp1 = _resolve_label(rec)
            if label_tp1 is None:
                continue
            resp = json.dumps({"holding_tp1": float(label_tp1)}, ensure_ascii=False)
            msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            if with_think:
                msgs.append({"role": "assistant", "content": "<think>...</think>", "loss": False})
            msgs.append({"role": "assistant", "content": resp, "loss": True})
            out = {"messages": msgs}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit:
                break
    return n


def _convert_prompts_to_grpo(inp: Path, outp: Path, *, system: str, no_think_example: bool = False, limit: Optional[int] = None) -> int:
    _ensure_dir(outp)
    n = 0
    with inp.open("r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as fout:
        for line in f:
            rec = json.loads(line)
            prompt = rec.get("prompt") or rec.get("query")
            if not prompt:
                continue
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            if not no_think_example:
                messages.append({
                    "role": "assistant",
                    "content": "<think>\n• Compare shifts in fundamentals (me, be, profit, Gat, beta).\n• Tie direction+magnitude to holding_t and recent deltas.\n• Check bounds: holding_tp1 >= 0, holding_delta >= -holding_t.\n</think>",
                    "loss": False,
                })
            out = {
                "messages": messages,
                "label_delta": rec.get("label_delta"),
                "label_tp1": rec.get("label_tp1") or rec.get("label"),
                "holding_t": rec.get("holding_t"),
                "mgrno": rec.get("mgrno"),
                "permno": rec.get("permno"),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit:
                break
    return n


def _count_zeros_in_prompts(fp: Path) -> tuple[int, int]:
    total = 0
    zeros = 0
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                ht = rec.get("holding_t")
                total += 1
                if ht is not None and float(ht) == 0.0:
                    zeros += 1
            except Exception:
                continue
    return total, zeros


def main():
    ap = argparse.ArgumentParser(description="Build SFT/GRPO/Test datasets for a single firm type in one command")
    ap.add_argument("--type", required=True, help="Firm type stem, e.g., 'banks'")
    ap.add_argument("--in-dir", type=str, default="data/processed/panel_quarter.parquet")
    ap.add_argument("--per-type-limit", type=int, default=1000)
    # date splits
    ap.add_argument("--sft-end", type=str, default="2016-12-31")
    ap.add_argument("--grpo-start", type=str, default="2017-01-01")
    ap.add_argument("--grpo-end", type=str, default="2018-12-31")
    ap.add_argument("--test-start", type=str, default="2019-01-01")
    # filter control
    ap.add_argument("--exclude-zero-holding-t", dest="exclude_zero", action="store_true")
    ap.add_argument("--include-zero-holding-t", dest="exclude_zero", action="store_false")
    ap.set_defaults(exclude_zero=True)
    # converter knobs
    ap.add_argument("--sft-with-think", action="store_true")
    ap.add_argument("--grpo-no-think-example", action="store_true")
    args = ap.parse_args()

    t = args.type
    in_dir = Path(args.in_dir)
    in_file = in_dir / f"{t}.parquet"
    if not in_file.exists():
        raise FileNotFoundError(f"input parquet not found: {in_file}")

    # load optional mapping
    mapping = None
    try:
        if load_mapping is not None:
            mp_path = Path("data/ticker_mapping.csv")
            if mp_path.exists():
                mapping = load_mapping(mp_path)
                print(f"[type-pipeline] loaded mapping: {len(mapping)}")
    except Exception as e:
        print(f"[type-pipeline] mapping load failed: {e}")

    # outputs
    ph_sft = Path("artifacts/prompts_hist_sft") / f"{t}.jsonl"
    ph_grpo = Path("artifacts/prompts_hist_grpo") / f"{t}.jsonl"
    ph_test = Path("artifacts/prompts_hist_test") / f"{t}.jsonl"
    sft_out = Path("artifacts/sft") / f"sft_train_{t}.jsonl"
    # Put test chat JSONL into a separate folder to avoid mixing with SFT train data
    sft_test_out = Path("artifacts/test") / f"test_{t}.jsonl"
    grpo_out = Path("artifacts/grpo") / f"grpo_{t}.jsonl"

    # 1) Build prompts for three splits
    print(f"[type-pipeline] building SFT prompts for {t} (<= {args.sft_end})")
    build_for_file(
        in_file=in_file,
        out_file=ph_sft,
        per_type_limit=args.per_type_limit,
        time_bins=10,
        cap_per_pair=3,
        seed=42,
        date_start=None,
        date_end=args.sft_end,
        head=None,
        progress_every=50000,
        use_tqdm=False,
        mapping=mapping,
        exclude_zero_holding_t=args.exclude_zero,
    )

    print(f"[type-pipeline] building GRPO prompts for {t} ({args.grpo_start}..{args.grpo_end})")
    build_for_file(
        in_file=in_file,
        out_file=ph_grpo,
        per_type_limit=args.per_type_limit,
        time_bins=10,
        cap_per_pair=3,
        seed=42,
        date_start=args.grpo_start,
        date_end=args.grpo_end,
        head=None,
        progress_every=50000,
        use_tqdm=False,
        mapping=mapping,
        exclude_zero_holding_t=args.exclude_zero,
    )

    print(f"[type-pipeline] building TEST prompts for {t} (>= {args.test_start})")
    build_for_file(
        in_file=in_file,
        out_file=ph_test,
        per_type_limit=args.per_type_limit,
        time_bins=10,
        cap_per_pair=3,
        seed=42,
        date_start=args.test_start,
        date_end=None,
        head=None,
        progress_every=50000,
        use_tqdm=False,
        mapping=mapping,
        exclude_zero_holding_t=args.exclude_zero,
    )

    # 2) Convert
    print(f"[type-pipeline] convert SFT -> chat ({sft_out})")
    _convert_prompts_to_sft(ph_sft, sft_out, system="You are a quantitative portfolio manager. Predict next-quarter absolute holding as valid JSON.", with_think=args.sft_with_think)

    print(f"[type-pipeline] convert TEST -> chat ({sft_test_out})")
    _convert_prompts_to_sft(ph_test, sft_test_out, system="You are a quantitative portfolio manager. Predict next-quarter absolute holding as valid JSON.")

    print(f"[type-pipeline] convert GRPO -> dataset ({grpo_out})")
    _convert_prompts_to_grpo(ph_grpo, grpo_out, system="You are a quantitative portfolio manager. Respond with valid JSON only.", no_think_example=args.grpo_no_think_example)

    # 3) Report stats
    for fp, name in [(ph_sft, "prompts_sft"), (ph_grpo, "prompts_grpo"), (ph_test, "prompts_test")]:
        if fp.exists():
            total, zeros = _count_zeros_in_prompts(fp)
            print(f"[type-pipeline] {name}: total={total} t==0={zeros} ratio={(zeros/total if total else 0):.3f}")

    print("[type-pipeline] done.")


if __name__ == "__main__":
    main()
