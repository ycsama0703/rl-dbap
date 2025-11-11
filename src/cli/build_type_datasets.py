from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Optional
import pandas as pd
from openai import OpenAI
import os
import re

from src.cli.build_history_prompts import build_for_file
from src.cli.prompts_to_sft import _build_auto_think

try:
    from src.cli.map_ticker_names import load_mapping  # type: ignore
except Exception:
    load_mapping = None  # type: ignore


def _role_from_type(inv_type: str | None) -> str:
    if not inv_type:
        return ""
    role = inv_type.replace("_", " ").strip()
    if role.endswith("s") and len(role) > 1:
        role = role[:-1]
    return role


def _build_system_prompt(inv_type: str | None) -> str:
    role = _role_from_type(inv_type)
    role_segment = f"{role} " if role else ""
    return (
        "A conversation between User and Assistant. The User provides financial data, and the Assistant, "
        f"acting as a {role_segment}quantitative portfolio manager, reasons carefully and predicts portfolio adjustments."
    )


def _format_row_for_prompt(row: dict | None) -> str:
    if not isinstance(row, dict):
        return "me=NA, be=NA, profit=NA, Gat=NA, beta=NA, aum=NA, outAUM=NA, holding=NA, price=NA"

    def _get(key: str) -> str:
        val = row.get(key)
        return "NA" if val is None else str(val)

    ordered_keys = ["me", "be", "profit", "Gat", "beta", "aum", "outAUM", "holding", "price"]
    return ", ".join(f"{key}={_get(key)}" for key in ordered_keys)


def _build_structured_prompt(rec: dict) -> str:
    history = rec.get("history_rows")
    ticker = rec.get("ticker") or rec.get("permno") or "{ticker}"
    company = rec.get("company") or "the company"
    ticker_str = str(ticker)
    company_str = str(company)

    if not isinstance(history, dict):
        return rec.get("prompt") or rec.get("query") or ""

    hist_parts = []
    for key_label in [("t-3", "{t-3}"), ("t-2", "{t-2}"), ("t-1", "{t-1}")]:
        tag, label = key_label
        row_line = _format_row_for_prompt(history.get(tag))
        hist_parts.append(f"{label}: {row_line}")
    hist_block = "\n".join(hist_parts)
    current_line = _format_row_for_prompt(history.get("t"))

    return (
        "Given recent historical fundamentals and the current data for "
        f"{ticker_str} ({company_str}), estimate the normalized log change in portfolio holding from time t to t+1.\n\n"
        "Past data:\n"
        f"{hist_block}\n\n"
        "Current (t):\n"
        f"{current_line}"
    )


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _estimate_total_records(fp: Path, limit: Optional[int]) -> Optional[int]:
    if limit is not None:
        return limit
    try:
        with fp.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def _create_progress_bar(label: str, total: Optional[int]):
    if not label:
        return None
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        return None
    return tqdm(total=total, desc=f"{label} (convert)")


def _convert_prompts_to_sft(
    inp: Path,
    outp: Path,
    *,
    system: str,
    with_think: bool = True,
    contract_mode: str = "delta",
    decimals: int = 2,
    think_template: str = "",
    limit: Optional[int] = None,
    label: str = "sft",
    progress_every: int = 100,
) -> int:
    from src.cli.prompts_to_sft import _resolve_absolute, _resolve_log_delta as _resolve_delta, _format_float


    if contract_mode not in {"absolute", "delta"}:
        raise ValueError(f"unknown contract_mode={contract_mode}")
    resolver = _resolve_delta if contract_mode == "delta" else _resolve_absolute
    answer_key = "holding_log_delta" if contract_mode == "delta" else "holding_tp1"


    _ensure_dir(outp)
    n = 0
    eff_progress = progress_every
    total_records = _estimate_total_records(inp, limit)
    pbar = _create_progress_bar(label, total_records)
    if pbar is not None:
        eff_progress = 0
    if limit and (not eff_progress or eff_progress > limit):
        eff_progress = max(1, limit // 5) if limit >= 5 else 1
    with inp.open("r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as fout:
        for line in f:
            rec = json.loads(line)
            prompt = _build_structured_prompt(rec)
            if not prompt:
                prompt = rec.get("prompt") or rec.get("query")
            if not prompt:
                continue
            label_val = resolver(rec)
            if label_val is None:
                continue

            value = _format_float(float(label_val), decimals)
            resp_json = json.dumps({answer_key: value}, ensure_ascii=False)
            msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            assistant_parts = []

            if with_think:
                think_text = rec.get("think")

                if not think_text:
                    try:
                        client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
                        clean_prompt = re.split(r"OUTPUT FORMAT", prompt, maxsplit=1)[0].strip()
                        ds_prompt = f"""
                            You are an experienced quantitative portfolio manager.

                            Below are historical fundamentals plus the true holding adjustment.
                            Briefly explain in English why this label is reasonable. Highlight only the insights you find most relevant and keep it under 4 sentences.

                            ---
                            {clean_prompt}

                            True label:
                            holding_log_delta = {label_val:.4f}
                            """
                        resp_ds = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[
                                {"role": "system", "content": "You are a financial reasoning assistant."},
                                {"role": "user", "content": ds_prompt},
                            ],
                            temperature=0.7,
                            max_tokens=200,
                        )
                        think_text = resp_ds.choices[0].message.content.strip()
                        if not think_text.startswith("<think>"):
                            think_text = f"<think>{think_text}</think>"
                    except Exception as e:
                        print(f"[WARN] DeepSeek generation failed: {e}")
                        think_text = _build_auto_think(prompt, decimals)

                if think_text:
                    think_lower = think_text.lower()
                    if not think_lower.startswith("<think>"):
                        think_text = f"<think>{think_text}</think>"
                    assistant_parts.append(think_text)

            # ---------------- 拼接最终输出 ----------------
            assistant_parts.append(f"<answer>\n{resp_json}\n</answer>")
            msgs.append({
                "role": "assistant",
                "content": "\n".join(assistant_parts),
                "loss": True,
            })
            out = {"messages": msgs}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
            if pbar is not None:
                pbar.update(1)
            if eff_progress and n % eff_progress == 0:
                limit_str = f"/{limit}" if limit else ""
                print(f"[type-pipeline] {label}: converted {n}{limit_str}")
            if limit and n >= limit:
                break
    if pbar is not None:
        pbar.close()
    print(f"[type-pipeline] {label}: done (total={n})")
    return n


def _convert_prompts_to_grpo(
    inp: Path,
    outp: Path,
    *,
    system: str,
    no_think_example: bool = False,
    limit: Optional[int] = None,
    label: str = "grpo",
    progress_every: int = 100,
) -> int:
    _ensure_dir(outp)
    n = 0
    eff_progress = progress_every
    total_records = _estimate_total_records(inp, limit)
    pbar = _create_progress_bar(label, total_records)
    if pbar is not None:
        eff_progress = 0
    if limit and (not eff_progress or eff_progress > limit):
        eff_progress = max(1, limit // 5) if limit >= 5 else 1
    with inp.open("r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as fout:
        for line in f:
            rec = json.loads(line)
            prompt = _build_structured_prompt(rec)
            if not prompt:
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
                    "content": "<think>\n[Concise reasoning: compare changes in fundamentals, interpret their direction, relate to holding_t, and infer expected log change.]\n</think>\n<answer>\n{\"holding_log_delta\": [predicted_value]}\n</answer>",
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
            if pbar is not None:
                pbar.update(1)
            if eff_progress and n % eff_progress == 0:
                limit_str = f"/{limit}" if limit else ""
                print(f"[type-pipeline] {label}: converted {n}{limit_str}")
            if limit and n >= limit:
                break
    if pbar is not None:
        pbar.close()
    print(f"[type-pipeline] {label}: done (total={n})")
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
    ap.add_argument("--sft-limit", type=int, default=None)
    ap.add_argument("--grpo-limit", type=int, default=None)
    ap.add_argument("--test-limit", type=int, default=None)

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
    ap.add_argument("--sft-with-think", dest="sft_with_think", action="store_true",
                    help="Force enabling <think> message in SFT data (default: enabled)")
    ap.add_argument("--sft-no-think", dest="sft_with_think", action="store_false",
                    help="Disable the <think> message in SFT data")
    ap.add_argument("--sft-contract-mode", choices=["absolute", "delta"], default="delta",
                    help="Target for SFT outputs (default: delta)")
    ap.add_argument("--sft-decimals", type=int, default=2,
                    help="Round SFT labels to this many decimals (default: 2)")
    ap.add_argument("--sft-think-template", type=str,
                    default="",
                    help="Template injected when --sft-with-think is active (leave blank to auto-generate reasoning)")
    ap.add_argument("--grpo-no-think-example", action="store_true")
    ap.set_defaults(sft_with_think=True)
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
    ph_sft_src = Path("artifacts/prompts_hist_sft") / f"{t}.jsonl"              # 构建原始 prompts 的输出（不带 think）
    ph_sft_enriched = Path("artifacts/prompts_hist_sft_with_think") / f"{t}.jsonl"  # DeepSeek 富集后的输入（带 think）
    #ph_sft = Path("artifacts/prompts_hist_sft_with_think") / f"{t}.jsonl"
    ph_grpo = Path("artifacts/prompts_hist_grpo") / f"{t}.jsonl"
    ph_test = Path("artifacts/prompts_hist_test") / f"{t}.jsonl"
    sft_out = Path("artifacts/sft") / f"sft_train_{t}.jsonl"
    # Put test chat JSONL into a separate folder to avoid mixing with SFT train data
    sft_test_out = Path("artifacts/test") / f"test_{t}.jsonl"
    grpo_out = Path("artifacts/grpo") / f"grpo_{t}.jsonl"

    system_prompt = _build_system_prompt(t)

    # 1) Build prompts for three splits
    print(f"[type-pipeline] building SFT prompts for {t} (<= {args.sft_end})")
    build_for_file(
        in_file=in_file,
        out_file=ph_sft_src,
        per_type_limit=args.sft_limit or args.per_type_limit,
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
        per_type_limit=args.grpo_limit or args.per_type_limit,
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
        per_type_limit=args.test_limit or args.per_type_limit,
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
    # ✅ 永远使用原始 SFT prompts，DeepSeek 由本脚本自动生成
    sft_input_for_convert = ph_sft_src
    print(f"[type-pipeline] SFT convert input = {sft_input_for_convert} (DeepSeek integrated generation)")


    _convert_prompts_to_sft(
        sft_input_for_convert,
        sft_out,
        system=system_prompt,
        with_think=args.sft_with_think,
        contract_mode=args.sft_contract_mode,
        decimals=args.sft_decimals,
        think_template=args.sft_think_template,
        label=f"sft_train_{t}",
        progress_every=100,
    )


    print(f"[type-pipeline] convert TEST -> chat ({sft_test_out})")
    _convert_prompts_to_sft(
        ph_test,
        sft_test_out,
        system=system_prompt,
        with_think=False,
        contract_mode="absolute",
        decimals=args.sft_decimals,
        label=f"test_chat_{t}",
        progress_every=100,
    )

    print(f"[type-pipeline] convert GRPO -> dataset ({grpo_out})")
    _convert_prompts_to_grpo(
        ph_grpo,
        grpo_out,
        system=system_prompt,
        no_think_example=args.grpo_no_think_example,
        label=f"grpo_{t}",
        progress_every=100,
    )

    # 3) Report stats
    stats_files = [
        (ph_sft_src, "prompts_sft_raw"),
        (ph_sft_enriched, "prompts_sft_with_think"),
        (ph_grpo, "prompts_grpo"),
        (ph_test, "prompts_test"),
    ]

    for fp, name in stats_files:
        if fp.exists():
            total, zeros = _count_zeros_in_prompts(fp)
            print(f"[type-pipeline] {name}: total={total} t==0={zeros} ratio={(zeros/total if total else 0):.3f}")


    print("[type-pipeline] done.")


if __name__ == "__main__":
    main()
