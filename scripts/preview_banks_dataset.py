"""Generate preview samples for updated SFT/GRPO banks datasets.

This script:
- Loads the first few entries from the existing SFT/GRPO banks JSONL files.
- Rewrites the user message to remove inline examples while keeping instructions.
- Calls scripts/deepseek.py to obtain high-quality assistant responses.
- Writes preview JSONL files under artifacts/samples/ for inspection.

Usage:
    python scripts/preview_banks_dataset.py --count 5 --temperature 0.4 --max-tokens 320
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SFT_PATH = REPO_ROOT / "artifacts" / "sft" / "sft_train_banks.jsonl"
GRPO_PATH = REPO_ROOT / "artifacts" / "grpo" / "grpo_banks.jsonl"
DEESEEK_SCRIPT = REPO_ROOT / "scripts" / "deepseek.py"
OUT_DIR = REPO_ROOT / "artifacts" / "samples"


def _read_jsonl(path: Path, limit: int) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _rewrite_user(original: str) -> str:
    head, *_ = original.partition("\n\nOUTPUT FORMAT")
    head = head.strip()
    instructions = (
        "\n\nRESPONSE FORMAT\n"
        "- Provide one <think>...</think> block with 3-5 concise sentences highlighting t-1->t numeric deltas.\n"
        "- Immediately follow with <answer>{\"holding_log_delta\": <float>}</answer> using 2 decimal places and no extra text.\n"
        "- Stop right after </answer>."
    )
    return head + instructions


def _call_deepseek(prompt: str, temperature: float, max_tokens: int, timeout: int) -> str:
    env = os.environ.copy()
    if not env.get("DEEPSEEK_API_KEY") and not env.get("OPENAI_API_KEY"):
        raise RuntimeError("环境变量 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 未设置，无法调用深度求解 API。")

    cmd = [
        sys.executable,
        str(DEESEEK_SCRIPT),
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(max_tokens),
    ]
    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"DeepSeek 调用失败 (code={proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def _build_sft_entries(records: Iterable[dict], temperature: float, max_tokens: int, timeout: int) -> list[dict]:
    new_records: list[dict] = []
    for rec in records:
        system_msg = rec["messages"][0]
        user_msg = rec["messages"][1]
        rewritten_user = _rewrite_user(user_msg["content"])
        assistant_content = _call_deepseek(rewritten_user, temperature, max_tokens, timeout)
        new_messages = [
            {"role": "system", "content": system_msg["content"]},
            {"role": "user", "content": rewritten_user},
            {"role": "assistant", "content": assistant_content, "loss": True},
        ]
        new_records.append({"messages": new_messages})
    return new_records


def _build_grpo_entries(records: Iterable[dict], temperature: float, max_tokens: int, timeout: int) -> list[dict]:
    new_records: list[dict] = []
    for rec in records:
        system_msg = rec["messages"][0]
        user_msg = rec["messages"][1]
        rewritten_user = _rewrite_user(user_msg["content"])
        assistant_content = _call_deepseek(rewritten_user, temperature, max_tokens, timeout)
        new_messages = [
            {"role": "system", "content": system_msg["content"]},
            {"role": "user", "content": rewritten_user},
            {"role": "assistant", "content": assistant_content, "loss": False},
        ]

        new_rec = dict(rec)
        new_rec["messages"] = new_messages
        new_records.append(new_rec)
    return new_records


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 SFT/GRPO banks 数据预览样本")
    ap.add_argument("--count", type=int, default=5, help="每个数据集生成样本数量")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--timeout", type=int, default=180, help="每次调用 DeepSeek 的超时时间（秒）")
    args = ap.parse_args()

    if not SFT_PATH.exists() or not GRPO_PATH.exists():
        raise FileNotFoundError("找不到原始 banks SFT/GRPO 数据文件。")

    sft_raw = _read_jsonl(SFT_PATH, args.count)
    grpo_raw = _read_jsonl(GRPO_PATH, args.count)

    print(f"Loaded {len(sft_raw)} SFT records and {len(grpo_raw)} GRPO records.")

    sft_new = _build_sft_entries(sft_raw, args.temperature, args.max_tokens, args.timeout)
    grpo_new = _build_grpo_entries(grpo_raw, args.temperature, args.max_tokens, args.timeout)

    sft_out = OUT_DIR / "banks_sft_preview.jsonl"
    grpo_out = OUT_DIR / "banks_grpo_preview.jsonl"
    _write_jsonl(sft_out, sft_new)
    _write_jsonl(grpo_out, grpo_new)

    print(f"Wrote {sft_out} and {grpo_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

