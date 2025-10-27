#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch inference utility that runs a base model (optionally with GRPO LoRA)
over a prompt JSONL exported via export_infer_prompts.py, parsing <think>
and <answer> blocks and saving the results to JSONL/CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backends.hf_infer import (  # noqa: E402
    load_model_and_tokenizer,
    infer_chat_batch,
    extract_y_true,
)
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting optional
    plt = None

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm missing
    tqdm = None

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_completion(text: str) -> Tuple[str | None, str | None, Dict[str, Any] | None]:
    think_match = THINK_RE.search(text)
    answer_match = ANSWER_RE.search(text)

    think = think_match.group(1).strip() if think_match else None
    answer = answer_match.group(1).strip() if answer_match else None

    payload = None
    candidate = answer
    if candidate:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            json_match = JSON_RE.search(candidate)
            if json_match:
                try:
                    payload = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    payload = None
    return think, answer, payload


def load_prompts(jsonl_path: Path, start: int = 0, limit: int | None = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start:
                continue
            if limit is not None and len(records) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_labels(labels_path: Path) -> Dict[int, float]:
    label_map: Dict[int, float] = {}
    with labels_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = rec.get("messages", [])
            assistants = [m for m in msgs if m.get("role") == "assistant"]
            if not assistants:
                continue
            target = next((m for m in assistants if m.get("loss") is True), assistants[-1])
            y_true = extract_y_true(target.get("content", ""))
            if y_true is not None:
                label_map[idx] = y_true
    return label_map


def main():
    parser = argparse.ArgumentParser(description="Run batch inference and export think/answer parsing results.")
    parser.add_argument("--jsonl", type=str, required=True, help="Prompt JSONL from export_infer_prompts.py.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model ID or local path.")
    parser.add_argument("--checkpoint", type=str, default="None", help="LoRA checkpoint directory (use None for base).")
    parser.add_argument("--labels", type=str, default="", help="Original eval JSONL to fetch ground-truth holding_tp1.")
    parser.add_argument("--out_jsonl", type=str, required=True, help="Destination JSONL to store outputs.")
    parser.add_argument("--out_csv", type=str, default="", help="Optional CSV path.")
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="",
        help="Optional directory to save evaluation plots (requires --labels and parsed holding_tp1).",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference.")
    parser.add_argument("--start", type=int, default=0, help="Start index within the prompt JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to process.")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Torch dtype for the base model.",
    )
    parser.add_argument(
        "--progress_log_steps",
        type=int,
        default=50,
        help="Frequency (in processed samples) to log running MAE/RMSE during inference (requires --labels).",
    )
    args = parser.parse_args()

    prompts_path = Path(args.jsonl)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompts_path}")

    prompts = load_prompts(prompts_path, args.start, args.limit)
    if not prompts:
        raise ValueError("No prompts loaded. Check --start/--limit and file contents.")

    label_map: Dict[int, float] = {}
    if args.labels:
        labels_path = Path(args.labels)
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        label_map = load_labels(labels_path)

    tokenizer, model = load_model_and_tokenizer(args.base_model, args.checkpoint, args.torch_dtype)

    outputs: List[Dict[str, Any]] = []
    iterator = range(0, len(prompts), args.batch_size)
    if tqdm:
        iterator = tqdm(iterator, desc="infer", total=(len(prompts) + args.batch_size - 1) // args.batch_size)

    preds_running: List[float] = []
    trues_running: List[float] = []
    running_counts: List[int] = []
    running_mae: List[float] = []
    running_rmse: List[float] = []

    processed = 0

    for i in iterator:
        batch = prompts[i : i + args.batch_size]
        messages = [
            [
                {"role": "system", "content": rec.get("system", "").strip()},
                {"role": "user", "content": rec.get("prompt", "").strip()},
            ]
            for rec in batch
        ]
        generations = infer_chat_batch(
            tokenizer,
            model,
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        for rec, text in zip(batch, generations):
            think, answer, payload = parse_completion(text)
            parsed_value = None
            parsed_key = None
            if isinstance(payload, dict):
                for key in ("holding_tp1", "holding_delta"):
                    if key in payload:
                        value = payload[key]
                        if isinstance(value, (int, float)):
                            parsed_value = float(value)
                            parsed_key = key
                            break
            label = label_map.get(rec.get("id")) if label_map else None
            outputs.append(
                {
                    "id": rec.get("id"),
                    "system": rec.get("system"),
                    "prompt": rec.get("prompt"),
                    "raw_output": text,
                    "think": think,
                    "answer": answer,
                    "parsed_key": parsed_key,
                    "parsed_value": parsed_value,
                    "label_tp1": label,
                }
            )

            processed += 1
            if label is not None and parsed_key == "holding_tp1" and parsed_value is not None:
                preds_running.append(parsed_value)
                trues_running.append(label)
                preds_arr = np.array(preds_running)
                trues_arr = np.array(trues_running)
                count = len(preds_running)
                mae = float(np.mean(np.abs(preds_arr - trues_arr)))
                rmse = float(np.sqrt(np.mean((preds_arr - trues_arr) ** 2)))
                running_counts.append(count)
                running_mae.append(mae)
                running_rmse.append(rmse)
                coverage = count / processed
                if args.progress_log_steps > 0 and processed % args.progress_log_steps == 0:
                    print(f"[progress] processed={processed} coverage={coverage:.1%} running_MAE={mae:.6f} running_RMSE={rmse:.6f}")
            elif args.progress_log_steps > 0 and processed % args.progress_log_steps == 0:
                coverage = (len(preds_running) / processed) if processed > 0 else 0.0
                print(f"[progress] processed={processed} coverage={coverage:.1%} (insufficient preds for MAE/RMSE)")

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for row in outputs:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.out_csv:
        csv_path = Path(args.out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["id", "parsed_key", "parsed_value", "label_tp1", "think", "answer", "raw_output"]
        with csv_path.open("w", encoding="utf-8", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()
            for row in outputs:
                writer.writerow({key: row.get(key) for key in fieldnames})

    print(f"Wrote {len(outputs)} rows -> {out_path}")
    if args.out_csv:
        print(f"Wrote {len(outputs)} rows -> {csv_path}")

    if args.plot_dir:
        if not label_map:
            print("Skipping plots: --labels required to compute metrics.", file=sys.stderr)
        else:
            plot_dir = Path(args.plot_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)
            preds = []
            trues = []
            for row in outputs:
                if row.get("parsed_key") == "holding_tp1" and row.get("parsed_value") is not None and row.get("label_tp1") is not None:
                    preds.append(row["parsed_value"])
                    trues.append(row["label_tp1"])
            if preds:
                preds_arr = np.array(preds)
                trues_arr = np.array(trues)
                mae = float(np.mean(np.abs(preds_arr - trues_arr)))
                rmse = float(np.sqrt(np.mean((preds_arr - trues_arr) ** 2)))
                coverage = len(preds) / len(outputs)

                print(f"Coverage: {coverage:.1%} | MAE: {mae:.6f} | RMSE: {rmse:.6f}")

                if plt:
                    resid = preds_arr - trues_arr
                    plt.figure(figsize=(6, 4))
                    plt.hist(resid, bins=50, alpha=0.8)
                    plt.title("Residuals (prediction - truth)")
                    plt.tight_layout()
                    plt.savefig(plot_dir / "residual_hist.png", dpi=150)
                    plt.close()

                    plt.figure(figsize=(5, 5))
                    lims = [
                        min(trues_arr.min(), preds_arr.min()),
                        max(trues_arr.max(), preds_arr.max()),
                    ]
                    plt.scatter(trues_arr, preds_arr, alpha=0.4, s=10)
                    plt.plot(lims, lims, "r--", label="Ideal")
                    plt.xlabel("True holding_tp1")
                    plt.ylabel("Predicted holding_tp1")
                    plt.title("Prediction vs Truth")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(plot_dir / "pred_vs_truth.png", dpi=150)
                    plt.close()

                    if running_counts:
                        plt.figure(figsize=(6, 4))
                        plt.plot(running_counts, running_mae, label="MAE")
                        plt.plot(running_counts, running_rmse, label="RMSE")
                        plt.xlabel("Samples with parsed holding_tp1")
                        plt.ylabel("Error")
                        plt.title("Running error vs samples")
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(plot_dir / "running_metrics.png", dpi=150)
                        plt.close()
                else:
                    print("matplotlib not available; skipping plot generation.", file=sys.stderr)
            else:
                print("No parsed holding_tp1 predictions. Plots skipped.", file=sys.stderr)


if __name__ == "__main__":
    main()
