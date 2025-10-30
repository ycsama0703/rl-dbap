#!/usr/bin/env python
"""
run_eval_strict.py

Evaluation script that enforces longer generation length and strict parsing,
avoiding fallback extraction issues seen in the legacy run_eval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.backends.hf_infer import (
    build_eval_inputs,
    load_model_and_tokenizer,
    infer_chat_batch,
    extract_pred,
)
from src.evaluation.metrics import basic_regression, topk


def chunked(indices: List[int], size: int):
    for i in range(0, len(indices), size):
        yield indices[i : i + size]


def run_eval_strict(
    model_id: str,
    lora_path: str | None,
    test_path: str,
    out_dir: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    torch_dtype: str,
    force_think: bool,
) -> pd.DataFrame:
    chat_inputs, y_true, quarters, ids, holding_ts = build_eval_inputs(test_path)

    tokenizer, model = load_model_and_tokenizer(model_id, lora_path, torch_dtype=torch_dtype)
    model.eval()

    raw_outputs: List[str] = []
    preds: List[float | None] = []

    indices = list(range(len(chat_inputs)))
    for batch in chunked(indices, batch_size):
        batch_msgs = [chat_inputs[i] for i in batch]
        completions = infer_chat_batch(
            tokenizer,
            model,
            batch_msgs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            force_think=force_think,
        )
        for idx, completion in zip(batch, completions):
            raw_outputs.append(completion)
            preds.append(extract_pred(completion, holding_ts[idx]))

    df = pd.DataFrame(
        {
            "id": ids,
            "quarter": quarters,
            "holding_t": holding_ts,
            "y_true": y_true,
            "raw_output": raw_outputs,
            "y_pred": preds[: len(y_true)],
        }
    ).set_index("id")
    df["valid"] = df["y_pred"].notna()
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict evaluation with longer generation length.")
    ap.add_argument("--test_path", type=str, required=True)
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora_path", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--torch_dtype", type=str, default="bfloat16")
    ap.add_argument("--force-think", dest="force_think", action="store_true")
    ap.add_argument("--no-force-think", dest="force_think", action="store_false")
    ap.set_defaults(force_think=True)
    args = ap.parse_args()

    df = run_eval_strict(
        args.base_model,
        args.lora_path,
        args.test_path,
        args.out_dir,
        args.batch_size,
        args.max_new_tokens,
        args.temperature,
        args.torch_dtype,
        args.force_think,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "pred_detail.csv", index=True)

    valid = df[df["valid"]]
    coverage = 100.0 * df["valid"].mean()

    if valid.empty:
        print("[run_eval_strict] No valid predictions; skipping metrics and plots.")
        pd.DataFrame(
            [
                {
                    "coverage%": coverage,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "sMAPE%": np.nan,
                    "IC": np.nan,
                    "RankIC": np.nan,
                    "Recall@50": np.nan,
                    "Precision@50": np.nan,
                    "NDCG@50": np.nan,
                }
            ]
        ).to_csv(out_dir / "metrics.csv", index=False)
        return

    mae, rmse, r2, sm, ic, ric = basic_regression(valid)
    rec, pre, ndcg = topk(valid, "quarter", k=50)

    pd.DataFrame(
        [
            {
                "coverage%": coverage,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "sMAPE%": sm,
                "IC": ic,
                "RankIC": ric,
                "Recall@50": rec,
                "Precision@50": pre,
                "NDCG@50": ndcg,
            }
        ]
    ).to_csv(out_dir / "metrics.csv", index=False)

    residuals = valid["y_pred"] - valid["y_true"]
    if not residuals.empty:
        plt.figure()
        residuals.hist(bins=50)
        plt.title("Residuals")
        plt.tight_layout()
        plt.savefig(out_dir / "residual_hist.png", dpi=150)
        plt.close()

    quarter_ic = (
        valid.groupby("quarter", group_keys=False)
        .apply(lambda g: g[["y_true", "y_pred"]].corr(method="spearman").iloc[0, 1])
        .dropna()
    )
    if not quarter_ic.empty:
        plt.figure()
        quarter_ic.plot(kind="bar", title="Quarterly IC")
        plt.tight_layout()
        plt.savefig(out_dir / "ic_by_quarter.png", dpi=150)
        plt.close()

    print(f"[run_eval_strict] Metrics saved to {out_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
