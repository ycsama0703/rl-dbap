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
    LOG_EPS,
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
            preds.append(extract_pred(completion))

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

    def to_tp1(log_delta, holding):
        if log_delta is None or holding is None:
            return np.nan
        try:
            return float(np.exp(log_delta) * (holding + LOG_EPS) - LOG_EPS)
        except Exception:
            return np.nan

    df["y_true_tp1"] = [
        to_tp1(log_val, ht) for log_val, ht in zip(df["y_true"], df["holding_t"])
    ]
    df["y_pred_tp1"] = [
        to_tp1(pred, ht) if pred is not None else np.nan for pred, ht in zip(df["y_pred"], df["holding_t"])
    ]
    df["abs_log_error"] = (df["y_pred"] - df["y_true"]).abs()
    if "y_pred_tp1" in df.columns:
        df["abs_tp1_error"] = (df["y_pred_tp1"] - df["y_true_tp1"]).abs()
    else:
        df["abs_tp1_error"] = np.nan
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
                    "MAE_log": np.nan,
                    "RMSE_log": np.nan,
                    "R2_log": np.nan,
                    "sMAPE_log%": np.nan,
                    "IC_log": np.nan,
                    "RankIC_log": np.nan,
                    "Recall@50_log": np.nan,
                    "Precision@50_log": np.nan,
                    "NDCG@50_log": np.nan,
                    "MAE_tp1": np.nan,
                    "RMSE_tp1": np.nan,
                    "R2_tp1": np.nan,
                    "sMAPE_tp1%": np.nan,
                    "IC_tp1": np.nan,
                    "RankIC_tp1": np.nan,
                    "Recall@50_tp1": np.nan,
                    "Precision@50_tp1": np.nan,
                    "NDCG@50_tp1": np.nan,
                }
            ]
        ).to_csv(out_dir / "metrics.csv", index=False)
        return

    mae_log, rmse_log, r2_log, sm_log, ic_log, ric_log = basic_regression(valid)
    rec_log, pre_log, ndcg_log = topk(valid, "quarter", k=50)

    valid_abs = valid.dropna(subset=["y_true_tp1", "y_pred_tp1"]).copy()
    if not valid_abs.empty:
        subset_cols = ["y_true_tp1", "y_pred_tp1", "quarter"]
        if "id" in valid_abs.columns:
            subset_cols.append("id")
        valid_abs_metrics = valid_abs[subset_cols].rename(
            columns={"y_true_tp1": "y_true", "y_pred_tp1": "y_pred"}
        )
        mae_tp1, rmse_tp1, r2_tp1, sm_tp1, ic_tp1, ric_tp1 = basic_regression(valid_abs_metrics)
        rec_tp1, pre_tp1, ndcg_tp1 = topk(valid_abs_metrics, "quarter", k=50)
    else:
        mae_tp1 = rmse_tp1 = r2_tp1 = sm_tp1 = ic_tp1 = ric_tp1 = np.nan
        rec_tp1 = pre_tp1 = ndcg_tp1 = np.nan

    pd.DataFrame(
        [
            {
                "coverage%": coverage,
                "MAE_log": mae_log,
                "RMSE_log": rmse_log,
                "R2_log": r2_log,
                "sMAPE_log%": sm_log,
                "IC_log": ic_log,
                "RankIC_log": ric_log,
                "Recall@50_log": rec_log,
                "Precision@50_log": pre_log,
                "NDCG@50_log": ndcg_log,
                "MAE_tp1": mae_tp1,
                "RMSE_tp1": rmse_tp1,
                "R2_tp1": r2_tp1,
                "sMAPE_tp1%": sm_tp1,
                "IC_tp1": ic_tp1,
                "RankIC_tp1": ric_tp1,
                "Recall@50_tp1": rec_tp1,
                "Precision@50_tp1": pre_tp1,
                "NDCG@50_tp1": ndcg_tp1,
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
