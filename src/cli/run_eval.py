# src/cli/run_eval.py
# -*- coding: utf-8 -*-
import os, argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from src.backends.hf_infer import load_model_and_tokenizer, infer_chat_batch, extract_pred, build_eval_inputs
from src.evaluation.metrics import basic_regression, topk

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None


def run_one(model_id: str, lora_path: str|None, test_path: str, out_dir: str,
            batch_size=8, max_new_tokens=48, temperature=0.0, torch_dtype="bfloat16",
            force_think: bool = False):

    chat_inputs, y_true, quarters, ids, holding_ts = build_eval_inputs(test_path)
    tok, mdl = load_model_and_tokenizer(model_id, lora_path, torch_dtype)
    preds=[]
    iterator = range(0, len(chat_inputs), batch_size)
    if tqdm:
        iterator = tqdm(iterator, desc=f"infer {Path(out_dir).name}", total=(len(chat_inputs)+batch_size-1)//batch_size)
    for i in iterator:
        outs = infer_chat_batch(
            tok,
            mdl,
            chat_inputs[i : i + batch_size],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            force_think=force_think,
        )
        for raw, ht in zip(outs, holding_ts[i:i+batch_size]):
            preds.append(extract_pred(raw, ht))

    df = pd.DataFrame({"id": ids, "quarter": quarters, "y_true": y_true, "y_pred": preds[:len(y_true)]}).set_index("id")
    df["valid"]=df["y_pred"].notna()
    valid=df[df["valid"]]

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(Path(out_dir)/"pred_detail.csv", index=True)

    coverage = 100.0 * df["valid"].mean()
    metrics_path = Path(out_dir) / "metrics.csv"

    if valid.empty:
        print("[run_eval] No valid predictions; skipping metric computation and plots.")
        pd.DataFrame([{
            "coverage%": coverage,
            "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "sMAPE%": np.nan,
            "IC": np.nan, "RankIC": np.nan,
            "Recall@50": np.nan, "Precision@50": np.nan, "NDCG@50": np.nan
        }]).to_csv(metrics_path, index=False)
        return df

    mae, rmse, r2, sm, ic, ric = basic_regression(valid)
    rec, pre, ndcg = topk(valid, "quarter", k=50)

    pd.DataFrame([{
        "coverage%": coverage,
        "MAE": mae, "RMSE": rmse, "R2": r2, "sMAPE%": sm,
        "IC": ic, "RankIC": ric,
        "Recall@50": rec, "Precision@50": pre, "NDCG@50": ndcg
    }]).to_csv(metrics_path, index=False)

    # plots
    residuals = valid["y_pred"] - valid["y_true"]
    if not residuals.empty:
        residuals.hist(bins=50)
        plt.title("Residuals")
        plt.tight_layout()
        plt.savefig(Path(out_dir)/"residual_hist.png", dpi=150)
        plt.close()

    quarter_ic = (
        valid.groupby("quarter", group_keys=False)
        .apply(lambda g: g[["y_true", "y_pred"]].corr(method="spearman").iloc[0, 1])
        .dropna()
    )
    if not quarter_ic.empty:
        quarter_ic.plot(kind="bar", title="Quarterly IC")
        plt.tight_layout()
        plt.savefig(Path(out_dir)/"ic_by_quarter.png", dpi=150)
        plt.close()

    return df


def compare(pre_csv: str, post_csv: str, out_dir: str):
    pre = pd.read_csv(pre_csv, index_col="id"); post = pd.read_csv(post_csv, index_col="id")
    df = pre[["quarter","y_true","y_pred"]].rename(columns={"y_pred":"pre"}).join(post[["y_pred"]].rename(columns={"y_pred":"post"}), how="inner")
    df = df.dropna()
    # 绝对误差提升
    ae_pre = (df["y_true"]-df["pre"]).abs().values
    ae_post= (df["y_true"]-df["post"]).abs().values
    imp = ae_pre.mean()-ae_post.mean()

    # 简单 bootstrap CI
    rng = np.random.default_rng(42)
    N=2000; diffs=[]
    for _ in range(N):
        idx = rng.integers(0, len(df), len(df))
        diffs.append(ae_pre[idx].mean()-ae_post[idx].mean())
    lo, hi = np.percentile(diffs, [2.5,97.5])

    with open(Path(out_dir)/"compare.txt","w",encoding="utf-8") as f:
        f.write(f"MAE improvement: {imp:.6f}\n95% CI: [{lo:.6f}, {hi:.6f}]\n")


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_path", type=str, default="artifacts/sft/test.jsonl")
    ap.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B-Instruct")
    ap.add_argument("--lora_path", type=str, default="None", help="None => Pre-SFT")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--torch_dtype", type=str, default="bfloat16")
    ap.add_argument("--post_csv_for_compare", type=str, default="")
    ap.add_argument("--force-think", dest="force_think", action="store_true",
                    help="Force generations to begin with <think>.")
    ap.add_argument("--no-force-think", dest="force_think", action="store_false",
                    help="Disable forced <think> prefix (default).")
    ap.set_defaults(force_think=False)
    args = ap.parse_args()

    df = run_one(
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

    if args.post_csv_for_compare:
        compare(str(Path(args.out_dir)/"pred_detail.csv"), args.post_csv_for_compare, args.out_dir)
