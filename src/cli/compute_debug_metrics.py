"""
Compute evaluation metrics for debug prediction CSVs and save them under output/metrics.
Supports optional trimming to keep only rows within a specified absolute-error percentile.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from src.evaluation.metrics import basic_regression, topk  # noqa: E402


def build_default_map(repo_root: Path) -> Dict[str, Path]:
    output_dir = repo_root / "output"
    return {
        "base": output_dir / "debug_eval_outputs_base.csv",
        "sft": output_dir / "debug_eval_outputs_sft.csv",
        "grpo": output_dir / "debug_eval_outputs_grpo.csv",
    }


def compute_metrics_from_df(df: pd.DataFrame, trim_pct: Optional[float] = None) -> Tuple[Dict[str, float], Optional[float]]:
    df = df.copy()
    if "parsed_pred" not in df.columns or "y_true" not in df.columns:
        raise ValueError("Required columns parsed_pred and y_true missing.")

    df["y_pred"] = pd.to_numeric(df["parsed_pred"], errors="coerce")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    if "holding_t" in df.columns:
        df["holding_t"] = pd.to_numeric(df["holding_t"], errors="coerce")

    if "abs_error" in df.columns:
        abs_errors = pd.to_numeric(df["abs_error"], errors="coerce")
    else:
        abs_errors = (df["y_pred"] - df["y_true"]).abs()

    trim_threshold: Optional[float] = None
    if trim_pct is not None:
        if not 0 < trim_pct <= 100:
            raise ValueError("trim_pct must be within (0, 100].")
        quantile = trim_pct / 100.0
        dropna_errors = abs_errors.dropna()
        if dropna_errors.empty:
            raise ValueError("Cannot compute trim threshold because all abs_error values are NaN.")
        trim_threshold = float(dropna_errors.quantile(quantile))
        df = df.loc[abs_errors <= trim_threshold].copy()
        if df.empty:
            raise ValueError(f"No data remaining after trimming at {trim_pct}% for provided CSV.")

    if "quarter" not in df.columns:
        df["quarter"] = "NA"
    df["quarter"] = df["quarter"].fillna("NA")

    total = len(df)
    valid = df[df["y_pred"].notna()].copy()
    coverage = 100.0 * len(valid) / total if total else np.nan

    if "id" in valid.columns:
        valid = valid.set_index("id", drop=False)

    mae, rmse, r2, smape, ic, ric = basic_regression(valid)
    rec, pre, ndcg = topk(valid, "quarter", k=50)

    return {
        "coverage%": coverage,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "sMAPE%": smape,
        "IC": ic,
        "RankIC": ric,
        "Recall@50": rec,
        "Precision@50": pre,
        "NDCG@50": ndcg,
    }, trim_threshold


def process_models(model_names: Iterable[str], csv_map: Dict[str, Path], metrics_dir: Path, trim_pct: Optional[float]) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for name in model_names:
        if name not in csv_map:
            raise KeyError(f"Unknown model '{name}'. Available: {', '.join(sorted(csv_map))}")

        csv_path = csv_map[name]
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing debug CSV for {name}: {csv_path}")

        df = pd.read_csv(csv_path)
        metrics, threshold = compute_metrics_from_df(df, trim_pct=trim_pct)

        suffix = ""
        if trim_pct is not None:
            pct_str = f"{trim_pct:g}"
            suffix = f"_trim{pct_str}"
        out_path = metrics_dir / f"metrics_from_debug_{name}{suffix}.csv"
        pd.DataFrame([metrics]).to_csv(out_path, index=False)

        if threshold is not None:
            print(f"Saved metrics for {name} to {out_path} (<= {trim_pct:g}th percentile abs_error ≈ {threshold:.4f})")
        else:
            print(f"Saved metrics for {name} to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute metrics for debug prediction CSVs.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of models to process (default: base sft grpo).",
    )
    parser.add_argument(
        "--metrics-dir",
        default=None,
        help="Directory to write metric CSVs (default: output/metrics).",
    )
    parser.add_argument(
        "--trim-pct",
        type=float,
        default=None,
        help="Optional percentile (0-100] of absolute error to keep, e.g., 95 to drop top 5%%.",
    )
    return parser.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    csv_map = build_default_map(repo_root)

    args = parse_args()
    models = args.models or sorted(csv_map.keys())
    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else repo_root / "output" / "metrics"

    process_models(models, csv_map, metrics_dir, trim_pct=args.trim_pct)


if __name__ == "__main__":
    main()
