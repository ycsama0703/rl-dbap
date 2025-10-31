"""
Compute error statistics for base, SFT, and GRPO debug evaluation CSVs,
including values restricted to the 95% interval (dropping the top 5% largest errors).
"""
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_abs_error(csv_path: Path) -> pd.Series:
    """Return absolute error series from a debug CSV."""
    df = pd.read_csv(csv_path)
    if "abs_error" in df.columns:
        return df["abs_error"].astype(float)
    if {"parsed_pred", "y_true"}.issubset(df.columns):
        return (df["parsed_pred"] - df["y_true"]).abs()
    raise ValueError(f"{csv_path.name} missing abs_error or parsed_pred/y_true columns.")


def compute_stats(series: pd.Series) -> Dict[str, float]:
    """Produce a consistent stats dictionary for a series."""
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "output"

    csv_files = {
        "Base": output_dir / "debug_eval_outputs_base.csv",
        "SFT": output_dir / "debug_eval_outputs_sft.csv",
        "GRPO": output_dir / "debug_eval_outputs_grpo.csv",
    }

    rows: List[Dict[str, float]] = []

    for label, path in csv_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV for {label}: {path}")

        errors = load_abs_error(path)
        threshold = float(errors.quantile(0.95))
        trimmed = errors[errors <= threshold]

        full_stats = compute_stats(errors)
        trimmed_stats = compute_stats(trimmed)

        rows.append(
            {
                "model": label,
                "scope": "Full",
                "threshold_p95": threshold,
                **full_stats,
            }
        )
        rows.append(
            {
                "model": label,
                "scope": "<=95%",
                "threshold_p95": threshold,
                **trimmed_stats,
            }
        )

        print(
            f"{label} - 95% threshold: {threshold:.4f} | "
            f"Trimmed mean: {trimmed_stats['mean']:.4f}, median: {trimmed_stats['median']:.4f}, "
            f"std: {trimmed_stats['std']:.4f}"
        )

    result_df = pd.DataFrame(rows)
    output_path = output_dir / "debug_error_stats_95.csv"
    result_df.to_csv(output_path, index=False)
    print(f"Saved detailed stats to {output_path}")


if __name__ == "__main__":
    main()

