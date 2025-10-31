"""
Plot error distributions for base, SFT, and GRPO debug evaluation CSVs.
"""
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_errors(csv_path: Path, label: str) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Load absolute errors for a model and compute summary statistics.

    Returns
    -------
    errors:
        Series containing absolute errors.
    stats:
        Dictionary with descriptive statistics for logging.
    """
    df = pd.read_csv(csv_path)
    if "abs_error" in df.columns:
        errors = df["abs_error"].astype(float)
    elif {"parsed_pred", "y_true"}.issubset(df.columns):
        errors = (df["parsed_pred"] - df["y_true"]).abs()
    else:
        raise ValueError(f"{csv_path.name} missing abs_error or parsed_pred/y_true columns.")

    stats = {
        "count": int(errors.count()),
        "mean": float(errors.mean()),
        "median": float(errors.median()),
        "std": float(errors.std(ddof=0)),
        "max": float(errors.max()),
        "p95": float(errors.quantile(0.95)),
        "p99": float(errors.quantile(0.99)),
    }

    print(f"{label} stats: {stats}")
    return errors, stats


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "output"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_files = {
        "Base": output_dir / "debug_eval_outputs_base.csv",
        "SFT": output_dir / "debug_eval_outputs_sft.csv",
        "GRPO": output_dir / "debug_eval_outputs_grpo.csv",
    }

    for label, path in csv_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV for {label}: {path}")

    errors = {}
    for label, path in csv_files.items():
        errors[label], _ = load_errors(path, label)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Histogram with log-scaled y to highlight tails
    bins = np.linspace(0, max(series.max() for series in errors.values()), 80)
    for label, series in errors.items():
        axes[0].hist(
            series,
            bins=bins,
            alpha=0.6,
            label=label,
            density=True,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Absolute Error")
    axes[0].set_ylabel("Density (log scale)")
    axes[0].set_title("Absolute Error Distribution")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Boxplot to show spread and outliers
    axes[1].boxplot(
        [errors[label] for label in errors],
        labels=list(errors.keys()),
        showfliers=True,
    )
    axes[1].set_ylabel("Absolute Error")
    axes[1].set_title("Absolute Error Spread by Model")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = plots_dir / "debug_error_distributions.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved error distribution plot to {output_path}")


if __name__ == "__main__":
    main()
