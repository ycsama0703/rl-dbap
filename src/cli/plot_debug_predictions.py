"""
Generate comparative plots of debug evaluation predictions for base, SFT, and GRPO models.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_predictions(csv_path: Path, prefix: str) -> pd.DataFrame:
    """
    Load a debug evaluation CSV and keep only columns needed for plotting.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    prefix:
        Prefix used when renaming the prediction column to identify the model.
    """
    df = pd.read_csv(csv_path)

    expected_cols = {"id", "parsed_pred"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {', '.join(sorted(missing))}")

    renamed = df[["id", "parsed_pred"]].rename(columns={"parsed_pred": f"{prefix}_pred"})
    if "y_true" in df.columns:
        renamed["y_true"] = df["y_true"]

    return renamed


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = repo_root / "output"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_files = {
        "base": output_dir / "debug_eval_outputs_base.csv",
        "sft": output_dir / "debug_eval_outputs_sft.csv",
        "grpo": output_dir / "debug_eval_outputs_grpo.csv",
    }

    for name, path in csv_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV for {name}: {path}")

    base_df = load_predictions(csv_files["base"], "base")
    sft_df = load_predictions(csv_files["sft"], "sft")
    grpo_df = load_predictions(csv_files["grpo"], "grpo")

    # Keep y_true from base only to avoid duplicate columns during merges.
    sft_df = sft_df.drop(columns=[col for col in sft_df.columns if col.startswith("y_true")])
    grpo_df = grpo_df.drop(columns=[col for col in grpo_df.columns if col.startswith("y_true")])

    combined = base_df.merge(sft_df, on="id", how="inner").merge(grpo_df, on="id", how="inner")

    if "y_true" not in combined.columns:
        raise ValueError("No ground truth column found in combined dataframe.")

    combined = combined.sort_values("id").reset_index(drop=True)

    x = combined.index
    y_true = combined["y_true"]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    axes[0].plot(x, y_true, label="Ground Truth", color="black", linewidth=1.8)
    axes[0].plot(x, combined["base_pred"], label="Base Prediction", alpha=0.8)
    axes[0].plot(x, combined["sft_pred"], label="SFT Prediction", alpha=0.8)
    axes[0].plot(x, combined["grpo_pred"], label="GRPO Prediction", alpha=0.8)
    axes[0].set_ylabel("Holding")
    axes[0].set_title("Model Predictions vs Ground Truth")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, combined["base_pred"] - y_true, label="Base - True")
    axes[1].plot(x, combined["sft_pred"] - y_true, label="SFT - True")
    axes[1].plot(x, combined["grpo_pred"] - y_true, label="GRPO - True")
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    axes[1].set_ylabel("Prediction Error")
    axes[1].set_xlabel("Sample Index")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    output_path = plots_dir / "debug_predictions_comparison.png"
    fig.savefig(output_path, dpi=200)

    print(f"Saved comparative plot to {output_path}")


if __name__ == "__main__":
    main()
