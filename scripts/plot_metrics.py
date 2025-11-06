import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import numpy as np

# === 路径 ===
metrics_dir = Path("output/distill_eval_output")
print(f"✅ Using metrics directory: {metrics_dir}")

metrics_files = sorted(metrics_dir.glob("metrics_from_*.csv"))
if not metrics_files:
    raise RuntimeError(f"❌ No metrics_from_*.csv found in {metrics_dir}")

print(f"✅ Found {len(metrics_files)} metric files")

dfs = []
for f in metrics_files:
    try:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        dfs.append(df)
    except Exception as e:
        print(f"⚠️ Failed to read {f.name}: {e}")

if not dfs:
    raise RuntimeError("❌ No valid CSVs loaded.")

df_all = pd.concat(dfs, ignore_index=True)
df_all.columns = [c.strip() for c in df_all.columns]

# === 提取模型名与 trim ===
df_all["model"] = df_all["source_file"].str.extract(r"metrics_from_debug_eval_([^.]*)")[0]
df_all["trim"] = df_all["source_file"].apply(
    lambda x: "99" if "trim99" in x else ("95" if "trim95" in x else "all")
)

# === 指标：去掉 IC 与 RankIC ===
metrics = [
    "MAE_log", "RMSE_log", "R2_log",
    "sMAPE_log%", "coverage_valid%", "coverage_filtered%"
]
available_metrics = [m for m in metrics if m in df_all.columns]
print(f"📊 Metrics available: {available_metrics}")

# === 聚合 ===
agg = df_all.groupby(["model", "trim"])[available_metrics].mean().reset_index()

# === 全局颜色映射（每种模型固定颜色） ===
model_list = sorted(agg["model"].unique())
n_models = len(model_list)
colors = cm.get_cmap("tab10", n_models)
color_map = {model: colors(i) for i, model in enumerate(model_list)}

# === trim 顺序 ===
trim_levels = ["all", "95", "99"]

# === 绘制每个 trim 的 2x3 面板 ===
for trim in trim_levels:
    sub = agg[agg["trim"] == trim]
    if sub.empty:
        continue

    sub_models = sorted(sub["model"].unique())
    n_sub_models = len(sub_models)
    colors_used = [color_map[m] for m in sub_models]

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(2, 3, figsize=(16, 6))
    fig.suptitle(f"Model Comparison — Trim {trim}", fontsize=16, y=1.02)

    for i, metric in enumerate(available_metrics):
        ax = axes[i // 3, i % 3]
        x = np.arange(n_sub_models)
        vals = [sub[sub["model"] == m][metric].values[0] for m in sub_models]

        # 关键：让柱状图变“细长”
        bar_width = 0.4 / max(1, n_sub_models / 6)
        bars = ax.bar(x, vals, color=colors_used, width=bar_width)

        ax.set_title(metric, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(sub_models, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # 数值标签
        for xi, yi in zip(x, vals):
            ax.text(xi, yi, f"{yi:.3f}", ha="center", va="bottom", fontsize=7)

    # 移除多余空轴（若指标不是6个）
    for j in range(n_metrics, 6):
        fig.delaxes(axes[j // 3, j % 3])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = metrics_dir / f"metrics_panel_trim{trim}_2x3.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved {out_path}")

print("🎯 Done! Each trim-level has a 2x3 grid with slim vertical bars.")
