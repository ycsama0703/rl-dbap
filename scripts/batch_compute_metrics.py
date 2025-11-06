import subprocess
from pathlib import Path

# 输入与输出都在同一目录
base_dir = Path("output/distill_eval_output")

# 遍历所有重新解析后的结果文件
for csv_path in base_dir.glob("*_reparsed.csv"):
    base_name = csv_path.stem.replace("_reparsed", "")
    base_out = base_dir / f"metrics_from_{base_name}.csv"

    print(f"\n📊 Processing: {csv_path.name}")

    # === 全样本 ===
    subprocess.run([
        "python", "scripts/compute_metrics_from_debug.py",
        "--debug-csv", str(csv_path),
        "--out-csv", str(base_out)
    ], check=True)

    # === Trim 95% ===
    subprocess.run([
        "python", "scripts/compute_metrics_from_debug.py",
        "--debug-csv", str(csv_path),
        "--out-csv", str(base_dir / f"metrics_from_{base_name}_trim95.csv"),
        "--error-quantile", "0.95"
    ], check=True)

    # === Trim 99% ===
    subprocess.run([
        "python", "scripts/compute_metrics_from_debug.py",
        "--debug-csv", str(csv_path),
        "--out-csv", str(base_dir / f"metrics_from_{base_name}_trim99.csv"),
        "--error-quantile", "0.99"
    ], check=True)

print("\n✅ All metrics computed successfully in output/distill_eval_output/")
