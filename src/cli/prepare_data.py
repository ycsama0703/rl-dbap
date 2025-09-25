"""
原始 raw/*.parquet 合并 → 季度对齐 → 生成 holding_t/holding_t1 → 清洗 → 按 type 分文件落到 processed/
可选读取 configs/data.yaml；命令行优先级最高。
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
try:
    import yaml  # 可选依赖
except Exception:
    yaml = None

import pandas as pd

from src.dataio.readers import load_raw_years
from src.dataio.writers import save_parquet, save_text
from src.dataio.schema import (
    align_to_quarter,
    finalize_schema,
    STANDARD_OUT_COLS,
)

DEFAULT_CONFIG = {
    "raw_dir": "data/raw",
    "processed_dir": "data/processed",
    "quarter_where": "end",         # "end" or "start"
    "dropna_label": True,           # 是否丢弃没有 t+1 的末期样本
    "write_combined": True,         # 是否写全集 all_investors.parquet
}

def load_cfg(cfg_path: str | None) -> dict:
    if not cfg_path:
        return DEFAULT_CONFIG.copy()
    p = Path(cfg_path)
    if not p.exists():
        return DEFAULT_CONFIG.copy()
    if yaml is None and p.suffix in {".yml", ".yaml"}:
        raise RuntimeError("未安装 pyyaml，无法读取 yaml 配置")
    with p.open("r", encoding="utf-8") as f:
        return (yaml.safe_load(f) if p.suffix in {".yml", ".yaml"} else json.load(f))

def generate_panel(df_raw: pd.DataFrame, quarter_where: str = "end", dropna_label: bool = True) -> pd.DataFrame:
    """
    基础转换：对齐季度、排序去重、生成 holding_t/holding_t1、裁剪列并返回标准面板
    """
    df = df_raw.copy()

    # 统一到季度
    df["date"] = align_to_quarter(df["fdate"], where=quarter_where)

    # 排序去重
    df = (df
          .sort_values(["mgrno", "permno", "date"])
          .drop_duplicates(subset=["mgrno", "permno", "date"], keep="first"))

    # 生成标签
    df["holding_t"]  = df["holding"]
    df["holding_t1"] = df.groupby(["mgrno", "permno"], sort=False)["holding_t"].shift(-1)

    # 基础清洗（可按需调整）
    # 负持仓改为 0（如果你认定数据不应为负）
    # df.loc[df["holding_t"]  < 0, "holding_t"]  = 0
    # df.loc[df["holding_t1"] < 0, "holding_t1"] = 0

    # 选择标准列顺序
    df = finalize_schema(df)

    # 丢末期无标签样本（可选）
    if dropna_label:
        before = len(df)
        df = df.dropna(subset=["holding_t1"]).reset_index(drop=True)
        print(f"[prepare] drop rows without holding_t1: {before - len(df)}")

    return df

def split_by_type_and_save(df_panel: pd.DataFrame, processed_dir: Path) -> list[Path]:
    """
    按 type 切成多份 parquet；返回写出的路径列表
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if "type" not in df_panel.columns:
        # 没有类型列，则全部写到 all_investors
        path = save_parquet(df_panel, processed_dir / "all_investors.parquet")
        paths.append(path)
        return paths

    types = (
        df_panel["type"]
        .astype("string")
        .fillna("Unknown")
        .str.strip()
    )
    unique_types = sorted(types.unique().tolist())

    for t in unique_types:
        sub = df_panel[types == t]
        # 文件名安全化
        fname = t.lower().replace(" ", "_")
        path = save_parquet(sub, processed_dir / f"{fname}.parquet")
        paths.append(path)
        print(f"[prepare] {t}: {len(sub):,} rows -> {path}")

    # 同时写一个全集
    path_all = save_parquet(df_panel, processed_dir / "all_investors.parquet")
    paths.append(path_all)
    return paths

def main():
    ap = argparse.ArgumentParser(description="Prepare panel data from raw yearly parquet files.")
    ap.add_argument("--config", type=str, default="configs/data.yaml", help="可选：配置文件路径（yaml/json）")
    ap.add_argument("--raw-dir", type=str, help="覆盖配置：原始数据目录")
    ap.add_argument("--processed-dir", type=str, help="覆盖配置：输出目录")
    ap.add_argument("--quarter", choices=["end", "start"], help="覆盖配置：季度对齐位置")
    ap.add_argument("--keep-nan-label", action="store_true", help="覆盖配置：保留无 holding_t1 的行")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.raw_dir:        cfg["raw_dir"] = args.raw_dir
    if args.processed_dir:  cfg["processed_dir"] = args.processed_dir
    if args.quarter:        cfg["quarter_where"] = args.quarter
    if args.keep_nan_label: cfg["dropna_label"] = False

    raw_dir = Path(cfg["raw_dir"])
    processed_dir = Path(cfg["processed_dir"])

    print(f"[prepare] load raw from {raw_dir}")
    # 只读可能用到的列（能省内存；若列名不同可在这里扩充）
    cols = ["mgrno","permno","fdate","holding","me","be","profit","Gat","beta",
            "prc","shares","type","aum","outaum"]
    df_raw = load_raw_years(raw_dir, usecols=[c for c in cols if (raw_dir / "dummy").exists() or True])  # 保守：内部会忽略不存在的列

    print(f"[prepare] build panel (align={cfg['quarter_where']}, dropna_label={cfg['dropna_label']})")
    df_panel = generate_panel(df_raw, quarter_where=cfg["quarter_where"], dropna_label=cfg["dropna_label"])

    print("[prepare] split by type & save to processed/")
    out_paths = split_by_type_and_save(df_panel, processed_dir=processed_dir)

    # 简单统计
    overview = {
        "rows_total": int(len(df_panel)),
        "types": [p.stem for p in out_paths if p.stem not in {"all_investors"}],
        "processed_dir": str(processed_dir),
        "columns": list(df_panel.columns),
    }
    save_text(json.dumps(overview, ensure_ascii=False, indent=2), Path("artifacts/stats/overview.json"))
    print(f"[prepare] done. total rows={overview['rows_total']:,}")

if __name__ == "__main__":
    main()
