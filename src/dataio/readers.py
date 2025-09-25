"""
读取 raw 分年 parquet；也可以读取 processed 的合并结果
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .schema import validate_min_cols, standardize_columns

def load_raw_years(raw_dir: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """
    扫描 raw_dir 下的所有 .parquet，选择性读列合并。
    强制使用 fastparquet 引擎，避免 pyarrow 兼容性问题。
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"未在 {raw_dir} 发现 .parquet")

    dfs = []
    for fp in files:
        if usecols:
            df = pd.read_parquet(fp, columns=usecols, engine="fastparquet")
        else:
            df = pd.read_parquet(fp, engine="fastparquet")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True, sort=False)

    # 基础校验 & 轻量标准化
    validate_min_cols(df_all)
    df_all = standardize_columns(df_all)
    return df_all
