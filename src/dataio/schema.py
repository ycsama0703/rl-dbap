"""
数据列契约 & 常用变换（对齐到季度、重命名因子、基本校验）
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

# 你原始数据里的列名 -> 统一列名（需要就改）
FACTOR_RENAME_MAP = {
    "me": "factor1",
    "be": "factor2",
    "profit": "factor3",
    "Gat": "factor4",
    "beta": "factor5",
}

REQUIRED_MIN_COLS = ["mgrno", "permno", "fdate", "holding"]  # 原始最小必需
STANDARD_FACTOR_COLS = ["factor1", "factor2", "factor3", "factor4", "factor5"]
OPTIONAL_COLS = ["prc", "shares", "type", "aum","outaum"]

STANDARD_OUT_COLS = (
    ["mgrno", "permno", "date", "holding_t", "holding_t1"]
    + STANDARD_FACTOR_COLS
    + OPTIONAL_COLS
)

def align_to_quarter(dt: pd.Series, where: str = "end") -> pd.Series:
    """
    把日期对齐到季度末/季度初，并统一转为 UTC timestamp
    """
    q = dt.dt.to_period("Q")
    ts = q.dt.asfreq("Q", where).dt.to_timestamp()
    return ts.dt.tz_localize("UTC")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """重命名因子列，保持 mgrno/permno 类型稳定"""
    print("[DEBUG] standardize_columns in:", df.columns.tolist())

    df = df.copy()
    # 统一因子名，但保留 aum/outaum 不动
    for old, new in FACTOR_RENAME_MAP.items():
        if old in df.columns and new not in df.columns and old not in ("aum", "outaum"):
            df.rename(columns={old: new}, inplace=True)

    # mgrno/permno 统一为整型或字符串之一（这里不强转为 int，避免 NaN/大整型溢出）
    for key in ("mgrno", "permno"):
        if key in df.columns:
            try:
                df[key] = pd.to_numeric(df[key], errors="ignore")
            except Exception:
                pass

    print("[DEBUG] standardize_columns out:", df.columns.tolist())
    # 仅在 aum/outaum 存在时打印
    if "aum" in df.columns and "outaum" in df.columns:
        print("[DEBUG] sample aum/outaum:\n", df[["aum","outaum"]].head())

    return df


def validate_min_cols(df: pd.DataFrame) -> None:
    miss = [c for c in REQUIRED_MIN_COLS if c not in df.columns]
    if miss:
        raise KeyError(f"缺少必要列: {miss}；现有列示例: {list(df.columns)[:20]}")

def finalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """补全缺失列并按标准顺序输出"""
    out = df.copy()
    for c in STANDARD_OUT_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[STANDARD_OUT_COLS]
