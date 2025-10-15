"""
通用写出：parquet / jsonl / 统计信息
"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def _json_default(o):
    """Make numpy/pandas scalar types JSON-serializable.
    Falls back to str for any remaining unsupported objects.
    """
    try:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        # pandas NA/NaT -> None
        if o is pd.NaT:
            return None
        if isinstance(o, (pd.Timestamp,)):
            # ISO string to avoid timezone issues
            return o.isoformat()
    except Exception:
        pass
    return str(o)


def save_jsonl(records: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
    return path


def save_text(s: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")
    return path

