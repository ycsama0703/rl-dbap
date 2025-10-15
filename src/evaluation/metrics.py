from __future__ import annotations
import numpy as np
import pandas as pd


def basic_regression(df: pd.DataFrame):
    """Compute basic regression metrics on a valid df with columns y_true, y_pred.
    Returns: MAE, RMSE, R2, sMAPE%, IC(Spearman), RankIC(Spearman on ranks)
    """
    y = df["y_true"].to_numpy()
    p = df["y_pred"].to_numpy()
    mae = float(np.mean(np.abs(y - p))) if len(y) else np.nan
    rmse = float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else np.nan
    # R2 (naive)
    if len(y) > 1:
        ss_res = np.sum((y - p) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
        r2 = 1 - ss_res / ss_tot
    else:
        r2 = np.nan
    # sMAPE in %
    smape = float(200.0 * np.mean(np.abs(y - p) / (np.abs(y) + np.abs(p) + 1e-12))) if len(y) else np.nan
    # IC (Spearman)
    try:
        ic = float(pd.Series(y).corr(pd.Series(p), method="spearman"))
    except Exception:
        ic = np.nan
    # RankIC
    try:
        ry = pd.Series(y).rank(method="average")
        rp = pd.Series(p).rank(method="average")
        ric = float(ry.corr(rp, method="spearman"))
    except Exception:
        ric = np.nan
    return mae, rmse, r2, smape, ic, ric


def topk(df: pd.DataFrame, group_col: str, k: int = 50):
    """Compute Recall/Precision/NDCG@k by group (e.g., quarter).
    Ground truth relevance by y_true descending; predictions by y_pred descending.
    """
    if df.empty:
        return np.nan, np.nan, np.nan
    recalls, precisions, ndcgs = [], [], []
    for _, g in df.groupby(group_col):
        g = g.dropna(subset=["y_true", "y_pred"]) 
        if g.empty:
            continue
        # ideal top-k by true
        ideal = g.sort_values("y_true", ascending=False).head(k)
        # predicted top-k by pred
        predk = g.sort_values("y_pred", ascending=False).head(k)
        rel_set = set(ideal.index)
        hit = sum(1 for idx in predk.index if idx in rel_set)
        recall = hit / max(len(ideal), 1)
        precision = hit / max(len(predk), 1)
        # NDCG (binary relevance)
        def dcg(n):
            return sum(1.0 / np.log2(i + 2) for i in range(n))
        dcg_val = sum((1.0 / np.log2(i + 2)) for i, idx in enumerate(predk.index) if idx in rel_set)
        idcg_val = dcg(min(len(ideal), len(predk)))
        ndcg = dcg_val / idcg_val if idcg_val > 0 else 0.0
        recalls.append(recall); precisions.append(precision); ndcgs.append(ndcg)
    if not recalls:
        return np.nan, np.nan, np.nan
    return float(np.mean(recalls)), float(np.mean(precisions)), float(np.mean(ndcgs))

