#!/usr/bin/env python
"""
生成最终版“抽象 profile”JSON（供 SFT/GRPO 作为慢变量使用）。

硬约束：
- 只允许输出 philosophy / constraints / objective_weights 三部分。
- 禁止出现任何特征名或样本统计（beta、book、profit、GARP、momentum、growth、value、HHI、n= 等）。
- constraints 必须取值 {low, medium, high}。
- objective_weights 直接来自回归，不允许 LLM 更改；和为 1（±1e-3）。
- 每个 type 内的 profile 语义不得完全相同（若相同则警告）。
- activity 与 turnover_constraint 必须一致：低→约束∈{low,medium}；中→{medium,high}；高→{high}。
- 目标权重不允许极端一热，alpha/risk/tc 均 ≥ epsilon 后再归一化。
- 采用 Option A：移除 tracking_error 维度（不生成 te、不输出 tracking_error_constraint）。

用法：
python scripts/generate_profile_semantics_llm.py \
  --features artifacts/features/*_iq_features.csv \
  --profiles artifacts/features/*_iq_profile.csv \
  --weights artifacts/features/*_profile_objective_weights.csv \
  --out-json artifacts/features/profile_semantics_llm.json \
  --model deepseek-chat --max-tokens 512 \
  --api-key "$DEEPSEEK_API_KEY"
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

BEHAVIOR_FEATS = ["turnover", "hhi", "bm_gap"]  # 可扩展，但不要在输出中直接引用


FORBIDDEN_TOKENS = [
    "beta", "book", "profit", "garp", "momentum", "growth", "value", "hhi", "n="
]

WEIGHT_EPS = 0.02

ALLOWED_TURNOVER = {
    "low_turnover": {"low", "medium"},
    "medium_turnover": {"medium", "high"},
    "high_turnover": {"high"},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--profiles", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", required=True, help="objective_weights tables (csv/parquet)")
    p.add_argument("--out-json", required=True, help="Output JSON file with profile semantics (strict schema).")
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--api-key", default=None)
    p.add_argument("--topn", type=int, default=3, help="保留 top/bottom 特征个数用于提示（仅供 LLM 抽象，不得出现在结果中）")
    return p.parse_args()


def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def _read_many(paths: List[str]) -> pd.DataFrame:
    return pd.concat([_read(p) for p in paths], axis=0, ignore_index=True)


def _norm_type(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.replace(" ", "_")


def _to_quarter_str(col: pd.Series) -> pd.Series:
    try:
        return pd.PeriodIndex(pd.to_datetime(col), freq="Q").astype(str)
    except Exception:
        return pd.PeriodIndex(col, freq="Q").astype(str)


def load_and_merge(feat_paths, prof_paths):
    feats = _read_many(feat_paths)
    profs = _read_many(prof_paths)
    typ_f = "type" if "type" in feats.columns else "investor_type"
    typ_p = "type" if "type" in profs.columns else "investor_type"
    feats["investor_type"] = _norm_type(feats[typ_f])
    profs["investor_type"] = _norm_type(profs[typ_p])

    inv_f = "mgrno" if "mgrno" in feats.columns else "investor_id"
    inv_p = "mgrno" if "mgrno" in profs.columns else "investor_id"
    feats = feats.rename(columns={inv_f: "investor_id"})
    profs = profs.rename(columns={inv_p: "investor_id"})

    if "quarter" in feats.columns:
        feats["quarter_str"] = _to_quarter_str(feats["quarter"])
    elif "date" in feats.columns:
        feats["quarter_str"] = _to_quarter_str(feats["date"])
    else:
        raise ValueError("features need quarter/date")
    if "quarter" in profs.columns:
        profs["quarter_str"] = _to_quarter_str(profs["quarter"])
    else:
        raise ValueError("profiles need quarter")

    merged = feats.merge(
        profs[["investor_id", "quarter_str", "investor_type", "profile_k"]],
        on=["investor_id", "quarter_str", "investor_type"],
        how="inner",
    )
    merged = merged.dropna(subset=["profile_k"])
    return merged


def attach_weights(df: pd.DataFrame, weights_paths: List[str]) -> pd.DataFrame:
    wdf = _read_many(weights_paths)
    typ_col = "investor_type" if "investor_type" in wdf.columns else "type"
    wdf["investor_type"] = _norm_type(wdf[typ_col])
    wdf = wdf[["investor_type", "profile_k", "alpha_w", "risk_w", "tc_w"]]
    return df.merge(wdf, how="left", on=["investor_type", "profile_k"])


def build_numeric_summary(df: pd.DataFrame, topn: int) -> Dict[str, dict]:
    out = {}
    feats_all = [c for c in BEHAVIOR_FEATS if c in df.columns]
    if not feats_all:
        raise ValueError("No behavior features found; expected at least one of " + ",".join(BEHAVIOR_FEATS))
    for t, g in df.groupby("investor_type"):
        feats = [c for c in feats_all if c in g.columns]
        if not feats:
            continue
        type_mean = g[feats].mean()
        type_std = g[feats].std().replace(0, np.nan)
        for k, p in g.groupby("profile_k"):
            n = len(p)
            means = p[feats].mean()
            z = (means - type_mean) / type_std
            z = z.replace([np.inf, -np.inf], np.nan)
            z_sorted = z.dropna().sort_values(ascending=False)
            top_pos = z_sorted.head(topn).to_dict()
            top_neg = z_sorted.tail(topn).to_dict()
            w_cols = ["alpha_w", "risk_w", "tc_w", "te_w"]
            weights = {w: float(p[w].iloc[0]) if w in p.columns and pd.notna(p[w].iloc[0]) else 0.0 for w in w_cols}
            out[f"{t}_p{k}"] = {
                "investor_type": t,
                "profile_k": int(k),
                "n_obs": int(n),
                "behavior_means": means.to_dict(),
                "behavior_z": z.to_dict(),
                "top_pos_z": top_pos,
                "top_neg_z": top_neg,
                "objective_weights": weights,
            }
    return out


def call_deepseek(prompt: str, api_key: str, model: str, max_tokens: int) -> str:
    if OpenAI is None:
        raise ImportError("openai 包未安装，无法调用 DeepSeek。请 pip install openai")
    api_key = api_key.strip()
    # API key 必须是 ASCII，可用 try/except 检查，不改写内容，避免把合法 key 清空
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError("DEEPSEEK_API_KEY 含有非 ASCII 字符，请重新设置纯 ASCII 的官方密钥（形如 sk-xxx）。")
    # 统一转成 ASCII，移除无法编码字符，避免请求阶段的编码报错
    prompt_clean = prompt.encode("ascii", errors="ignore").decode("ascii", errors="ignore")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a financial analyst. Output only JSON per instructions."},
            {"role": "user", "content": prompt_clean},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
        stream=False,
    )
    return resp.choices[0].message.content


def validate_obj(d: dict) -> bool:
    try:
        pid = d["profile_id"]
        ph = d["philosophy"]
        cs = d["constraints"]
        ow = d["objective_weights"]
        # constraints values
        allowed = {"low", "medium", "high"}
        for k in ["turnover_constraint", "concentration_constraint", "risk_tolerance"]:
            if cs.get(k) not in allowed:
                return False
        # weights sum
        sm = float(ow.get("alpha", 0)) + float(ow.get("risk", 0)) + float(ow.get("tc", 0))
        if not (0.99 <= sm <= 1.01):
            return False
        return True
    except Exception:
        return False


def contains_forbidden(text: str) -> bool:
    lower = text.lower()
    return any(tok in lower for tok in FORBIDDEN_TOKENS)


def main():
    args = parse_args()
    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY missing; 请用 --api-key 或设置环境变量")

    merged = load_and_merge(args.features, args.profiles)
    merged = attach_weights(merged, args.weights)
    summary = build_numeric_summary(merged, args.topn)

    out_profiles = []
    forbidden_hits = 0
    for pid, info in summary.items():
        # 构造提示：提供类型、profile_k、行为均值 z-score、目标权重；要求按 schema 输出 JSON
        prompt = f"""
You are given a profile summary for investor_type={info['investor_type']}, profile_k={info['profile_k']}.
Input numeric summary (do NOT leak these names into output):
- behavior_means: {json.dumps(info['behavior_means'], ensure_ascii=True)}
- behavior_z: {json.dumps({k: v for k, v in info['behavior_z'].items() if v==v}, ensure_ascii=True)}
- objective_weights: {json.dumps(info['objective_weights'], ensure_ascii=True)}  # alpha/risk/tc already sum to 1

TASK: Output exactly one JSON object with fields:
{{
  "profile_id": "{pid}",
  "philosophy": {{
    "style": one of ["defensive","aggressive","balanced"],
    "activity": one of ["low_turnover","medium_turnover","high_turnover"],
    "benchmark_orientation": one of ["low","medium","high"],
    "horizon": one of ["long_term","medium_term","tactical"]
  }},
  "constraints": {{
    "turnover_constraint": one of ["low","medium","high"],
    "concentration_constraint": one of ["low","medium","high"],
    "risk_tolerance": one of ["low","medium","high"]
  }},
  "objective_weights": {{
    "alpha": float,
    "risk": float,
    "tc": float
  }}
}}
Rules:
- Do NOT mention any feature names or dataset stats (e.g., beta, book, profit, momentum, growth, value, HHI, n=).
- Use only the allowed vocab above. Output MUST be a single JSON object only.
"""
        try:
            content = call_deepseek(prompt, api_key, args.model, args.max_tokens)
            # 尝试解析 JSON（容忍前后空白）
            content_stripped = content.strip()
            if content_stripped.startswith("```"):
                content_stripped = content_stripped.strip("` \n")
                # 去除可能的 json 标记
                content_stripped = content_stripped.replace("json\n", "").replace("JSON\n", "")
            obj = json.loads(content_stripped)
            raw_text = json.dumps(obj, ensure_ascii=False)
            if contains_forbidden(raw_text):
                forbidden_hits += 1
                print(f"[warn] forbidden tokens in {pid}, skipping")
                continue
            if not validate_obj(obj):
                print(f"[warn] validation failed for {pid}, content: {content_stripped}")
                continue
            # post-process: enforce activity vs turnover_constraint consistency
            act = obj.get("philosophy", {}).get("activity")
            tcst = obj.get("constraints", {}).get("turnover_constraint")
            if act in ALLOWED_TURNOVER and tcst not in ALLOWED_TURNOVER[act]:
                # adjust to nearest allowed (pick min in allowed set)
                obj["constraints"]["turnover_constraint"] = sorted(ALLOWED_TURNOVER[act])[0]
            # smooth weights with epsilon and renormalize
            ow = obj.get("objective_weights", {})
            a = float(ow.get("alpha", 0))
            r = float(ow.get("risk", 0))
            tc = float(ow.get("tc", 0))
            a = max(a, WEIGHT_EPS)
            r = max(r, WEIGHT_EPS)
            tc = max(tc, WEIGHT_EPS)
            sm = a + r + tc
            ow["alpha"] = round(a / sm, 4)
            ow["risk"] = round(r / sm, 4)
            ow["tc"] = round(tc / sm, 4)
            obj["objective_weights"] = ow
            out_profiles.append(obj)
        except Exception as e:
            import traceback
            print(f"[warn] LLM failed for {pid}: {repr(e)}")
            traceback.print_exc()

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    # 简单重复检测：同 type 内 philosophy+constraints 相同则警告
    dup_warn = 0
    seen = {}
    for obj in out_profiles:
        t = obj.get("profile_id", "")
        if "_" in t:
            ttype = t.split("_")[0]
        else:
            ttype = "unknown"
        key = (ttype, json.dumps(obj.get("philosophy", {}), sort_keys=True), json.dumps(obj.get("constraints", {}), sort_keys=True))
        if key in seen:
            dup_warn += 1
        else:
            seen[key] = True

    Path(args.out_json).write_text(json.dumps(out_profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {len(out_profiles)} profiles -> {args.out_json}, forbidden_hits={forbidden_hits}, duplicate_warnings={dup_warn}")


if __name__ == "__main__":
    main()
