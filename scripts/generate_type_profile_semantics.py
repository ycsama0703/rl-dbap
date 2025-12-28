#!/usr/bin/env python
"""
Generate semantic summaries for type-level profiles (one profile per type) from
artifacts/features/type_profiles.csv.

Outputs JSON with fields:
  profile_id, investor_type, profile_k, objective_weights, summary

If --use-llm is provided (with --api-key), summaries are produced by the model;
otherwise a deterministic template summary is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate semantic summaries for type-level profiles.")
    p.add_argument("--weights", required=True, help="type_profiles.csv or parquet with risk_aversion/herd_behavior/profit_driven")
    p.add_argument("--out-json", required=True, help="Output JSON path")
    p.add_argument("--use-llm", action="store_true", help="If set, call LLM for summaries (requires --api-key)")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=256)
    return p.parse_args()


def read_weights(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    required = {"type", "risk_aversion", "herd_behavior", "profit_driven"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"weights file missing columns: {missing}")
    df["type"] = df["type"].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    df["profile_k"] = 0  # single profile per type
    return df


def build_prompt(row: pd.Series) -> str:
    return (
        "You are an analyst. Summarize this type's portfolio objective preferences in two short sentences. "
        "Avoid any numbers; use qualitative words (e.g., strongest, moderate, weakest). "
        "Only talk about risk_aversion, herd_behavior, profit_driven.\n\n"
        f"Type: {row['type']}\n"
        f"risk_aversion weight: {row['risk_aversion']:.4f}\n"
        f"herd_behavior weight: {row['herd_behavior']:.4f}\n"
        f"profit_driven weight: {row['profit_driven']:.4f}\n"
    )


def call_llm(prompt: str, api_key: str, model: str, temperature: float, max_tokens: int) -> str:
    if OpenAI is None:
        raise ImportError("openai package not installed; pip install openai to use --use-llm")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def deterministic_summary(row: pd.Series) -> str:
    items = [
        ("risk_aversion", row["risk_aversion"]),
        ("herd_behavior", row["herd_behavior"]),
        ("profit_driven", row["profit_driven"]),
    ]
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
    strongest, mid, weakest = sorted_items
    return (
        f"{row['type']} tilts most toward {strongest[0].replace('_', ' ')}, "
        f"has moderate emphasis on {mid[0].replace('_', ' ')}, "
        f"and the lowest weight on {weakest[0].replace('_', ' ')}."
    )


def main():
    args = parse_args()
    df = read_weights(Path(args.weights))

    out_profiles: List[Dict] = []

    for _, row in df.iterrows():
        if args.use_llm:
            if not args.api_key:
                raise ValueError("api-key required when --use-llm is set")
            summary = call_llm(build_prompt(row), args.api_key, args.model, args.temperature, args.max_tokens)
        else:
            summary = deterministic_summary(row)

        out_profiles.append(
            {
                "profile_id": f"{row['type']}_p0",
                "investor_type": row["type"],
                "profile_k": int(row["profile_k"]),
                "objective_weights": {
                    "risk_aversion": float(row["risk_aversion"]),
                    "herd_behavior": float(row["herd_behavior"]),
                    "profit_driven": float(row["profit_driven"]),
                },
                "summary": summary,
            }
        )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_profiles, ensure_ascii=False, indent=2))
    print(f"[ok] wrote {len(out_profiles)} profiles -> {out_path}")


if __name__ == "__main__":
    main()
