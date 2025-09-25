# src/cli/build_prompts.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd


def _get(row: pd.Series, name: str):
    """Safe getter for a row field"""
    return row[name] if name in row.index else None


def build_structured_prompt(row: pd.Series) -> dict:
    """
    Build one English structured prompt with original factor names + AUM/outAUM.
    Returns dict: {"prompt": str, "label": float}
    """

    # ---- try original factor names; fallback to factor1..5 ----
    me      = _get(row, "me")      or _get(row, "factor1")
    be      = _get(row, "be")      or _get(row, "factor2")
    profit  = _get(row, "profit")  or _get(row, "factor3")
    Gat     = _get(row, "Gat")     or _get(row, "factor4")
    beta    = _get(row, "beta")    or _get(row, "factor5")

    # ---- portfolio scale ----
    aum    = _get(row, "aum")
    outaum = _get(row, "outaum")
    prc    = _get(row, "prc")
    holding_t = float(row.get("holding_t", float("nan")))
    pos_val, pos_pct_of_aum = None, None
    try:
        if pd.notna(holding_t) and prc is not None and pd.notna(prc):
            pos_val = holding_t * float(prc)
        if pos_val is not None and aum is not None and pd.notna(aum) and float(aum) != 0:
            pos_pct_of_aum = pos_val / float(aum)
    except Exception:
        pass

    investor_role = _get(row, "type") or "Unknown"
    mgrno = row.get("mgrno", "NA")
    permno = row.get("permno", "NA")
    date_t = row.get("date", "NA")
    quarter = row.get("quarter")

    def fmt(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "NA"
        try:
            return f"{float(x):.6g}"
        except Exception:
            return str(x)

    prompt = f"""
Act as a quantitative portfolio manager at a {investor_role} institution.

Goal
- Predict the next-quarter absolute holding `holding_(t+1)`.
- Output **valid JSON only** with a single field `holding_tp1` (it can be 0). No explanation.

Identifiers & Timeline
- Investor: investor_id (mgrno) = {mgrno}
- Stock: permno = {permno}
- Timeline: (t) {str(date_t)}{f"  (quarter {quarter})" if quarter else ""}

Recent realized holdings & features
- Recent holding (t): {fmt(holding_t)}
- Fundamentals (t):
  - me (market equity) = {fmt(me)}
  - be (book equity)   = {fmt(be)}
  - profit             = {fmt(profit)}
  - Gat                = {fmt(Gat)}
  - beta               = {fmt(beta)}
- Portfolio scale:
  - AUM (total)        = {fmt(aum)}
  - outAUM (ex-stock)  = {fmt(outaum)}
  - position value (holding_t * price) = {fmt(pos_val)}
  - position / AUM     = {fmt(pos_pct_of_aum)}

Constraints & Guidance
- Non-negativity: `holding_(t+1) ≥ 0`.
- Directional intuition (soft):
  - If profit increases or beta decreases ⇒ `holding_(t+1)` tends to be higher than `holding_(t)`.
  - If profit decreases or beta increases ⇒ `holding_(t+1)` tends to be lower.
- Consider portfolio scale information (AUM and outAUM) when proposing a plausible magnitude.

Output (valid JSON only)
{{"holding_tp1": <float>}}
""".strip()

    label = None
    if "holding_t1" in row.index and pd.notna(row["holding_t1"]):
        label = float(row["holding_t1"])

    return {"prompt": prompt, "label": label}


def main():
    ap = argparse.ArgumentParser(description="Build one prompt with factor names + AUM/outAUM.")
    ap.add_argument("--in-file", type=str,
                    default="data/processed/panel_quarter.parquet/banks.parquet",
                    help="Input parquet for one investor type.")
    ap.add_argument("--out-file", type=str,
                    default="artifacts/prompts/test.jsonl",
                    help="Output JSONL (one record).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling one row.")
    args = ap.parse_args()

    in_path = Path(args.in_file)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    row = df.sample(1, random_state=args.seed).iloc[0]
    rec = build_structured_prompt(row)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Wrote 1 prompt with label to {out_path}")


if __name__ == "__main__":
    main()
