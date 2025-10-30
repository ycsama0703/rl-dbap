from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import pandas as pd


@dataclass
class PromptRow:
    mgrno: str | int
    permno: str | int
    investor_type: str
    # current t snapshot
    holding_t: Optional[float]
    holding_t1: Optional[float]
    me: Optional[float] = None
    be: Optional[float] = None
    profit: Optional[float] = None
    Gat: Optional[float] = None
    beta: Optional[float] = None
    aum: Optional[float] = None
    outaum: Optional[float] = None
    prc: Optional[float] = None
    date: Optional[str] = None


def _fmt(x):
    if x is None:
        return "NA"
    try:
        v = float(x)
        if pd.isna(v):
            return "NA"
        # decimal formatting: up to 6 decimals, no scientific notation
        s = f"{v:.6f}".rstrip('0').rstrip('.')
        if s == "-0":
            s = "0"
        return s
    except Exception:
        return str(x)


def build_history_prompt(
    hist_rows: Sequence[PromptRow],
    hide_date: bool = True,
    target: str = "delta",  # "delta" or "absolute"
    strict_contract: bool = True,
) -> tuple[str, dict]:
    """Build a prompt like the figure: t-3, t-2, t-1 history + new data.

    Returns (prompt_text, extras_dict) where extras includes fields used for labels.
    hist_rows: length >= 4, order [t-3, t-2, t-1, t]
    """
    assert len(hist_rows) >= 4, "need at least 4 rows: t-3..t"
    r_tm3, r_tm2, r_tm1, r_t = hist_rows[-4], hist_rows[-3], hist_rows[-2], hist_rows[-1]

    role_raw = r_t.investor_type or "Unknown"
    role = role_raw.strip() if isinstance(role_raw, str) else str(role_raw)
    role_phrase = role.lower()
    if role_phrase.endswith("s"):
        role_phrase = role_phrase[:-1]
    mgrno = r_t.mgrno
    permno = r_t.permno

    def block(title: str, r: PromptRow) -> str:
        return (
            f"{title}\n"
            f"me={_fmt(r.me)}, be={_fmt(r.be)}, profit={_fmt(r.profit)}, Gat={_fmt(r.Gat)}, beta={_fmt(r.beta)}\n"
            f"aum={_fmt(r.aum)}, outAUM={_fmt(r.outaum)}, holding={_fmt(r.holding_t)}"
            + (f", price={_fmt(r.prc)}" if r.prc is not None else "")
        )

    # history changes (if available)
    def change(prev: PromptRow, curr: PromptRow) -> str:
        try:
            if prev.holding_t is None or curr.holding_t is None:
                return "NA"
            if pd.isna(prev.holding_t) or pd.isna(curr.holding_t):
                return "NA"
            diff = float(curr.holding_t) - float(prev.holding_t)
            s = f"{diff:.2f}".rstrip('0').rstrip('.')
            if s == "-0":
                s = "0"
            return s
        except Exception:
            return "NA"

    if strict_contract:
        header = (
            f"Act as a {role_phrase} portfolio manager.\n\n"
            f"Your goal is to adjust your portfolio holdings for {{ticker}} according to the change in fundamental data.\n"
            f"\"holding_delta\" means the ABSOLUTE change in holdings (same units as holding).\n"
            f"Formula: holding_delta = holding_tp1 - holding_t.\n"
            f"Direction is given by the sign of holding_delta; lower bound holding_delta ≥ -holding_t.\n\n"
        )
    else:
        header = (
            f"You are a {role}.\n\n"
            f"Your goal is to adjust your portfolio holdings for {{ticker}} according to the change in fundamental data.\n\n"
        )

    hist = (
        f"{{t-3 data}}\n{block('', r_tm3)}\nholding_delta: {change(r_tm3, r_tm2)}\n\n"
        f"{{t-2 data}}\n{block('', r_tm2)}\nholding_delta: {change(r_tm2, r_tm1)}\n\n"
        f"{{t-1 data}}\n{block('', r_tm1)}\nholding_delta: {change(r_tm1, r_t)}\n\n"
    )

    now_block = f"Now, these are the new data:\n{block('{t data}', r_t)}\n\n"

    if strict_contract:
        instructions = (
            "OUTPUT FORMAT & REASONING GUIDE\n\n"
            "You MUST follow the exact output structure below.\n\n"
            "1. Begin with a single reasoning block enclosed in <think>...</think>.\n"
            "   - Purpose: explain numerically and briefly how the given fundamentals justify your final holding adjustment.\n"
            "   - Keep it concise (<= 3 sentences, numeric precision 2 decimals).\n"
            "   - Example style: <think>me↓ (-15%), profit↑ (+0.02) → moderate increase in holding.</think>\n\n"
            "2. Immediately after </think>, output a single <answer>...</answer> block.\n"
            "   - Inside <answer>, output exactly ONE valid JSON object:\n"
            "     {\"holding_delta\": <float>}\n"
            "   - The float must have 2 decimal places, no scientific notation.\n"
            "   - Example: <answer>{\"holding_delta\": 0.31}</answer>\n\n"
            "The final output should look EXACTLY like this example (structure only):\n\n"
            "<think>...</think>\n"
            "<answer>{\"holding_delta\": 0.00}</answer>\n\n"
        )
    else:
        instructions = ""

    prompt = header
    prompt += (f"Ticker: {permno}\n\n" if permno is not None else "")
    prompt += hist + now_block
    prompt += instructions

    extras = {
        "mgrno": mgrno,
        "permno": permno,
        "holding_t": r_t.holding_t,
        "label_tp1": (float(r_t.holding_t1) if r_t.holding_t1 is not None and not pd.isna(r_t.holding_t1) else None),
        "label_delta": None,
    }
    if extras["holding_t"] is not None and extras["label_tp1"] is not None:
        try:
            extras["label_delta"] = float(extras["label_tp1"]) - float(extras["holding_t"])  # type: ignore
        except Exception:
            extras["label_delta"] = None

    return prompt, extras
