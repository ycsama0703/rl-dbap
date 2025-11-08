from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import math
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


LOG_EPS = 1e-6


def _safe_nonneg(x: float | int | None) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
        if pd.isna(v) or v < 0:
            return None
        return v
    except Exception:
        return None


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
        prev_val = _safe_nonneg(prev.holding_t)
        curr_val = _safe_nonneg(curr.holding_t)
        if prev_val is None or curr_val is None:
            return "NA"
        try:
            ratio = math.log((curr_val + LOG_EPS) / (prev_val + LOG_EPS))
        except Exception:
            return "NA"
        s = f"{ratio:.4f}".rstrip('0').rstrip('.')
        if s == "-0":
            s = "0"
        return s

    if strict_contract:
        header = (
            f"Act as a {role_phrase} portfolio manager.\n\n"
            f"Your goal is to adjust your portfolio holdings for {{ticker}} according to the change in fundamental data.\n"
            f"\"holding_log_delta\" means the log change in holdings.\n"
            f"Formula: holding_log_delta = log((holding_tp1 + {LOG_EPS:.0e}) / (holding_t + {LOG_EPS:.0e})).\n"
            f"This smooths scale differences while keeping the adjustment direction.\n\n"
        )
    else:
        header = (
            f"You are a {role}.\n\n"
            f"Your goal is to adjust your portfolio holdings for {{ticker}} according to the change in fundamental data.\n\n"
        )

    hist = (
        f"{{t-3 data}}\n{block('', r_tm3)}\nholding_log_delta: {change(r_tm3, r_tm2)}\n\n"
        f"{{t-2 data}}\n{block('', r_tm2)}\nholding_log_delta: {change(r_tm2, r_tm1)}\n\n"
        f"{{t-1 data}}\n{block('', r_tm1)}\nholding_log_delta: {change(r_tm1, r_t)}\n\n"
    )

    now_block = f"Now, these are the new data:\n{block('{t data}', r_t)}\n\n"

    if strict_contract:
        instructions = (
            "OUTPUT FORMAT & REASONING GUIDE\n\n"
            "1. Begin with a single reasoning block enclosed in <think>...</think>. Keep concise (≤ 3 sentences).\n"
            "2. Then output a single <answer>...</answer> block with exactly one JSON object: "
            '{"holding_delta": <float>}.\n'
            "3. The float must have 2 decimal places, no scientific notation.\n"
            "The final output should look EXACTLY like this example (structure only):\n\n"
            "<think>...</think>\n"
            "<answer>{\"holding_log_delta\": 0.00}</answer>\n\n"
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
        "label_log_delta": None,
        "label_delta_absolute": None,
    }
    if extras["holding_t"] is not None and extras["label_tp1"] is not None:
        try:
            ht = float(extras["holding_t"])  # type: ignore[arg-type]
            tp1 = float(extras["label_tp1"])  # type: ignore[arg-type]
            extras["label_delta_absolute"] = tp1 - ht
            if ht >= 0 and tp1 >= 0:
                log_delta = math.log((tp1 + LOG_EPS) / (ht + LOG_EPS))
                extras["label_log_delta"] = log_delta
                extras["label_delta"] = log_delta
        except Exception:
            extras["label_delta"] = None
            extras["label_log_delta"] = None
            extras["label_delta_absolute"] = None

    return prompt, extras
