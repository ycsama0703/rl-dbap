import json
import math
import re
from typing import List

from swift.plugin.orm import ORM, orms

import numpy as np


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
THINK_CONTENT_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)


def _extract_answer_body(text: str) -> str | None:
    m = ANSWER_RE.search(text or "")
    if not m:
        return None
    return m.group(1).strip()


def _extract_json_from_answer(text: str):
    """Try to extract a JSON object either from <answer>...</answer> or anywhere in text.

    This makes the reward robust when the model does not emit <answer> tags yet.
    """
    def _try_load(s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    # 1) Prefer content inside <answer> tags
    body = _extract_answer_body(text)
    candidates: list[str] = []
    if isinstance(body, str) and body:
        # strip surrounding code fences if present
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\n|```$", "", body.strip())
        candidates.append(cleaned)
        # substring between first { and last }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(cleaned[start:end + 1])

    # 2) Fallback: search whole completion
    whole = text or ""
    whole_cleaned = re.sub(r"^```[a-zA-Z0-9]*\n|```$", "", whole.strip())
    start2 = whole_cleaned.find("{")
    end2 = whole_cleaned.rfind("}")
    if start2 != -1 and end2 != -1 and end2 > start2:
        candidates.append(whole_cleaned[start2:end2 + 1])

    # Try load in order
    for s in candidates:
        obj = _try_load(s)
        if isinstance(obj, dict):
            return obj
    return None


class ContractHoldingsORM(ORM):
    """Format/number contract for holdings output while allowing reasoning.

    - Extracts the first occurrence of {"holding_delta": <num>} from the <answer> body (tolerates extra text).
    - Numeric value must be a finite decimal with up to 6 decimals.
    - Bounds: holding_delta >= -holding_t (when provided).
    Reward: +1.0 if a valid value is found and passes bounds, else -1.0.
    """

    _DELTA_SEARCH = re.compile(r'"holding_delta"\s*:\s*(-?\d+(?:\.\d{1,6})?)', re.IGNORECASE)

    def __call__(self, completions, holding_t=None, **kwargs) -> List[float]:
        rewards: List[float] = []
        if not isinstance(holding_t, list):
            holding_t = [holding_t] * len(completions)
        for comp, ht in zip(completions, holding_t):
            try:
                text = comp or ""
                lower_text = text.lower()
                if "<answer>" not in lower_text or ("</answer>" in lower_text and "<answer>" not in lower_text):
                    rewards.append(-1.0)
                    continue
                think_match = THINK_CONTENT_RE.search(text)
                if not think_match or not think_match.group(1).strip():
                    rewards.append(-1.0)
                    continue
                answer_pos = lower_text.find("<answer>")
                if answer_pos > -1:
                    prefix = text[:answer_pos]
                    cleaned_prefix = THINK_CONTENT_RE.sub("", prefix).strip()
                    if "assistant" in cleaned_prefix.lower() or "</answer>" in cleaned_prefix.lower():
                        rewards.append(-1.0)
                        continue

                # Accept values inside <answer> or anywhere in completion as fallback
                body = _extract_answer_body(comp) or comp or ""

                m = self._DELTA_SEARCH.search(body)
                if m:
                    val = float(m.group(1))
                    if ht is not None and val < -float(ht):
                        rewards.append(-1.0)
                        continue
                    rewards.append(1.0)
                    continue

                rewards.append(-1.0)
            except Exception:
                rewards.append(-1.0)
        return rewards


class HoldingsDeltaORM(ORM):
    """Composite reward = w_mag * R_mag + w_dir * R_dir.

    - Magnitude reward R_mag: normalized Huber on error e=pred-target with adaptive/robust scale.
    - Direction reward R_dir: sigmoid on scaled pred aligned with sign(target).
    Accepts holding_delta values only.
    """

    def __init__(self):
        self._ema_abs_err = None  # type: float | None
        self._ema_abs_r = None    # type: float | None

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = max(min(x, 20.0), -20.0)
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _huber(e: float, c: float) -> float:
        ae = abs(e)
        if ae <= c:
            return 0.5 * e * e
        return c * (ae - 0.5 * c)

    @staticmethod
    def _robust_scale(vals: list[float], method: str, k: float, eps: float = 1e-8) -> float:
        v = [abs(x) for x in vals if x is not None]
        if not v:
            return eps
        v.sort()
        if method == 'iqr':
            q1 = v[int(0.25 * (len(v) - 1))]
            q3 = v[int(0.75 * (len(v) - 1))]
            iqr = max(q3 - q1, eps)
            return k * iqr
        # mad
        med = v[len(v) // 2]
        mad = [abs(x - med) for x in v]
        mad.sort()
        m = mad[len(mad) // 2]
        return k * max(m, eps)

    def __call__(self, completions, label_delta=None, label_tp1=None, holding_t=None, **kwargs) -> List[float]:
        k_mag = float(kwargs.get('k_mag', 1.5))
        k_dir = float(kwargs.get('k_dir', 1.0))
        ema_lambda = float(kwargs.get('ema_lambda', 0.9))
        alpha = float(kwargs.get('alpha', 5.0))
        margin = float(kwargs.get('margin', 0.0))
        w_mag = float(kwargs.get('w_mag', 0.6))
        w_dir = float(kwargs.get('w_dir', 0.4))
        robust_mode = str(kwargs.get('robust_mode', 'ema'))  # 'ema', 'mad', 'iqr'

        rewards: List[float] = []
        if not isinstance(label_delta, list):
            label_delta = [label_delta] * len(completions)
        if not isinstance(label_tp1, list):
            label_tp1 = [label_tp1] * len(completions)
        if not isinstance(holding_t, list):
            holding_t = [holding_t] * len(completions)

        preds: list[float | None] = []
        targets: list[float | None] = []
        for comp, gt_delta, gt_tp1, ht in zip(completions, label_delta, label_tp1, holding_t):
            pred = None
            try:
                obj = _extract_json_from_answer(comp)
                if isinstance(obj, dict):
                    if obj.get('holding_delta') is not None:
                        pred = float(obj['holding_delta'])
            except Exception:
                pred = None

            tgt = None
            try:
                if gt_delta is not None:
                    tgt = float(gt_delta)
            except Exception:
                tgt = None

            preds.append(pred)
            targets.append(tgt)

        es: list[float] = []
        rs: list[float] = []
        for pred, tgt in zip(preds, targets):
            if pred is None or tgt is None:
                es.append(None)  # type: ignore
                rs.append(None)  # type: ignore
            else:
                es.append(pred - tgt)
                rs.append(tgt)

        eps = 1e-6
        if robust_mode == 'ema':
            def _mean_abs(vs):
                v = [abs(x) for x in vs if x is not None]
                return (sum(v) / len(v)) if v else None

            mean_abs_e = _mean_abs(es)
            mean_abs_r = _mean_abs(rs)
            if mean_abs_e is not None:
                self._ema_abs_err = (
                    float(mean_abs_e) if self._ema_abs_err is None else ema_lambda * float(self._ema_abs_err)
                    + (1 - ema_lambda) * float(mean_abs_e)
                )
            if mean_abs_r is not None:
                self._ema_abs_r = (
                    float(mean_abs_r) if self._ema_abs_r is None else ema_lambda * float(self._ema_abs_r)
                    + (1 - ema_lambda) * float(mean_abs_r)
                )
            c_mag = k_mag * float(self._ema_abs_err if self._ema_abs_err is not None else 1.0)
            c_dir = k_dir * float(self._ema_abs_r if self._ema_abs_r is not None else 1.0)
        else:
            method = 'mad' if robust_mode == 'mad' else 'iqr'
            c_mag = self._robust_scale([x for x in es if x is not None], method, k_mag, eps)
            c_dir = self._robust_scale([x for x in rs if x is not None], method, k_dir, eps)
        c_mag = max(c_mag, eps)
        c_dir = max(c_dir, eps)

        for pred, tgt, e, r_val in zip(preds, targets, es, rs):
            if pred is None or tgt is None or e is None or r_val is None:
                rewards.append(-1.0)
                continue
            sig = math.copysign(math.log1p(abs(e)), e)
            sim01 = 1.0 - min((sig * sig) / 4.0, 1.0)
            r_mag = 2.0 * sim01 - 1.0

            s = (pred / c_dir) * (1.0 if r_val >= 0 else -1.0)
            r_dir = self._sigmoid(alpha * (s - margin))
            rewards.append(float(w_mag * r_mag + w_dir * (2.0 * r_dir - 1.0)))

        return rewards


# register names for --reward_funcs
orms['external_holdings'] = HoldingsDeltaORM
orms['contract_holdings'] = ContractHoldingsORM


class MSEHoldingsORM(ORM):
    """Pure MSE reward on holding_delta (or derived from holding_tp1).

    - Prediction is parsed from JSON in <answer>...</answer> (fallback: anywhere),
      using keys: `holding_delta` or `holding_tp1` with `holding_t`.
    - Target is `label_delta` or (`label_tp1` - `holding_t`).
    - Reward scaled to [-1, +1]. If pred/target is missing, reward = -1.0.
    """

    def __call__(self, completions, label_delta=None, label_tp1=None, holding_t=None, **kwargs) -> List[float]:
        if not isinstance(label_delta, list):
            label_delta = [label_delta] * len(completions)
        if not isinstance(label_tp1, list):
            label_tp1 = [label_tp1] * len(completions)
        if not isinstance(holding_t, list):
            holding_t = [holding_t] * len(completions)

        errors: list[float | None] = []
        for comp, gt_delta, gt_tp1, ht in zip(completions, label_delta, label_tp1, holding_t):
            pred = None
            try:
                obj = _extract_json_from_answer(comp)
                if isinstance(obj, dict):
                    if obj.get('holding_delta') is not None:
                        pred = float(obj['holding_delta'])
            except Exception:
                pred = None

            tgt = None
            try:
                if gt_delta is not None:
                    tgt = float(gt_delta)
            except Exception:
                tgt = None

            if pred is None or tgt is None:
                errors.append(None)
                continue

            errors.append(pred - tgt)

        rewards: List[float] = []
        valid_errors = [abs(e) for e in errors if e is not None]
        if valid_errors:
            scale = float(np.percentile(valid_errors, 95))
        else:
            scale = 1.0
        scale = max(scale, 1e-6)

        for err in errors:
            if err is None:
                rewards.append(-1.0)
                continue
            x = err / scale
            sim01 = 1.0 - min(x * x, 4.0) / 4.0  # clamp when |x| >= 2
            r_mse = 2.0 * sim01 - 1.0
            rewards.append(float(r_mse))

        return rewards


# register: pure MSE reward
orms['mse_holdings'] = MSEHoldingsORM
