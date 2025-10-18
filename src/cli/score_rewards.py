from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, List

import re

# Import SWIFT registries
from swift.plugin.orm import orms, ORM


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _extract_answer_body(text: str | None) -> str | None:
    if not text:
        return None
    m = ANSWER_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _maybe_load_external_plugins(paths: list[str] | None) -> None:
    if not paths:
        return
    for p in paths:
        pth = Path(p)
        if not pth.exists():
            print(f"[warn] external plugin not found: {pth}", file=sys.stderr)
            continue
        spec = importlib.util.spec_from_file_location(pth.stem, pth)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
        else:
            print(f"[warn] cannot import plugin: {pth}", file=sys.stderr)


def _resolve_completions(
    ds: list[dict[str, Any]],
    completions_rows: list[dict[str, Any]] | None,
    completion_field: str,
    strict_answer_only: bool,
) -> list[str]:
    comps: list[str] = []
    if completions_rows is not None:
        if len(completions_rows) < len(ds):
            print(f"[warn] completions rows ({len(completions_rows)}) < dataset rows ({len(ds)}); will truncate",
                  file=sys.stderr)
        for row in completions_rows[: len(ds)]:
            # Heuristics: allow nested fields like a.b.c
            cur: Any = row
            for key in completion_field.split('.'):
                if isinstance(cur, dict) and key in cur:
                    cur = cur[key]
                else:
                    cur = row.get('completion') or row.get('text') or row.get('response') or row.get('output')
                    break
            if isinstance(cur, list):
                cur = cur[0] if cur else ''
            comp = str(cur) if cur is not None else ''
            if strict_answer_only:
                comp = _extract_answer_body(comp) or ''
            comps.append(comp)
    else:
        raise SystemExit("--completions is required for now. Provide a JSONL with a 'completion' field.")
    return comps


def _batch_call(orm: ORM, completions: list[str], kwargs_list: list[dict[str, Any]], batch_size: int = 64) -> list[float]:
    scores: list[float] = []
    n = len(completions)
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        batch_comps = completions[i:j]
        # merge kwargs by turning scalar columns into lists of aligned values
        merged_kwargs: dict[str, Any] = {}
        keys = set().union(*[kw.keys() for kw in kwargs_list[i:j]])
        for k in keys:
            col = [kwargs_list[t].get(k) for t in range(i, j)]
            merged_kwargs[k] = col
        try:
            scores.extend(list(orm(batch_comps, **merged_kwargs)))
        except TypeError:
            # Some ORMs accept fewer kwargs; try removing unknown keys progressively
            filtered = {
                k: v
                for k, v in merged_kwargs.items()
                if k in orm.__call__.__code__.co_varnames  # type: ignore[attr-defined]
            }
            scores.extend(list(orm(batch_comps, **filtered)))
        i = j
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description="Score model completions against GRPO reward functions")
    ap.add_argument('--dataset', type=str, required=True, help='GRPO dataset jsonl (with messages + labels)')
    ap.add_argument('--completions', type=str, required=True, help='JSONL file with model completions aligned to dataset')
    ap.add_argument('--completion_field', type=str, default='completion', help='Field name for completion in JSONL')
    ap.add_argument('--external_plugins', type=str, nargs='*', default=['src/plugins/grpo/holdings_plugin.py'],
                    help='Paths to external plugin files to register custom ORMs')
    ap.add_argument('--reward_funcs', type=str, nargs='+', required=True,
                    help='Reward function names, e.g. contract_holdings external_holdings format')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--strict_answer_only', action='store_true',
                    help='Extract and score only the <answer>...</answer> body (recommended for strict checking)')
    args = ap.parse_args()

    _maybe_load_external_plugins(args.external_plugins)

    ds_path = Path(args.dataset)
    comp_path = Path(args.completions)
    ds_rows = _load_jsonl(ds_path)
    comp_rows = _load_jsonl(comp_path)
    if args.limit:
        ds_rows = ds_rows[: args.limit]
        comp_rows = comp_rows[: args.limit]

    completions = _resolve_completions(ds_rows, comp_rows, args.completion_field, args.strict_answer_only)

    # Prepare kwargs per row (透传字段)
    kwargs_list: list[dict[str, Any]] = []
    for r in ds_rows:
        kw = {k: v for k, v in r.items() if k not in ('messages',)}
        kwargs_list.append(kw)

    # Build reward instances
    reward_insts: list[tuple[str, ORM]] = []
    for name in args.reward_funcs:
        if name not in orms:
            print(f"[error] reward function '{name}' not found in registry", file=sys.stderr)
            print(f"Available: {sorted(orms.keys())[:50]}...", file=sys.stderr)
            raise SystemExit(2)
        reward_insts.append((name, orms[name]()))

    # Score
    print(f"Scoring {len(completions)} samples against rewards: {', '.join(args.reward_funcs)}")
    all_scores: dict[str, list[float]] = {}
    for name, inst in reward_insts:
        scores = _batch_call(inst, completions, kwargs_list)
        all_scores[name] = scores
        nz = sum(1 for s in scores if s != 0.0)
        mu = mean(scores) if scores else 0.0
        sd = pstdev(scores) if len(scores) > 1 else 0.0
        print(f"- {name}: mean={mu:.4f} std={sd:.4f} nonzero={nz}/{len(scores)}")

    # Optional: show a few failures for debugging
    def _short(s: str, n: int = 160) -> str:
        s = s.replace('\n', ' ')
        return s[:n] + ('…' if len(s) > n else '')

    print("\nExamples with zero reward (up to 5 per reward):")
    for name, scores in all_scores.items():
        idxs = [i for i, sc in enumerate(scores) if sc == 0.0][:5]
        print(f"\n{name}:")
        for i in idxs:
            print(f"  - idx={i} completion[short]=" + _short(completions[i]))


if __name__ == '__main__':
    main()

