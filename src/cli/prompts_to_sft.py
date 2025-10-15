# src/cli/prompts_to_sft.py

from __future__ import annotations
import argparse, json
from pathlib import Path


def _resolve_label(rec: dict) -> float | None:
    """Resolve absolute label for SFT from multiple possible fields.
    Priority:
      1) rec['label'] -> absolute holding_tp1
      2) rec['label_tp1'] -> absolute
      3) rec['label_delta'] + rec['holding_t'] -> absolute = holding_t + delta
    """
    try:
        if rec.get("label") is not None:
            return float(rec["label"])  # absolute
        if rec.get("label_tp1") is not None:
            return float(rec["label_tp1"])  # absolute
        if rec.get("label_delta") is not None and rec.get("holding_t") is not None:
            return float(rec["holding_t"]) + float(rec["label_delta"])  # reconstruct
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, required=True,
                    help="input prompts jsonl or directory with jsonl files")
    ap.add_argument("--out", dest="out", type=str, default="artifacts/sft/sft_train.jsonl")
    ap.add_argument("--system", type=str,
                    default="You are a quantitative portfolio manager. Predict next-quarter absolute holding as valid JSON.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--with-think", action="store_true",
                    help="Add a non-loss assistant <think> message before the JSON answer")
    args = ap.parse_args()

    inp = Path(args.inp)
    files = [inp] if inp.is_file() else sorted(inp.glob("*.jsonl"))
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with outp.open("w", encoding="utf-8") as fout:
        for fp in files:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    prompt = rec.get("prompt") or rec.get("query")
                    if not prompt:
                        continue
                    label_tp1 = _resolve_label(rec)
                    if label_tp1 is None:
                        continue   # need an absolute label
                    resp = json.dumps({"holding_tp1": float(label_tp1)}, ensure_ascii=False)

                    msgs = [
                        {"role": "system", "content": args.system},
                        {"role": "user", "content": prompt},
                    ]
                    if args.with_think:
                        msgs.append({"role": "assistant", "content": "<think>...</think>", "loss": False})
                    msgs.append({"role": "assistant", "content": resp, "loss": True})

                    out = {"messages": msgs}

                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    n += 1
                    if args.limit and n >= args.limit:
                        break
            if args.limit and n >= args.limit:
                break
    print(f"wrote {n} samples -> {outp}")


if __name__ == "__main__":
    main()
