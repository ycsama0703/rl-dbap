from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Convert prompts JSONL to GRPO dataset (messages + labels)")
    ap.add_argument("--in", dest="inp", type=str, required=True,
                    help="input prompts jsonl file or directory of jsonl files")
    ap.add_argument("--out", dest="out", type=str, default="artifacts/grpo/grpo.jsonl")
    ap.add_argument("--system", type=str,
                    default="You are a quantitative portfolio manager. Respond with valid JSON only.")
    ap.add_argument("--limit", type=int, default=None)
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
                    out = {
                        "messages": [
                            {"role": "system", "content": args.system},
                            {"role": "user", "content": prompt},
                        ],
                        # pass-through fields for reward
                        "label_delta": rec.get("label_delta"),
                        "label_tp1": rec.get("label_tp1") or rec.get("label"),
                        "holding_t": rec.get("holding_t"),
                        # optional metadata
                        "mgrno": rec.get("mgrno"),
                        "permno": rec.get("permno"),
                    }
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    n += 1
                    if args.limit and n >= args.limit:
                        break
            if args.limit and n >= args.limit:
                break

    print(f"wrote {n} GRPO samples -> {outp}")


if __name__ == "__main__":
    main()

