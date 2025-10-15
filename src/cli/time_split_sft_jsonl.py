# src/cli/time_split_sft_jsonl.py
import argparse, json, re, sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

RG_MGRNO  = re.compile(r"mgrno[^\n=]*=\s*([0-9]+(?:\.[0-9]+)?)", re.I)
RG_PERMNO = re.compile(r"permno\s*=\s*([0-9]+)", re.I)
RG_DATE   = re.compile(r"Timeline:\s*\(t\)\s*([0-9]{4}-[0-9]{2}-[0-9]{2}[^\n]*)", re.I)

SYS_MSG = "You are a quantitative portfolio manager. Predict next-quarter absolute holding as valid JSON."

def norm_mgr(x: str) -> str:
    try:
        v = float(x)
        return str(int(v)) if v.is_integer() else str(v)
    except Exception:
        return x

def parse_one(rec):
    """从一条 {prompt,label} 解析 mgrno/permno/date_t"""
    p = rec.get("prompt","")
    m_mgr = RG_MGRNO.search(p)
    m_per = RG_PERMNO.search(p)
    m_dt  = RG_DATE.search(p)

    mgr   = norm_mgr(m_mgr.group(1)) if m_mgr else None
    perm  = m_per.group(1) if m_per else None
    date  = m_dt.group(1).strip() if m_dt else None

    # 解析 datetime 作排序
    dt = None
    if date:
        try:
            date_clean = date.split("+")[0].strip()
            dt = datetime.fromisoformat(date_clean)
        except Exception:
            pass
    return mgr, perm, date, dt

def to_chat(rec):
    """把 {prompt,label} 转 chat 格式 messages"""
    lbl = rec.get("label", 0)
    return {
        "messages": [
            {"role":"system","content": SYS_MSG},
            {"role":"user","content": rec["prompt"]},
            {"role":"assistant","content": json.dumps({"holding_tp1": float(lbl)}, ensure_ascii=False), "loss": True},
        ]
    }

def load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            yield json.loads(line)

def save_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="输入 jsonl，含 {prompt,label}")
    ap.add_argument("--train_out", required=True)
    ap.add_argument("--val_out", required=True)
    ap.add_argument("--test_out", required=False, help="三段切分时必填（global_3cutoff）")

    ap.add_argument("--mode", choices=["global_cutoff","per_group_tail","global_3cutoff"], required=True)

    ap.add_argument("--cutoff", help="mode=global_cutoff 时的日期，如 2016-12-31")
    ap.add_argument("--per_group_k", type=int, default=1, help="mode=per_group_tail 时每组尾部留 K 个到验证")
    ap.add_argument("--min_series_len", type=int, default=1, help="每个 (mgrno,permno) 最小序列长度，不足则全进训练")

    ap.add_argument("--train_end", help="mode=global_3cutoff 时训练集结束日期 (YYYY-MM-DD)")
    ap.add_argument("--val_end", help="mode=global_3cutoff 时验证集结束日期 (YYYY-MM-DD)")

    args = ap.parse_args()

    src = Path(args.inp)
    items = []
    for rec in load_jsonl(src):
        mgr, perm, date_str, dt = parse_one(rec)
        if not (mgr and perm and dt):
            continue
        items.append({
            "mgrno": mgr,
            "permno": perm,
            "date_t": date_str,
            "dt": dt,
            "raw": rec
        })

    groups = defaultdict(list)
    for it in items:
        groups[(it["mgrno"], it["permno"])].append(it)

    train, val = [], []

    if args.mode == "global_cutoff":
        if not args.cutoff:
            print("--cutoff 必填（例如 2016-12-31）", file=sys.stderr); sys.exit(1)
        cutoff_dt = datetime.fromisoformat(args.cutoff+" 23:59:59")
        for gkey, rows in groups.items():
            rows.sort(key=lambda x: x["dt"])
            for r in rows:
                (train if r["dt"] <= cutoff_dt else val).append(to_chat(r["raw"]))

        save_jsonl(Path(args.train_out), train)
        save_jsonl(Path(args.val_out),   val)
        print(f"[OK] train={len(train)}  val={len(val)}  groups={len(groups)}  mode={args.mode}")
        return

    elif args.mode == "per_group_tail":
        K = max(1, args.per_group_k)
        for gkey, rows in groups.items():
            rows.sort(key=lambda x: x["dt"])
            if len(rows) < args.min_series_len:
                train.extend(to_chat(r["raw"]) for r in rows)
                continue
            if len(rows) <= K:
                val.extend(to_chat(r["raw"]) for r in rows)
            else:
                train.extend(to_chat(r["raw"]) for r in rows[:-K])
                val.extend(to_chat(r["raw"]) for r in rows[-K:])

        save_jsonl(Path(args.train_out), train)
        save_jsonl(Path(args.val_out),   val)
        print(f"[OK] train={len(train)}  val={len(val)}  groups={len(groups)}  mode={args.mode}")
        return

    elif args.mode == "global_3cutoff":
        if not (args.train_end and args.val_end and args.test_out):
            print("--train_end / --val_end / --test_out 都是 mode=global_3cutoff 必填", file=sys.stderr)
            sys.exit(1)
        train_end_dt = datetime.fromisoformat(args.train_end + " 23:59:59")
        val_end_dt   = datetime.fromisoformat(args.val_end   + " 23:59:59")

        test = []
        for gkey, rows in groups.items():
            rows.sort(key=lambda x: x["dt"])
            for r in rows:
                if r["dt"] <= train_end_dt:
                    train.append(to_chat(r["raw"]))
                elif r["dt"] <= val_end_dt:
                    val.append(to_chat(r["raw"]))
                else:
                    test.append(to_chat(r["raw"]))

        save_jsonl(Path(args.train_out), train)
        save_jsonl(Path(args.val_out),   val)
        save_jsonl(Path(args.test_out),  test)
        print(f"[OK] train={len(train)}  val={len(val)}  test={len(test)}  groups={len(groups)}  mode={args.mode}")
        return

    else:
        print(f"未知 mode: {args.mode}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

