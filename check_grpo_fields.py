import json
import pathlib

TYPES = [
    "banks",
    "mutual_funds",
    "investment_advisors",
    "insurance_companies",
    "pension_funds",
    "other",
]

# Values considered missing/NA for stock-level fields
NA = {None, "NA", "", "nan"}


def check_file(path: pathlib.Path):
    """Return total count and per-field missing stats for a GRPO jsonl file."""
    miss = {
        "label_delta": 0,
        "holding_log_delta": 0,
        "stock_vol": 0,
        "stock_ln_vol": 0,
        "vix": 0,
        "ln_mv": 0,
        "messages": 0,
    }
    tot = 0
    for line in path.open():
        if not line.strip():
            continue
        rec = json.loads(line)
        tot += 1

        if rec.get("label_delta") is None:
            miss["label_delta"] += 1
        if rec.get("holding_log_delta") is None:
            miss["holding_log_delta"] += 1

        cur = rec.get("history_rows", {}).get("t", {}) or {}
        if cur.get("stock_vol_q_prev") in NA:
            miss["stock_vol"] += 1
        if cur.get("stock_ln_volume_q_prev") in NA:
            miss["stock_ln_vol"] += 1

        if rec.get("vix_q_prev") is None:
            miss["vix"] += 1
        if rec.get("ln_market_volume_q_prev") is None:
            miss["ln_mv"] += 1

        if not rec.get("messages"):
            miss["messages"] += 1

    return tot, miss


def main():
    for t in TYPES:
        p = pathlib.Path(f"artifacts/grpo/grpo_{t}.jsonl")
        tot, miss = check_file(p)
        print(
            f"{t}: total={tot} "
            f"label_delta_missing={miss['label_delta']} "
            f"holding_log_delta_missing={miss['holding_log_delta']} "
            f"stock_vol_missing={miss['stock_vol']} "
            f"stock_ln_volume_missing={miss['stock_ln_vol']} "
            f"vix_missing={miss['vix']} "
            f"ln_market_volume_missing={miss['ln_mv']} "
            f"messages_missing={miss['messages']}"
        )


if __name__ == "__main__":
    main()
