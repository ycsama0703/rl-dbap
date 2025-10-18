#!/usr/bin/env bash
set -euo pipefail

# One-shot pipeline: per-investor-type SFT -> GRPO
# Usage:
#   bash scripts/run_per_type.sh -t <TYPE> \
#     -m <MODEL> \
#     [-sft_end YYYY-MM-DD] [-grpo_start YYYY-MM-DD] [-grpo_end YYYY-MM-DD] \
#     [-g <NUM_GENERATIONS>] [-l <MAX_COMPLETION_LEN>] \
#     [--per_type_limit 1000] [--time_bins 10] [--cap_per_pair 3]

TYPE=""
MODEL="Qwen/Qwen2.5-7B-Instruct"
SFT_END="2016-12-31"
GRPO_START="2017-01-01"
GRPO_END="2018-12-31"
NUM_GENERATIONS=4
MAX_COMPLETION_LEN=512
PER_TYPE_LIMIT=1000
TIME_BINS=10
CAP_PER_PAIR=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--type) TYPE="$2"; shift 2;;
    -m|--model) MODEL="$2"; shift 2;;
    -sft_end) SFT_END="$2"; shift 2;;
    -grpo_start) GRPO_START="$2"; shift 2;;
    -grpo_end) GRPO_END="$2"; shift 2;;
    -g) NUM_GENERATIONS="$2"; shift 2;;
    -l) MAX_COMPLETION_LEN="$2"; shift 2;;
    --per_type_limit) PER_TYPE_LIMIT="$2"; shift 2;;
    --time_bins) TIME_BINS="$2"; shift 2;;
    --cap_per_pair) CAP_PER_PAIR="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$TYPE" ]]; then
  echo "Error: -t <TYPE> is required (e.g., banks)"; exit 1
fi

IN_DIR="data/processed/panel_quarter.parquet"

# SFT stage (per-type)
SFT_PROMPTS_DIR="artifacts/prompts_hist_sft_${TYPE}"
SFT_JSONL="artifacts/sft/sft_train_${TYPE}.jsonl"
SFT_OUT="outputs/sft_${TYPE}"

echo "[pipeline] Building SFT prompts for type=${TYPE} until ${SFT_END} ..."
python -m src.cli.build_history_prompts \
  --in-dir "$IN_DIR" \
  --out-dir "$SFT_PROMPTS_DIR" \
  --include-types "$TYPE" \
  --date-end "$SFT_END" \
  --per-type-limit "$PER_TYPE_LIMIT" \
  --time-bins "$TIME_BINS" \
  --cap-per-pair "$CAP_PER_PAIR"

echo "[pipeline] Converting SFT prompts -> ${SFT_JSONL} ..."
python -m src.cli.prompts_to_sft --in "$SFT_PROMPTS_DIR/${TYPE}.jsonl" --out "$SFT_JSONL"

echo "[pipeline] Running SFT training -> ${SFT_OUT} ..."
bash "$(dirname "$0")/sft.sh" -m "$MODEL" -d "$SFT_JSONL" -o "$SFT_OUT"

# GRPO stage (per-type)
GRPO_PROMPTS_DIR="artifacts/prompts_hist_grpo_${TYPE}"
GRPO_JSONL="artifacts/grpo/grpo_${TYPE}.jsonl"
GRPO_OUT="outputs/grpo_${TYPE}"

echo "[pipeline] Building GRPO prompts for type=${TYPE} in [${GRPO_START}, ${GRPO_END}] ..."
python -m src.cli.build_history_prompts \
  --in-dir "$IN_DIR" \
  --out-dir "$GRPO_PROMPTS_DIR" \
  --include-types "$TYPE" \
  --date-start "$GRPO_START" \
  --date-end "$GRPO_END" \
  --per-type-limit "$PER_TYPE_LIMIT" \
  --time-bins "$TIME_BINS" \
  --cap-per-pair "$CAP_PER_PAIR"

echo "[pipeline] Converting GRPO prompts -> ${GRPO_JSONL} ..."
python -m src.cli.prompts_to_grpo --in "$GRPO_PROMPTS_DIR/${TYPE}.jsonl" --out "$GRPO_JSONL"

echo "[pipeline] Running GRPO training from SFT adapters -> ${GRPO_OUT} ..."
bash "$(dirname "$0")/grpo.sh" -m "$MODEL" -d "$GRPO_JSONL" -o "$GRPO_OUT" -g "$NUM_GENERATIONS" -l "$MAX_COMPLETION_LEN" -a "$SFT_OUT"

echo "[pipeline] Done. SFT=${SFT_OUT}  GRPO=${GRPO_OUT}"

