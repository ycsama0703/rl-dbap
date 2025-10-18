#!/usr/bin/env bash
set -euo pipefail

# Minimal per-type training pipeline (data already prepared)
# Assumes the following files exist:
#   artifacts/sft/sft_train_<TYPE>.jsonl
#   artifacts/grpo/grpo_<TYPE>.jsonl
# Runs: SFT -> GRPO (GRPO initialized from SFT adapters)

# Usage:
#   bash scripts/train_per_type.sh -t <TYPE> -m <MODEL> \
#       [-sft_json artifacts/sft/sft_train_<TYPE>.jsonl] \
#       [-grpo_json artifacts/grpo/grpo_<TYPE>.jsonl] \
#       [-g <NUM_GENERATIONS>] [-l <MAX_COMPLETION_LEN>]

TYPE=""
MODEL="Qwen/Qwen2.5-7B-Instruct"
SFT_JSON=""
GRPO_JSON=""
NUM_GENERATIONS=4
MAX_COMPLETION_LEN=512

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--type) TYPE="$2"; shift 2;;
    -m|--model) MODEL="$2"; shift 2;;
    -sft_json) SFT_JSON="$2"; shift 2;;
    -grpo_json) GRPO_JSON="$2"; shift 2;;
    -g) NUM_GENERATIONS="$2"; shift 2;;
    -l) MAX_COMPLETION_LEN="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$TYPE" ]]; then
  echo "Error: -t <TYPE> is required (e.g., banks)"; exit 1
fi

if [[ -z "$SFT_JSON" ]]; then
  SFT_JSON="artifacts/sft/sft_train_${TYPE}.jsonl"
fi
if [[ -z "$GRPO_JSON" ]]; then
  GRPO_JSON="artifacts/grpo/grpo_${TYPE}.jsonl"
fi

if [[ ! -f "$SFT_JSON" ]]; then
  echo "SFT dataset not found: $SFT_JSON"; exit 1
fi
if [[ ! -f "$GRPO_JSON" ]]; then
  echo "GRPO dataset not found: $GRPO_JSON"; exit 1
fi

SFT_OUT="outputs/sft_${TYPE}"
GRPO_OUT="outputs/grpo_${TYPE}"

echo "[train-only] SFT: model=${MODEL} dataset=${SFT_JSON} -> ${SFT_OUT}"
bash "$(dirname "$0")/sft.sh" -m "$MODEL" -d "$SFT_JSON" -o "$SFT_OUT"

echo "[train-only] GRPO: model=${MODEL} dataset=${GRPO_JSON} adapters=${SFT_OUT} -> ${GRPO_OUT}"
bash "$(dirname "$0")/grpo.sh" -m "$MODEL" -d "$GRPO_JSON" -o "$GRPO_OUT" -g "$NUM_GENERATIONS" -l "$MAX_COMPLETION_LEN" -a "$SFT_OUT"

echo "[train-only] Done. SFT=$SFT_OUT  GRPO=$GRPO_OUT"

