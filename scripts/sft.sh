#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/sft.sh -m <MODEL> -d <DATASET_JSONL> -o <OUTPUT_DIR>

MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET="artifacts/sft/sft_train.jsonl"
OUTPUT_DIR="outputs/sft_qwen2.5_7b"
EXTRA_ARGS=()

while getopts ":m:d:o:" opt; do
  case ${opt} in
    m) MODEL="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    *) echo "Usage: $0 -m <MODEL> -d <DATASET_JSONL> -o <OUTPUT_DIR>" ; exit 1 ;;
  esac
done

shift $((OPTIND - 1))
if [[ $# -gt 0 ]]; then
  EXTRA_ARGS=("$@")
fi

swift sft \
  --model "${MODEL}" \
  --train_type lora \
  --dataset "${DATASET}" \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --logging_steps 20 \
  --save_steps 500 \
  --save_total_limit 2 \
  --max_length 2048 \
  --output_dir "${OUTPUT_DIR}" \
  --system "You are a quantitative portfolio manager." \
  "${EXTRA_ARGS[@]}"
