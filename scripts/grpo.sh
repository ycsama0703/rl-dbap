#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/grpo.sh -m <MODEL> -d <DATASET_JSONL> -o <OUTPUT_DIR> [-g <NUM_GENERATIONS>] [-l <MAX_COMPLETION_LEN>] [-v]
#   -v: use vLLM colocate mode

MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET="artifacts/grpo/grpo.jsonl"
OUTPUT_DIR="outputs/grpo_qwen2.5_7b"
NUM_GENERATIONS=4
MAX_COMPLETION_LEN=512
USE_VLLM=0

while getopts ":m:d:o:g:l:v" opt; do
  case ${opt} in
    m) MODEL="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    g) NUM_GENERATIONS="$OPTARG" ;;
    l) MAX_COMPLETION_LEN="$OPTARG" ;;
    v) USE_VLLM=1 ;;
    *) echo "Usage: $0 -m <MODEL> -d <DATASET_JSONL> -o <OUTPUT_DIR> [-g <NUM_GENERATIONS>] [-l <MAX_COMPLETION_LEN>] [-v]" ; exit 1 ;;
  esac
done

EXTRA_VLLM_ARGS=()
if [[ "$USE_VLLM" == "1" ]]; then
  EXTRA_VLLM_ARGS+=(--use_vllm true --vllm_mode colocate)
fi

swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL}" \
  --external_plugins src/plugins/grpo/holdings_plugin.py \
  --reward_funcs contract_holdings external_holdings format \
  --train_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --dataset "${DATASET}" \
  --load_from_cache_file true \
  --max_completion_length "${MAX_COMPLETION_LEN}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-6 \
  --gradient_accumulation_steps 8 \
  --logging_steps 5 \
  --save_steps 100 \
  --save_total_limit 2 \
  --max_length 2048 \
  --output_dir "${OUTPUT_DIR}" \
  --warmup_ratio 0.05 \
  --dataset_num_proc 2 \
  --num_generations "${NUM_GENERATIONS}" \
  --temperature 0.9 \
  --beta 0.04 \
  --log_completions true \
  "${EXTRA_VLLM_ARGS[@]}"

