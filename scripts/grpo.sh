#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/grpo.sh -m <MODEL> -d <DATASET_JSONL> -o <OUTPUT_DIR> \
#        [-a <SFT_ADAPTER_DIR>] [-g <NUM_GENERATIONS>] [-l <MAX_COMPLETION_LEN>] [-r <CKPT_DIR>] [-v] \
#        [-b <PER_DEVICE_BATCH>] [-A <GRAD_ACCUM>] [-R <LORA_RANK>] [-L <LORA_ALPHA>] \
#        [-S "</answer>"|"}"] [-T 0.9]
#   -a: initialize from existing SFT LoRA adapters (e.g., outputs/sft_*)
#   -r: resume GRPO from a checkpoint dir (e.g., outputs/grpo_*/checkpoint-1000)
#   -v: use vLLM colocate mode
#   -F: reward funcs (space-separated, order matters)
#   -W: reward weights (space-separated, align with -F)
#   -S: stop words (single string)
#   -T: temperature (float)

MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET="artifacts/grpo/grpo.jsonl"
OUTPUT_DIR="outputs/grpo_qwen2.5_7b"
NUM_GENERATIONS=4
MAX_COMPLETION_LEN=512
USE_VLLM=0
ADAPTERS=""
RESUME_FROM=""
TEMPERATURE=0.9
STOP_WORDS=""
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUM_STEPS=4
LORA_RANK=16
LORA_ALPHA=64
# 固定使用格式/数值奖励组合
REWARD_FUNCS=(contract_holdings mse_holdings)
REWARD_WEIGHTS=(0.4 0.6)

while getopts ":m:d:o:g:l:a:r:vS:T:b:A:R:L:" opt; do
  case ${opt} in
    m) MODEL="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    g) NUM_GENERATIONS="$OPTARG" ;;
    l) MAX_COMPLETION_LEN="$OPTARG" ;;
    a) ADAPTERS="$OPTARG" ;;
    r) RESUME_FROM="$OPTARG" ;;
    v) USE_VLLM=1 ;;
    S) STOP_WORDS="$OPTARG" ;;
    T) TEMPERATURE="$OPTARG" ;;
    b) PER_DEVICE_TRAIN_BATCH_SIZE="$OPTARG" ;;
    A) GRADIENT_ACCUM_STEPS="$OPTARG" ;;
    R) LORA_RANK="$OPTARG" ;;
    L) LORA_ALPHA="$OPTARG" ;;
    *) echo "Usage: $0 -m <MODEL> -d <DATASET_JSONL> -o <OUTPUT_DIR> [-a <SFT_ADAPTER_DIR>] [-g <NUM_GENERATIONS>] [-l <MAX_COMPLETION_LEN>] [-r <CKPT_DIR>] [-v] [-b <PER_DEVICE_BATCH>] [-A <GRAD_ACCUM>] [-R <LORA_RANK>] [-L <LORA_ALPHA>] [-S <stop words>] [-T <temperature>]" ; exit 1 ;;
  esac
done

EXTRA_VLLM_ARGS=()
if [[ "$USE_VLLM" == "1" ]]; then
  EXTRA_VLLM_ARGS+=(--use_vllm true --vllm_mode colocate)
fi

ADAPTER_ARGS=()
if [[ -n "$ADAPTERS" ]]; then
  ADAPTER_ARGS+=(--adapters "$ADAPTERS")
fi

RESUME_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint "$RESUME_FROM")
fi

# Auto-detect latest checkpoint if none provided and output dir has checkpoints
if [[ -z "$RESUME_FROM" ]]; then
  latest_ckpt=""
  if compgen -G "${OUTPUT_DIR}/checkpoint-*" > /dev/null; then
    latest_ckpt=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sed -E 's/.*checkpoint-([0-9]+)/\1 \0/' | sort -k1,1n | awk '{print $2}' | tail -1)
  fi
  if [[ -n "$latest_ckpt" ]]; then
    RESUME_ARGS=(--resume_from_checkpoint "$latest_ckpt")
  fi
fi

swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL}" \
  --external_plugins src/plugins/grpo/holdings_plugin.py \
  --reward_funcs ${REWARD_FUNCS[@]} \
  --train_type lora \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --dataset "${DATASET}" \
  --load_from_cache_file true \
  --max_completion_length "${MAX_COMPLETION_LEN}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --learning_rate 1e-6 \
  --gradient_accumulation_steps "${GRADIENT_ACCUM_STEPS}" \
  --logging_steps 5 \
  --save_steps 100 \
  --save_total_limit 2 \
  --max_length 2048 \
  --output_dir "${OUTPUT_DIR}" \
  --warmup_ratio 0.05 \
  --dataset_num_proc 2 \
  --num_generations "${NUM_GENERATIONS}" \
  --temperature "${TEMPERATURE}" \
  --beta 0.02 \
  --log_completions true \
  --reward_weights ${REWARD_WEIGHTS[@]} \
  $( [[ -n "$STOP_WORDS" ]] && echo --stop_words "$STOP_WORDS" ) \
  "${EXTRA_VLLM_ARGS[@]}" \
  "${ADAPTER_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
