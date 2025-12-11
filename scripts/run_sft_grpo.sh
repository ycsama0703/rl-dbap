#!/usr/bin/env bash
# Sequential SFT -> GRPO runner.
# 1) Runs SFT.
# 2) Picks the latest SFT checkpoint.
# 3) Runs GRPO initialized from that checkpoint.
#
# You can override the default commands by exporting SFT_CMD and GRPO_CMD.
# Make sure you are in the repo root before running: bash scripts/run_sft_grpo.sh

set -euo pipefail

# -------- Config: adjust as needed --------
SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR:-outputs/sft_mutual_funds}
GRPO_OUTPUT_DIR=${GRPO_OUTPUT_DIR:-outputs/grpo_mutual_funds}

# Default training commands (edit to your needs or override via env).
# We reuse the provided sft.sh / grpo.sh wrappers (swift).
SFT_CMD=${SFT_CMD:-"bash scripts/sft.sh -m Qwen/Qwen2.5-7B-Instruct -d artifacts/sft/sft_train_mutual_funds.jsonl -o ${SFT_OUTPUT_DIR} -- --num_train_epochs 1"}
GRPO_CMD_PREFIX=${GRPO_CMD_PREFIX:-"bash scripts/grpo.sh -m Qwen/Qwen2.5-7B-Instruct -d artifacts/grpo/grpo_mutual_funds.jsonl -o ${GRPO_OUTPUT_DIR} -g 4 -l 512 -b 2 -A 4"}
# ------------------------------------------

echo "[run] SFT -> GRPO pipeline start"

echo "[run] Step 1: SFT training"
echo "[run] cmd: ${SFT_CMD}"
eval "${SFT_CMD}"

echo "[run] Step 2: locate latest SFT checkpoint"
SFT_CKPT=$(ls -td ${SFT_OUTPUT_DIR}/v*/checkpoint-* 2>/dev/null | head -n1 || true)
if [[ -z "${SFT_CKPT}" ]]; then
  echo "[error] no SFT checkpoint found under ${SFT_OUTPUT_DIR}"
  exit 1
fi
echo "[run] using SFT checkpoint: ${SFT_CKPT}"

echo "[run] Step 3: GRPO training (init from SFT checkpoint)"
GRPO_CMD="${GRPO_CMD_PREFIX} --init-model ${SFT_CKPT}"
echo "[run] cmd: ${GRPO_CMD}"
eval "${GRPO_CMD}"

echo "[run] done"
