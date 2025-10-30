# RL-DBAP Pipeline (Qwen3-8B)

This README只保留最新流程，按顺序执行即可（bash 環境）。假定當前目錄為 `/workspace/rl-dbap`。

## 0. 基礎環境
```bash
export PYTHONPATH=.
```

## 1. 生成 2000 條 Prompt
```bash
python -m src.cli.build_history_prompts \
  --in-dir data/processed/panel_quarter.parquet \
  --out-dir artifacts/prompts_hist_sft \
  --include-types banks \
  --per-type-limit 2000 \
  --date-end 2016-12-31 \
  --exclude-zero-holding-t

python -m src.cli.build_history_prompts \
  --in-dir data/processed/panel_quarter.parquet \
  --out-dir artifacts/prompts_hist_grpo \
  --include-types banks \
  --per-type-limit 2000 \
  --date-start 2017-01-01 \
  --date-end 2018-12-31 \
  --exclude-zero-holding-t

python -m src.cli.build_history_prompts \
  --in-dir data/processed/panel_quarter.parquet \
  --out-dir artifacts/prompts_hist_test \
  --include-types banks \
  --per-type-limit 2000 \
  --date-start 2019-01-01 \
  --exclude-zero-holding-t
```

## 2. 轉換為 SFT / GRPO / Test 數據
```bash
python -m src.cli.prompts_to_sft \
  --in artifacts/prompts_hist_sft/banks.jsonl \
  --out artifacts/sft/sft_train_banks.jsonl

python -m src.cli.prompts_to_grpo \
  --in artifacts/prompts_hist_grpo/banks.jsonl \
  --out artifacts/grpo/grpo_banks.jsonl

python -m src.cli.prompts_to_sft \
  --in artifacts/prompts_hist_test/banks.jsonl \
  --out artifacts/test/test_banks.jsonl \
  --contract-mode absolute \
  --no-think
```

## 3. SFT 訓練（Qwen/Qwen3-8B-Instruct）
```bash
bash scripts/sft.sh \
  -m "Qwen/Qwen3-8B-Instruct" \
  -d artifacts/sft/sft_train_banks.jsonl \
  -o outputs/sft_banks_qwen3_8b \
  -- \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --lora_rank 16 \
  --lora_alpha 64 \
  --num_train_epochs 4
```
完成後記下最新 checkpoint，例如 `outputs/sft_banks_qwen3_8b/checkpoint-500`。

## 4. GRPO 訓練
```bash
bash scripts/grpo.sh \
  -m "Qwen/Qwen3-8B-Instruct" \
  -d artifacts/grpo/grpo_banks.jsonl \
  -o outputs/grpo_banks_qwen3_8b \
  -a outputs/sft_banks_qwen3_8b/checkpoint-500 \
  -S "</answer>"
```
Reward 權重已在腳本中設為格式:數值 = 0.3 : 0.7。

## 5. 評測
```bash
# GRPO 後模型
python -m src.cli.run_eval \
  --test_path artifacts/test/test_banks.jsonl \
  --base_model Qwen/Qwen3-8B-Instruct \
  --lora_path outputs/grpo_banks_qwen3_8b/checkpoint-500 \
  --out_dir artifacts/eval_grpo_banks_qwen3_8b

# （可選）SFT-only
python -m src.cli.run_eval \
  --test_path artifacts/test/test_banks.jsonl \
  --base_model Qwen/Qwen3-8B-Instruct \
  --lora_path outputs/sft_banks_qwen3_8b/checkpoint-500 \
  --out_dir artifacts/eval_sft_banks_qwen3_8b

# （可選）原始模型
python -m src.cli.run_eval \
  --test_path artifacts/test/test_banks.jsonl \
  --base_model Qwen/Qwen3-8B-Instruct \
  --lora_path None \
  --out_dir artifacts/eval_base_banks_qwen3_8b
```
評測輸出包含 `pred_detail.csv`、`metrics.csv` 及圖表。

## 6. 抽樣檢查輸出（可選）
```bash
python scripts/compare_base_vs_lora.py \
  --test-path artifacts/test/test_banks.jsonl \
  --base-model Qwen/Qwen3-8B-Instruct \
  --lora-path outputs/grpo_banks_qwen3_8b/checkpoint-500 \
  --limit 20 \
  --max-new-tokens 128 \
  --out-csv outputs/banks_base_vs_grpo_qwen3_8b.csv

python scripts/inspect_eval_outputs.py \
  --test-path artifacts/test/test_banks.jsonl \
  --base-model Qwen/Qwen3-8B-Instruct \
  --lora-path outputs/grpo_banks_qwen3_8b/checkpoint-500 \
  --limit 20 \
  --max-new-tokens 128 \
  --out-jsonl outputs/banks_grpo_samples_qwen3_8b.jsonl
```

以上命令涵蓋了最新 prompt 模板、SFT/GRPO 訓練與評測流程。如需重新開始，只要按順序重跑即可。
