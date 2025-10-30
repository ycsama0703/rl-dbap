# RL-DBAP Pipeline (Qwen2.5-7B)

Step-by-step commands to regenerate prompts, train SFT/GRPO, and run evaluation with `Qwen/Qwen2.5-7B-Instruct`. Run everything from the project root `/workspace/rl-dbap` in **bash**.

## 0. Environment
```bash
export PYTHONPATH=.
```

## 1. Generate 2000 Prompts
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

## 2. Convert to SFT / GRPO / Test Chat Data
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

## 3. SFT Training (`Qwen/Qwen2.5-7B-Instruct`)
```bash
bash scripts/sft.sh \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -d artifacts/sft/sft_train_banks.jsonl \
  -o outputs/sft_banks_qwen2p5 \
  -- \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --lora_rank 16 \
  --lora_alpha 64 \
  --num_train_epochs 4
```
Record the latest checkpoint (e.g. `outputs/sft_banks_qwen2p5/checkpoint-500`).

## 4. GRPO Training (format:numeric = 0.3 : 0.7)
```bash
bash scripts/grpo.sh \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -d artifacts/grpo/grpo_banks.jsonl \
  -o outputs/grpo_banks_qwen2p5 \
  -a outputs/sft_banks_qwen2p5/checkpoint-500 \
  -S "</answer>"
```

## 5. Evaluation
```bash
# GRPO model
python -m src.cli.run_eval \
  --test_path artifacts/test/test_banks.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path outputs/grpo_banks_qwen2p5/checkpoint-500 \
  --out_dir artifacts/eval_grpo_banks_qwen2p5

# (Optional) SFT-only
python -m src.cli.run_eval \
  --test_path artifacts/test/test_banks.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path outputs/sft_banks_qwen2p5/checkpoint-500 \
  --out_dir artifacts/eval_sft_banks_qwen2p5

# (Optional) Base model
python -m src.cli.run_eval \
  --test_path artifacts/test/test_banks.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path None \
  --out_dir artifacts/eval_base_banks_qwen2p5
```

## 6. Sample Outputs (Optional)
```bash
python scripts/compare_base_vs_lora.py \
  --test-path artifacts/test/test_banks.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --lora-path outputs/grpo_banks_qwen2p5/checkpoint-500 \
  --limit 20 \
  --max-new-tokens 128 \
  --out-csv outputs/banks_base_vs_grpo_qwen2p5.csv

python scripts/inspect_eval_outputs.py \
  --test-path artifacts/test/test_banks.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --lora-path outputs/grpo_banks_qwen2p5/checkpoint-500 \
  --limit 20 \
  --max-new-tokens 128 \
  --out-jsonl outputs/banks_grpo_samples_qwen2p5.jsonl
```

以上命令涵盖更新后的 prompt 模板、SFT/GRPO 训练与评测流程。需要重新开始时按顺序执行即可。
