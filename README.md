RL-DBAP Workflow (Banks, Qwen2.5-7B)
====================================

All commands below assume a bash shell (WSL or Linux) running in `/workspace/rl-dbap`.

0. Environment
--------------
```bash
export PYTHONPATH=.
# Install core dependencies (includes ms-swift + SwanLab)
pip install -r requirements.txt
```

1. Build Prompt Windows
-----------------------
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

2. Convert Prompts to Chat Datasets
-----------------------------------
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

3. SFT Training (Qwen/Qwen2.5-7B-Instruct)
-----------------------------------------
```bash
bash scripts/sft.sh \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -d artifacts/sft/sft_train_banks.jsonl \
  -o outputs/sft_banks_qwen2p5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --lora_rank 16 \
  --lora_alpha 64 \
  --num_train_epochs 4 \
  --report-to-swanlab \
  --swanlab-project rl-dbap \
  --swanlab-exp-name banks-sft
```
The main checkpoint referenced later is `outputs/sft_banks_qwen2p5/checkpoint-500`.

4. GRPO Training (Reward Mix 0.3 Format / 0.7 Value)
----------------------------------------------------
```bash
bash scripts/grpo.sh \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -d artifacts/grpo/grpo_banks.jsonl \
  -o outputs/grpo_banks_v4 \
  -a outputs/sft_banks_qwen2p5/checkpoint-500 \
  -S "</answer>" \
  --report-to-swanlab \
  --swanlab-project rl-dbap \
  --swanlab-exp-name banks-grpo
```

5. Strict Evaluation (No Fallback Parsing)
-----------------------------------------
Use `scripts/run_eval_strict.py` to avoid legacy parsing paths. Delete old artifacts before each run.

```bash
rm -rf artifacts/eval_grpo_banks_v4_strict
python scripts/run_eval_strict.py \
  --test_path artifacts/test/test_banks.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path outputs/grpo_banks_v4/v0-20251030-062734/checkpoint-500 \
  --out_dir artifacts/eval_grpo_banks_v4_strict

rm -rf artifacts/eval_sft_banks_v4_strict
python scripts/run_eval_strict.py \
  --test_path artifacts/test/test_banks.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path outputs/sft_banks_qwen2p5/checkpoint-500 \
  --out_dir artifacts/eval_sft_banks_v4_strict

rm -rf artifacts/eval_base_banks_v4_strict
python scripts/run_eval_strict.py \
  --test_path artifacts/test/test_banks.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path None \
  --out_dir artifacts/eval_base_banks_v4_strict
```
Each directory contains `pred_detail.csv` and `metrics.csv`.

6. Capture Raw Outputs for Debugging
------------------------------------
```bash
python scripts/debug_eval_outputs.py \
  --test-path artifacts/test/test_banks.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --lora-path outputs/grpo_banks_v4/v0-20251030-062734/checkpoint-500 \
  --max-new-tokens 128 \
  --force-think \
  --out-csv outputs/debug_eval_outputs_grpo.csv

python scripts/debug_eval_outputs.py \
  --test-path artifacts/test/test_banks.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --lora-path outputs/sft_banks_qwen2p5/checkpoint-500 \
  --max-new-tokens 128 \
  --force-think \
  --out-csv outputs/debug_eval_outputs_sft.csv

python scripts/debug_eval_outputs.py \
  --test-path artifacts/test/test_banks.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --lora-path None \
  --max-new-tokens 128 \
  --force-think \
  --out-csv outputs/debug_eval_outputs_base.csv
```
Each CSV records the raw generation, parsed prediction, label, and absolute error.

7. Recompute Metrics from Debug CSVs
------------------------------------
```bash
python scripts/compute_metrics_from_debug.py \
  --debug-csv outputs/debug_eval_outputs_grpo.csv \
  --out-csv outputs/metrics_from_debug_grpo.csv

python scripts/compute_metrics_from_debug.py \
  --debug-csv outputs/debug_eval_outputs_sft.csv \
  --out-csv outputs/metrics_from_debug_sft.csv

python scripts/compute_metrics_from_debug.py \
  --debug-csv outputs/debug_eval_outputs_base.csv \
  --out-csv outputs/metrics_from_debug_base.csv
```

## Real-time Monitoring with SwanLab (optional)

1. Install and authenticate (once per machine):
   ```bash
   pip install swanlab
   swanlab login
   ```
2. Enable logging through the helper scripts:
   - Command-line flags: `--report-to-swanlab`, `--swanlab-project <name>`, `--swanlab-token <api-key>`, `--swanlab-workspace <team>`, `--swanlab-exp-name <run>`, `--swanlab-mode <cloud|local>`.
   - Or environment variables (same semantics):
     ```bash
     export SWIFT_REPORT_TO=swanlab
     export SWIFT_SWANLAB_PROJECT=rl-dbap
     export SWIFT_SWANLAB_EXP_NAME=banks-sft
     ```
   - Any remaining Swift CLI arguments can still be appended as usual.

Metrics continue to be written locally (`runs/*.tfevents`, `logging.jsonl`), so SwanLab can be used alongside TensorBoard if desired.
The metric CSVs report coverage, MAE, RMSE, R2, sMAPE, IC, RankIC, Recall@50, Precision@50, and NDCG@50.
