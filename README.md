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
  --date-start 2015-01-01 \
  --date-end 2018-12-31 \
  --exclude-zero-holding-t

python -m src.cli.build_history_prompts \
  --in-dir data/processed/panel_quarter.parquet \
  --out-dir artifacts/prompts_hist_grpo \
  --include-types banks \
  --per-type-limit 2000 \
  --date-start 2019-01-01 \
  --date-end 2022-12-31 \
  --exclude-zero-holding-t

python -m src.cli.build_history_prompts \
  --in-dir data/processed/panel_quarter.parquet \
  --out-dir artifacts/prompts_hist_test \
  --include-types banks \
  --per-type-limit 2000 \
  --date-start 2023-01-01 \
  --date-end 2024-10-01 \
  --exclude-zero-holding-t
```

2. Convert Prompts to Chat Datasets
-----------------------------------
All conversion scripts now emit `holding_log_delta` (signed log change) and keep `<think>` in a separate loss-free assistant message.

```bash
python -m src.cli.prompts_to_sft \
  --in artifacts/prompts_hist_sft/banks.jsonl \
  --out artifacts/sft/sft_train_banks.jsonl \
  --contract-mode log_delta

python -m src.cli.prompts_to_grpo \
  --in artifacts/prompts_hist_grpo/banks.jsonl \
  --out artifacts/grpo/grpo_banks.jsonl

python -m src.cli.prompts_to_sft \
  --in artifacts/prompts_hist_test/banks.jsonl \
  --out artifacts/test/test_banks.jsonl \
  --contract-mode log_delta \
  --no-think
```

3. SFT Training (Qwen/Qwen2.5-7B-Instruct)
-----------------------------------------
```bash
export SWIFT_REPORT_TO=swanlab
export SWIFT_SWANLAB_PROJECT=rl-dbap
export SWIFT_SWANLAB_EXP_NAME=banks-sft

CUDA_VISIBLE_DEVICES=0,1 \
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
The main checkpoint referenced later is `outputs/sft_banks_qwen2p5/checkpoint-500`.

4. GRPO Training (Format + Direction + Magnitude Rewards)
---------------------------------------------------------
```bash
# reuse SWIFT_REPORT_TO / SWIFT_SWANLAB_PROJECT from above, only change the run name
export SWIFT_SWANLAB_EXP_NAME=banks-grpo

CUDA_VISIBLE_DEVICES=0,1 \
`scripts/grpo.sh` now defaults to a two-part reward stack: `contract_holdings` (format contract) and `mse_holdings` (squared-error mapper) with weights `0.3/0.7`. Override via `-F/-W` or adjust scaling with `--reward_kwargs mse_cap=<value>`.
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -d artifacts/grpo/grpo_banks.jsonl \
  -o outputs/grpo_banks_v4 \
  -a outputs/sft_banks_qwen2p5/checkpoint-500 \
  -S "</answer>" \
  --
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
- `pred_detail.csv` includes `y_true`/`y_pred` (log deltas), `y_true_tp1`/`y_pred_tp1` (reconstructed holdings), and both `abs_log_error`/`abs_tp1_error`.
- `metrics.csv` reports mirrored statistics for log deltas (`*_log`) and reconstructed holdings (`*_tp1`).

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
Each CSV records the raw generations, log-space labels/predictions, reconstructed `holding_tp1`, and both log/absolute errors for downstream analysis.

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
The emitted metrics columns match `run_eval_strict`, covering both log-space and reconstructed holding statistics.

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
