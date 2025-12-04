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

3. SFT Training (single GPU example)
------------------------------------
```bash
export SWIFT_REPORT_TO=swanlab
export SWIFT_SWANLAB_PROJECT=rl-dbap
export SWIFT_SWANLAB_EXP_NAME=banks-sft

CUDA_VISIBLE_DEVICES=0 bash scripts/sft.sh \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -d artifacts/sft/sft_train_banks.jsonl \
  -o outputs/sft_banks_qwen2p5 \
  -- \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --num_train_epochs 4
```
This produces LoRA checkpoints under `outputs/sft_banks_qwen2p5/.../checkpoint-###` (e.g. `checkpoint-500`).

4. GRPO Training (contract + MSE rewards)
-----------------------------------------
`scripts/grpo.sh` defaults to the reward pair `contract_holdings` (format contract) and `mse_holdings` (squared-error mapper) with weights `0.3/0.7`. Adjust via `-F/-W` if you want to experiment with other reward mixes.

```bash
export SWIFT_REPORT_TO=swanlab
export SWIFT_SWANLAB_PROJECT=rl-dbap
export SWIFT_SWANLAB_EXP_NAME=banks-grpo

CUDA_VISIBLE_DEVICES=0 bash scripts/grpo.sh \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -d artifacts/grpo/grpo_banks.jsonl \
  -o outputs/grpo_banks_qwen2p5 \
  -a outputs/sft_banks_qwen2p5/v0-20251103-052552/checkpoint-500 \
  -S "</answer>" \
  -- \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8
```
Checkpoints are written to `outputs/grpo_banks_qwen2p5/.../checkpoint-*` (e.g. `checkpoint-500` or `checkpoint-400`).

5. Strict Evaluation (legacy, optional)
---------------------------------------
You can still run `scripts/run_eval_strict.py` for a one-shot eval; it writes the same log/absolute metrics as the debug pipeline.

```bash
rm -rf artifacts/eval_grpo_banks_latest
PYTHONPATH=. python scripts/run_eval_strict.py \
  --test_path artifacts/test/test_banks.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path outputs/grpo_banks_qwen2p5/v3-20251103-130248/checkpoint-500 \
  --out_dir artifacts/eval_grpo_banks_latest

rm -rf artifacts/eval_sft_banks_latest
PYTHONPATH=. python scripts/run_eval_strict.py \
  --test_path artifacts/test/test_banks.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path outputs/sft_banks_qwen2p5/v0-20251103-052552/checkpoint-500 \
  --out_dir artifacts/eval_sft_banks_latest

rm -rf artifacts/eval_base_banks_latest
PYTHONPATH=. python scripts/run_eval_strict.py \
  --test_path artifacts/test/test_banks.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path None \
  --out_dir artifacts/eval_base_banks_latest
```
The outputs still contain `pred_detail.csv` (log deltas + reconstructed holdings) and `metrics.csv` (`*_log` / `*_tp1`).

6. Generate Debug Outputs (official Test pipeline)
--------------------------------------------------
The recommended test flow is: (1) dump model generations with parse results, then (2) compute metrics from those CSVs. The example below evaluates Base, SFT checkpoint-500, and GRPO checkpoint-500; swap the LoRA path/filenames if you want to inspect another checkpoint (e.g. `checkpoint-400`).

```bash
export PYTHONPATH=.

python scripts/debug_eval_outputs.py \
  --test-path artifacts/test/test_banks.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --lora-path None \
  --max-new-tokens 128 \
  --force-think \
  --out-csv outputs/debug_eval_outputs_base.csv

python scripts/debug_eval_outputs.py \
  --test-path artifacts/test/test_banks.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --lora-path outputs/sft_banks_qwen2p5/v0-20251103-052552/checkpoint-500 \
  --max-new-tokens 128 \
  --force-think \
  --out-csv outputs/debug_eval_outputs_sft_ck500.csv

python scripts/debug_eval_outputs.py \
  --test-path artifacts/test/test_banks.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --lora-path outputs/grpo_banks_qwen2p5/v3-20251103-130248/checkpoint-500 \
  --max-new-tokens 128 \
  --force-think \
  --out-csv outputs/debug_eval_outputs_grpo_ck500.csv
```
Each CSV records the raw generations, log-space labels/predictions, reconstructed `holding_tp1`, and both log/absolute errors for downstream analysis.

7. Compute Metrics from Debug CSVs
----------------------------------
```bash
python scripts/compute_metrics_from_debug.py \
  --debug-csv outputs/debug_eval_outputs_base.csv \
  --out-csv outputs/metrics_from_debug_base.csv

python scripts/compute_metrics_from_debug.py \
  --debug-csv outputs/debug_eval_outputs_sft_ck500.csv \
  --out-csv outputs/metrics_from_debug_sft_ck500.csv

python scripts/compute_metrics_from_debug.py \
  --debug-csv outputs/debug_eval_outputs_grpo_ck500.csv \
  --out-csv outputs/metrics_from_debug_grpo_ck500.csv
```
The emitted metrics mirror what `run_eval_strict.py` produces, covering both log-space and reconstructed holding statistics.

## Custom Test Flow (top-k permno universe)

For a focused evaluation on a fixed stock universe (e.g., S&P500 weight top 10):
- Generate test prompts with permno filtering (default uses `data/sp500_top10_panel_2015_2024.csv` if present):
  ```bash
  python -m src.cli.build_type_datasets \
    --type mutual_funds \
    --in-dir data/processed/panel_quarter.parquet \
    --history-len 2 \
    --test-start 2023-01-01 \
    --exclude-zero-holding-t \
    --test-permnos-file data/sp500_top10_panel_2015_2024.csv \
    --sft-limit 0 --grpo-limit 0 \
    --per-type-limit 100000000 --test-limit 100000000 --cap-per-pair 100000000
  ```
  This writes `artifacts/prompts_hist_test/mutual_funds.jsonl` and `artifacts/test/test_mutual_funds.jsonl` with all available windows for the listed permnos.
- Run inference and dump per-sample outputs:
  ```bash
  python scripts/debug_eval_outputs.py \
    --test-path artifacts/test/test_mutual_funds.jsonl \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --lora-path outputs/grpo_mutual_funds/<run-id>/checkpoint-XXX \
    --max-new-tokens 128 --force-think \
    --out-csv outputs/debug_eval_outputs_topk.csv
  ```
- Aggregate by `permno,date` and compute per-stock errors:
  ```bash
  python scripts/aggregate_predictions.py \
    --debug-csv outputs/debug_eval_outputs_topk.csv \
    --out-prefix outputs/agg_topk
  ```
  Outputs:
  - `outputs/agg_topk.by_date_permno.csv`: summed true/pred holdings per date/permno with errors.
  - `outputs/agg_topk.per_stock.csv`: per-stock MAE/WAPE and sample counts.

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
