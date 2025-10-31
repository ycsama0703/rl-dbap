# Technical Pipeline Overview

This note documents the modelling pipeline behind the holdings adjustment project.
It complements the operational README by focusing on how data is prepared, how each
training stage is configured, and how evaluation metrics are computed.

---

## 1. Data Preparation

### 1.1 Source panel and window construction
- Raw features come from per-type parquet slices under `data/processed/panel_quarter.parquet`.
- `build_history_prompts.py` reads only the required columns (`type`, `mgrno`, `permno`, `date`, `holding_t`,
  `holding_t1`, fundamentals, AUM metrics, price) for efficiency (`src/cli/build_history_prompts.py:37`).
- Continuous quarterly windows (t-3..t) are materialised by `build_continuous_windows`, which enforces four
  consecutive quarters per `(type, mgrno, permno)` and records metadata such as the t-quarter AUM and delta sign
  (`src/prompts/sampler.py:36`).

### 1.2 Stratified sampling
- `stratified_sample_windows` draws at most `per_type_limit` windows per firm type while balancing across time.
  Time buckets are defined by quantiles of the quarter id and capped at 12 bins (`src/prompts/sampler.py:125`).
- Sampling uses round-robin selection across `(mgrno, permno)` pairs with a configurable per-pair cap to avoid
  over-sampling any single manager or ticker (`src/prompts/sampler.py:181`).
- Windows with zero position at t can be dropped (`--exclude-zero-holding-t`) to remove degenerate reward cases
  (`src/prompts/sampler.py:145`).

### 1.3 Prompt materialisation
- Each sampled window becomes a prompt through `build_history_prompt`, which writes the history blocks, the
  contract for `holding_delta`, and guardrails on bounds (`src/prompts/builder.py:43`).
- Auxiliary fields (`holding_t`, `label_tp1`, `label_delta`, identifiers) are stored alongside the prompt so they can
  be reused when constructing SFT or GRPO datasets and when computing rewards (`src/prompts/builder.py:118`).

---

## 2. Dataset Splits

`build_type_datasets.py` orchestrates the full split for a single firm type (`src/cli/build_type_datasets.py:55`):

- **SFT**: windows ending on or before `--sft-end` (default `2016-12-31`), converted to chat format with labels rounded
  to configurable decimals (`src/cli/build_type_datasets.py:125`).
- **GRPO**: windows between `--grpo-start` and `--grpo-end` are converted into RLHF samples while preserving label
  columns for reward computation (`src/cli/build_type_datasets.py:141`).
- **Test**: windows after `--test-start` are exported to chat JSONL (absolute target) for deterministic evaluation
  (`src/cli/build_type_datasets.py:133`).
- All conversions reuse the `prompts_to_sft` helpers so that think templates and numeric formatting stay consistent
  (`src/cli/build_type_datasets.py:78`).

By default the pipeline samples up to 1,000 windows per type, balances them across time, and filters out zero holdings.

---

## 3. Supervised Fine-Tuning (SFT)

### 3.1 Chat conversion
- `prompts_to_sft.py` attaches the system instruction, the user prompt, and an assistant reply that optionally begins
  with an auto-generated `<think>...</think>` block summarising recent factor deltas (`src/cli/prompts_to_sft.py:146`).
- Targets can be emitted either as `holding_delta` (default) or reconstructed absolute `holding_tp1`
  (`src/cli/prompts_to_sft.py:123`).

### 3.2 Training recipe
- `scripts/sft.sh` launches `swift sft` with LoRA rank 8, alpha 32, bfloat16, and effective batch size 16 via gradient
  accumulation while keeping the context length at 2048 (`scripts/sft.sh:25`).
- Training runs for one epoch with learning rate `1e-4` and checkpoints every 500 steps.

---

## 4. GRPO Reinforcement Learning

### 4.1 Dataset format
- `prompts_to_grpo.py` (and the wrapper in `build_type_datasets.py`) keeps the system and user turns, optionally inserts
  a non-loss `<think>` exemplar, and passes through `label_delta`, `label_tp1`, and `holding_t` for reward functions
  (`src/cli/prompts_to_grpo.py:18`).

### 4.2 Reward functions
Custom ORMs live in `src/plugins/grpo/holdings_plugin.py`:

- `contract_holdings` rewards completions that obey the `<think>` plus `<answer>{"holding_delta":...}</answer>` contract
  and respect the lower bound `holding_delta >= -holding_t` (`src/plugins/grpo/holdings_plugin.py:63`).
- `mse_holdings` computes a robustly scaled squared-error score on the predicted delta, clipping via the 95th percentile
  to map errors into `[-1, 1]` (`src/plugins/grpo/holdings_plugin.py:254`).
- `external_holdings` (optional) mixes a Huber-style magnitude reward with a direction reward using adaptive EMA
  scaling; it is registered but not enabled in the default launcher (`src/plugins/grpo/holdings_plugin.py:115`).

`score_rewards.py` can audit model completions against any registered reward by replaying datasets and printing score
statistics (`src/cli/score_rewards.py:113`).

### 4.3 GRPO training recipe
- `scripts/grpo.sh` launches `swift rlhf --rlhf_type grpo` with LoRA rank 32 (alpha 128), eight samples per prompt,
  temperature 0.9, and beta 0.02 (`scripts/grpo.sh:80`).
- Format and accuracy rewards are combined with weights 0.3 (`contract_holdings`) and 0.7 (`mse_holdings`)
  (`scripts/grpo.sh:31`).
- The launcher can resume from the latest checkpoint automatically and optionally warm start from the SFT adapters.

---

## 5. Evaluation and Diagnostics

### 5.1 Generation and parsing
- `run_eval.py` loads the base model plus optional LoRA adapters, generates answers, and re-parses either
  `holding_tp1` directly or reconstructs it from `holding_delta + holding_t` if needed (`src/backends/hf_infer.py:42`).
- Coverage is tracked by counting rows where parsing succeeds before computing metrics (`src/cli/run_eval.py:36`).

### 5.2 Metrics
- `basic_regression` returns MAE, RMSE, R2, sMAPE, IC, and RankIC; `topk` adds Recall, Precision, and NDCG@50 by quarter
  (`src/evaluation/metrics.py:6`).
- `compute_debug_metrics.py` re-runs the same metrics on debug CSVs and supports trimming out large residuals based on
  an absolute-error percentile (`src/cli/compute_debug_metrics.py:30`).
- `run_eval_suite.py` evaluates base, SFT, and GRPO checkpoints on a shared test set and emits per-stage CSVs plus
  bootstrap comparisons (`scripts/run_eval_suite.py:44`).

### 5.3 Visuals and debugging aids
- `run_eval.py` writes residual histograms and quarterly IC bar charts for quick inspection (`src/cli/run_eval.py:70`).
- Reward score audits log mean, standard deviation, and sample zero-reward completions (`src/cli/score_rewards.py:155`).

---

## 6. End-to-End Flow

1. **Sampling** - select balanced quarterly windows per firm type (`build_history_prompts.py` and `sampler.py`).
2. **Prompting** - render history prompts and convert them to SFT, GRPO, and test chat datasets with consistent
   formatting.
3. **SFT** - train LoRA adapters on pre-2017 data (`scripts/sft.sh`).
4. **GRPO** - continue training with GRPO rewards on 2017-2018 windows (`scripts/grpo.sh`).
5. **Evaluation** - score checkpoints on 2019-and-later test prompts, compute metrics, and optionally inspect rewards or
   run trimmed debug analyses.

This document should give new contributors enough context to inspect or modify each stage without digging through the
codebase piecemeal.
