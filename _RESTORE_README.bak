RL-DBAP: Prompted Holdings Prediction �?Quick Guide

概览
- 目标：把季度持仓面板数据转成“历史窗口型”Prompts，完�?SFT 热身�?GRPO 强化训练，并评估 MAE/IC 等指标�?- 入口：严格模板与分层采样�?`src/cli/build_history_prompts.py`�?
环境准备
- Python 3.10+；数据以 parquet 提供�?
1) 准备数据（对齐到季度 + 生成标签�?- 配置：`configs/data.yaml`
- 运行：`python -m src.cli.prepare_data --config configs/data.yaml`
- 产物：`data/processed/panel_quarter.parquet/*.parquet`

2) 生成严格模板 Prompts
- 严格模板：提示包含历�?`t-3..t`、推理指令与“STRICT OUTPUT CONTRACT”。模型必须在 `<answer>` 中给�?`{"holding_delta": <float>}` �?`{"holding_tp1": <float>}`（≤6 小数、非科学计数法）�?- 构建�? - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist --per-type-limit 1000 --time-bins 10 --cap-per-pair 3 --seed 42 --use-tqdm`
- 输出：`artifacts/prompts_hist/{type}.jsonl`

采样策略（Data Sampling�?- 连续窗口：按 `(type, mgrno, permno)` 排序后，仅保留严格连续季度的 `t-3,t-2,t-1,t` 窗口（断档会被跳过）�?- 分层时间桶：对每个投资者类型，�?`qid_t`（季�?id）做分位数切分为 `B=time_bins` 个时间桶（自动夹�?[3,12]），确保时间上的均匀覆盖�?- 桶内配额：`per_type_limit` 在各桶之间均匀分配（余数前置），保证每个时间段都有样本�?- 对对儿上限：同一 `(mgrno,permno)` 的窗口在全局最多取 `cap_per_pair` 个，防止单一账户/单一股票主导；桶内采用“轮�?+ 随机打乱”挑选，提升多样性�?- 随机性与可复现：使用 `--seed` 固定 `numpy.random.default_rng`；可复现同一划分结果�?- 窗口大小与标签：若存�?`holding_t1`，会计算 `label_delta=holding_t1−holding_t` 与符号，供奖励或分析使用；缺失标签不影响样本生成�?- 关键参数�?  - `--per-type-limit` 每个投资者类型的样本上限（推�?1000�?000）�?  - `--time-bins` 时间分桶数（推荐 8�?2）�?  - `--cap-per-pair` 每个 `(mgrno,permno)` 全局上限（推�?2�?）�?  - `--include-types/--exclude-types` 控制类型；`--date-start/--date-end` 控制时间范围；`--max-files/--head` 用于快速调试�?  - 进度：`--use-tqdm` �?`--progress-every` 显示“构建窗�?写出”进度�?
3) 数据划分（推荐“同分布，不重叠”）
- 按时间三段：
  - SFT：`--date-end 2016-12-31` 输出�?`artifacts/prompts_hist_sft`
  - GRPO：`--date-start 2017-01-01 --date-end 2018-12-31` 输出�?`artifacts/prompts_hist_grpo`
  - Test：`--date-start 2019-01-01` 输出�?`artifacts/prompts_hist_test`

4) 转数据格�?- SFT：`python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_sft --out artifacts/sft/sft_train.jsonl`
- GRPO：`python -m src.cli.prompts_to_grpo --in artifacts/prompts_hist_grpo --out artifacts/grpo/grpo.jsonl`
- Test：`python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test --out artifacts/sft/test.jsonl`

5) 训练
- SFT（示例，LoRA）：执行 `scripts/sft.ps1`，将 `--dataset` 指向 `artifacts/sft/sft_train.jsonl`�?- GRPO：执行：
  - `.\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 4 -MaxCompletionLen 512`
  - 默认 `--reward_funcs contract_holdings external_holdings format`

6) 奖励函数（ms-swift�?- `contract_holdings`（格式契约，硬约束，0/1�?  - 只允许一�?`<answer>�?/answer>` 区块，且区块内只�?JSON 对象�?  - 允许两种键之一且仅出现一次：`{"holding_delta": <float>}` �?`{"holding_tp1": <float>}`；键小写�?  - 数值必须是十进制浮点，�? 位小数；禁止科学计数法（�?e/E）；无多余逗号/字段�?  - 约束：`holding_tp1 �?0`；若提供 `holding_t`，则 `holding_delta �?-holding_t`�?  - 满足全部规则奖励 1.0，否�?0.0（用于抑制格式走样与越界）�?- `external_holdings`（数值型复合：量�?+ 方向�?  - 预测/目标获取�?    - 预测 pred：优�?`holding_delta`；否则用 `holding_tp1 �?holding_t`（需�?holding_t）�?    - 目标 target：优�?`label_delta`；否则用 `label_tp1 �?holding_t`�?    - 误差 `e = pred �?target`；目标幅�?`r = target`�?  - 量级奖励 R_mag（自适应 Huber，方差无关）�?    - Huber 损失：`ℓ_Huber(e;c) = 0.5 e^2 (|e|≤c)�?c(|e|�?.5c) (|e|>c)`�?    - 阈�?c 的稳健尺度：默认 `c = k_mag · EMA_λ(|e|)`；或�?`robust_mode �?{mad, iqr}` 使用 `k·MAD` / `k·IQR`�?    - 归一化到 [0,1]：`R_mag = 1 �?min(ℓ_Huber / (0.5 c^2), 1)`�?  - 方向奖励 R_dir（无方差替代）：
    - 方向评分：`s = (pred / c_dir) · sign(target)`，`c_dir = k_dir · EMA_λ(|target|)`（或 MAD/IQR）�?    - 平滑打分：`R_dir = sigmoid(α (s �?m))`，其�?α 控陡峭度（默�?5），m 为正向边际（默认 0）�?  - 总奖励：`R = w_mag · R_mag + w_dir · R_dir`（默�?`w_mag=0.6, w_dir=0.4`）�?  - 可调超参（kwargs）：`k_mag, k_dir, ema_lambda, alpha, margin, w_mag, w_dir, robust_mode`；内部维�?EMA 状态以自适应尺度�?
7) 评估与测�?- 生成测试集（建议取较晚年份，例如 2019+）：
  - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_test --date-start 2019-01-01 --use-tqdm`
  - 转换为评测用 chat 格式（assistant 含绝对标签）�?    - `python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test --out artifacts/sft/test.jsonl`
- 运行评测�?  - Base（无 LoRA）：
    - `python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path None --out_dir artifacts/eval_base`
  - SFT LoRA�?    - `python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/sft_qwen2.5_7b --out_dir artifacts/eval_sft --post_csv_for_compare artifacts/eval_base/pred_detail.csv`
  - GRPO LoRA�?    - `python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/grpo_qwen2.5_7b --out_dir artifacts/eval_grpo --post_csv_for_compare artifacts/eval_base/pred_detail.csv`
- 评测产物�?  - `metrics.csv`（覆盖率、MAE、RMSE、R2、sMAPE%、IC、RankIC、Recall/Precision/NDCG@50�?  - `pred_detail.csv`（逐样�?y_true/y_pred/quarter/valid�?  - `residual_hist.png`、`ic_by_quarter.png`
  - `compare.txt`（如指定 `--post_csv_for_compare`，输出相�?Base �?MAE 改善�?95% CI�?
加速与进度
- `--use-tqdm` �?`--progress-every N` 打印进度；`--date-start/--date-end`、`--max-files/--head` 可快速子集�?
文件导航
- 生成/采样：`src/cli/build_history_prompts.py`、`src/prompts/sampler.py`、`src/prompts/builder.py`
- 数据准备：`src/cli/prepare_data.py`、`src/dataio/*`
- SFT：`src/cli/prompts_to_sft.py`
- GRPO：`src/cli/prompts_to_grpo.py`
- 切分：`src/cli/time_split_sft_jsonl.py`、`src/cli/split_sft_jsonl.py`
- 评估：`src/cli/run_eval.py`、`src/backends/hf_infer.py`、`src/evaluation/metrics.py`
- 奖励：`ms-swift/examples/train/grpo/plugin/plugin.py`（`contract_holdings`、`external_holdings`�?
运行 SFT/GRPO（标准与最小）
- 标准 SFT（LoRA�?  - `swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset artifacts/sft/sft_train.jsonl \
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
    --output_dir outputs/sft_qwen2.5_7b \
    --system "You are a quantitative portfolio manager. Respond with valid JSON only."`
- 标准 GRPO（LoRA�?  - `.\\scripts\\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 4 -MaxCompletionLen 512`

- 最�?SFT（快速打通链路）
  - 生成 mini 数据�?    - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_sftmini --include-types banks --date-end 2016-12-31 --per-type-limit 100 --time-bins 6 --cap-per-pair 2 --head 200000 --use-tqdm`
    - `python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_sftmini --out artifacts/sft/sft_train_mini.jsonl --limit 1000`
  - 训练命令�?    - `swift sft --model Qwen/Qwen2.5-7B-Instruct --train_type lora --dataset artifacts/sft/sft_train_mini.jsonl --torch_dtype bfloat16 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 2e-4 --lora_rank 8 --lora_alpha 16 --target_modules all-linear --max_length 1024 --logging_steps 10 --save_steps 200 --save_total_limit 2 --output_dir outputs/sft_debug`

- 最�?GRPO（快速打通链路）
  - 生成 mini 数据�?    - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_grpomini --include-types banks --date-start 2017-01-01 --date-end 2018-12-31 --per-type-limit 100 --time-bins 6 --cap-per-pair 2 --head 200000 --use-tqdm`
    - `python -m src.cli.prompts_to_grpo --in artifacts/prompts_hist_grpomini --out artifacts/grpo/grpo_mini.jsonl --limit 1500`
  - 训练命令�?    - `.\\scripts\\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo_mini.jsonl" -OutputDir "outputs/grpo_debug" -NumGenerations 2 -MaxCompletionLen 256`

如何判断“已跑通�?- 产物存在�?  - SFT：`outputs/sft_qwen2.5_7b` �?`outputs/sft_debug` 出现 `adapter_config.json`、`adapter_model.safetensors` 等�?  - GRPO：`outputs/grpo_qwen2.5_7b` �?`outputs/grpo_debug` 出现 checkpoint 与日志�?- 日志信号�?  - SFT：loss 正常下降/收敛，`Saving state at step ...` 正常打印�?  - GRPO：能看到 `contract_holdings`/`external_holdings` 奖励；合同奖励通过�?0 且随迭代提升，R_mag/R_dir 均值上升�?- 样本统计�?  - `SFT:  (Get-ChildItem artifacts/prompts_hist_sft*.jsonl | Get-Content).Count`
  - `GRPO: (Get-ChildItem artifacts/prompts_hist_grpo*.jsonl | Get-Content).Count`

算力建议
- 单卡 24GB（A10/3090/4090）：可直接跑 7B LoRA（SFT/GRPO），长度 1024�?048�?- 16GB：仍�?7B LoRA，建议将 `--max_length` 降到 768/1024，或�?`lora_rank`=4�?- 8�?2GB：改用小模型（如 `Qwen/Qwen2.5-1.5B-Instruct` �?`Qwen/Qwen2.5-3B-Instruct`）；GRPO �?`-NumGenerations` 设为 1�?�?
