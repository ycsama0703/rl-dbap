RL-DBAP: Prompted Holdings Prediction — Quick Guide

概览
- 目标：把季度持仓面板数据转成“历史窗口型”Prompts，完成 SFT 热身与 GRPO 强化训练，并评估 MAE/IC 等指标。
- 入口：严格模板与分层采样在 `src/cli/build_history_prompts.py`。

环境准备
- Python 3.10+；数据以 parquet 提供。

1) 准备数据（对齐到季度 + 生成标签）
- 配置：`configs/data.yaml`
- 运行：`python -m src.cli.prepare_data --config configs/data.yaml`
- 产物：`data/processed/panel_quarter.parquet/*.parquet`

2) 生成严格模板 Prompts
- 严格模板：提示包含历史 `t-3..t`、推理指令与“STRICT OUTPUT CONTRACT”。模型必须在 `<answer>` 中给出 `{"holding_delta": <float>}` 或 `{"holding_tp1": <float>}`（≤6 小数、非科学计数法）。
- 构建：
  - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist --per-type-limit 1000 --time-bins 10 --cap-per-pair 3 --seed 42 --use-tqdm`
- 输出：`artifacts/prompts_hist/{type}.jsonl`

2.1) Ticker → 公司名映射（自动）
- 生成严格模板 Prompts 时会自动从 `data/ticker_mapping.csv` 读取映射（列名不区分大小写，包含 `PERMNO, COMNAM, TICKER`），并将 `Ticker: {permno}` 行替换为：`Ticker: {TICKER或PERMNO} | Company: {COMNAM}`。
- 若映射文件不存在或个别 PERMNO 缺失，对应行将保留原始 `Ticker: {permno}` 文本。

2.1) Ticker → 公司名映射（可选）
- 目的：将模板中的 `Ticker: {permno}` 行替换为带公司名（和/或交易代码）的可读文本，便于人工检查与演示。
- 映射来源：`data/ticker_mapping.csv`（列名不区分大小写，包含 `PERMNO, COMNAM, TICKER`）。
- 运行（目录就地批量转换）：
  - `python -m src.cli.map_ticker_names --in artifacts/prompts_hist --out artifacts/prompts_hist_named --mapping data/ticker_mapping.csv --mode append`
  - `--mode append`：示例行会变成 `Ticker: {TICKER或PERMNO} | Company: {COMNAM}`；`--mode replace` 则直接替换为 `Company: {COMNAM}`。
- 后续若希望在 SFT/GRPO 使用带公司名的 prompts，请把 `--in` 指向 `artifacts/prompts_hist_named`。
- 覆盖率检查：
  - 基于 parquet 源数据：
    - `python -m src.cli.check_ticker_mapping --parquet-dir data/processed/panel_quarter.parquet --mapping data/ticker_mapping.csv --export-missing artifacts/mapping_missing_permno.csv`
  - 基于 prompts（若已生成）：
    - `python -m src.cli.check_ticker_mapping --prompts-dir artifacts/prompts_hist --mapping data/ticker_mapping.csv`

 采样策略（Data Sampling）
- 连续窗口：按 `(type, mgrno, permno)` 排序后，仅保留严格连续季度的 `t-3,t-2,t-1,t` 窗口（断档会被跳过）。
- 分层时间桶：对每个投资者类型，按 `qid_t`（季度 id）做分位数切分为 `B=time_bins` 个时间桶（自动夹在 [3,12]），确保时间上的均匀覆盖。
- 桶内配额：`per_type_limit` 在各桶之间均匀分配（余数前置），保证每个时间段都有样本。
- 对对儿上限：同一 `(mgrno,permno)` 的窗口在全局最多取 `cap_per_pair` 个，防止单一账户/单一股票主导；桶内采用“轮转 + 随机打乱”挑选，提升多样性。
- 随机性与可复现：使用 `--seed` 固定 `numpy.random.default_rng`；可复现同一划分结果。
- 窗口大小与标签：若存在 `holding_t1`，会计算 `label_delta=holding_t1−holding_t` 与符号，供奖励或分析使用；缺失标签不影响样本生成。
- 关键参数：
  - `--per-type-limit` 每个投资者类型的样本上限（推荐 1000–3000）。
  - `--time-bins` 时间分桶数（推荐 8–12）。
  - `--cap-per-pair` 每个 `(mgrno,permno)` 全局上限（推荐 2–4）。
  - `--include-types/--exclude-types` 控制类型；`--date-start/--date-end` 控制时间范围；`--max-files/--head` 用于快速调试。
  - 进度：`--use-tqdm` 或 `--progress-every` 显示“构建窗口/写出”进度。

3) 数据划分（推荐“同分布，不重叠”）
- 按时间三段：
  - SFT：`--date-end 2016-12-31` 输出到 `artifacts/prompts_hist_sft`
  - GRPO：`--date-start 2017-01-01 --date-end 2018-12-31` 输出到 `artifacts/prompts_hist_grpo`
  - Test：`--date-start 2019-01-01` 输出到 `artifacts/prompts_hist_test`

4) 转数据格式
- SFT：`python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_sft --out artifacts/sft/sft_train.jsonl`
- GRPO：`python -m src.cli.prompts_to_grpo --in artifacts/prompts_hist_grpo --out artifacts/grpo/grpo.jsonl`
- Test：`python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test --out artifacts/sft/test.jsonl`

常用命令速查（生成 SFT/GRPO/Test 数据集）
- 说明：Ticker→公司名映射已内置于 `build_history_prompts`，无需额外步骤或参数。
- 生成三段 prompts：
  - SFT：`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_sft --date-end 2016-12-31 --use-tqdm`
  - GRPO：`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_grpo --date-start 2017-01-01 --date-end 2018-12-31 --use-tqdm`
  - Test：`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_test --date-start 2019-01-01 --use-tqdm`
- 转换为训练/评测集：
  - SFT：`python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_sft --out artifacts/sft/sft_train.jsonl [--with-think]`
  - GRPO：`python -m src.cli.prompts_to_grpo --in artifacts/prompts_hist_grpo --out artifacts/grpo/grpo.jsonl`
  - Test：`python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test --out artifacts/sft/test.jsonl`

5) 训练
- SFT（示例，LoRA）：执行 `scripts/sft.ps1`，将 `--dataset` 指向 `artifacts/sft/sft_train.jsonl`。
- GRPO：执行：
  - `.\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 4 -MaxCompletionLen 512`
  - 默认 `--reward_funcs contract_holdings external_holdings format`

6) 奖励函数（ms-swift）
- `contract_holdings`（格式契约，硬约束，0/1）
  - 只允许一个 `<answer>…</answer>` 区块，且区块内只有 JSON 对象。
  - 允许两种键之一且仅出现一次：`{"holding_delta": <float>}` 或 `{"holding_tp1": <float>}`；键小写。
  - 数值必须是十进制浮点，≤6 位小数；禁止科学计数法（含 e/E）；无多余逗号/字段。
  - 约束：`holding_tp1 ≥ 0`；若提供 `holding_t`，则 `holding_delta ≥ -holding_t`。
  - 满足全部规则奖励 1.0，否则 0.0（用于抑制格式走样与越界）。
- `external_holdings`（数值型复合：量级 + 方向）
  - 预测/目标获取：
    - 预测 pred：优先 `holding_delta`；否则用 `holding_tp1 − holding_t`（需有 holding_t）。
    - 目标 target：优先 `label_delta`；否则用 `label_tp1 − holding_t`。
    - 误差 `e = pred − target`；目标幅度 `r = target`。
  - 量级奖励 R_mag（自适应 Huber，方差无关）：
    - Huber 损失：`ℓ_Huber(e;c) = 0.5 e^2 (|e|≤c)； c(|e|−0.5c) (|e|>c)`。
    - 阈值 c 的稳健尺度：默认 `c = k_mag · EMA_λ(|e|)`；或设 `robust_mode ∈ {mad, iqr}` 使用 `k·MAD` / `k·IQR`。
    - 归一化到 [0,1]：`R_mag = 1 − min(ℓ_Huber / (0.5 c^2), 1)`。
  - 方向奖励 R_dir（无方差替代）：
    - 方向评分：`s = (pred / c_dir) · sign(target)`，`c_dir = k_dir · EMA_λ(|target|)`（或 MAD/IQR）。
    - 平滑打分：`R_dir = sigmoid(α (s − m))`，其中 α 控陡峭度（默认 5），m 为正向边际（默认 0）。
  - 总奖励：`R = w_mag · R_mag + w_dir · R_dir`（默认 `w_mag=0.6, w_dir=0.4`）。
  - 可调超参（kwargs）：`k_mag, k_dir, ema_lambda, alpha, margin, w_mag, w_dir, robust_mode`；内部维护 EMA 状态以自适应尺度。

7) 评估与测试
- 生成测试集（建议取较晚年份，例如 2019+）：
  - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_test --date-start 2019-01-01 --use-tqdm`
  - 转换为评测用 chat 格式（assistant 含绝对标签）：
    - `python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test --out artifacts/sft/test.jsonl`
- 运行评测：
  - Base（无 LoRA）：
    - `python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path None --out_dir artifacts/eval_base`
  - SFT LoRA：
    - `python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/sft_qwen2.5_7b --out_dir artifacts/eval_sft --post_csv_for_compare artifacts/eval_base/pred_detail.csv`
  - GRPO LoRA：
    - `python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/grpo_qwen2.5_7b --out_dir artifacts/eval_grpo --post_csv_for_compare artifacts/eval_base/pred_detail.csv`
- 评测产物：
  - `metrics.csv`（覆盖率、MAE、RMSE、R2、sMAPE%、IC、RankIC、Recall/Precision/NDCG@50）
  - `pred_detail.csv`（逐样本 y_true/y_pred/quarter/valid）
  - `residual_hist.png`、`ic_by_quarter.png`

SFT/GRPO 流程（含 think）
- 生成 SFT 训练集（在严格模板 prompts 基础上，附加非监督的 <think>）
  - `python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_sft --out artifacts/sft/sft_train.jsonl --with-think`
  - 说明：messages 将包含两条连续的 assistant 消息：
    - 第一条 `<think>...</think>`，标记 `loss=False`（不计损）
    - 第二条 `{"holding_tp1": ...}`，标记 `loss=True`（仅对该值计损）
- 训练 SFT（示例）
  - `powershell .\scripts\sft.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/sft/sft_train.jsonl" -OutputDir "outputs/sft_qwen2.5_7b"`
- 生成 GRPO 数据（保持 <think>+<answer> 模板，奖励只读取 <answer>）
  - `python -m src.cli.prompts_to_grpo --in artifacts/prompts_hist_grpo --out artifacts/grpo/grpo.jsonl`
  - 训练 GRPO（示例）：
    - `powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 4 -MaxCompletionLen 512`
  - 说明：`contract_holdings` 仅校验 <answer> JSON 结构与边界，`external_holdings` 仅从 <answer> 解析数值；`format` 奖励鼓励存在 `<think>…</think>` 与 `<answer>…</answer>`。
- 评测（只解析答案，不看 think）
  - `python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/sft_qwen2.5_7b --out_dir artifacts/eval_sft`
  - 提示：评测解析会优先选取 messages 中 `loss=True` 的 assistant 作为标签/对齐的对象，忽略 `<think>`。
