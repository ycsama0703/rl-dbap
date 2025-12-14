# Experiment Pipeline (Detailed Narrative)

本文件描述当前实验的完整方法与动机，方便对照新 scope 识别差距。

## 1) 研究目标与思路
- **核心假设**：直接学价格不现实，改为学“像真实投资者一样调整持仓”，让模型学到需求侧的决策逻辑，再推断价格。  
- **三段式流程**：  
  1) **数据与样本**：构造 (t-1, t) → t+1 的持仓调整样本。  
  2) **行为学习**：SFT 学基础格式/策略；GRPO 强化数值与方向；可选蒸馏复制行为。  
  3) **评估与对照**：base vs SFT vs GRPO；有 think vs 无 think；top30 训练抽样 vs top10 全量测试。  
- **下一步（尚未实现）**：显式“动机/理由”建模（meta-reasoning）、区分不同投资者 archetype 的风格。

## 2) 数据构建与切分
- **原始来源**：季度面板 `data/processed/panel_quarter.parquet`。  
- **白名单**：训练用 `data/sp500_top30_panel_2015_2024.csv`，测试用 `data/sp500_top10_panel_2015_2024.csv`。  
- **窗口**：history_len=2（t-1、t 都有数据），预测 t+1 持仓变化。  
- **时间切片**：SFT ≤ 2018-12-31；GRPO 2019-01-01..2022-12-31；Test ≥ 2023-01-01。  
- **过滤与采样**：排除 holding_t=0；训练采样上限 SFT 2000 / GRPO 2000；测试全量（~6109 样本，10 permno）。  
- **生成结果**：  
  - 含 think：`artifacts/sft/sft_train_mutual_funds.jsonl`，`artifacts/grpo/grpo_mutual_funds.jsonl`，`artifacts/test/test_mutual_funds.jsonl`  
  - 无 think：对应 `*_no_think.jsonl`（仅移除 `<think>` 文本，其余字段、标签一致）  
  - Base 极简测试：`artifacts/test/test_mutual_funds_base_min.jsonl`（同样本，System 简化，只要求输出 `holding_tp1`）

## 3) Prompt 设计（为什么这样写）
- **System**：定义“你是某投资经理”，可选要求先 `<think>` 后 `<answer>`，鼓励显式推理。  
- **User**：提供 t-1、t 的核心变量：公司特征 (me, be, profit, Gat, beta)、组合规模 (aum, outAUM)、指数权重 spx_weight、当前持仓/价格。目标是估计 t+1 持仓调整。  
- **Assistant 标签**：  
  - 含 think：`<think>...</think>` 解释；`<answer>{"holding_log_delta": ...}</answer>` 为监督目标。  
  - 无 think：移除 think，只留 `<answer>`。  
  - Base_min：保留标签消息，但 System 极简，便于解析。

## 4) 模型训练策略
- **SFT（监督，LoRA）**：先学格式与基础策略；有/无 think 两套。基座：`Qwen2.5-7B-Instruct`。  
- **GRPO（强化）**：在 SFT 基础上强化，奖励包含：  
  - 格式/约束（contract_holdings）  
  - 数值偏差（huber_holdings，主权重）  
  - 方向一致性（direction_holdings）  
  默认权重 5% / 60% / 35%。  
- **SwanLab 监控**：通过环境变量指定 workspace/project/token/exp_name 记录实验。

## 5) 评估与指标解释
- **推理输出**：`scripts/debug_eval_outputs.py` 生成 CSV，包含 raw_output、解析后的 log_delta / pred_tp1、真值及误差。  
- **聚合**：`aggregate_predictions.py`（SFT/GRPO）与 `aggregate_predictions_base.py`（base）按日期/permno 和按股票汇总：  
  - MAE（绝对误差），WAPE（加权绝对百分比误差），真实/预测持仓总和。  
- **测试集选择**：base 用极简版 `test_mutual_funds_base_min.jsonl`；SFT/GRPO 用标准版 `test_mutual_funds.jsonl`（或无 think 版）。  
- **对照点**：有/无 think 的表现差异；base vs SFT vs GRPO 的提升；top30 训练抽样 → top10 全量测试的泛化。

## 6) 蒸馏（可选）
- 合并 GRPO LoRA 作为教师；学生 1.5B 用 `gkd/train_gkd.py`，蒸馏数据可用 GRPO 副本；输出模型可直接推理、便于多 agent 或资源受限场景。
