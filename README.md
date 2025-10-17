RL-DBAP: Prompted Holdings Prediction — Quick Guide

概览
- 目标：把季度持仓面板数据转成“历史窗口型”Prompts，完成 SFT 热身、GRPO 强化训练，并评估 MAE/IC 等指标。
- 入口：严格模板与分层采样：`src/cli/build_history_prompts.py`

环境准备
- Python 3.10+；数据以 parquet 提供。
- 建议使用虚拟环境并按以下步骤安装依赖。

Windows（PowerShell）
- 创建并激活环境：
  - `python -m venv .venv`
  - `./.venv/Scripts/Activate.ps1`
- 安装 PyTorch（选择与你机器匹配的 CUDA/CPU 版本）：
  - CUDA 12.1 示例：`pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`
  - CPU 环境：`pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio`
- 安装其余依赖：
  - `pip install -r requirements.txt`
- 如 ms-swift 从 PyPI 安装失败，改用源码安装：
  - `pip install git+https://github.com/modelscope/ms-swift.git`

Linux/macOS（可选）
- 创建并激活环境：
  - `python3 -m venv .venv && source .venv/bin/activate`
- 参照 https://pytorch.org 的命令安装 `torch/torchvision/torchaudio`（匹配 CUDA/CPU），再执行：
  - `pip install -r requirements.txt`
- 如需源码安装 ms-swift：
  - `pip install git+https://github.com/modelscope/ms-swift.git`

MS‑SWIFT 固定路径安装（方案 A，推荐）
- 适用：你在算力平台 clone 本仓库后，`ms-swift` 目录为空或未被同步，且希望以“固定路径 + 开发模式”安装，确保 `swift` 与奖励接口可用。
- 将环境变量 `REPO_ROOT` 替换为本仓库根目录（例如 `/root/rl-dbap`）。

Linux（bash）
- 激活你的虚拟环境：
  - `source /path/to/venv/bin/activate`
- 删除旧目录（若存在）：
  - `rm -rf $REPO_ROOT/ms-swift`
- 以相同路径名重新克隆官方仓库（浅克隆加速）：
  - `git clone --depth 1 https://github.com/modelscope/ms-swift.git $REPO_ROOT/ms-swift`
- 开发模式安装：
  - `pip install -e $REPO_ROOT/ms-swift`

示例（与你的环境一致）
- 假设仓库在 `/root/rl-dbap`，虚拟环境在 `/workspace/rl-dabp-vastai`：
  - `source /workspace/rl-dabp-vastai/bin/activate`
  - `rm -rf /root/rl-dbap/ms-swift`
  - `git clone --depth 1 https://github.com/modelscope/ms-swift.git /root/rl-dbap/ms-swift`
  - `pip install -e /root/rl-dbap/ms-swift`

快速校验
- `swift --help` 与 `swift sft --help` 可正常运行。
- 奖励插件可加载：
  - `python - << 'PY'
from swift.plugin.orm import orms
import src.plugins.grpo.holdings_plugin
print('contract_holdings' in orms, 'external_holdings' in orms)
PY`

基础模型下载与配置
- 在线环境：首次运行会自动从 Hugging Face 下载基础模型（如 `Qwen/Qwen2.5-7B-Instruct`），无需额外操作。
- 离线/受限环境：建议提前拉取并本地化，训练与评测均可直接指向本地目录。

Windows（PowerShell）
- 安装 CLI：`pip install "huggingface_hub[cli]"`（一次即可）
- 登录（如需私有/受限下载）：`huggingface-cli login`
- 下载到本地目录（示例）：
  - `huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct --local-dir-use-symlinks False`
- 训练/评测时改为使用本地路径：
  - SFT：`powershell .\scripts\sft.ps1 -Model ".\models\Qwen2.5-7B-Instruct" -Dataset "artifacts/sft/sft_train.jsonl" -OutputDir "outputs/sft_qwen2.5_7b"`
  - GRPO：`powershell .\scripts\grpo.ps1 -Model ".\models\Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 4 -MaxCompletionLen 512`
  - 评测：`python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model .\models\Qwen2.5-7B-Instruct --lora_path outputs/sft_qwen2.5_7b --out_dir artifacts/eval_sft`

Linux/macOS（bash）
- `pip install "huggingface_hub[cli]"`
- `huggingface-cli login`（如需）
- `huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct --local-dir-use-symlinks False`
- 训练/评测命令中的 `-Model/--base_model` 改为 `./models/Qwen2.5-7B-Instruct`

可选：统一缓存目录
- 设置缓存路径，避免多次下载与便于迁移：
  - Windows：`$env:HF_HOME="D:/hf-cache"`; `pip cache dir` 查看缓存
  - Linux：`export HF_HOME=/data/hf-cache`

1) 准备数据（对齐到季度 + 生成标签）
- 配置：`configs/data.yaml`
- 运行：`python -m src.cli.prepare_data --config configs/data.yaml`
- 产物：`data/processed/panel_quarter.parquet/*.parquet`

2) 生成严格模板 Prompts
- 严格模板：提示包含历史 `t-3..t`、推理指令与 STRICT OUTPUT CONTRACT。模型必须在 `<answer>` 中给出 `{"holding_delta": <float>}` 或 `{"holding_tp1": <float>}`（≤6 小数，非科学计数法）。
- 构建：`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist --per-type-limit 1000 --time-bins 10 --cap-per-pair 3 --seed 42 --use-tqdm`
- 输出：`artifacts/prompts_hist/{type}.jsonl`

2.1) Ticker → 公司名映射（自动）
- 生成严格模板 Prompts 时会自动从 `data/ticker_mapping.csv` 读取映射（列名不区分大小写，包含 `PERMNO, COMNAM, TICKER`），并将 `Ticker: {permno}` 行替换为：`Ticker: {TICKER或PERMNO} | Company: {COMNAM}`。
- 若映射文件不存在或个别 PERMNO 缺失，对应行将保留原始 `Ticker: {permno}` 文本。

采样策略（Data Sampling）
- 连续窗口：按 `(type, mgrno, permno)` 排序后，仅保留严格连续季度的 `t-3,t-2,t-1,t` 窗口（断档跳过）。
- 分层时间桶：对每个投资者类型，对 `qid_t` 做分位数切分为 `B=time_bins` 个时间桶（自动夹在 [3,12]），确保时间均匀覆盖。
- 桶内配额：`per_type_limit` 在各桶间均匀分配（余数前置），保证每段时间都有样本。
- 对对儿上限：同一 `(mgrno,permno)` 全局最多取 `cap_per_pair` 个；桶内“轮转 + 随机打乱”。
- 随机性与可复现：`--seed` 固定 `numpy.random.default_rng`。
- 标签：若存 `holding_t1`，计算 `label_delta=holding_t1−holding_t`。
- 关键参数：`--per-type-limit`、`--time-bins`、`--cap-per-pair`、`--include/exclude-types`、`--date-start/--date-end`、`--max-files/--head`、`--use-tqdm`。

3) 数据划分（推荐“同分布，不重叠”）
- 按时间三段：
  - SFT：`--date-end 2016-12-31` 输出 `artifacts/prompts_hist_sft`
  - GRPO：`--date-start 2017-01-01 --date-end 2018-12-31` 输出 `artifacts/prompts_hist_grpo`
  - Test：`--date-start 2019-01-01` 输出 `artifacts/prompts_hist_test`

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
- SFT（示例，LoRA）：执行 `scripts/sft.ps1`，将 `--dataset` 指向 `artifacts/sft/sft_train.jsonl`
- GRPO（示例）：
  - `powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 4 -MaxCompletionLen 512`
  - 说明：脚本使用仓库内置插件 `src/plugins/grpo/holdings_plugin.py` 作为 `--external_plugins`，避免依赖 ms-swift 目录下的示例插件。
  - 默认 `--reward_funcs contract_holdings external_holdings format`

6) 评估与测试
- 生成测试集：
  - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_test --date-start 2019-01-01 --use-tqdm`
  - `python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test --out artifacts/sft/test.jsonl`
- 运行评测：
  - Base（无 LoRA）：`python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path None --out_dir artifacts/eval_base`
  - SFT LoRA：`python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/sft_qwen2.5_7b --out_dir artifacts/eval_sft --post_csv_for_compare artifacts/eval_base/pred_detail.csv`
  - GRPO LoRA：`python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/grpo_qwen2.5_7b --out_dir artifacts/eval_grpo --post_csv_for_compare artifacts/eval_base/pred_detail.csv`
- 评测产物：`metrics.csv`（覆盖率、MAE、RMSE、R2、sMAPE%、IC、RankIC、Recall/Precision/NDCG@50）、`pred_detail.csv`、`residual_hist.png`、`ic_by_quarter.png`

一步跑通（从零到评测）
- 数据准备：`python -m src.cli.prepare_data --config configs/data.yaml`
- 生成 prompts：
  - SFT：`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_sft --date-end 2016-12-31`
  - GRPO：`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_grpo --date-start 2017-01-01 --date-end 2018-12-31`
  - Test：`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_test --date-start 2019-01-01`
- 转换：
  - `python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_sft --out artifacts/sft/sft_train.jsonl`
  - `python -m src.cli.prompts_to_grpo --in artifacts/prompts_hist_grpo --out artifacts/grpo/grpo.jsonl`
  - `python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test --out artifacts/sft/test.jsonl`
- 训练：
  - `powershell .\scripts\sft.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/sft/sft_train.jsonl" -OutputDir "outputs/sft_qwen2.5_7b"`
  - `powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 4 -MaxCompletionLen 512`
- 评测：
  - `python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/sft_qwen2.5_7b --out_dir artifacts/eval_sft`
  - `python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/grpo_qwen2.5_7b --out_dir artifacts/eval_grpo --post_csv_for_compare artifacts/eval_base/pred_detail.csv`

提示：评测解析会优先选取 messages 中 `loss=True` 的 assistant 作为标签/对齐对象，忽略 `<think>`。
