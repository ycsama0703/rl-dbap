RL-DBAP: Prompted Holdings Prediction — Quick Guide

概览
- 目标：把季度持仓面板数据转成“历史窗口型”Prompts，完成 SFT 热身、GRPO 强化训练，并评估 MAE/IC 等指标。
- 核心入口：严格模板与分层采样（`src/cli/build_history_prompts.py`）。

一步跑通（从 clone 到评测）
- 获取代码并进入目录：
  - `git clone <YOUR_REPO_URL> rl-dbap && cd rl-dbap`
- 建环境与依赖（Windows PowerShell）：
  - `python -m venv .venv && ./.venv/Scripts/Activate.ps1`
  - 安装 PyTorch（任选其一）：
    - CUDA 12.1：`pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`
    - CPU：`pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio`
  - 其余依赖：`pip install -r requirements.txt`
- 可选：ms-swift 源码/固定路径安装
  - 源码安装：`pip install git+https://github.com/modelscope/ms-swift.git`
  - 固定路径安装：
    - `rm -rf ./ms-swift`
    - `git clone --depth 1 https://github.com/modelscope/ms-swift.git ./ms-swift`
    - `pip install -e ./ms-swift`
- 可选：提前下载基础模型（离线/限网时）
  - `pip install "huggingface_hub[cli]"`
  - `huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct --local-dir-use-symlinks False`
- 数据准备：
  - `python -m src.cli.prepare_data --config configs/data.yaml`
- 生成 prompts：
  - SFT：`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_sft --date-end 2016-12-31`
  - GRPO：`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_grpo --date-start 2017-01-01 --date-end 2018-12-31`
  - Test：`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_test --date-start 2019-01-01`
- 转换：
  - `python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_sft --out artifacts/sft/sft_train.jsonl`
  - `python -m src.cli.prompts_to_grpo --in artifacts/prompts_hist_grpo --out artifacts/grpo/grpo.jsonl`
  - `python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test --out artifacts/sft/test.jsonl`
- 训练：
  - SFT：`powershell .\scripts\sft.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/sft/sft_train.jsonl" -OutputDir "outputs/sft_qwen2.5_7b"`
  - GRPO：`powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 4 -MaxCompletionLen 512`
- 评测：
  - Base：`python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path None --out_dir artifacts/eval_base`
  - SFT：`python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/sft_qwen2.5_7b --out_dir artifacts/eval_sft --post_csv_for_compare artifacts/eval_base/pred_detail.csv`
  - GRPO：`python -m src.cli.run_eval --test_path artifacts/sft/test.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/grpo_qwen2.5_7b --out_dir artifacts/eval_grpo --post_csv_for_compare artifacts/eval_base/pred_detail.csv`

环境安装（完整说明）
- Windows（PowerShell）：见上文“一步跑通”。
- Linux/macOS：
  - `python3 -m venv .venv && source .venv/bin/activate`
  - 参照 https://pytorch.org 安装适配的 `torch/torchvision/torchaudio`
  - `pip install -r requirements.txt`
- ms-swift：`pip install ms-swift` 或 `pip install git+https://github.com/modelscope/ms-swift.git`
- 快速校验：`swift --help`、`swift sft --help` 可运行。

基础模型下载与加载
- 在线：首次运行自动拉取。
- 离线：使用 huggingface_hub 下载到 `models/Qwen2.5-7B-Instruct` 并在训练/评测命令中把 `-Model/--base_model` 改为该目录。
- 缓存：可设置 `HF_HOME` 避免重复下载。

数据与 Prompt 生成
- 准备数据：`python -m src.cli.prepare_data --config configs/data.yaml`
- 严格模板 Prompts：
  - `python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist --per-type-limit 1000 --time-bins 10 --cap-per-pair 3 --seed 42 --use-tqdm`
- 自动公司名映射：从 `data/ticker_mapping.csv` 读取，将 `Ticker: {permno}` 替换为 `Ticker: {TICKER或PERMNO} | Company: {COMNAM}`。

转换为训练/评测集
- SFT：`python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_sft --out artifacts/sft/sft_train.jsonl [--with-think]`
- GRPO：`python -m src.cli.prompts_to_grpo --in artifacts/prompts_hist_grpo --out artifacts/grpo/grpo.jsonl`
- Test：`python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test --out artifacts/sft/test.jsonl`

训练与评估（细化）
- SFT（LoRA）：见上文“一步跑通”。
- GRPO（LoRA）：奖励函数使用 `contract_holdings external_holdings format`，外部插件路径为仓库内置 `src/plugins/grpo/holdings_plugin.py`。
- 评估产物：`metrics.csv`、`pred_detail.csv`、`residual_hist.png`、`ic_by_quarter.png`。

模块职责一览
- `src/cli/prepare_data.py`：对齐到季度、生成标签，输出 parquet 面板
- `src/cli/build_history_prompts.py`：采样、构造严格模板 t-3..t 窗口型 prompt（自动公司名映射）
- `src/prompts/sampler.py`：连续窗口与分层时间桶采样
- `src/prompts/builder.py`：严格模板 prompt 拼装、十进制格式、输出契约
- `src/cli/prompts_to_sft.py`：将 prompts 转为 SFT chat 格式（可选 `<think>` 不计损）
- `src/cli/prompts_to_grpo.py`：将 prompts 转为 GRPO 数据（messages + labels）
- `src/plugins/grpo/holdings_plugin.py`：自定义奖励（`contract_holdings`、`external_holdings`）
- `scripts/sft.ps1`、`scripts/grpo.ps1`：训练脚本（ms‑swift CLI）
- `src/cli/run_eval.py`、`src/backends/hf_infer.py`：评测与推理

提示
- 评测解析优先选取 messages 中 `loss=True` 的 assistant 作为标签/对齐对象，忽略 `<think>`。

附录与完整说明
- 原始完整记录（包含更详细的动机、参数、奖励定义与命令清单）保存在仓库文件：`_RESTORE_README.md`。本 README 仅做结构化整理与快速上手，未删除任何历史信息。
- 如需我将 `_RESTORE_README.md` 的内容直接合并进本 README 作为“详细版”章节，请告知，我会无损合并并保留所有原段落。
