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
    - `cd /workspace/rl-dbap` 改成你自己的路径
    - `rm -rf ./ms-swift` 删掉原始空文件夹
    - `git clone https://github.com/modelscope/ms-swift.git ms-swift` 直接clone就行
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
- GRPO（LoRA）：默认奖励为 `mse_holdings`（纯 MSE），外部插件路径为仓库内置 `src/plugins/grpo/holdings_plugin.py`。
- 评估产物：`metrics.csv`、`pred_detail.csv`、`residual_hist.png`、`ic_by_quarter.png`。

模块职责一览
- `src/cli/prepare_data.py`：对齐到季度、生成标签，输出 parquet 面板
- `src/cli/build_history_prompts.py`：采样、构造严格模板 t-3..t 窗口型 prompt（自动公司名映射）
- `src/prompts/sampler.py`：连续窗口与分层时间桶采样
- `src/prompts/builder.py`：严格模板 prompt 拼装、十进制格式、输出契约
- `src/cli/prompts_to_sft.py`：将 prompts 转为 SFT chat 格式（可选 `<think>` 不计损）
- `src/cli/prompts_to_grpo.py`：将 prompts 转为 GRPO 数据（messages + labels）
- `src/plugins/grpo/holdings_plugin.py`：自定义奖励（`mse_holdings`、`contract_holdings`、`external_holdings`）
- `scripts/sft.ps1`、`scripts/grpo.ps1`：训练脚本（ms‑swift CLI）
- `src/cli/run_eval.py`、`src/backends/hf_infer.py`：评测与推理

提示
- 评测解析优先选取 messages 中 `loss=True` 的 assistant 作为标签/对齐对象，忽略 `<think>`。

一条命令：按投资者类型 SFT+GRPO 管道
- Linux/macOS：
  - `bash scripts/run_per_type.sh -t banks -m "Qwen/Qwen2.5-7B-Instruct" -sft_end 2016-12-31 -grpo_start 2017-01-01 -grpo_end 2018-12-31 -g 4 -l 512`
  - 自动执行：构建 per‑type SFT prompts → 转 SFT jsonl → 运行 SFT；构建 per‑type GRPO prompts → 转 GRPO jsonl → 以 SFT 适配器为起点运行 GRPO。
- Windows/PowerShell：
  - `powershell .\scripts\run_per_type.ps1 -Type banks -Model "Qwen/Qwen2.5-7B-Instruct" -SftEnd "2016-12-31" -GrpoStart "2017-01-01" -GrpoEnd "2018-12-31" -NumGenerations 4 -MaxCompletionLen 512`
  - 同样串联执行 per‑type 的 SFT 与 GRPO。

最简训练（仅训练，数据已备）
- 场景：已存在 per‑type 的 SFT 与 GRPO 数据集（如 `artifacts/sft/sft_train_banks.jsonl`、`artifacts/grpo/grpo_banks.jsonl`）。
- Linux/macOS：
  - `bash scripts/train_per_type.sh -t banks -m "Qwen/Qwen2.5-7B-Instruct"`
- Windows/PowerShell：
  - `powershell .\scripts\train_per_type.ps1 -Type banks -Model "Qwen/Qwen2.5-7B-Instruct"`
- 说明：脚本默认读取 `artifacts/sft/sft_train_<TYPE>.jsonl` 与 `artifacts/grpo/grpo_<TYPE>.jsonl`，先运行 SFT 输出到 `outputs/sft_<TYPE>`，随后以该 SFT 适配器为起点运行 GRPO，输出到 `outputs/grpo_<TYPE>`。

GRPO 承接 SFT 与断点续训（补充）
- 从 SFT LoRA 继续做 GRPO：
  - PowerShell：`powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 4 -MaxCompletionLen 512 -Adapters "outputs/sft_qwen2.5_7b"`
  - Bash：`bash scripts/grpo.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/grpo/grpo.jsonl -o outputs/grpo_qwen2.5_7b -g 4 -l 512 -a outputs/sft_qwen2.5_7b`
- GRPO 断点续训：
  - PowerShell：`powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 4 -MaxCompletionLen 512 -ResumeFrom "outputs/grpo_qwen2.5_7b/checkpoint-1000"`
  - Bash：`bash scripts/grpo.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/grpo/grpo.jsonl -o outputs/grpo_qwen2.5_7b -g 4 -l 512 -r outputs/grpo_qwen2.5_7b/checkpoint-1000`

按投资者类型（per‑type）训练
- 场景：希望捕捉不同投资者类型的差异，每次训练仅使用单一类型的数据（SFT 与 GRPO 皆可）。
- 步骤（以 `banks` 为例）：
  1) 只为该类型生成 prompts（按需切分时间）
     - SFT：`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_sft --include-types banks --date-end 2016-12-31`
     - GRPO：`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_grpo --include-types banks --date-start 2017-01-01 --date-end 2018-12-31`
  2) 仅转换该类型的 jsonl 为训练集
     - SFT：`python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_sft/banks.jsonl --out artifacts/sft/sft_train_banks.jsonl`
     - GRPO：`python -m src.cli.prompts_to_grpo --in artifacts/prompts_hist_grpo/banks.jsonl --out artifacts/grpo/grpo_banks.jsonl`
  3) 使用 per‑type 数据启动训练
     - Windows/PS（SFT）：`powershell .\scripts\sft.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/sft/sft_train_banks.jsonl" -OutputDir "outputs/sft_banks"`
     - Windows/PS（GRPO）：`powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo_banks.jsonl" -OutputDir "outputs/grpo_banks" -NumGenerations 4 -MaxCompletionLen 512`
     - Linux/macOS（SFT）：`bash scripts/sft.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/sft/sft_train_banks.jsonl -o outputs/sft_banks`
     - Linux/macOS（GRPO）：`bash scripts/grpo.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/grpo/grpo_banks.jsonl -o outputs/grpo_banks -g 4 -l 512`
  4) 按类型评测（可选，仅 banks）
     - 生成测试集（仅 banks）：`python -m src.cli.build_history_prompts --in-dir data/processed/panel_quarter.parquet --out-dir artifacts/prompts_hist_test_banks --include-types banks --date-start 2019-01-01`
     - 转换：`python -m src.cli.prompts_to_sft --in artifacts/prompts_hist_test_banks/banks.jsonl --out artifacts/sft/test_banks.jsonl`
     - 评测：`python -m src.cli.run_eval --test_path artifacts/sft/test_banks.jsonl --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/sft_banks --out_dir artifacts/eval_sft_banks`

说明
- `--include-types` 接受以逗号分隔的文件名 stem（如 `banks,mutual_funds`）；单类型时仅填一个。
- 转换阶段如果 `--in` 指向目录，会合并目录内全部 jsonl；因此做 per‑type 训练时，务必把 `--in` 指向单个类型的 jsonl 文件路径。

训练（Python 执行 与 .sh 启动）
- Python 执行（SFT）：
  - `python -m swift.cli.sft --model "Qwen/Qwen2.5-7B-Instruct" --train_type lora --dataset artifacts/sft/sft_train.jsonl --torch_dtype bfloat16 --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 16 --learning_rate 1e-4 --lora_rank 8 --lora_alpha 32 --target_modules all-linear --logging_steps 20 --save_steps 500 --save_total_limit 2 --max_length 2048 --output_dir outputs/sft_qwen2.5_7b --system "You are a quantitative portfolio manager. Respond with valid JSON only."`
- Python 执行（GRPO）：
- `python -m swift.cli.rlhf --rlhf_type grpo --model "Qwen/Qwen2.5-7B-Instruct" --external_plugins src/plugins/grpo/holdings_plugin.py --reward_funcs mse_holdings --train_type lora --lora_rank 8 --lora_alpha 32 --target_modules all-linear --torch_dtype bfloat16 --dataset artifacts/grpo/grpo.jsonl --load_from_cache_file true --max_completion_length 512 --num_train_epochs 1 --per_device_train_batch_size 1 --learning_rate 1e-6 --gradient_accumulation_steps 8 --logging_steps 5 --save_steps 100 --save_total_limit 2 --max_length 2048 --output_dir outputs/grpo_qwen2.5_7b --warmup_ratio 0.05 --dataset_num_proc 2 --num_generations 4 --temperature 0.9 --beta 0.04 --log_completions true`
- Linux/macOS（.sh 启动）：
  - SFT：`bash scripts/sft.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/sft/sft_train_banks.jsonl -o outputs/sft_qwen2.5_7b`
  - GRPO：`bash scripts/grpo.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/grpo/grpo_banks.jsonl -o outputs/grpo_qwen2.5_7b -g 4 -l 512`（加 `-v` 启用 vLLM colocate）
  - macOS 若无 `bash` 可改用 `zsh` 执行上述命令。
~- Windows（PowerShell）：保持使用现有 `.ps1` 脚本 `scripts/sft.ps1` 与 `scripts/grpo.ps1`。

附录与完整说明
- 原始完整记录（包含更详细的动机、参数、奖励定义与命令清单）保存在仓库文件：`_RESTORE_README.md`。本 README 仅做结构化整理与快速上手，未删除任何历史信息。
- 如需我将 `_RESTORE_README.md` 的内容直接合并进本 README 作为“详细版”章节，请告知，我会无损合并并保留所有原段落。
Manual Reward Scoring
- Manually inspect reward scores for sampled completions aligned with the GRPO dataset:
- `python -m src.cli.score_rewards \
  `  --dataset artifacts/grpo/grpo_banks.jsonl \
  `  --completions outputs/grpo_qwen2.5_7b/v1-20251018-073116/completions.jsonl \
  `  --external_plugins src/plugins/grpo/holdings_plugin.py \
  `  --reward_funcs mse_holdings \
  `  --completion_field completion.0 \
  `  --limit 5`
- Notes: `--completion_field completion.0` selects the first sampled completion when the completions JSONL stores a list. Add `--strict_answer_only` to score only the <answer>...</answer> body. Adjust the completions path to your actual run directory.

数值优先的 GRPO 训练
- 目标：让 HoldingsDeltaORM（数值精度）主导优化，ContractHoldingsORM 仅作格式/边界兜底。
- PowerShell（Windows）：
  - `powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 6 -MaxCompletionLen 96 -Temperature 0.3 -StopWords "}"`
- Bash（Linux/macOS）：
  - `bash scripts/grpo.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/grpo/grpo.jsonl -o outputs/grpo_qwen2.5_7b -g 6 -l 96 -T 0.3 -S "}"`
- 提示：
  - 默认使用 `mse_holdings`。如需用 `contract_holdings` 稍作格式牵引，可添加：`-F "contract_holdings mse_holdings" -W "0.05 1.0"`（PowerShell 对应 `-RewardFuncs`/`-RewardWeights`）。
  - 纯 JSON 输出建议 `-S/--StopWords '}'`；若使用 XML 两区块，改为 `'</answer>'` 并保持数据模板一致。
  - 温度建议 0.3–0.6，避免冗长输出；`MaxCompletionLen` 设定为能完整容纳 JSON 的最小足够长度以减少截断。

在 SFT 基础上热启动 / 从 GRPO 断点续训
- Bash（Linux/macOS）：
  - 基于已有 SFT LoRA 继续（请替换 `<SFT_ADAPTER_DIR>` 为你的 SFT 适配器目录，例如 `outputs/sft_qwen2.5_7b`）
    - `bash scripts/grpo.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/grpo/grpo.jsonl -o outputs/grpo_qwen2.5_7b -g 6 -l 96 -T 0.3 -S "}" -a <SFT_ADAPTER_DIR>`
  - 从已有 GRPO 检查点继续（请替换 `<GRPO_CKPT_DIR>` 为你的断点路径，例如 `outputs/grpo_qwen2.5_7b/checkpoint-1000`）
    - `bash scripts/grpo.sh -m "Qwen/Qwen2.5-7B-Instruct" -d artifacts/grpo/grpo.jsonl -o outputs/grpo_qwen2.5_7b -g 6 -l 96 -T 0.3 -S "}" -r <GRPO_CKPT_DIR>`
  - 说明：如未显式传入 `-r`，脚本会在输出目录中自动检测最新的 `checkpoint-*` 作为断点。
- PowerShell（Windows）：
  - 基于 SFT LoRA：
    - `powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 6 -MaxCompletionLen 96 -RewardFuncs contract_holdings,external_holdings -RewardWeights 0.1,1.0 -Temperature 0.3 -StopWords "}" -Adapters <SFT_ADAPTER_DIR>`
  - 断点续训：
    - `powershell .\scripts\grpo.ps1 -Model "Qwen/Qwen2.5-7B-Instruct" -Dataset "artifacts/grpo/grpo.jsonl" -OutputDir "outputs/grpo_qwen2.5_7b" -NumGenerations 6 -MaxCompletionLen 96 -RewardFuncs contract_holdings,external_holdings -RewardWeights 0.1,1.0 -Temperature 0.3 -StopWords "}" -ResumeFrom <GRPO_CKPT_DIR>`
