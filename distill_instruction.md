# 🧠 Qwen2.5-7B → Qwen1.8B 知识蒸馏实验 (fdistill 框架)

本项目展示如何使用 [MANGA-UOFA/fdistill](https://github.com/MANGA-UOFA/fdistill) 框架  
将经过 **GRPO 微调的 Qwen2.5-7B + LoRA** 蒸馏为更轻量的 **Qwen1.8B Student** 模型。  
流程针对单卡 **A100-40GB** 环境优化，包含数据混合、Teacher 输出生成与在线蒸馏训练。

---

## 📦 1. 环境准备

```bash
conda create -n qwen_kd python=3.10 -y
conda activate qwen_kd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.43.3 peft accelerate datasets bitsandbytes
pip install sentencepiece tqdm nlg-eval
```

---

## 📂 2. 获取 fdistill 仓库

```bash
git clone https://github.com/MANGA-UOFA/fdistill.git
cd fdistill
```

---

## 🧰 3. 数据准备（70% GRPO + 30% SFT）

仓库内的原始数据已经按公司类别拆分存放：

- GRPO：`artifacts/grpo/grpo_<category>.jsonl`
- SFT：`artifacts/sft/sft_train_<category>.jsonl`
- `<category>` 取值：`banks`、`households`、`insurance_companies`、`investment_advisors`、`mutual_funds`、`other`、`pension_funds`

运行下列脚本，将每个公司类别按 70% GRPO + 30% SFT 混合，输出到 `artifacts/distill_data/raw/train_mix_<category>.jsonl`：

```bash
python - <<'PY'
import json
import random
from pathlib import Path

repo_root = Path.cwd()
grpo_dir = repo_root / "artifacts" / "grpo"
sft_dir = repo_root / "artifacts" / "sft"
output_dir = repo_root / "artifacts" / "distill_data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)

categories = [
    "banks",
    "households",
    "insurance_companies",
    "investment_advisors",
    "mutual_funds",
    "other",
    "pension_funds",
]

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

random.seed(42)
for cat in categories:
    grpo_path = grpo_dir / f"grpo_{cat}.jsonl"
    sft_path = sft_dir / f"sft_train_{cat}.jsonl"

    if not grpo_path.exists() or not sft_path.exists():
        print(f"[skip] {cat}: missing source file")
        continue

    grpo_records = load_jsonl(grpo_path)
    sft_records = load_jsonl(sft_path)
    if not grpo_records or not sft_records:
        raise RuntimeError(f"{cat}: empty source data")

    grpo_take = min(int(len(grpo_records) * 0.7), len(grpo_records))
    sft_take = min(int(len(sft_records) * 0.3), len(sft_records))

    mix = random.sample(grpo_records, grpo_take) + random.sample(sft_records, sft_take)
    random.shuffle(mix)

    out_path = output_dir / f"train_mix_{cat}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in mix:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[ok] {cat}: {len(mix)} records -> {out_path}")
PY
```

脚本执行完后，`artifacts/distill_data/raw/` 下将生成各公司的 `train_mix_<category>.jsonl`，后续步骤会基于这些文件生成教师伪标签并转成模型训练所需的 `.source/.target` 格式。

---

## 🧠 4. 生成 Teacher 输出（离线伪标签）

1. **导出 Prompt**  
   ```bash
   python scripts/export_infer_prompts.py \
     --in artifacts/distill_data/raw/train_mix_banks.jsonl \
     --out-dir artifacts/distill_data/raw \
     --stem banks_teacher
   ```
   `banks` 换成其它公司即可。脚本会生成 `banks_teacher_prompts_base.jsonl`（与 `_grpo` 内容相同，保留一份即可）。

2. **批量推理**  
   ```bash
   python scripts/batch_infer.py \
     --jsonl artifacts/distill_data/raw/banks_teacher_prompts_base.jsonl \
     --base_model Qwen/Qwen2.5-7B-Instruct \
     --checkpoint outputs/grpo_banks_qwen2p5/v3-20251103-130248/checkpoint-500 \
     --out_jsonl artifacts/distill_data/raw/teacher_outputs_banks.jsonl \
     --batch_size 4 \
     --max_new_tokens 512 \
     --temperature 0.7 \
     --torch_dtype bfloat16
   ```
   - `--base_model` 可换成本地缓存目录。
   - 生成的 `teacher_outputs_<category>.jsonl` 包含完整 `<think>/<answer>` 文本以及解析出的 `holding_log_delta`。

3. **转换为 `.source/.target`**  
   ```bash
   python - <<'PY'
   import json
   from pathlib import Path

   root = Path("artifacts/distill_data")
   raw_dir = root / "raw"
   processed_dir = root / "processed"
   processed_dir.mkdir(parents=True, exist_ok=True)

   categories = [
       "banks",
       "households",
       "insurance_companies",
       "investment_advisors",
       "mutual_funds",
       "other",
       "pension_funds",
   ]

   for cat in categories:
       prompt_path = raw_dir / f"{cat}_teacher_prompts_base.jsonl"
       output_path = raw_dir / f"teacher_outputs_{cat}.jsonl"
       if not prompt_path.exists() or not output_path.exists():
           print(f"[skip] {cat}: missing prompts or teacher outputs")
           continue

       prompts = {}
       with prompt_path.open("r", encoding="utf-8") as f:
           for line in f:
               rec = json.loads(line)
               prompts[rec["id"]] = rec

       generations = []
       with output_path.open("r", encoding="utf-8") as f:
           for line in f:
               generations.append(json.loads(line))

       out_dir = processed_dir / cat
       out_dir.mkdir(parents=True, exist_ok=True)

       with (out_dir / "train.source").open("w", encoding="utf-8") as f_src, \
            (out_dir / "train.target").open("w", encoding="utf-8") as f_tgt:
           for row in sorted(generations, key=lambda x: x["id"]):
               prompt = prompts[row["id"]]
               system = (prompt.get("system") or "").strip()
               user = (prompt.get("prompt") or "").strip()
               teacher = (row.get("raw_output") or "").rstrip()
               f_src.write(f"{system}\n\n{user}\n")
               f_tgt.write(f"{teacher}\n")

       print(f"[ok] wrote {cat}: {len(generations)} samples -> {out_dir}")
   PY
   ```

最终，蒸馏脚本的 `--data_dir` 可以直接指向 `artifacts/distill_data/processed/<category>`，其中包含 `train.source` / `train.target`（以及按需扩展的 `val.*`、`test.*`）。

---

## 🔥 5. 启动蒸馏训练 (KL Divergence)

```bash
cat > run_qwen_kd.sh <<'SH'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python train_kd.py \
  --teacher_model "<base model path>" \
  --teacher_lora "<checkpoint path>" \
  --student_model "Qwen1.8B" \
  --dataset_path "data/mixed/teacher_outputs.json" \
  --output_dir "./output/student_kd" \
  --temperature 2.0 \
  --alpha 0.5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --lr 5e-5 \
  --fp16 True \
  --teacher_8bit True \
  --gradient_checkpointing True
SH

bash run_qwen_kd.sh
```

### 参数说明
| 参数 | 含义 | 推荐值 |
|------|------|--------|
| `temperature` | KL Soft label 平滑度 | 2.0 |
| `alpha` | KL 与 CE 权重 | 0.5 |
| `batch_size` | 单卡 batch | 2 |
| `gradient_accumulation_steps` | 累积步数 | 8 |
| `lr` | 学习率 | 5e-5 |
| `num_train_epochs` | 训练轮数 | 3 |

预计显存占用：**约 36–38 GB (fp16)**。

---

## 📈 6. 模型评估与调试

### 6.1 构建 Chat 评测集
`debug_eval_outputs.py` 要求输入为 chat 格式。若手上只有 prompt/teacher 输出，可先拼成 `messages` 结构：

```bash
PYTHONPATH=/workspace/rl-dbap python - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/distill_data")
prompt_path = root / "raw" / "test_prompts_banks.jsonl"
teacher_path = root / "raw" / "teacher_outputs_banks.jsonl"
out_path = root / "processed" / "test_banks_chat.jsonl"

prompts = {}
with prompt_path.open() as f:
    for line in f:
        rec = json.loads(line)
        prompts[rec["id"]] = rec

rows = []
with teacher_path.open() as f:
    for line in f:
        rec = json.loads(line)
        prompt = prompts.get(rec["id"])
        if not prompt:
            continue
        system = prompt.get("system", "")
        user = prompt.get("prompt", "")
        raw = (rec.get("raw_output") or "").strip()
        think = (rec.get("think") or "").strip()
        answer = (rec.get("answer") or raw).strip()
        assistant_msgs = []
        if think:
            if not think.lower().startswith("<think>"):
                think = f"<think>{think}</think>"
            assistant_msgs.append({"role": "assistant", "content": think, "loss": False})
        if not answer.lower().startswith("<answer>"):
            answer = f"<answer>{answer}</answer>"
        assistant_msgs.append({"role": "assistant", "content": answer, "loss": True})
        rows.append({
            "id": rec["id"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                *assistant_msgs,
            ],
        })

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"[ok] wrote {len(rows)} rows -> {out_path}")
PY
```

为其它类别替换相应的 `test_prompts_*.jsonl` 与 `teacher_outputs_*.jsonl` 即可。

### 6.2 跑推理（Student / Baseline / Teacher）

以下命令默认在 `/workspace/rl-dbap` 目录执行，可按需调整 batch size：

```bash
# Student (蒸馏后模型)
PYTHONPATH=/workspace/rl-dbap python scripts/debug_eval_outputs.py \
  --test-path artifacts/distill_data/processed/test_banks_chat.jsonl \
  --base-model /workspace/rl-dbap/outputs/student_seqkd_1p5b/best_tfmr \
  --out-csv outputs/debug_eval_student_seqkd.csv \
  --batch-size 4 --max-new-tokens 128 --torch-dtype bfloat16

# Qwen2.5-1.5B 原始基座
PYTHONPATH=/workspace/rl-dbap python scripts/debug_eval_outputs.py \
  --test-path artifacts/distill_data/processed/test_banks_chat.jsonl \
  --base-model /workspace/models/Qwen2.5-1.5B-Instruct \
  --out-csv outputs/debug_eval_qwen15b_base.csv \
  --batch-size 4 --max-new-tokens 128 --torch-dtype bfloat16

# Qwen2.5-7B 基座（参考上限）
PYTHONPATH=/workspace/rl-dbap python scripts/debug_eval_outputs.py \
  --test-path artifacts/distill_data/processed/test_banks_chat.jsonl \
  --base-model /workspace/models/Qwen2.5-7B-Instruct \
  --out-csv outputs/debug_eval_qwen7b_base.csv \
  --batch-size 2 --max-new-tokens 128 --torch-dtype bfloat16

# GRPO Teacher（LoRA 形式）
PYTHONPATH=/workspace/rl-dbap python scripts/debug_eval_outputs.py \
  --test-path artifacts/distill_data/processed/test_banks_chat.jsonl \
  --base-model /workspace/models/Qwen2.5-7B-Instruct \
  --lora-path /workspace/rl-dbap/outputs/grpo_banks_qwen2p5/v3-20251103-130248/checkpoint-500 \
  --out-csv outputs/debug_eval_grpo_qwen7b.csv \
  --batch-size 2 --max-new-tokens 128 --torch-dtype bfloat16
```

可加上 `--limit 50` 快速 sanity check，再移除跑全量。

### 6.3 过滤不合规样本（可选）
`compute_metrics_from_debug.py` 已内置过滤逻辑；若想提前生成过滤后的 CSV 以便人工排查，可运行：

```bash
PYTHONPATH=/workspace/rl-dbap python - <<'PY'
import pandas as pd
from pathlib import Path
for name in ["student_seqkd", "qwen15b_base", "qwen7b_base", "grpo_qwen7b"]:
    path = Path(f"outputs/debug_eval_{name}.csv")
    if not path.exists():
        continue
    df = pd.read_csv(path)
    filt = df["raw_output"].astype(str).str.contains("holding_log_delta", case=False, na=False)
    out = path.with_name(path.stem + "_filtered.csv")
    df[filt].to_csv(out, index=False)
    print(f"[{name}] coverage = {filt.mean():.4f} ({filt.sum()}/{len(df)} rows) -> {out}")
PY
```

### 6.4 计算评测指标

[`scripts/compute_metrics_from_debug.py`](scripts/compute_metrics_from_debug.py) 在内部会先过滤掉 `raw_output` 中未包含 `holding_log_delta` 的样本，再计算与 `run_eval` 一致的指标。推荐对四套模型统一按 99% 分位裁剪误差：

```bash
export PYTHONPATH=/workspace/rl-dbap

for name in student_seqkd qwen15b_base qwen7b_base grpo_qwen7b; do
  python scripts/compute_metrics_from_debug.py \
    --debug-csv /workspace/rl-dbap/outputs/debug_eval_${name}.csv \
    --out-csv   /workspace/rl-dbap/outputs/metrics_${name}_q99.csv \
    --error-quantile 0.99
done
```

若想查看“全样本”表现，可省略 `--error-quantile` 并另存一份，例如 `metrics_${name}_full.csv`。如需完全关闭过滤，可显式传入 `--filter-substring ''`。

### 6.5 指标解读建议

- **coverage_filtered%**：过滤后仍保留的样本比例，可视作模型遵守输出合同的合规率。  
- **coverage_valid%**：在过滤结果上能成功解析为数字的占比。  
- **MAE/RMSE（log & tp1）**：用于衡量误差水平，推荐对比 `metrics_*_q99.csv` 结果。  
- **IC / RankIC / Top‑K**：衡量排序相关性与投资决策指标，需结合误差一起观察。  
- **全样本 vs. 分位裁剪**：全样本统计有助于发现极端异常（未合规、误解析等），而 0.99 分位提供相对稳健的模型对比。

实践中，GRPO Teacher 与 Student（蒸馏模型）在合规覆盖率与误差上最接近；原始 7B 基座次之；1.5B 基座在未对齐情况下合规率与误差都会明显劣化。

---

## 💾 7. 显存占用参考 (A100-40GB)

| 模式 | 说明 | 显存 | 备注 |
|------|------|------|------|
| 生成伪标签 | 仅 Teacher (8bit) | ~18 GB | 离线生成 |
| 在线蒸馏 | Teacher + Student (fp16) | ~36 GB | 主训练阶段 |
| 单模型评估 | 仅 Student | ~20 GB | 验证阶段 |

---

## 📊 8. 推荐目录结构

```
fdistill/
├── data/
│   ├── grpo_data.json
│   ├── sft_data.json
│   └── mixed/
│       ├── train_mix.json
│       └── teacher_outputs.json
├── output/
│   └── student_kd/
├── generate_teacher_outputs.py
├── run_qwen_kd.sh
└── README.md
```

---

## 🧩 9. 实验总结

| 阶段 | 模型组合 | 模式 | 目标 |
|------|------------|--------|--------|
| 1️⃣ | Qwen2.5-7B + LoRA | 推理 | 生成伪标签 |
| 2️⃣ | Teacher + Qwen1.8B | KL 蒸馏 | 学习 Teacher 分布 |
| 3️⃣ | Qwen1.8B | 评估 | 保留 GRPO 行为与通用能力 |

---

## ✅ 10. 运行提示

- 若遇 OOM，可调小 `batch_size` 或增大 `gradient_accumulation_steps`。  
- 若 Teacher 加载慢，可先手动 merge LoRA 权重至 base model。  
- 若 Student 收敛慢，可将 `alpha` 调低至 0.3，增强 CE 学习。

---

**最终输出路径：**
```
output/student_kd/
```
即为蒸馏完成的 Qwen1.8B 模型，可直接用于下游任务或推理部署。
