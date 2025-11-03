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

## 📈 6. 模型评估

```bash
python eval_model.py \
  --model_path ./output/student_kd \
  --data_path data/val.json \
  --metrics bleu rouge ppl
```

或使用 nlg-eval:
```bash
nlg-eval --hypothesis=student_output.txt --references=ref.txt
```

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
