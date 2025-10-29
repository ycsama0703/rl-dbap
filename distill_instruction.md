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

```bash
mkdir -p data/mixed
python - <<'PY'
import json, random
grpo = json.load(open("data/grpo_data.json"))
sft  = json.load(open("data/sft_data.json"))
N = int(len(grpo) * 0.7)
M = int(len(sft)  * 0.3)
mix = random.sample(grpo, N) + random.sample(sft, M)
random.shuffle(mix)
json.dump(mix, open("data/mixed/train_mix.json", "w"), ensure_ascii=False, indent=2)
PY
```

---

## 🧠 4. 生成 Teacher 输出 (离线伪标签)

```bash
cat > generate_teacher_outputs.py <<'PY'
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, tqdm

base_model_path = "<base model path>"
lora_path = "<checkpoint path>"
data_path = "data/mixed/train_mix.json"
output_path = "data/mixed/teacher_outputs.json"

device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=dtype, device_map="auto", load_in_8bit=True
)
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

dataset = json.load(open(data_path))
outputs = []
for item in tqdm.tqdm(dataset):
    prompt = item.get("prompt", item.get("instruction", ""))
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    outputs.append({"prompt": prompt, "teacher_output": text})

json.dump(outputs, open(output_path, "w"), ensure_ascii=False, indent=2)
PY

python generate_teacher_outputs.py
```

输出文件：`data/mixed/teacher_outputs.json`

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
