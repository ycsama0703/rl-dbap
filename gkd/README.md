GKD Distillation
================

Minimal wrapper to run TRL GKD using our chat JSONL datasets (`messages` field).

Example (local models, mutual_funds SFT as train, test as eval):

```bash
PYTHONPATH=. python gkd/train_gkd.py \
  --student /root/autodl-tmp/rl-dbap/models/Qwen2.5-7B-Instruct \
  --teacher /root/autodl-tmp/rl-dbap/outputs/grpo_mutual_funds/v0-20251204-233156/checkpoint-1000 \
  --train-path artifacts/sft/sft_train_mutual_funds.jsonl \
  --eval-path artifacts/test/test_mutual_funds.jsonl \
  --output-dir outputs/gkd_mutual_funds \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-5 \
  --num-train-epochs 1 \
  --max-seq-length 2048 \
  --bf16
```

Notes:
- Input JSONL must contain `messages` (system/user/assistant with loss labels if needed).
- Use local model paths to avoid network pulls.
- Adjust batch size/grad accumulation per GPU memory; set `save_steps`/`eval_steps` as needed.
