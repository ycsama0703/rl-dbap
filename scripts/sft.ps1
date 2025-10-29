param(
  [string]$Model = "Qwen/Qwen2.5-7B-Instruct",
  [string]$Dataset = "artifacts/sft/sft_train.jsonl",
  [string]$OutputDir = "outputs/sft_qwen2.5_7b"
)

swift sft `
  --model $Model `
  --train_type lora `
  --dataset $Dataset `
  --torch_dtype bfloat16 `
  --num_train_epochs 1 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 16 `
  --learning_rate 1e-4 `
  --lora_rank 8 `
  --lora_alpha 32 `
  --target_modules all-linear `
  --logging_steps 20 `
  --save_steps 500 `
  --save_total_limit 2 `
  --max_length 2048 `
  --output_dir $OutputDir `
  --system "You are a quantitative portfolio manager."
