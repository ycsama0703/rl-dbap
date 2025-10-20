param(
  [string]$Model = "Qwen/Qwen2.5-7B-Instruct",
  [string]$Dataset = "artifacts/grpo/grpo.jsonl",
  [string]$OutputDir = "outputs/grpo_qwen2.5_7b",
  [int]$NumGenerations = 4,
  [int]$MaxCompletionLen = 512,
  [switch]$UseVllm,
  [string]$Adapters = "",
  [string]$ResumeFrom = "",
  # New: make rewards configurable (defaults keep backward compatibility)
  # Default to pure MSE reward unless overridden
  [string[]]$RewardFuncs = @("mse_holdings"),
  [double[]]$RewardWeights = @(),
  [string]$StopWords = "",
  [double]$Temperature = 0.9
)

# Auto-detect latest checkpoint if -ResumeFrom not provided
if (-not $ResumeFrom -and (Test-Path $OutputDir)) {
  $ckpts = Get-ChildItem -Path $OutputDir -Directory -Filter "checkpoint-*" | ForEach-Object {
    $m = [regex]::Match($_.Name, 'checkpoint-(\d+)');
    if ($m.Success) { [pscustomobject]@{ Path = $_.FullName; Step = [int]$m.Groups[1].Value } }
  } | Sort-Object Step
  if ($ckpts -and $ckpts.Count -gt 0) {
    $ResumeFrom = $ckpts[-1].Path
  }
}

swift rlhf `
  --rlhf_type grpo `
  --model $Model `
  --external_plugins src/plugins/grpo/holdings_plugin.py `
  --reward_funcs $RewardFuncs `
  --train_type lora `
  --lora_rank 8 `
  --lora_alpha 32 `
  --target_modules all-linear `
  --torch_dtype bfloat16 `
  --dataset $Dataset `
  --load_from_cache_file true `
  --max_completion_length $MaxCompletionLen `
  --num_train_epochs 1 `
  --per_device_train_batch_size 1 `
  --learning_rate 1e-6 `
  --gradient_accumulation_steps 8 `
  --logging_steps 5 `
  --save_steps 100 `
  --save_total_limit 2 `
  --max_length 2048 `
  --output_dir $OutputDir `
  --warmup_ratio 0.05 `
  --dataset_num_proc 2 `
  --num_generations $NumGenerations `
  --temperature $Temperature `
  --beta 0.02 `
  --log_completions true `
  $(if ($RewardWeights.Count -gt 0) { "--reward_weights $($(($RewardWeights) -join ' '))" }) `
  $(if ($StopWords -ne "") { "--stop_words `"$StopWords`"" }) `
  $(if ($Adapters -ne "") { "--adapters `"$Adapters`"" }) `
  $(if ($ResumeFrom -ne "") { "--resume_from_checkpoint `"$ResumeFrom`"" }) `
  $(if ($UseVllm) { "--use_vllm true --vllm_mode colocate" })
