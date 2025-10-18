param(
  [Parameter(Mandatory=$true)][string]$Type,
  [string]$Model = "Qwen/Qwen2.5-7B-Instruct",
  [string]$SftJson = "",
  [string]$GrpoJson = "",
  [int]$NumGenerations = 4,
  [int]$MaxCompletionLen = 512
)

$ErrorActionPreference = 'Stop'

if (-not $SftJson -or $SftJson -eq "") { $SftJson = "artifacts/sft/sft_train_$Type.jsonl" }
if (-not $GrpoJson -or $GrpoJson -eq "") { $GrpoJson = "artifacts/grpo/grpo_$Type.jsonl" }

if (-not (Test-Path $SftJson)) { throw "SFT dataset not found: $SftJson" }
if (-not (Test-Path $GrpoJson)) { throw "GRPO dataset not found: $GrpoJson" }

$SftOut = "outputs/sft_$Type"
$GrpoOut = "outputs/grpo_$Type"

Write-Host "[train-only] SFT: model=$Model dataset=$SftJson -> $SftOut"
powershell .\scripts\sft.ps1 -Model $Model -Dataset $SftJson -OutputDir $SftOut

Write-Host "[train-only] GRPO: model=$Model dataset=$GrpoJson adapters=$SftOut -> $GrpoOut"
powershell .\scripts\grpo.ps1 -Model $Model -Dataset $GrpoJson -OutputDir $GrpoOut -NumGenerations $NumGenerations -MaxCompletionLen $MaxCompletionLen -Adapters $SftOut

Write-Host "[train-only] Done. SFT=$SftOut  GRPO=$GrpoOut"

