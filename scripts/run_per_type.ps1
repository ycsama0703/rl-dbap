param(
  [Parameter(Mandatory=$true)][string]$Type,
  [string]$Model = "Qwen/Qwen2.5-7B-Instruct",
  [string]$SftEnd = "2016-12-31",
  [string]$GrpoStart = "2017-01-01",
  [string]$GrpoEnd = "2018-12-31",
  [int]$NumGenerations = 4,
  [int]$MaxCompletionLen = 512,
  [int]$PerTypeLimit = 1000,
  [int]$TimeBins = 10,
  [int]$CapPerPair = 3
)

$ErrorActionPreference = 'Stop'

$InDir = "data/processed/panel_quarter.parquet"

# SFT
$SftPrompts = "artifacts/prompts_hist_sft_$Type"
$SftJsonl = "artifacts/sft/sft_train_$Type.jsonl"
$SftOut = "outputs/sft_$Type"

Write-Host "[pipeline] Building SFT prompts for type=$Type until $SftEnd ..."
python -m src.cli.build_history_prompts `
  --in-dir $InDir `
  --out-dir $SftPrompts `
  --include-types $Type `
  --date-end $SftEnd `
  --per-type-limit $PerTypeLimit `
  --time-bins $TimeBins `
  --cap-per-pair $CapPerPair

Write-Host "[pipeline] Converting SFT prompts -> $SftJsonl ..."
python -m src.cli.prompts_to_sft --in "$SftPrompts/$Type.jsonl" --out $SftJsonl

Write-Host "[pipeline] Running SFT training -> $SftOut ..."
powershell .\scripts\sft.ps1 -Model $Model -Dataset $SftJsonl -OutputDir $SftOut

# GRPO
$GrpoPrompts = "artifacts/prompts_hist_grpo_$Type"
$GrpoJsonl = "artifacts/grpo/grpo_$Type.jsonl"
$GrpoOut = "outputs/grpo_$Type"

Write-Host "[pipeline] Building GRPO prompts for type=$Type in [$GrpoStart, $GrpoEnd] ..."
python -m src.cli.build_history_prompts `
  --in-dir $InDir `
  --out-dir $GrpoPrompts `
  --include-types $Type `
  --date-start $GrpoStart `
  --date-end $GrpoEnd `
  --per-type-limit $PerTypeLimit `
  --time-bins $TimeBins `
  --cap-per-pair $CapPerPair

Write-Host "[pipeline] Converting GRPO prompts -> $GrpoJsonl ..."
python -m src.cli.prompts_to_grpo --in "$GrpoPrompts/$Type.jsonl" --out $GrpoJsonl

Write-Host "[pipeline] Running GRPO from SFT adapters -> $GrpoOut ..."
powershell .\scripts\grpo.ps1 -Model $Model -Dataset $GrpoJsonl -OutputDir $GrpoOut -NumGenerations $NumGenerations -MaxCompletionLen $MaxCompletionLen -Adapters $SftOut

Write-Host "[pipeline] Done. SFT=$SftOut  GRPO=$GrpoOut"

