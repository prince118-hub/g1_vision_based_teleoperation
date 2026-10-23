# Run A evaluation: checkpoint selection + Experiments 1 and 2 for all nine n40 runs.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\eval_scripted_v3_n40.ps1
#
# Safe to start at ANY time, before or after tools\train_scripted_v3_n40.ps1: it
# WAITS (tools\eval_scripted.py --wait-for-training) until training has finished -
# every run complete, or runs\scripted_v3\summary_n40.txt has an "ended" line
# after its last "started" line - and nothing heavy is running, then evaluates,
# one deploy at a time, under lock rule v3. Same procedure as Run B (the 160
# runs); see tools\eval_scripted.py's docstring for the selection rule and the
# outputs (runs\scripted_v3\eval\<run>\selection.json, exp1.jsonl, exp2.jsonl,
# runs\scripted_v3\eval\summary.txt). Restart-safe: finished files are kept.
# A run whose training did not finish is skipped and reported.
param([int]$PollSeconds = 300)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Runs = "bc_n40_seed0 bc_n40_seed1 bc_n40_seed2 act_n40_seed0 act_n40_seed1 act_n40_seed2 act_lstm_n40_seed0 act_lstm_n40_seed1 act_lstm_n40_seed2"
$Log = Join-Path $Root "runs\scripted_v3\eval\run_a_n40_$(Get-Date -Format yyyyMMdd-HHmmss).log"
New-Item -ItemType Directory -Force (Split-Path $Log) | Out-Null
Write-Host "log: $Log"
& python -u tools\eval_scripted.py --wait-for-training --poll $PollSeconds --lock-rule v3 --runs $Runs.Split(" ") 2>&1 |
    Tee-Object -FilePath $Log
