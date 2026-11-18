# Evaluate ONE finished human-data run: checkpoint selection, then Exp 1 and Exp 2 with
# the human evaluation cap (ceil(1.3 x longest accepted human demo)). See tools\human_runs.py.
#
#   powershell -ExecutionPolicy Bypass -File tools\eval_human.ps1 -Model act_lstm_full -Seed 0
#   powershell -ExecutionPolicy Bypass -File tools\eval_human.ps1 -Model act_orig -Seed 0 -Data human_v2
#
# Refuses if the run's training is not finished, or while another training or evaluation
# runs. Already evaluated -> says so. Interrupted -> run the same command again: finished
# result files are kept, an unfinished one is redone from its first seed.
# Writes runs\human_v1\summaries\<run>.eval.txt and a line in runs\human_v1\summary_human.txt.
param(
    [Parameter(Mandatory = $true)][ValidateSet("bc", "act_orig", "act_lstm_full")][string]$Model,
    [Parameter(Mandatory = $true)][ValidateSet(0, 1, 2)][int]$Seed,
    [ValidateSet("human_v1", "human_v2")][string]$Data = "human_v1"
)
Set-Location (Split-Path -Parent $PSScriptRoot)
& python -u tools\human_runs.py eval --data $Data --model $Model --seed $Seed
exit $LASTEXITCODE
