# Train ONE human-data run (model + seed), then stop. See tools\human_runs.py.
#
#   powershell -ExecutionPolicy Bypass -File tools\run_human.ps1 -Model act_lstm_full -Seed 0
#   powershell -ExecutionPolicy Bypass -File tools\run_human.ps1 -Model act_lstm_full -Seed 0 -Restart
#   powershell -ExecutionPolicy Bypass -File tools\run_human.ps1 -Model act_orig -Seed 0 -Data human_v2
#
# -Data (default human_v1): human_v2 = the same seeds with smoothed walking labels (a TEST),
# runs\human_v2\<model>_human40v2_seed<k> - never touches runs\human_v1. human_v3 = human_v2
# + robot-frame task features (fix #1, 56-D observation), runs\human_v3\<model>_human40v3_seed<k>.
#
# Already finished -> says so and does nothing. Interrupted earlier -> REFUSES and tells
# you to add -Restart, which moves the incomplete run to runs\human_v1\_failed\ (kept)
# and trains it again from step 0. Refuses while another training or evaluation runs.
# Writes runs\human_v1\summaries\<run>.train.txt and a line in runs\human_v1\summary_human.txt.
param(
    [Parameter(Mandatory = $true)][ValidateSet("bc", "act_orig", "act_lstm_full")][string]$Model,
    [Parameter(Mandatory = $true)][ValidateSet(0, 1, 2)][int]$Seed,
    [switch]$Restart,
    [ValidateSet("human_v1", "human_v2", "human_v3")][string]$Data = "human_v1"
)
Set-Location (Split-Path -Parent $PSScriptRoot)
$a = @("-u", "tools\human_runs.py", "train", "--data", $Data, "--model", $Model, "--seed", $Seed)
if ($Restart) { $a += "--restart" }
& python @a
exit $LASTEXITCODE
