# Train + evaluate a LIST of human-data runs back to back, one heavy process at a time.
# See tools\human_runs.py (queue).
#
#   powershell -ExecutionPolicy Bypass -File tools\queue_human.ps1 -Data human_v3
#   powershell -ExecutionPolicy Bypass -File tools\queue_human.ps1 -Data human_v3 -Plan
#   powershell -ExecutionPolicy Bypass -File tools\queue_human.ps1 -Data human_v3 -Restart
#   powershell -ExecutionPolicy Bypass -File tools\queue_human.ps1 -Data human_v3 -Models act_orig -Seeds 0,1
#
# Default list: bc, act_orig, act_lstm_full x seeds 0, 1, 2, SEED-major (all seed-0 runs
# first). For each run: train (skipped if finished), then evaluate (skipped if finished).
# RESUMABLE: run the same command again after a stop or a reboot.
#   - a finished training is never redone; an INTERRUPTED one is REFUSED and the queue stops;
#     add -Restart to move it to runs\<data>\_failed\ (kept) and retrain it from step 0;
#   - an interrupted evaluation redoes only its unfinished file.
# Waits (and says so) while another training or evaluation runs; -NoWait refuses instead.
# Ctrl+C stops the current step; that run then counts as interrupted.
param(
    [ValidateSet("human_v1", "human_v2", "human_v3")][string]$Data = "human_v1",
    [string]$Models = "bc,act_orig,act_lstm_full",   # comma-separated (powershell -File passes one string)
    [string]$Seeds = "0,1,2",                        # comma-separated
    [switch]$Restart,
    [switch]$NoWait,
    [switch]$Plan
)
Set-Location (Split-Path -Parent $PSScriptRoot)
$m = @($Models -split "[,\s]+" | Where-Object { $_ })
$s = @($Seeds -split "[,\s]+" | Where-Object { $_ })
$a = @("-u", "tools\human_runs.py", "queue", "--data", $Data, "--models") + $m + @("--seeds") + $s
if ($Restart) { $a += "--restart" }
if ($NoWait) { $a += "--no-wait" }
if ($Plan) { $a += "--plan" }
& python @a
exit $LASTEXITCODE
