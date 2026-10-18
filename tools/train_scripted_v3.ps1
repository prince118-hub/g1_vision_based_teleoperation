# Train on scripted_v3 (spec g1-spec-1.2.0, lock rule v2, setting A, 2026-10-18).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\train_scripted_v3.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\train_scripted_v3.ps1 -DryRun
#
# Five runs, ONE AT A TIME (7.7 GB RAM), in this order:
#   act_lstm_160_seed0   ACT-LSTM on the 160-episode train split
#   act_160_seed0        ACT on the 160
#   bc_160_seed0         BC on the 160
#   act_lstm_n40_seed0   ACT-LSTM on the nested 40 (a stratified subset of the 160)
#   act_n40_seed0        ACT on the nested 40
# Both sizes validate on the SAME 40 episodes; each size has its own norm stats
# (fitted on its own training split). Settings are scripted_v2's: seed 0, W_o 12,
# batch 8; BC 300 epochs, checkpoint every 30; ACT / ACT-LSTM K=100, 200k steps,
# checkpoint every 20k, no stop rule - checkpoints are selected later by task
# success (tools/deploy.py --selection). Disclose: with the step budget fixed,
# the 160 runs see ~4x fewer epochs than the n40 runs.
#
# Each run writes runs\scripted_v3\<model>_<size>_seed0\ (train.log, metrics.jsonl,
# result.json, metadata.json, best.pt, last.pt, state.pt, ckpt_*.pt). If one run
# fails the next still starts. Success is read from result.json "status", NOT
# from the exit code: the cuDNN LSTM crashes Python AT EXIT (0xC0000409) after a
# clean finish (CLAUDE.md section 8, 2026-09-29).
#
# The summary line of every run also carries the PEAK RAM of the training
# process and the peak GPU memory in use (nvidia-smi), polled every 5 s.
#
# -DryRun: every run at --max-steps 200 (window 100, no periodic checkpoints)
# into runs\scripted_v3_dryrun\ - a memory/time check. Delete that folder after.
#
# train_bc.py refuses a run directory that already holds files, so a re-launch
# never overwrites a finished run: move or rename the old directory first.
param([switch]$DryRun)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Tag = if ($DryRun) { "scripted_v3_dryrun" } else { "scripted_v3" }
$Out = Join-Path $Root "runs\$Tag"
New-Item -ItemType Directory -Force $Out | Out-Null
$Summary = Join-Path $Out "summary.txt"

$Data160 = "--episodes recordings/scripted_v3 --splits data/processed/splits_scripted_v3.json --norm data/processed/norm_stats_scripted_v3.npz"
$Data40 = "--episodes recordings/scripted_v3 --splits data/processed/splits_scripted_v3_n40.json --norm data/processed/norm_stats_scripted_v3_n40.npz"
$Common = "--obs-window 12 --batch 8"
$Why = "scripted_v3 (spec 1.2.0, lock rule v2): scripted_v2 settings"
$BcBudget = "--epochs 300 --checkpoint-every 30 --budget-reason `"$Why, 300 epochs, checkpoint every 30, selected later by task success`""
$ActBudget = "--max-steps 200000 --window-steps 1000 --checkpoint-every 20000 --budget-reason `"$Why, 200k steps, checkpoint every 20k, no stop rule, selected later by task success`""
if ($DryRun) {
    $BcBudget = "--max-steps 200 --window-steps 100 --budget-reason `"DRY RUN: 200-step memory and time check, deleted after`""
    $ActBudget = $BcBudget
}

$Runs = @(
    @{ Name = "act_lstm_160_seed0"; Args = "--model act_lstm --chunk 100 $Common $ActBudget $Data160" },
    @{ Name = "act_160_seed0";      Args = "--model act --chunk 100 $Common $ActBudget $Data160" },
    @{ Name = "bc_160_seed0";       Args = "--model bc --chunk 1 $Common $BcBudget $Data160" },
    @{ Name = "act_lstm_n40_seed0"; Args = "--model act_lstm --chunk 100 $Common $ActBudget $Data40" },
    @{ Name = "act_n40_seed0";      Args = "--model act --chunk 100 $Common $ActBudget $Data40" }
)

$dirty = git status --porcelain --untracked-files=no
if ($dirty) {
    Write-Host "WARNING: uncommitted changes to tracked files - the runs' git_commit will not identify the code exactly:" -ForegroundColor Yellow
    Write-Host $dirty
}

"$Tag training started $(Get-Date -Format s)  commit $(git rev-parse --short HEAD)" | Out-File -Encoding utf8 $Summary

function GpuUsedMiB {
    try {
        $v = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        return [int]($v | Select-Object -First 1)
    } catch { return -1 }
}

foreach ($r in $Runs) {
    $Dir = Join-Path $Out $r.Name
    New-Item -ItemType Directory -Force $Dir | Out-Null
    $Log = Join-Path $Dir "train.log"
    # stderr beside the run, not in it: train_bc.py refuses a run directory
    # holding anything but train.log.
    $ErrLog = Join-Path $Out "$($r.Name).stderr.log"
    $Start = Get-Date
    Write-Host ""
    Write-Host "=== $($r.Name) started $(Get-Date -Format s) -> $Dir" -ForegroundColor Cyan

    $ArgLine = "-u tools\train_bc.py train $($r.Args) --out-dir `"$Dir`""
    $p = Start-Process -FilePath "python" -ArgumentList $ArgLine -NoNewWindow -PassThru `
        -RedirectStandardOutput $Log -RedirectStandardError $ErrLog
    $null = $p.Handle                 # keep the handle so ExitCode is readable after exit
    $PeakRam = 0
    $PeakGpu = GpuUsedMiB
    while (-not $p.HasExited) {
        try { $p.Refresh(); if ($p.PeakWorkingSet64 -gt $PeakRam) { $PeakRam = $p.PeakWorkingSet64 } } catch {}
        $g = GpuUsedMiB
        if ($g -gt $PeakGpu) { $PeakGpu = $g }
        Start-Sleep -Seconds 5
    }
    $p.WaitForExit()
    $Code = $p.ExitCode

    $Status = "no result.json"
    $Steps = ""
    $ResultFile = Join-Path $Dir "result.json"
    if (Test-Path $ResultFile) {
        try {
            $Res = Get-Content $ResultFile -Raw | ConvertFrom-Json
            $Status = $Res.status
            $Steps = $Res.optimizer_steps
        } catch {
            $Status = "unreadable result.json"
        }
    }
    $Mins = [math]::Round(((Get-Date) - $Start).TotalMinutes, 1)
    $RamGB = [math]::Round($PeakRam / 1GB, 2)
    $Line = "{0,-20} status={1,-12} steps={2,-8} exit={3,-12} {4} min  peak RAM {5} GB  peak GPU {6} MiB" -f $r.Name, $Status, $Steps, $Code, $Mins, $RamGB, $PeakGpu
    Write-Host $Line
    $Line | Out-File -Encoding utf8 -Append $Summary
    if ($Status -ne "finished" -and $Status -ne "hard_cap") {
        Write-Host "  $($r.Name) did not finish cleanly - see $Log and $ErrLog. Moving on." -ForegroundColor Yellow
    }
}

"$Tag training ended $(Get-Date -Format s)" | Out-File -Encoding utf8 -Append $Summary
Write-Host ""
Get-Content $Summary
