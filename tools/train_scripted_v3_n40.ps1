# Run A training: the nested-40 split, more seeds (2026-10-23).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\train_scripted_v3_n40.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\train_scripted_v3_n40.ps1 -DryRun
#
# Seven runs, ONE AT A TIME, into runs\scripted_v3\<model>_n40_seed<k>\:
#   bc_n40_seed0, bc_n40_seed1, bc_n40_seed2,
#   act_n40_seed1, act_lstm_n40_seed1, act_n40_seed2, act_lstm_n40_seed2
# (act_n40_seed0 and act_lstm_n40_seed0 exist, from tools\train_scripted_v3.ps1.)
# Settings are EXACTLY tools\train_scripted_v3.ps1's - same data, W_o 12, batch 8,
# BC 300 epochs / checkpoint every 30, ACT and ACT-LSTM K=100, 200k steps,
# checkpoint every 20k, no stop rule, same budget-reason text - plus
# train_bc.py --seed k (initial weights, data order, dropout). Disclose: BC is
# epoch-budgeted, so BC-n40 takes ~4x fewer optimizer steps than BC-160
# (823k vs 3.31M); ACT/ACT-LSTM are step-budgeted, as before.
#
# RESTART-SAFE. A run whose result.json says "finished" or "hard_cap" is SKIPPED.
# A non-empty run folder without one (an interrupted run) is MOVED to
# runs\scripted_v3\_failed\<name>_<timestamp>\ (kept, never deleted) with its
# stderr log, and the run starts again from scratch.
#
# ONE HEAVY PROCESS AT A TIME: before the first run it WAITS, checking every
# -PollSeconds, until no python process runs train_bc.py or deploy.py (Run B's
# evaluation, or an earlier launcher), seen idle on two checks 60 s apart.
#
# Success is read from result.json "status", NOT from the exit code: the cuDNN
# LSTM crashes Python AT EXIT (0xC0000409) after a clean finish (CLAUDE.md
# section 8, 2026-09-29). Summary lines go to runs\scripted_v3\summary_n40.txt;
# its "ended" line is what tools\eval_scripted_v3_n40.ps1 waits for.
#
# -DryRun: every run at --max-steps 200 (window 100, no periodic checkpoints)
# into runs\scripted_v3_n40_dryrun\ - a plumbing/memory/time check. Delete that
# folder after.
param([switch]$DryRun, [int]$PollSeconds = 300)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Tag = if ($DryRun) { "scripted_v3_n40_dryrun" } else { "scripted_v3" }
$Out = Join-Path $Root "runs\$Tag"
New-Item -ItemType Directory -Force $Out | Out-Null
$Summary = Join-Path $Out "summary_n40.txt"

# ---- identical to tools\train_scripted_v3.ps1 ---------------------------------
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
    @{ Name = "bc_n40_seed0";       Args = "--model bc --chunk 1 --seed 0 $Common $BcBudget $Data40" },
    @{ Name = "bc_n40_seed1";       Args = "--model bc --chunk 1 --seed 1 $Common $BcBudget $Data40" },
    @{ Name = "bc_n40_seed2";       Args = "--model bc --chunk 1 --seed 2 $Common $BcBudget $Data40" },
    @{ Name = "act_n40_seed1";      Args = "--model act --chunk 100 --seed 1 $Common $ActBudget $Data40" },
    @{ Name = "act_lstm_n40_seed1"; Args = "--model act_lstm --chunk 100 --seed 1 $Common $ActBudget $Data40" },
    @{ Name = "act_n40_seed2";      Args = "--model act --chunk 100 --seed 2 $Common $ActBudget $Data40" },
    @{ Name = "act_lstm_n40_seed2"; Args = "--model act_lstm --chunk 100 --seed 2 $Common $ActBudget $Data40" }
)

# ---- wait until no other heavy process runs ------------------------------------
function Heavy {
    @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match '^python' -and $_.CommandLine -match '(train_bc|deploy)\.py' })
}
$idle = 0
while ($true) {
    $h = @(Heavy)
    if ($h.Count -gt 0) {
        $idle = 0
        $what = ($h | ForEach-Object { if ($_.CommandLine -match '(train_bc|deploy)\.py') { $Matches[0] } }) -join ', '
        Write-Host "[$(Get-Date -Format s)] busy: $what - next check in $PollSeconds s"
        Start-Sleep -Seconds $PollSeconds
        continue
    }
    $idle++
    if ($idle -ge 2) { break }
    Write-Host "[$(Get-Date -Format s)] nothing heavy running - confirming in 60 s"
    Start-Sleep -Seconds 60
}

$dirty = git status --porcelain --untracked-files=no
if ($dirty) {
    Write-Host "WARNING: uncommitted changes to tracked files - the runs' git_commit will not identify the code exactly:" -ForegroundColor Yellow
    Write-Host $dirty
}

"$Tag n40 training started $(Get-Date -Format s)  commit $(git rev-parse --short HEAD)" | Out-File -Encoding utf8 -Append $Summary

function GpuUsedMiB {
    try {
        $v = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        return [int]($v | Select-Object -First 1)
    } catch { return -1 }
}

function RunStatus($Dir) {
    $f = Join-Path $Dir "result.json"
    if (-not (Test-Path $f)) { return "no result.json" }
    try { return (Get-Content $f -Raw | ConvertFrom-Json).status } catch { return "unreadable result.json" }
}

foreach ($r in $Runs) {
    $Dir = Join-Path $Out $r.Name
    $ErrLog = Join-Path $Out "$($r.Name).stderr.log"
    $st = RunStatus $Dir
    if ($st -eq "finished" -or $st -eq "hard_cap") {
        $Line = "{0,-20} SKIPPED: already complete (status={1})" -f $r.Name, $st
        Write-Host $Line -ForegroundColor Yellow
        $Line | Out-File -Encoding utf8 -Append $Summary
        continue
    }
    if ((Test-Path $Dir) -and @(Get-ChildItem $Dir -Force).Count -gt 0) {
        $Failed = Join-Path $Out "_failed"
        New-Item -ItemType Directory -Force $Failed | Out-Null
        $Dest = Join-Path $Failed "$($r.Name)_$(Get-Date -Format yyyyMMdd-HHmmss)"
        Move-Item $Dir $Dest
        if (Test-Path $ErrLog) { Move-Item $ErrLog "$Dest.stderr.log" }
        $Line = "{0,-20} incomplete folder ({1}) moved to {2}; restarting" -f $r.Name, $st, $Dest
        Write-Host $Line -ForegroundColor Yellow
        $Line | Out-File -Encoding utf8 -Append $Summary
    }
    New-Item -ItemType Directory -Force $Dir | Out-Null
    $Log = Join-Path $Dir "train.log"
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

    $Status = RunStatus $Dir
    $Steps = ""
    if ($Status -eq "finished" -or $Status -eq "hard_cap") {
        try { $Steps = (Get-Content (Join-Path $Dir "result.json") -Raw | ConvertFrom-Json).optimizer_steps } catch {}
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

"$Tag n40 training ended $(Get-Date -Format s)" | Out-File -Encoding utf8 -Append $Summary
Write-Host ""
Get-Content $Summary
