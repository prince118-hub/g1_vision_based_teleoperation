# The three 160-episode runs of tools\train_scripted_v3.ps1, run AFTER it.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\train_scripted_v3_rest.ps1
#
# Why this exists (2026-10-18): act_lstm_160, act_160 and bc_160 were stopped by
# Ctrl+C in the first launch, and the relaunch refused their non-empty folders
# (moved to runs\scripted_v3\_failed\, kept). The relaunch went on to the n40
# runs. This script runs ONLY the three 160 runs, with the SAME arguments,
# output folders (runs\scripted_v3\<model>_160_seed0\) and summary line format
# as train_scripted_v3.ps1, appending to the same runs\scripted_v3\summary.txt.
#
# ONE TRAINING AT A TIME (7.7 GB RAM): it first WAITS, checking every 5 minutes,
# until no train_bc.py process exists AND the main launcher has finished. The
# main launcher may have been started with -NoExit, so its window can outlive
# the script: "finished" is read from summary.txt (its last "scripted_v3
# training started" line has a later "scripted_v3 training ended" line), not from
# the window's process. No train_bc.py may be seen on two checks 60 s apart, so
# the gap between two runs of the main launcher is never mistaken for the end.
#
# Run order: act_lstm_160, act_160, bc_160. Success is read from result.json
# "status", NOT the exit code: the cuDNN LSTM crashes Python AT EXIT (0xC0000409)
# after a clean finish (CLAUDE.md section 8, 2026-09-29). train_bc.py refuses a
# run folder that already holds files, so a finished run is never overwritten.
param([int]$PollSeconds = 300)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Out = Join-Path $Root "runs\scripted_v3"
New-Item -ItemType Directory -Force $Out | Out-Null
$Summary = Join-Path $Out "summary.txt"

# ---- identical to tools\train_scripted_v3.ps1 ---------------------------------
$Data160 = "--episodes recordings/scripted_v3 --splits data/processed/splits_scripted_v3.json --norm data/processed/norm_stats_scripted_v3.npz"
$Common = "--obs-window 12 --batch 8"
$Why = "scripted_v3 (spec 1.2.0, lock rule v2): scripted_v2 settings"
$BcBudget = "--epochs 300 --checkpoint-every 30 --budget-reason `"$Why, 300 epochs, checkpoint every 30, selected later by task success`""
$ActBudget = "--max-steps 200000 --window-steps 1000 --checkpoint-every 20000 --budget-reason `"$Why, 200k steps, checkpoint every 20k, no stop rule, selected later by task success`""
$Runs = @(
    @{ Name = "act_lstm_160_seed0"; Args = "--model act_lstm --chunk 100 $Common $ActBudget $Data160" },
    @{ Name = "act_160_seed0";      Args = "--model act --chunk 100 $Common $ActBudget $Data160" },
    @{ Name = "bc_160_seed0";       Args = "--model bc --chunk 1 $Common $BcBudget $Data160" }
)

# ---- wait for the main launcher ----------------------------------------------
function Trainers {
    @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'train_bc\.py' })
}

function MainLauncherFinished {
    # Running copies of the main launcher (not this script). None at all = finished.
    $mine = $PID
    $main = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
              Where-Object { $_.ProcessId -ne $mine -and $_.CommandLine -match 'train_scripted_v3\.ps1' })
    if ($main.Count -eq 0) { return $true }
    # A window kept open with -NoExit: finished once summary.txt says so.
    if (-not (Test-Path $Summary)) { return $false }
    $lines = @(Get-Content $Summary)
    $s = -1; $e = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^\W*scripted_v3 training started') { $s = $i }
        if ($lines[$i] -match '^\W*scripted_v3 training ended') { $e = $i }
    }
    return ($s -ge 0 -and $e -gt $s)
}

Write-Host "Waiting for the current training to finish (checking every $PollSeconds s)..."
$idle = 0
while ($true) {
    $busy = @(Trainers).Count -gt 0 -or -not (MainLauncherFinished)
    if ($busy) {
        $idle = 0
        $t = @(Trainers) | ForEach-Object { if ($_.CommandLine -match 'out-dir\s+"?([^"]+)"?') { Split-Path -Leaf $Matches[1] } }
        Write-Host "[$(Get-Date -Format s)] busy: $(@($t) -join ', ') - next check in $PollSeconds s"
        Start-Sleep -Seconds $PollSeconds
        continue
    }
    $idle++
    if ($idle -ge 2) { break }
    Write-Host "[$(Get-Date -Format s)] no training seen - confirming in 60 s"
    Start-Sleep -Seconds 60
}
Write-Host "[$(Get-Date -Format s)] the main launcher has finished - starting the three 160 runs." -ForegroundColor Cyan

# ---- the runs (same loop as train_scripted_v3.ps1) ----------------------------
$dirty = git status --porcelain --untracked-files=no
if ($dirty) {
    Write-Host "WARNING: uncommitted changes to tracked files - the runs' git_commit will not identify the code exactly:" -ForegroundColor Yellow
    Write-Host $dirty
}

"scripted_v3 (rest: the 160 runs) started $(Get-Date -Format s)  commit $(git rev-parse --short HEAD)" | Out-File -Encoding utf8 -Append $Summary

function GpuUsedMiB {
    try {
        $v = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
        return [int]($v | Select-Object -First 1)
    } catch { return -1 }
}

foreach ($r in $Runs) {
    $Dir = Join-Path $Out $r.Name
    if ((Test-Path $Dir) -and @(Get-ChildItem $Dir -Force).Count -gt 0) {
        $Line = "{0,-20} SKIPPED: $Dir already holds files (move it away to re-run)" -f $r.Name
        Write-Host $Line -ForegroundColor Yellow
        $Line | Out-File -Encoding utf8 -Append $Summary
        continue
    }
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

"scripted_v3 (rest: the 160 runs) ended $(Get-Date -Format s)" | Out-File -Encoding utf8 -Append $Summary
Write-Host ""
Get-Content $Summary
