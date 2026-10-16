# Train BC, then ACT, then ACT-LSTM on scripted_v1 (PHASE5 step 8).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\train_scripted_v1.ps1
#
# Each model writes runs\scripted_v1\<model>_seed0\ (train.log, metrics.jsonl,
# result.json, metadata.json, best.pt, last.pt, state.pt and every periodic
# ckpt_*.pt). If one model fails, the next one still runs.
#
# Success is read from result.json "status", NOT from the exit code: the cuDNN
# LSTM crashes Python AT EXIT (0xC0000409) after a clean finish (CLAUDE.md
# section 8, 2026-09-29), so ACT-LSTM's exit code says nothing about its run.
#
# A model whose run directory already holds files is refused by train_bc.py
# (--out-dir), so a re-launch never overwrites a finished run: move or rename
# the old directory first.

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Out = Join-Path $Root "runs\scripted_v1"
New-Item -ItemType Directory -Force $Out | Out-Null
$Summary = Join-Path $Out "summary.txt"

$Data = "--episodes recordings/scripted_v1 --splits data/processed/splits_scripted_v1.json --norm data/processed/norm_stats_scripted_v1.npz"
$Common = "--obs-window 12 --batch 8"

$Runs = @(
    @{ Name = "bc_seed0"; Args = "--model bc --chunk 1 $Common --epochs 300 --checkpoint-every 30 --budget-reason `"PHASE5 step 8: 300 epochs, checkpoint every 30, selected later by task success`"" },
    @{ Name = "act_seed0"; Args = "--model act --chunk 100 $Common --max-steps 200000 --window-steps 1000 --checkpoint-every 20000 --budget-reason `"PHASE5 step 8: 200k steps, checkpoint every 20k, no stop rule, selected later by task success`"" },
    @{ Name = "act_lstm_seed0"; Args = "--model act_lstm --chunk 100 $Common --max-steps 200000 --window-steps 1000 --checkpoint-every 20000 --budget-reason `"PHASE5 step 8: 200k steps, checkpoint every 20k, no stop rule, selected later by task success`"" }
)

$dirty = git status --porcelain --untracked-files=no
if ($dirty) {
    Write-Host "WARNING: uncommitted changes to tracked files - the runs' git_commit will not identify the code exactly:" -ForegroundColor Yellow
    Write-Host $dirty
}

"scripted_v1 training started $(Get-Date -Format s)  commit $(git rev-parse --short HEAD)" | Out-File -Encoding utf8 $Summary

foreach ($r in $Runs) {
    $Dir = Join-Path $Out $r.Name
    New-Item -ItemType Directory -Force $Dir | Out-Null
    $Log = Join-Path $Dir "train.log"
    $Start = Get-Date
    Write-Host ""
    Write-Host "=== $($r.Name) started $(Get-Date -Format s) -> $Dir" -ForegroundColor Cyan

    $Cmd = "python -u tools\train_bc.py train $($r.Args) $Data --out-dir `"$Dir`" > `"$Log`" 2>&1"
    cmd /c $Cmd
    $Code = $LASTEXITCODE

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
    $Line = "{0,-16} status={1,-12} steps={2,-8} exit={3,-12} {4} min" -f $r.Name, $Status, $Steps, $Code, $Mins
    Write-Host $Line
    $Line | Out-File -Encoding utf8 -Append $Summary
    if ($Status -ne "finished" -and $Status -ne "hard_cap") {
        Write-Host "  $($r.Name) did not finish cleanly - see $Log. Moving on." -ForegroundColor Yellow
    }
}

"scripted_v1 training ended $(Get-Date -Format s)" | Out-File -Encoding utf8 -Append $Summary
Write-Host ""
Get-Content $Summary
