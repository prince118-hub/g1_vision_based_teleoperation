# Memory-design runs on the nested-40 split (2026-10-24): act_orig and act_lstm_full.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\train_memory_design_n40.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\train_memory_design_n40.ps1 -DryRun
#
# act_orig      ACT on the original paper's input: the current observation only (W_o = 1)
# act_lstm_full act_orig + an LSTM over the whole episode history 0..t (g1_model/act_memory.py)
#
# Eight steps, ONE AT A TIME, in this order, into runs\scripted_v3_memory_design\:
#   1. train act_orig_n40_seed0
#   2. train act_lstm_full_n40_seed0
#   3. eval  act_orig_n40_seed0 act_lstm_full_n40_seed0
#   4. train act_orig_n40_seed1        5. train act_orig_n40_seed2
#   6. train act_lstm_full_n40_seed1   7. train act_lstm_full_n40_seed2
#   8. eval  act_orig_n40_seed1 act_orig_n40_seed2 act_lstm_full_n40_seed1 act_lstm_full_n40_seed2
# "eval" = tools\eval_scripted.py: checkpoint selection on the 20 selection seeds, then
# Exp 1 and Exp 2 on the chosen checkpoint, lock rule v3, --allow-lock-rule-mismatch
# (scripted_v3 data is v2; the experiments run v3 by decision, CLAUDE.md section 8
# 2026-10-24) -> runs\scripted_v3_memory_design\eval\<run>\ (selection.json, exp1.jsonl,
# exp2.jsonl, summary.txt). A separate root, so Run A / Run B and their analysis are untouched.
#
# Training settings are ACT's in tools\train_scripted_v3_n40.ps1 - same data, K = 100,
# batch 8, 200k steps, window 1000, checkpoint every 20k, no stop rule, --seed k - with
# the model's own W_o = 1 (stated: --obs-window 1).
#
# RESTART-SAFE. Completion is read from the files a step writes when it FINISHES, never
# from the exit code or the process window (the cuDNN LSTM crashes Python AT EXIT,
# 0xC0000409, after a clean finish - CLAUDE.md section 8, 2026-09-29):
#   training  the run's summary file result.json says "finished" or "hard_cap" -> SKIPPED.
#             A non-empty run folder without one (interrupted) is MOVED to
#             runs\scripted_v3_memory_design\_failed\<name>_<timestamp>\ (kept, never
#             deleted) with its stderr log and any evaluation of it (<dest>.eval), and
#             the run starts again from scratch.
#   eval      every run of the step has selection.json and complete exp1.jsonl and
#             exp2.jsonl (one line per seed) -> SKIPPED. Otherwise eval_scripted.py is
#             run, and it is restart-safe itself: finished files are kept, a stale .part
#             is redone, selection.json fixes the choice once written.
# Every step appends to runs\scripted_v3_memory_design\summary_memory_design_n40.txt
# (training lines in tools\write_resources.py's format).
#
# ONE HEAVY PROCESS AT A TIME: before the first step it WAITS, checking every
# -PollSeconds, until no python process runs train_bc.py or deploy.py, seen idle on two
# checks 60 s apart (as tools\train_scripted_v3_n40.ps1 does; an eval_scripted.py that
# is only waiting with --wait-for-training is not heavy, and itself waits for THIS
# launcher - eval_scripted.HEAVY names it).
#
# -DryRun: every training at --max-steps 200 (window 100, no periodic checkpoints), and
# each eval step replaced by ONE short deploy per run (selection seed 679, 100 ticks, on
# best.pt), into runs\scripted_v3_memory_design_dryrun\ - a plumbing/memory/time check of
# every path the real launch takes. Delete that folder after.
param([switch]$DryRun, [int]$PollSeconds = 300)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Tag = if ($DryRun) { "scripted_v3_memory_design_dryrun" } else { "scripted_v3_memory_design" }
$RunRoot = "runs\$Tag"
$Out = Join-Path $Root $RunRoot
New-Item -ItemType Directory -Force $Out | Out-Null
$Summary = Join-Path $Out "summary_memory_design_n40.txt"
$EvalRoot = Join-Path $Out "eval"

# ---- ACT's settings in tools\train_scripted_v3_n40.ps1, with W_o = 1 ------------
$Data40 = "--episodes recordings/scripted_v3 --splits data/processed/splits_scripted_v3_n40.json --norm data/processed/norm_stats_scripted_v3_n40.npz"
$Common = "--chunk 100 --obs-window 1 --batch 8"
$Why = "scripted_v3 (spec 1.2.0, lock rule v2): memory-design pair, ACT settings of the n40 runs"
$ActBudget = "--max-steps 200000 --window-steps 1000 --checkpoint-every 20000 --budget-reason `"$Why, 200k steps, checkpoint every 20k, no stop rule, selected later by task success`""
if ($DryRun) {
    $ActBudget = "--max-steps 200 --window-steps 100 --budget-reason `"DRY RUN: 200-step memory and time check, deleted after`""
}

function Train($Name, $Model, $Seed) {
    @{ Kind = "train"; Name = $Name; Args = "--model $Model --seed $Seed $Common $ActBudget $Data40" }
}
$Steps = @(
    (Train "act_orig_n40_seed0" "act_orig" 0),
    (Train "act_lstm_full_n40_seed0" "act_lstm_full" 0),
    @{ Kind = "eval"; Runs = @("act_orig_n40_seed0", "act_lstm_full_n40_seed0") },
    (Train "act_orig_n40_seed1" "act_orig" 1),
    (Train "act_orig_n40_seed2" "act_orig" 2),
    (Train "act_lstm_full_n40_seed1" "act_lstm_full" 1),
    (Train "act_lstm_full_n40_seed2" "act_lstm_full" 2),
    @{ Kind = "eval"; Runs = @("act_orig_n40_seed1", "act_orig_n40_seed2", "act_lstm_full_n40_seed1", "act_lstm_full_n40_seed2") }
)

# ---- wait until no other heavy process runs -------------------------------------
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

"$Tag memory_design n40 training started $(Get-Date -Format s)  commit $(git rev-parse --short HEAD)" | Out-File -Encoding utf8 -Append $Summary

function Say($Line) {
    # Out-File utf8, never Tee-Object -Append (UTF-16 in Windows PowerShell 5.1):
    # tools\write_resources.py and eval_scripted.py read this file as UTF-8
    Write-Host $Line
    $Line | Out-File -Encoding utf8 -Append $Summary
}

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

function Done($Status) { return ($Status -eq "finished" -or $Status -eq "hard_cap") }

function LineCount($Path) {
    if (-not (Test-Path $Path)) { return -1 }
    return @(Get-Content $Path | Where-Object { $_.Trim() -ne "" }).Count
}

function SeedCount($Path) {
    $j = Get-Content $Path -Raw | ConvertFrom-Json
    if ($j -is [System.Array]) { return $j.Count }
    return @($j.seeds).Count
}
$NExp1 = SeedCount (Join-Path $Root "data\eval\exp1_seeds.json")
$NExp2 = SeedCount (Join-Path $Root "data\eval\exp2_seeds.json")

function Evaluated($Name) {
    $d = Join-Path $EvalRoot $Name
    return ((Test-Path (Join-Path $d "selection.json")) -and
            ((LineCount (Join-Path $d "exp1.jsonl")) -eq $NExp1) -and
            ((LineCount (Join-Path $d "exp2.jsonl")) -eq $NExp2))
}

function RunTraining($r) {
    $Dir = Join-Path $Out $r.Name
    $ErrLog = Join-Path $Out "$($r.Name).stderr.log"
    $st = RunStatus $Dir
    if (Done $st) {
        Say ("{0,-24} SKIPPED: already complete (status={1})" -f $r.Name, $st)
        return
    }
    if ((Test-Path $Dir) -and @(Get-ChildItem $Dir -Force).Count -gt 0) {
        $Failed = Join-Path $Out "_failed"
        New-Item -ItemType Directory -Force $Failed | Out-Null
        $Dest = Join-Path $Failed "$($r.Name)_$(Get-Date -Format yyyyMMdd-HHmmss)"
        Move-Item $Dir $Dest
        if (Test-Path $ErrLog) { Move-Item $ErrLog "$Dest.stderr.log" }
        # an evaluation of the moved weights must not count for the retrained run
        $OldEval = Join-Path $EvalRoot $r.Name
        if (Test-Path $OldEval) { Move-Item $OldEval "$Dest.eval" }
        Say ("{0,-24} incomplete folder ({1}) moved to {2}; restarting" -f $r.Name, $st, $Dest)
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
    $StepsDone = ""
    if (Done $Status) {
        try { $StepsDone = (Get-Content (Join-Path $Dir "result.json") -Raw | ConvertFrom-Json).optimizer_steps } catch {}
    }
    $Mins = [math]::Round(((Get-Date) - $Start).TotalMinutes, 1)
    $RamGB = [math]::Round($PeakRam / 1GB, 2)
    Say ("{0,-24} status={1,-12} steps={2,-8} exit={3,-12} {4} min  peak RAM {5} GB  peak GPU {6} MiB" -f $r.Name, $Status, $StepsDone, $Code, $Mins, $RamGB, $PeakGpu)
    if (-not (Done $Status)) {
        Write-Host "  $($r.Name) did not finish cleanly - see $Log and $ErrLog. Moving on." -ForegroundColor Yellow
    }
}

function RunEval($s) {
    $Names = $s.Runs
    if ($DryRun) {
        # one short deploy per run: the deploy path of the new models, not an evaluation
        foreach ($n in $Names) {
            $Dir = Join-Path $Out $n
            $Jsonl = Join-Path $EvalRoot "$n\dryrun_deploy.jsonl"
            if (-not (Done (RunStatus $Dir))) {
                Say ("{0,-24} DRY-RUN deploy SKIPPED: training status {1}" -f $n, (RunStatus $Dir))
                continue
            }
            if ((LineCount $Jsonl) -eq 1) {
                Say ("{0,-24} DRY-RUN deploy SKIPPED: already done" -f $n)
                continue
            }
            if (Test-Path $Jsonl) { Remove-Item $Jsonl }
            New-Item -ItemType Directory -Force (Split-Path $Jsonl) | Out-Null
            & python -u tools\deploy.py --run "$RunRoot\$n" --ckpt best.pt --seed-list 679 --max-ticks 100 --lock-rule v3 --allow-lock-rule-mismatch --out $Jsonl 2>&1 |
                Tee-Object -FilePath (Join-Path $EvalRoot "$n\dryrun_deploy.log")
            Say ("{0,-24} DRY-RUN deploy lines={1} (1 = ok)" -f $n, (LineCount $Jsonl))
        }
        return
    }
    $Todo = @($Names | Where-Object { -not (Evaluated $_) })
    foreach ($n in $Names) {
        if (-not ($Todo -contains $n)) {
            Say ("{0,-24} EVAL SKIPPED: selection.json, exp1.jsonl and exp2.jsonl complete" -f $n)
        }
    }
    if ($Todo.Count -eq 0) { return }
    New-Item -ItemType Directory -Force $EvalRoot | Out-Null
    $Log = Join-Path $EvalRoot "memory_design_$(Get-Date -Format yyyyMMdd-HHmmss).log"
    Say "eval started $(Get-Date -Format s): $($Todo -join ' ')  log $Log"
    & python -u tools\eval_scripted.py --run-root $RunRoot --lock-rule v3 --allow-lock-rule-mismatch --runs $Todo 2>&1 |
        Tee-Object -FilePath $Log
    foreach ($n in $Todo) {
        $ok = if (Evaluated $n) { "complete" } else { "INCOMPLETE - see $Log" }
        Say ("{0,-24} eval {1}" -f $n, $ok)
    }
}

foreach ($s in $Steps) {
    if ($s.Kind -eq "train") { RunTraining $s } else { RunEval $s }
}

"$Tag memory_design n40 training ended $(Get-Date -Format s)" | Out-File -Encoding utf8 -Append $Summary
Write-Host ""
Get-Content $Summary
