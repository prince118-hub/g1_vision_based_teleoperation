# Re-run of tools\train_memory_design_n40.ps1 AFTER the current pass (2026-10-25).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\train_memory_design_n40_rerun.ps1
#
# Why: act_lstm_full_n40_seed0 died at ~25k steps (cuDNN packed-LSTM native abort,
# NOTES 2026-10-25c; fixed in fdc1cdc). Its folder still holds result.json "running".
# Running the launcher AGAIN moves that folder (and its eval) to
# runs\scripted_v3_memory_design\_failed\, retrains it from scratch with the fix,
# evaluates it, and SKIPS everything already complete (its own restart logic).
#
# ONE HEAVY PROCESS AT A TIME. This script WAITS, checking every -PollSeconds, until:
#   1. the CURRENT pass has fully ended: in summary_memory_design_n40.txt the LAST
#      "memory_design n40 training started" line has a later "... training ended" line.
#      Read from the file, never from the launcher's window process - that window may
#      be kept open with -NoExit after the pass ends.
#   2. no python process runs train_bc.py, deploy.py or eval_scripted.py,
#   3. both seen on two checks -ConfirmSeconds (60 s) apart.
# Then it re-checks once more and runs the launcher IN THIS WINDOW (the launcher also
# waits for heavy processes itself). If the launcher window is closed mid-pass, no
# "ended" line is ever written and this keeps waiting: it never guesses.
#
# Testing (never launches anything):
#   -TestOnly              print WOULD LAUNCH instead of launching
#   -Summary <path>        a different summary file
#   -HeavyPattern <regex>  a different "heavy" command-line pattern
#   -MaxChecks <n>         give up after n waiting checks (exit 3)
param([int]$PollSeconds = 300, [int]$ConfirmSeconds = 60, [switch]$TestOnly,
      [string]$Summary = "", [string]$HeavyPattern = '(train_bc|deploy|eval_scripted)\.py',
      [int]$MaxChecks = 0)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (-not $Summary) { $Summary = Join-Path $Root "runs\scripted_v3_memory_design\summary_memory_design_n40.txt" }
$Launcher = Join-Path $Root "tools\train_memory_design_n40.ps1"
$Log = Join-Path $Root ("runs\memory_design_rerun_waiter_{0}.log" -f (Get-Date -Format yyyyMMdd-HHmmss))

function Say($msg) {
    $line = "[$(Get-Date -Format s)] $msg"
    Write-Host $line
    if (-not $TestOnly) { $line | Out-File -Encoding utf8 -Append $Log }
}

function PassEnded {
    # The last "started" line has a later "ended" line.
    if (-not (Test-Path $Summary)) { return $false }
    $lines = @(Get-Content $Summary)
    $s = -1; $e = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match 'memory_design n40 training started') { $s = $i }
        if ($lines[$i] -match 'memory_design n40 training ended') { $e = $i }
    }
    return ($s -ge 0 -and $e -gt $s)
}

function Heavy {
    @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match '^python' -and $_.CommandLine -match $HeavyPattern })
}

Say "waiter started (poll $PollSeconds s, confirm $ConfirmSeconds s, summary $Summary$(if ($TestOnly) { ', TEST ONLY' }))"
$idle = 0
$checks = 0
while ($true) {
    $ended = PassEnded
    $h = @(Heavy)
    if (-not $ended -or $h.Count -gt 0) {
        $idle = 0
        $checks++
        $what = @()
        if (-not $ended) { $what += "current pass not ended" }
        if ($h.Count -gt 0) {
            $what += "heavy: " + (($h | ForEach-Object { if ($_.CommandLine -match $HeavyPattern) { $Matches[0] } }) -join ', ')
        }
        Say ("waiting: {0} - next check in {1} s" -f ($what -join '; '), $PollSeconds)
        if ($MaxChecks -gt 0 -and $checks -ge $MaxChecks) { Say "gave up after $checks check(s) (-MaxChecks)"; exit 3 }
        Start-Sleep -Seconds $PollSeconds
        continue
    }
    $idle++
    if ($idle -ge 2) { break }
    Say "pass ended and nothing heavy - confirming in $ConfirmSeconds s"
    Start-Sleep -Seconds $ConfirmSeconds
}

# Final re-check immediately before launching.
if (-not (PassEnded) -or @(Heavy).Count -gt 0) {
    Say "state changed at the last moment - NOT launching; re-run this script"
    exit 4
}
if ($TestOnly) {
    Say "WOULD LAUNCH: powershell -NoProfile -ExecutionPolicy Bypass -File $Launcher -PollSeconds 60"
    exit 0
}
Say "launching $Launcher (this window)"
& powershell -NoProfile -ExecutionPolicy Bypass -File $Launcher -PollSeconds 60
Say "launcher returned (exit $LASTEXITCODE); its summary: $Summary"
