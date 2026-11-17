# Status of every human-data run and evaluation: done / not started / incomplete / running.
#
#   powershell -ExecutionPolicy Bypass -File tools\status_human.ps1
Set-Location (Split-Path -Parent $PSScriptRoot)
& python -u tools\human_runs.py status
exit $LASTEXITCODE
