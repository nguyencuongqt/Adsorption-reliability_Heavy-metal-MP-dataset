$ErrorActionPreference = "Stop"

python -m venv .venv

$activateScript = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
. $activateScript

python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host ""
Write-Host "Environment ready."
Write-Host "Activate later with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Run the reproducible nested-tuning sensitivity analysis with:"
Write-Host "  python scripts/13_run_nested_tuning.py"
