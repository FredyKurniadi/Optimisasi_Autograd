$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Push-Location $root
if (-not (Test-Path ".venv\Scripts\python.exe")) {
  throw "Python environment belum siap. Jalankan scripts/setup_all.ps1 dulu."
}

.\.venv\Scripts\python.exe train/src/optimize.py `
  --train-config train/configs/train.yaml `
  --model-config train/configs/model.yaml

Pop-Location
