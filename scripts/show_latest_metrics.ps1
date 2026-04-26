$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$modelsRoot = Join-Path $root "models"

if (-not (Test-Path $modelsRoot)) {
  Write-Host "models folder belum ada"
  exit 0
}

$dirs = Get-ChildItem $modelsRoot -Directory | Where-Object { $_.Name -match '^model_\d{3}$' } | Sort-Object Name
if (-not $dirs -or $dirs.Count -eq 0) {
  Write-Host "belum ada output model"
  exit 0
}

$latest = $dirs[-1].FullName
$metricsPath = Join-Path $latest "metrics.json"

if (-not (Test-Path $metricsPath)) {
  Write-Host "metrics.json tidak ditemukan pada $latest"
  exit 1
}

Write-Host "Latest model folder: $latest"
Get-Content $metricsPath
