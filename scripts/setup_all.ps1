param(
  [switch]$RecreateVenv = $false
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Write-Host "[setup] project root: $root"
Push-Location $root

$pythonCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
  try {
    py -3.11 -c "import sys; print(sys.version)" | Out-Null
    $pythonCmd = "py -3.11"
  } catch {
    $pythonCmd = $null
  }
}

if (-not $pythonCmd) {
  $pythonExe = Get-Command python -ErrorAction SilentlyContinue
  if ($pythonExe) {
    try {
      python -c "import sys; assert sys.version_info[:2] >= (3, 10)" | Out-Null
      $pythonCmd = "python"
    } catch {
      $pythonCmd = $null
    }
  }
}

if (-not $pythonCmd) {
  throw "Python 3.10+ tidak ditemukan. Install Python lalu jalankan lagi."
}

if ($RecreateVenv -and (Test-Path ".venv")) {
  Remove-Item -Recurse -Force .venv
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  Invoke-Expression "$pythonCmd -m venv .venv"
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r train/requirements.txt
.\.venv\Scripts\python.exe -c "import numpy, yaml, pytest; print('python deps ok')"

Pop-Location
Write-Host "[setup] done"
