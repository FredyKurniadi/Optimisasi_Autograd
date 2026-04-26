# Installation

## Prasyarat
- Windows PowerShell
- Python 3.10+ (direkomendasikan 3.11)

## Setup
```powershell
./scripts/setup_all.ps1
```

Script ini akan:
1. Membuat virtual environment `.venv`
2. Install dependency dari `train/requirements.txt`
3. Memvalidasi import package inti

## Catatan
Jika sudah memiliki `.venv` tetapi ingin clean setup:
```powershell
./scripts/setup_all.ps1 -RecreateVenv
```
