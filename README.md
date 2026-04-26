# Numerical Optimization Portfolio (Multi-Objective, Constrained)

Proyek ini menunjukkan pemahaman optimisasi numerik untuk kasus ekonomi multi-objective dengan kendala ganda, diselesaikan menggunakan Adam optimizer.

## Tujuan
- Memformulasikan trade-off profit vs emisi dalam satu objective weighted-sum.
- Mengubah kendala menjadi bentuk differentiable penalty agar cocok untuk optimisasi gradien.
- Menjalankan sweep bobot lambda untuk mendapatkan pendekatan Pareto frontier.
- Menyimpan output hasil optimisasi terstruktur per run (`models/model_xxx`).

## Struktur
- `docs`: formulasi matematika, arsitektur, dan metrik.
- `train`: source code optimizer, konfigurasi, dan test.
- `scripts`: otomatisasi setup, run, test, dan ringkasan hasil.
- `models`: output versioned (`model_001`, `model_002`, dst).
- `datasets`: placeholder folder agar struktur konsisten dengan proyek lain.

## Commands
Jalankan dari root folder `OPTIM`:

```powershell
./scripts/setup_all.ps1
./scripts/run_optimize.ps1
./scripts/show_latest_metrics.ps1
./scripts/run_all_tests.ps1
```

## Output Optimisasi
Setiap run membuat folder baru di `models/model_xxx/` berisi:
- `solutions.json`: solusi untuk setiap nilai lambda.
- `metrics.json`: ringkasan feasible rate, best profit, best emission, dan Pareto set.
- `config_snapshot.yaml`: snapshot konfigurasi yang dipakai.
