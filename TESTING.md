# Testing Strategy

## Unit Tests
Framework: `pytest`

Jalankan:
```powershell
./scripts/run_all_tests.ps1
```

Cakupan minimum:
- Validasi fungsi objective weighted-sum.
- Validasi fungsi penalty kendala.
- Smoke test Adam optimizer menghasilkan solusi feasible.

## Smoke Test End-to-End
1. Setup env
2. Jalankan optimisasi
3. Pastikan `models/model_xxx/metrics.json` terbentuk
