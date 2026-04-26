# Arsitektur Solusi

Pipeline optimisasi:
1. Baca konfigurasi `train/configs/train.yaml`.
2. Untuk tiap nilai `lambda` pada grid, jalankan Adam pada objective weighted-sum + penalty constraints.
3. Simpan solusi per lambda (`x`, `y`, profit, emisi, feasible).
4. Hitung ringkasan metrik dan Pareto set aproksimasi.
5. Simpan artifact ke `models/model_xxx/`.

Komponen kode:
- `train/src/objective.py`: fungsi profit, emisi, weighted objective, gradient.
- `train/src/constraints.py`: fungsi kendala dan penalty differentiable.
- `train/src/adam_optimizer.py`: implementasi Adam dari nol (numpy).
- `train/src/optimize.py`: runner end-to-end, snapshot config, metrics.
- `train/src/evaluate.py`: utilitas ringkasan metrik dan Pareto filtering.
