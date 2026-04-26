# Metrics Report

Metrik utama yang dihasilkan:
- `n_solutions`: jumlah total solusi dari sweep lambda.
- `feasible_rate`: proporsi solusi yang memenuhi semua kendala.
- `best_profit`: solusi feasible dengan profit tertinggi.
- `best_emission`: solusi feasible dengan emisi terendah.
- `pareto_count`: jumlah solusi non-dominated (aproksimasi frontier).

Interpretasi:
- `feasible_rate` tinggi menunjukkan penalty/optimizer stabil.
- `pareto_count` lebih besar menunjukkan trade-off yang lebih kaya.
- Bandingkan `best_profit` vs `best_emission` untuk menilai ekstrem trade-off.
