# Training / Optimization Spec

## Decision Variables
- `x`: jumlah produk A
- `y`: jumlah produk B

## Objective (Weighted Sum)
Kita ubah multi-objective menjadi satu scalar objective untuk minimisasi:

- Profit (maksimasi):
  - `P(x,y) = 50x + 40y - (2x^2 + y^2)`
- Environmental cost (minimisasi):
  - `E(x,y) = 3x + 5y + 0.1y^2`

Loss inti:
`L_base(x,y) = -lambda * P_norm + (1-lambda) * E_norm`

dengan:
- `P_norm = P / profit_scale`
- `E_norm = E / emission_scale`

## Constraint Handling
Kendala inequality diubah ke soft penalty kuadrat:
`L_penalty = rho * sum(relu(g_i(x,y))^2)`

Total loss:
`L_total = L_base + L_penalty`

## Kendala yang digunakan
- `x + y <= 100`
- `2x + y <= 150`
- `x >= 20`
- `y >= 10`
- `x <= 80`
- `10x + 8y <= 800`

Semua kendala ditulis sebagai `g_i(x,y) <= 0` untuk penalty.

## Optimizer
Adam dengan hyperparameter dari config:
- `learning_rate`
- `beta1`, `beta2`
- `eps`
- `steps`

## Output
- `solutions.json`: hasil untuk setiap `lambda`.
- `metrics.json`: best-profit, best-emission, feasible-rate, pareto-count.
- `config_snapshot.yaml`: config run.
