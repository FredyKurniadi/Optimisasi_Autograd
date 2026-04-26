from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from train.src.constraints import ConstraintConfig, is_feasible, penalty_loss
from train.src.objective import NormalizationConfig, weighted_base_loss
from train.src.optimize import run_single_lambda


def test_weighted_loss_finite_value():
  norm = NormalizationConfig(profit_scale=2000.0, emission_scale=1000.0)
  value = weighted_base_loss(30.0, 20.0, lam=0.6, norm_cfg=norm)
  assert isinstance(value, float)


def test_penalty_loss_zero_when_feasible():
  cfg = ConstraintConfig(
    capacity_total=100.0,
    labor_limit=150.0,
    min_x=20.0,
    min_y=10.0,
    max_x=80.0,
    budget_limit=800.0,
    budget_x_coeff=10.0,
    budget_y_coeff=8.0,
  )
  loss = penalty_loss(40.0, 20.0, cfg=cfg, rho=100.0)
  assert abs(loss) < 1e-9


def test_adam_optimizer_returns_feasible_solution():
  train_cfg = {
    "optimizer": {
      "learning_rate": 0.08,
      "beta1": 0.9,
      "beta2": 0.999,
      "eps": 1e-8,
      "steps": 1200,
      "penalty_rho": 120.0,
      "feasible_tolerance": 1e-5,
      "x0": 30.0,
      "y0": 20.0,
      "clip_min": 0.0,
      "clip_max": 200.0,
      "log_every": 200,
    }
  }
  norm_cfg = NormalizationConfig(profit_scale=2000.0, emission_scale=1000.0)
  cons_cfg = ConstraintConfig(
    capacity_total=100.0,
    labor_limit=150.0,
    min_x=20.0,
    min_y=10.0,
    max_x=80.0,
    budget_limit=800.0,
    budget_x_coeff=10.0,
    budget_y_coeff=8.0,
  )

  sol = run_single_lambda(0.7, train_cfg, norm_cfg, cons_cfg)
  assert is_feasible(sol["x"], sol["y"], cons_cfg, tol=1e-4)
  assert sol["profit"] > 0.0
