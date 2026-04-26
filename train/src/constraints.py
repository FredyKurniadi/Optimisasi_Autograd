from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class ConstraintConfig:
  capacity_total: float
  labor_limit: float
  min_x: float
  min_y: float
  max_x: float
  budget_limit: float
  budget_x_coeff: float
  budget_y_coeff: float


def violations(x: float, y: float, cfg: ConstraintConfig) -> Dict[str, float]:
  return {
    "capacity": x + y - cfg.capacity_total,
    "labor": 2.0 * x + y - cfg.labor_limit,
    "min_x": cfg.min_x - x,
    "min_y": cfg.min_y - y,
    "max_x": x - cfg.max_x,
    "budget": cfg.budget_x_coeff * x + cfg.budget_y_coeff * y - cfg.budget_limit,
  }


def penalty_loss(x: float, y: float, cfg: ConstraintConfig, rho: float) -> float:
  vals = violations(x, y, cfg)
  total = 0.0
  for value in vals.values():
    if value > 0.0:
      total += value * value
  return rho * total


def penalty_grad(x: float, y: float, cfg: ConstraintConfig, rho: float) -> Tuple[float, float]:
  vals = violations(x, y, cfg)
  grad_x = 0.0
  grad_y = 0.0

  if vals["capacity"] > 0.0:
    grad_x += 2.0 * rho * vals["capacity"] * 1.0
    grad_y += 2.0 * rho * vals["capacity"] * 1.0

  if vals["labor"] > 0.0:
    grad_x += 2.0 * rho * vals["labor"] * 2.0
    grad_y += 2.0 * rho * vals["labor"] * 1.0

  if vals["min_x"] > 0.0:
    grad_x += 2.0 * rho * vals["min_x"] * (-1.0)

  if vals["min_y"] > 0.0:
    grad_y += 2.0 * rho * vals["min_y"] * (-1.0)

  if vals["max_x"] > 0.0:
    grad_x += 2.0 * rho * vals["max_x"] * 1.0

  if vals["budget"] > 0.0:
    grad_x += 2.0 * rho * vals["budget"] * cfg.budget_x_coeff
    grad_y += 2.0 * rho * vals["budget"] * cfg.budget_y_coeff

  return grad_x, grad_y


def is_feasible(x: float, y: float, cfg: ConstraintConfig, tol: float = 1.0e-6) -> bool:
  vals = violations(x, y, cfg)
  return all(value <= tol for value in vals.values())
