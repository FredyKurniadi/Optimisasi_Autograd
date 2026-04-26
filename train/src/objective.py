from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class NormalizationConfig:
  profit_scale: float
  emission_scale: float


def profit(x: float, y: float) -> float:
  return 50.0 * x + 40.0 * y - (2.0 * x * x + y * y)


def emission(x: float, y: float) -> float:
  return 3.0 * x + 5.0 * y + 0.1 * y * y


def weighted_base_loss(
  x: float,
  y: float,
  lam: float,
  norm_cfg: NormalizationConfig,
) -> float:
  p_norm = profit(x, y) / norm_cfg.profit_scale
  e_norm = emission(x, y) / norm_cfg.emission_scale
  return -lam * p_norm + (1.0 - lam) * e_norm


def weighted_base_grad(
  x: float,
  y: float,
  lam: float,
  norm_cfg: NormalizationConfig,
) -> Tuple[float, float]:
  dprofit_dx = 50.0 - 4.0 * x
  dprofit_dy = 40.0 - 2.0 * y

  demission_dx = 3.0
  demission_dy = 5.0 + 0.2 * y

  dloss_dx = -lam * (dprofit_dx / norm_cfg.profit_scale) + (1.0 - lam) * (
    demission_dx / norm_cfg.emission_scale
  )
  dloss_dy = -lam * (dprofit_dy / norm_cfg.profit_scale) + (1.0 - lam) * (
    demission_dy / norm_cfg.emission_scale
  )
  return dloss_dx, dloss_dy
