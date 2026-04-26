from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


LossGradFn = Callable[[float, float], Tuple[float, float, float]]


@dataclass
class AdamConfig:
  learning_rate: float
  beta1: float
  beta2: float
  eps: float
  steps: int
  x0: float
  y0: float
  clip_min: float
  clip_max: float
  log_every: int


def optimize_with_adam(loss_grad_fn: LossGradFn, cfg: AdamConfig) -> Dict[str, object]:
  x = float(cfg.x0)
  y = float(cfg.y0)

  m_x = 0.0
  m_y = 0.0
  v_x = 0.0
  v_y = 0.0

  history: List[Dict[str, float]] = []
  best = {"loss": float("inf"), "x": x, "y": y}

  for t in range(1, cfg.steps + 1):
    loss, gx, gy = loss_grad_fn(x, y)

    if loss < best["loss"]:
      best = {"loss": loss, "x": x, "y": y}

    m_x = cfg.beta1 * m_x + (1.0 - cfg.beta1) * gx
    m_y = cfg.beta1 * m_y + (1.0 - cfg.beta1) * gy
    v_x = cfg.beta2 * v_x + (1.0 - cfg.beta2) * (gx * gx)
    v_y = cfg.beta2 * v_y + (1.0 - cfg.beta2) * (gy * gy)

    mhat_x = m_x / (1.0 - cfg.beta1**t)
    mhat_y = m_y / (1.0 - cfg.beta1**t)
    vhat_x = v_x / (1.0 - cfg.beta2**t)
    vhat_y = v_y / (1.0 - cfg.beta2**t)

    x -= cfg.learning_rate * mhat_x / ((vhat_x**0.5) + cfg.eps)
    y -= cfg.learning_rate * mhat_y / ((vhat_y**0.5) + cfg.eps)

    x = min(max(x, cfg.clip_min), cfg.clip_max)
    y = min(max(y, cfg.clip_min), cfg.clip_max)

    if t == 1 or t % cfg.log_every == 0 or t == cfg.steps:
      history.append(
        {
          "step": float(t),
          "loss": float(loss),
          "x": float(x),
          "y": float(y),
          "grad_norm": float((gx * gx + gy * gy) ** 0.5),
        }
      )

  return {
    "x": float(best["x"]),
    "y": float(best["y"]),
    "best_loss": float(best["loss"]),
    "history": history,
  }
