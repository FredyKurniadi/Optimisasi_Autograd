from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from train.src.adam_optimizer import AdamConfig, optimize_with_adam
from train.src.constraints import ConstraintConfig, is_feasible, penalty_grad, penalty_loss
from train.src.evaluate import summarize_metrics
from train.src.objective import (
  NormalizationConfig,
  emission,
  profit,
  weighted_base_grad,
  weighted_base_loss,
)


def parse_args():
  parser = argparse.ArgumentParser(description="Run weighted-sum constrained optimization")
  parser.add_argument("--train-config", default="train/configs/train.yaml")
  parser.add_argument("--model-config", default="train/configs/model.yaml")
  return parser.parse_args()


def load_yaml(path: str) -> dict:
  with open(path, "r", encoding="utf-8") as file:
    return yaml.safe_load(file)


def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)


def latest_model_id(models_dir: Path, prefix: str) -> int:
  current = 0
  for path in models_dir.glob(f"{prefix}[0-9][0-9][0-9]"):
    suffix = path.name.replace(prefix, "")
    if suffix.isdigit():
      current = max(current, int(suffix))
  return current


def build_loss_and_grad(
  lam: float,
  norm_cfg: NormalizationConfig,
  cons_cfg: ConstraintConfig,
  penalty_rho: float,
):
  def _inner(x: float, y: float):
    base = weighted_base_loss(x, y, lam=lam, norm_cfg=norm_cfg)
    pen = penalty_loss(x, y, cfg=cons_cfg, rho=penalty_rho)

    gbx, gby = weighted_base_grad(x, y, lam=lam, norm_cfg=norm_cfg)
    gpx, gpy = penalty_grad(x, y, cfg=cons_cfg, rho=penalty_rho)

    total_loss = base + pen
    return total_loss, gbx + gpx, gby + gpy

  return _inner


def run_single_lambda(
  lam: float,
  train_cfg: Dict[str, object],
  norm_cfg: NormalizationConfig,
  cons_cfg: ConstraintConfig,
) -> Dict[str, object]:
  opt_cfg = train_cfg["optimizer"]
  adam_cfg = AdamConfig(
    learning_rate=float(opt_cfg["learning_rate"]),
    beta1=float(opt_cfg["beta1"]),
    beta2=float(opt_cfg["beta2"]),
    eps=float(opt_cfg["eps"]),
    steps=int(opt_cfg["steps"]),
    x0=float(opt_cfg["x0"]),
    y0=float(opt_cfg["y0"]),
    clip_min=float(opt_cfg["clip_min"]),
    clip_max=float(opt_cfg["clip_max"]),
    log_every=int(opt_cfg["log_every"]),
  )

  loss_grad = build_loss_and_grad(
    lam=lam,
    norm_cfg=norm_cfg,
    cons_cfg=cons_cfg,
    penalty_rho=float(opt_cfg["penalty_rho"]),
  )

  result = optimize_with_adam(loss_grad_fn=loss_grad, cfg=adam_cfg)
  x = float(result["x"])
  y = float(result["y"])

  p = profit(x, y)
  e = emission(x, y)
  feasible = is_feasible(x, y, cfg=cons_cfg, tol=float(opt_cfg["feasible_tolerance"]))

  return {
    "lambda": float(lam),
    "x": x,
    "y": y,
    "profit": float(p),
    "emission": float(e),
    "best_loss": float(result["best_loss"]),
    "feasible": bool(feasible),
    "history": result["history"],
  }


def main() -> None:
  args = parse_args()
  train_cfg = load_yaml(args.train_config)
  model_cfg = load_yaml(args.model_config)

  set_seed(int(train_cfg["seed"]))

  norm_cfg = NormalizationConfig(
    profit_scale=float(train_cfg["normalization"]["profit_scale"]),
    emission_scale=float(train_cfg["normalization"]["emission_scale"]),
  )
  cons_cfg = ConstraintConfig(**train_cfg["constraints"])

  lambdas: List[float] = [float(v) for v in train_cfg["lambdas"]]
  solutions = [run_single_lambda(lam, train_cfg, norm_cfg, cons_cfg) for lam in lambdas]

  metrics = summarize_metrics(solutions)
  metrics["project_name"] = model_cfg.get("project_name", "optim")
  metrics["method"] = model_cfg.get("method", "weighted_sum_with_adam")

  models_root = Path(train_cfg["model_output_dir"])
  models_root.mkdir(parents=True, exist_ok=True)
  next_id = latest_model_id(models_root, train_cfg["model_counter_prefix"]) + 1
  model_dir = models_root / f"{train_cfg['model_counter_prefix']}{next_id:03d}"
  model_dir.mkdir(parents=True, exist_ok=False)

  with open(model_dir / "solutions.json", "w", encoding="utf-8") as file:
    json.dump(solutions, file, indent=2)

  with open(model_dir / "metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

  with open(model_dir / "config_snapshot.yaml", "w", encoding="utf-8") as file:
    yaml.safe_dump({"train": train_cfg, "model": model_cfg}, file, sort_keys=False)

  print("Optimization complete")
  print(f"Model output: {model_dir}")
  print(f"Feasible rate: {metrics['feasible_rate']:.3f}")
  print(f"Pareto count: {metrics['pareto_count']}")


if __name__ == "__main__":
  main()
