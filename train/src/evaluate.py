from __future__ import annotations

from typing import Dict, List


def non_dominated_solutions(solutions: List[Dict[str, float]]) -> List[Dict[str, float]]:
  feasible = [s for s in solutions if s.get("feasible", False)]
  pareto: List[Dict[str, float]] = []

  for s in feasible:
    dominated = False
    for other in feasible:
      if other is s:
        continue
      better_or_equal_profit = other["profit"] >= s["profit"]
      better_or_equal_emission = other["emission"] <= s["emission"]
      strictly_better = (
        other["profit"] > s["profit"] or other["emission"] < s["emission"]
      )
      if better_or_equal_profit and better_or_equal_emission and strictly_better:
        dominated = True
        break
    if not dominated:
      pareto.append(s)

  pareto.sort(key=lambda item: item["lambda"])
  return pareto


def summarize_metrics(solutions: List[Dict[str, float]]) -> Dict[str, object]:
  feasible = [s for s in solutions if s.get("feasible", False)]
  feasible_rate = 0.0 if not solutions else len(feasible) / len(solutions)

  if feasible:
    best_profit = max(feasible, key=lambda item: item["profit"])
    best_emission = min(feasible, key=lambda item: item["emission"])
  else:
    best_profit = None
    best_emission = None

  pareto = non_dominated_solutions(solutions)

  return {
    "n_solutions": len(solutions),
    "n_feasible": len(feasible),
    "feasible_rate": feasible_rate,
    "best_profit": best_profit,
    "best_emission": best_emission,
    "pareto_count": len(pareto),
    "pareto_solutions": pareto,
  }
