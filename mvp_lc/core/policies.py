# File: mvp_lc/core/policies.py
from typing import List, Tuple, Optional
import numpy as np

from .models import CurveModel

def slope_last_k(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    if len(x) < 2 or k < 1:
        return 0.0
    k = min(k, len(x) - 1)
    xk, yk = x[-k-1:], y[-k-1:]
    return float(np.polyfit(xk, yk, 1)[0])

def bootstrap_std(y: np.ndarray, n_boot: int = 200, random_state: int = 42) -> float:
    rng = np.random.default_rng(random_state)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        sample = y[rng.integers(0, n, size=n)]
        stats.append(float(np.std(sample)))
    return float(np.mean(stats))

class DataAcquisitionPolicy:
    def __init__(self, target_metric: float):
        self.target_metric = target_metric

    def required_samples(self, model: CurveModel, x_obs: np.ndarray, x_max: float) -> Tuple[float, bool]:
        x_min = float(np.min(x_obs)) if len(x_obs) else 1.0
        return model.resource_for_target(self.target_metric, x_min, x_max)

class EarlyStoppingPolicy:
    def __init__(self, k: int = 3, slope_tol: float = 1e-4, noise_std: Optional[float] = None):
        self.k = k
        self.slope_tol = slope_tol
        self.noise_std = noise_std

    def should_stop(self, x: np.ndarray, y: np.ndarray) -> Tuple[bool, dict]:
        s = slope_last_k(x, y, k=self.k)
        stop = abs(s) < self.slope_tol
        noise = self.noise_std or bootstrap_std(y)
        recent_gain = y[-1] - y[max(0, len(y)-1-self.k)]
        reason = {
            "slope_last_k": s,
            "stop_by_slope": stop,
            "recent_gain": recent_gain,
            "noise_std": noise
        }
        return stop, reason

class EarlyDiscardingPolicy:
    def __init__(self, incumbent_score: float, remaining_budget: int, safety_margin: float = 0.0):
        self.incumbent_score = incumbent_score
        self.remaining_budget = remaining_budget
        self.safety_margin = safety_margin

    def should_discard(self, model: CurveModel, x_obs: np.ndarray) -> Tuple[bool, dict]:
        x_min = float(np.min(x_obs)) if len(x_obs) else 1.0
        x_max = float(x_min + self.remaining_budget)
        yhat = model.predict(np.array([x_max]))[0]
        try:
            asym = model.asymptote()
            upper_bound = min(yhat, asym)
        except Exception:
            upper_bound = yhat
        discard = (upper_bound < (self.incumbent_score + self.safety_margin))
        return discard, {
            "predicted_at_budget": yhat,
            "upper_bound": upper_bound,
            "incumbent": self.incumbent_score,
            "safety_margin": self.safety_margin
        }
