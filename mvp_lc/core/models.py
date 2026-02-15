# File: mvp_lc/core/models.py
from typing import List, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class CurveModel:
    def __init__(self, name: str):
        self.name = name
        self.fitted = False

    def fit(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def asymptote(self) -> float:
        return float("inf")

    def resource_for_target(self, target: float, x_min: float, x_max: float) -> Tuple[float, bool]:
        grid = np.linspace(x_min, x_max, 2000)
        yhat = self.predict(grid)
        idx = np.where(yhat >= target)[0]
        if len(idx) == 0:
            return np.nan, False
        return float(grid[idx[0]]), True

class MonotoneSplineModel(CurveModel):
    def __init__(self, increasing: bool = False):
        super().__init__(name=f"monotone_spline_{'inc' if increasing else 'dec'}")
        self._interp = None
        self.increasing = increasing

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        assert x.ndim == 1 and y.ndim == 1 and len(x) == len(y)
        order = np.argsort(x)
        xo, yo = x[order], y[order]

        # Try Scipy monotone cubic Hermite spline; fallback to linear interpolation
        try:
            from scipy.interpolate import CubicHermiteSpline
            dy = np.gradient(yo, xo)
            self._interp = CubicHermiteSpline(xo, yo, dy, monotone=True)
        except Exception:
            # Linear interpolation (monotone in x)
            def _interp_fn(xq):
                xq = np.asarray(xq, dtype=float)
                return np.interp(xq, xo, yo)
            self._interp = _interp_fn
        self.fitted = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("fit first")
        x = np.asarray(x, dtype=float)
        return self._interp(x)

    def asymptote(self) -> float:
        # No closed-form asymptote; approximate as last observed value
        return float("inf")

class PowerLawModel(CurveModel):
    # y ≈ a - b * x^(-c), with c>0 (for accuracy-like curves increasing with x)
    def __init__(self):
        super().__init__(name="power_law")

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        L = np.max(y) + 1e-6
        Y = np.log(np.maximum(L - y, 1e-9))
        X = np.log(np.maximum(x, 1e-9))
        mask = np.isfinite(Y) & np.isfinite(X)
        if np.sum(mask) < 2:
            raise ValueError("Not enough finite points for power-law fit")
        reg = LinearRegression().fit(X[mask].reshape(-1, 1), Y[mask])
        c = float(reg.coef_[0])
        logb = float(reg.intercept_)
        b = np.exp(logb)
        self.L_ = L
        self.a_ = L - b  # because L - y = b * x^{-c} => y = (L - b) + b * x^{-c}
        self.c_ = c
        self.fitted = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("fit first")
        x = np.asarray(x, dtype=float)
        return self.a_ + (self.L_ - self.a_) * np.power(np.maximum(x, 1e-9), -self.c_)

    def asymptote(self) -> float:
        return float(self.a_)

class ExponentialSaturatingModel(CurveModel):
    # y ≈ L - b * exp(-c*x), c>0
    def __init__(self):
        super().__init__(name="exp_saturating")

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        L = float(np.max(y) + 1e-6)
        Y = np.log(np.maximum(L - y, 1e-9))
        reg = LinearRegression().fit(x.reshape(-1, 1), Y)
        c = float(-reg.coef_[0])  # because Y=log(b) - c*x
        logb = float(reg.intercept_)
        b = np.exp(logb)
        self.L_ = L
        self.b_ = b
        self.c_ = c
        self.fitted = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("fit first")
        x = np.asarray(x, dtype=float)
        return self.L_ - self.b_ * np.exp(-self.c_ * x)

    def asymptote(self) -> float:
        return float(self.L_)

class InversePowerModel(CurveModel):
    # y ≈ L - b / x^c
    def __init__(self):
        super().__init__(name="inverse_power")

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        L = float(np.max(y) + 1e-6)
        Y = np.log(np.maximum(L - y, 1e-9))
        X = np.log(np.maximum(x, 1e-9))
        reg = LinearRegression().fit(X.reshape(-1, 1), Y)
        c = float(reg.coef_[0])
        logb = float(reg.intercept_)
        b = np.exp(logb)
        self.L_ = L
        self.b_ = b
        self.c_ = c
        self.fitted = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("fit first")
        x = np.asarray(x, dtype=float)
        return self.L_ - self.b_ * np.power(np.maximum(x, 1e-9), -self.c_)

    def asymptote(self) -> float:
        return float(self.L_)

class BestCurveModel(CurveModel):
    def __init__(self, models: List[CurveModel]):
        super().__init__(name="best_of")
        self.models = models
        self.best = None
        self.best_rmse = np.inf

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.best = None
        self.best_rmse = np.inf
        y = np.asarray(y, dtype=float)
        for m in self.models:
            try:
                m.fit(x, y)
                yhat = m.predict(x)
                rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
                if rmse < self.best_rmse:
                    self.best_rmse = rmse
                    self.best = m
            except Exception:
                continue
        if self.best is None:
            self.best = MonotoneSplineModel(increasing=True)
            self.best.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.best is None:
            raise RuntimeError("fit first")
        return self.best.predict(x)

    def asymptote(self) -> float:
        if self.best is None:
            raise RuntimeError("fit first")
        return self.best.asymptote()

    def resource_for_target(self, target: float, x_min: float, x_max: float) -> Tuple[float, bool]:
        if self.best is None:
            raise RuntimeError("fit first")
        return self.best.resource_for_target(target, x_min, x_max)
