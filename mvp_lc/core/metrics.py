# File: mvp_lc/core/metrics.py
import numpy as np
from sklearn.metrics import r2_score

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))

def evaluate_curve_fit(model, x: np.ndarray, y: np.ndarray) -> dict:
    yhat = model.predict(x)
    return {"rmse": rmse(y, yhat), "r2": r2(y, yhat)}
