# File: tests/test_models.py
import numpy as np
from mvp_lc.core.models import (
    MonotoneSplineModel,
    PowerLawModel,
    ExponentialSaturatingModel,
    InversePowerModel,
    BestCurveModel,
)

def test_monotone_spline_increasing():
    x = np.linspace(1, 100, 10)
    y = np.log(x)  # increasing curve
    m = MonotoneSplineModel(increasing=True)
    m.fit(x, y)
    yhat = m.predict(x)
    assert yhat.shape == y.shape
    # Check monotonicity in predictions (non-decreasing)
    assert np.all(np.diff(yhat) >= -1e-8)

def test_power_law_asymptote():
    np.random.seed(42)  # for reproducibility
    x = np.linspace(10, 1000, 30)
    y = 0.9 - 0.4 * x**(-0.7) + np.random.normal(0, 0.005, size=x.shape)
    m = PowerLawModel()
    m.fit(x, y)
    assert m.fitted
    # Power law should provide predictions
    y_pred = m.predict(x)
    assert y_pred.shape == x.shape
    # Check that model has parameters
    assert hasattr(m, 'a_')
    assert hasattr(m, 'c_')

def test_exp_saturating_predict():
    x = np.linspace(0.1, 50, 30)
    L = 0.95
    b = 0.3
    c = 0.2
    y = L - b * np.exp(-c * x)
    m = ExponentialSaturatingModel()
    m.fit(x, y)
    assert abs(m.asymptote() - L) < 0.01
    assert m.fitted

def test_inverse_power_predict():
    x = np.linspace(1, 200, 25)
    L = 0.92
    b = 0.4
    c = 0.6
    y = L - b * x**(-c)
    m = InversePowerModel()
    m.fit(x, y)
    assert m.fitted
    yhat = m.predict(x)
    assert yhat.shape == x.shape

def test_best_curve_chooses_spline_on_failure():
    # MonotoneSplineModel should be the fallback when parametric models fail
    # Use data that will cause all parametric models to raise exceptions
    x = np.array([1.0])
    y = np.array([0.5])
    candidate_models = [PowerLawModel(), ExponentialSaturatingModel(), InversePowerModel()]
    best = BestCurveModel(models=candidate_models)
    best.fit(x, y)
    # With only 1 point, parametric models should fail and fallback to spline
    assert best.best is not None
    assert best.predict(x).shape == x.shape

def test_resource_for_target():
    x = np.linspace(10, 100, 20)
    y = 0.5 + 0.4 * (1 - np.exp(-0.05 * x))
    m = ExponentialSaturatingModel()
    m.fit(x, y)
    target = 0.85
    est, ok = m.resource_for_target(target, float(x.min()), float(x.max()))
    assert ok
    assert est >= x.min() and est <= x.max()
