import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from src.gradpulse.hardware import (
    BayesianCalibrationLoop,
    expected_improvement,
    probability_of_improvement,
    upper_confidence_bound,
    SimulatedBackend,
    GateMeasurement
)
from src.gradpulse.parametric import ParametricCouplerProfile

def apply_params(base_waveform, params):
    # Parameter is just a scaling factor
    return base_waveform * params[0]

def test_bayesian_calibration_loop_runs():
    prof = ParametricCouplerProfile()
    # Mock backend to speed things up and directly test BO loop functionality
    class MockBackend:
        def measure_gate(self, waveform, dt_ns=1.0, meta=None):
            # A dummy function: optimal scale is around 1.0. Distance from 1.0 lowers fidelity
            scale = waveform[0, 0] if waveform.shape[0] > 0 else 0
            f_avg = 1.0 - (scale - 1.0)**2
            return GateMeasurement(f_avg=f_avg, source="mock")

    backend = MockBackend()
    base_waveform = np.ones((10, 3))

    loop = BayesianCalibrationLoop(backend, base_waveform, dt_ns=1.0, acquisition_fn="EI")

    # We set bounds such that the optimal 1.0 is inside the bounds [0.5, 1.5]
    bounds = [[0.5, 1.5]]
    res = loop.run(bounds=bounds, n_iterations=10, apply_params_fn=apply_params, n_init=3)

    # Check that best parameters found are close to 1.0
    assert abs(res["best_params"][0] - 1.0) < 0.2
    assert "history" in res
    assert len(res["history"]) == 10

def test_acquisition_functions():
    gpr = GaussianProcessRegressor()
    X_train = np.array([[0.1], [0.5], [0.9]])
    y_train = np.array([0.2, 0.8, 0.1])
    gpr.fit(X_train, y_train)

    X_test = np.array([[0.2], [0.4], [0.6], [0.8]])

    ei = expected_improvement(X_test, X_train, y_train, gpr)
    pi = probability_of_improvement(X_test, X_train, y_train, gpr)
    ucb = upper_confidence_bound(X_test, X_train, y_train, gpr)

    assert ei.shape == (4,)
    assert pi.shape == (4,)
    assert ucb.shape == (4,)

    assert np.all(ei >= 0)
    assert np.all(pi >= 0) and np.all(pi <= 1)
