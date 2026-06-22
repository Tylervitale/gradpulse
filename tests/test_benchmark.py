"""Tests for the qutip-qtrl head-to-head benchmark.

Fast checks that the gradpulse autodiff engine reaches the optimum on the standard
problem and that the comparison driver runs (and includes qutip-qtrl when present).
"""
import numpy as np
import pytest

from gradpulse import benchmark as bm


def test_standard_problem_shapes():
    H0, Hc, Ut, n_ts, T = bm.standard_two_qubit_problem("cnot")
    assert H0.shape == (4, 4)
    assert len(Hc) == 4
    assert Ut.shape == (4, 4)
    assert np.allclose(Ut.conj().T @ Ut, np.eye(4))


def test_unitary_fidelity_metric():
    Ut = bm.standard_two_qubit_problem("cnot")[2]
    assert abs(bm._unitary_fidelity(Ut, Ut) - 1.0) < 1e-12
    assert bm._unitary_fidelity(np.eye(4), Ut) < 0.6


def test_gradpulse_engine_reaches_optimum():
    H0, Hc, Ut, n_ts, T = bm.standard_two_qubit_problem("cnot")
    r = bm.grape_autodiff(H0, Hc, Ut, n_ts, T, lbfgs_iters=100, seed=0)
    assert r["fidelity"] > 0.99
    assert r["wall_s"] > 0.0
    assert "CPU" in r["method"]                 # runs on the correct tier for 4x4


def test_run_benchmark_includes_both_when_qutip_qtrl_present():
    pytest.importorskip("qutip_qtrl")
    out = bm.run_benchmark("iswap", max_iter=150, verbose=False)
    assert out["qutip_qtrl_available"]
    fids = [r["fidelity"] for r in out["results"]]
    assert all(f > 0.99 for f in fids)          # both engines reach the optimum
    # the gap should be a small constant, not orders of magnitude
    gp = next(r for r in out["results"] if "gradpulse" in r["method"])
    qt = next(r for r in out["results"] if "qutip-qtrl" in r["method"])
    assert gp["wall_s"] < 20 * qt["wall_s"] + 1.0   # same ballpark, not 100x
