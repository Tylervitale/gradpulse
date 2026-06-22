"""Divergence guards + convergence diagnostics on the optimizers.

The guards exist so a single non-finite loss/gradient (which, via the shared
scalar loss, would otherwise poison every seed on opt.step()) is rolled back and
never corrupts the returned best_fidelity. We test both that the diagnostics are
reported and that an injected NaN is actually caught.
"""
import math

import pytest
import torch

from gradpulse import (ParametricCouplerProfile, ParametricCZOptimizer,
                       CrossResonanceProfile, CrossResonanceZXOptimizer,
                       MultiQubitProfile, MultiQubitOptimizer)


def test_parametric_reports_convergence_diagnostics():
    opt = ParametricCZOptimizer(ParametricCouplerProfile())
    r = opt.optimize_multi_seed(n_seeds=1, iterations=12, lbfgs_polish=False)
    for k in ("history", "converged", "final_grad_norm", "n_nonfinite_steps"):
        assert k in r
    assert len(r["history"]) == 12
    assert math.isfinite(r["best_fidelity"])
    # history is the running best -> monotonically non-decreasing
    assert all(b >= a - 1e-9 for a, b in zip(r["history"], r["history"][1:]))


def test_cross_resonance_reports_convergence_diagnostics():
    opt = CrossResonanceZXOptimizer(CrossResonanceProfile())
    r = opt.optimize(n_slices=24, dt_ns=1.0, iterations=10, n_seeds=1)
    for k in ("history", "converged", "final_grad_norm", "n_nonfinite_steps"):
        assert k in r
    assert math.isfinite(r["best_fidelity"])


def test_multiqubit_reports_convergence_diagnostics():
    prof = MultiQubitProfile(n_qubits=2, freqs_ghz=[4.8, 5.0],
                             anharm_mhz=[-200, -200], t1_ns=[3e4, 3e4],
                             t2_ns=[2e4, 2e4], couplings={(0, 1): 12.0})
    opt = MultiQubitOptimizer(prof, open_system=False, verbose=False)
    r = opt.optimize(n_slices=20, iterations=10, n_seeds=1)
    for k in ("history", "converged", "final_grad_norm", "n_nonfinite_steps"):
        assert k in r
    assert math.isfinite(r["best_fidelity"])


def test_divergence_guard_catches_injected_nan():
    """Inject a NaN into the fidelity on one iteration; the guard must roll the
    step back, count it, and still return a finite best_fidelity."""
    opt = ParametricCZOptimizer(ParametricCouplerProfile())
    real_pf = opt._process_fidelity
    state = {"calls": 0}

    def poisoned(rho):
        fids = real_pf(rho)
        state["calls"] += 1
        if state["calls"] == 4:           # poison one mid-run evaluation
            return fids * float("nan")
        return fids

    opt._process_fidelity = poisoned
    r = opt.optimize_multi_seed(n_seeds=1, iterations=10, lbfgs_polish=False)
    assert r["n_nonfinite_steps"] >= 1, "guard did not catch the injected NaN"
    assert math.isfinite(r["best_fidelity"]), "non-finite best leaked through the guard"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
