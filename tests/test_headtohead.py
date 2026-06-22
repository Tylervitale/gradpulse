"""Tests for the coherent-only (#1) and dephasing-robust (#2) objectives and the
decoherence-in-the-loop head-to-head built on them.

Structural/guard tests run with tiny settings; the magnitude tests
(``*_beats_*`` / ``*_ignores_*``) use a deliberately decoherence-pressured regime so
the effects are real but stay quick. Tolerances are derived from a measured run and
left with comfortable margin.
"""
import math

import numpy as np
import pytest

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer
from gradpulse.headtohead import run_head_to_head


def _pressured_profile():
    # A few-microsecond T1/T2 pair: decoherence competes with leakage.
    return ParametricCouplerProfile(
        t1_ns_q1=4000.0, t1_ns_q2=4000.0, t2_ns_q1=3000.0, t2_ns_q2=3000.0,
        notes=["test: decoherence-pressured"],
    )


# --------------------------------------------------------------------------- #
# #2 guards (no optimization needed -- they raise before the loop)
# --------------------------------------------------------------------------- #
def test_dephasing_robust_requires_process_fidelity():
    opt = ParametricCZOptimizer(_pressured_profile())
    with pytest.raises(ValueError, match="use_process_fidelity"):
        opt.optimize_multi_seed(n_seeds=1, iterations=2, n_slices=20,
                                 use_process_fidelity=False,
                                 robust_dephasing_sigma_mhz=0.5, lbfgs_polish=False)


def test_dephasing_robust_rejects_jitter_combo():
    opt = ParametricCZOptimizer(_pressured_profile())
    with pytest.raises(ValueError, match="standalone"):
        opt.optimize_multi_seed(n_seeds=1, iterations=2, n_slices=20,
                                 robust_dephasing_sigma_mhz=0.5,
                                 robust_freq_jitter_mhz=0.3, lbfgs_polish=False)


# --------------------------------------------------------------------------- #
# #2 correctness: the in-loop objective IS the scorer (same Gauss-Hermite grid)
# --------------------------------------------------------------------------- #
def test_dephasing_robust_objective_matches_quasi_static_estimator():
    """best_fidelity from the dephasing-robust run must equal
    quasi_static_fidelity scored on the same pulse with the same node count --
    proving 'optimise-against' and 'measure-against' are the identical estimator."""
    sigma, nodes = 0.5, 3
    opt = ParametricCZOptimizer(_pressured_profile(), precision="double")
    res = opt.optimize_multi_seed(n_seeds=1, iterations=20, n_slices=60, lr=0.02,
                                  robust_dephasing_sigma_mhz=sigma,
                                  robust_dephasing_nodes=nodes, lbfgs_polish=True,
                                  lbfgs_iters=15)
    qs = opt.quasi_static_fidelity(res["best_raw_param"], sigma_mhz=sigma,
                                   n_nodes=nodes)
    # Same pulse, same grid -> the objective value and the score coincide.
    assert abs(res["best_fidelity"] - qs["F_proc"]) < 5e-4


# --------------------------------------------------------------------------- #
# head-to-head structure (tiny config)
# --------------------------------------------------------------------------- #
def test_head_to_head_summary_structure():
    out = run_head_to_head(_pressured_profile(), [80.0, 160.0], n_seeds=1,
                           iterations=30, lbfgs_iters=10, precision="single",
                           verbose=False)
    assert len(out["rows"]) == 2
    s = out["summary"]
    for k in ("inloop_best_f_avg", "multiply_delivered_f_avg",
              "delivered_fidelity_gain_vs_multiply", "pulse_shaping_gain",
              "duration_selection_loss_of_multiply", "multiplicative_overprediction"):
        assert k in s and math.isfinite(s[k])
    for r in out["rows"]:
        for k in ("f_inloop", "f_coherent", "f_predicted", "f_delivered"):
            assert 0.0 < r[k] <= 1.0
    # The in-loop edge is exactly its decomposition (definitional identity).
    assert abs(s["delivered_fidelity_gain_vs_multiply"]
               - (s["pulse_shaping_gain"]
                  + s["duration_selection_loss_of_multiply"])) < 1e-9


# --------------------------------------------------------------------------- #
# #1 magnitude: the coherent-only objective ignores decoherence, and the in-loop
# objective delivers a real (open-system) gate at least as good as the coherent
# pulse scored open.
# --------------------------------------------------------------------------- #
def test_coherent_only_objective_ignores_decoherence():
    opt = ParametricCZOptimizer(_pressured_profile())
    res = opt.optimize_multi_seed(n_seeds=1, iterations=40, n_slices=100, lr=0.02,
                                  diss_scale=0.0, lbfgs_polish=True, lbfgs_iters=20)
    eb = opt.error_budget(res["best_raw_param"])
    # diss_scale=0 optimises the CLOSED (coherent) fidelity exactly.
    assert abs(res["best_fidelity"] - eb["F_proc_closed"]) < 1e-3
    # ...and on this decoherence-pressured device that closed fidelity is well
    # above what the same pulse actually delivers open -- the gap the multiply
    # recipe has to estimate blind.
    assert eb["F_proc_closed"] - eb["F_proc"] > 5e-3


def test_inloop_delivers_at_least_coherent_open():
    import torch
    from gradpulse.parametric import DEVICE
    opt = ParametricCZOptimizer(_pressured_profile())
    cfg = dict(n_seeds=1, iterations=50, n_slices=120, lr=0.02,
               lbfgs_polish=True, lbfgs_iters=25)
    a = opt.optimize_multi_seed(diss_scale=1.0,
                                rng=torch.Generator(device=DEVICE).manual_seed(0), **cfg)
    b = opt.optimize_multi_seed(diss_scale=0.0,
                                rng=torch.Generator(device=DEVICE).manual_seed(0), **cfg)
    eb_a = opt.error_budget(a["best_raw_param"])
    eb_b = opt.error_budget(b["best_raw_param"])
    # The in-loop pulse, scored open, is at least as good as the coherent-optimal
    # pulse scored open (it optimised that very objective); tol absorbs optimizer
    # scatter at low iteration counts.
    assert eb_a["F_proc"] >= eb_b["F_proc"] - 3e-3
    # And the regime is non-trivial: the coherent pulse genuinely loses fidelity
    # to decoherence (so the comparison is not vacuous).
    assert eb_b["F_proc_closed"] - eb_b["F_proc"] > 5e-3


# --------------------------------------------------------------------------- #
# #2 magnitude: the dephasing-robust objective produces a pulse that is more
# tolerant of quasi-static dephasing than a nominal-optimised pulse.
# --------------------------------------------------------------------------- #
def test_dephasing_robust_beats_nominal_under_dephasing():
    import torch
    from gradpulse.parametric import DEVICE
    # sigma=1.0 MHz puts the comparison in a regime where hardening is decisive
    # rather than marginal. At sigma=0.6 the converged advantage is only ~4e-3,
    # and a single under-converged seed leaves a ~1e-3 margin -- the same order as
    # the cross-platform float scatter between two separate optimizations, so the
    # sign of the test was effectively round-off-dependent (it passed locally at
    # +1.1e-3 but failed on CI at +9.5e-4). Two seeds at sigma=1.0 measure the
    # real physical effect instead of optimizer noise: nom ~0.815 -> rob ~0.834,
    # a ~2e-2 win that reproduces across platforms.
    sigma = 1.0
    opt = ParametricCZOptimizer(_pressured_profile())
    cfg = dict(n_seeds=2, iterations=60, n_slices=90, lr=0.02,
               lbfgs_polish=True, lbfgs_iters=20)
    nom = opt.optimize_multi_seed(diss_scale=1.0,
                                  rng=torch.Generator(device=DEVICE).manual_seed(0), **cfg)
    rob = opt.optimize_multi_seed(diss_scale=1.0, robust_dephasing_sigma_mhz=sigma,
                                  robust_dephasing_nodes=3,
                                  rng=torch.Generator(device=DEVICE).manual_seed(0), **cfg)
    qs_nom = opt.quasi_static_fidelity(nom["best_raw_param"], sigma_mhz=sigma, n_nodes=5)
    qs_rob = opt.quasi_static_fidelity(rob["best_raw_param"], sigma_mhz=sigma, n_nodes=5)
    # Hardening against the dephasing distribution in the loop beats ignoring it,
    # by a margin (~2e-2) comfortably above optimizer/platform scatter.
    assert qs_rob["F_proc"] > qs_nom["F_proc"] + 5e-3


# --------------------------------------------------------------------------- #
# warm-start across the sweep: resampling helper + chaining preserves structure
# --------------------------------------------------------------------------- #
def test_resample_pulse_shape_endpoints_and_identity():
    from gradpulse.headtohead import resample_pulse
    wf = np.stack([np.linspace(0.1, 0.9, 60), np.linspace(0.9, 0.2, 60),
                   np.full(60, 0.5)], axis=1)                       # [60, 3]
    up = resample_pulse(wf, 90)
    assert up.shape == (90, 3)
    # align_corners: endpoints are preserved exactly under linear resampling.
    assert np.allclose(up[0], wf[0]) and np.allclose(up[-1], wf[-1])
    # Identity when the slice count is unchanged.
    same = resample_pulse(wf, 60)
    assert same.shape == (60, 3) and np.allclose(same, wf)
    # A monotone channel stays monotone (no interpolation overshoot).
    assert (np.diff(up[:, 0]) >= -1e-9).all()


def test_warm_start_chain_preserves_summary_structure():
    """Chaining each duration from the previous solution must still produce a
    well-formed head-to-head (same keys, same exact decomposition identity, valid
    fidelities). It changes the optimisation path, not the contract."""
    out = run_head_to_head(_pressured_profile(), [80.0, 120.0, 160.0], n_seeds=1,
                           iterations=25, lbfgs_iters=8, precision="single",
                           verbose=False, warm_start_chain=True)
    assert len(out["rows"]) == 3
    s = out["summary"]
    for k in ("inloop_best_f_avg", "multiplicative_overprediction",
              "pulse_shaping_gain", "duration_selection_loss_of_multiply"):
        assert k in s and math.isfinite(s[k])
    for r in out["rows"]:
        for k in ("f_inloop", "f_coherent", "f_predicted", "f_delivered"):
            assert 0.0 < r[k] <= 1.0
    assert abs(s["delivered_fidelity_gain_vs_multiply"]
               - (s["pulse_shaping_gain"]
                  + s["duration_selection_loss_of_multiply"])) < 1e-9
