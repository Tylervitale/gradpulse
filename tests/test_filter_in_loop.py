"""Filter-function dephasing put INSIDE the GRAPE gradient (robust_filter_*).

Companion to test_filter_function.py (which validates the *scorer*). Where
robust_dephasing_sigma_mhz hardens only F(0) -- the slow limit -- this hardens the
whole noise band via the leakage-inclusive filter function. Two cheap correctness
gates plus one slow end-to-end:

  * ALIGNMENT (the rigor gate): the in-loop infidelity _filter_dephasing_infidelity
    IS the quantity filter_function_fidelity scores -- same shared estimator, so
    optimise-against and measure-against cannot drift. Asserted to machine precision.
  * DIFFERENTIABLE: the in-loop infidelity carries a finite, non-zero gradient
    through the toggling-frame propagators (it is built under no_grad in the scorer,
    so this is the check that autograd actually flows the in-loop path).
  * GUARDS: standalone-objective contract (needs process fidelity; not combinable
    with robust_dephasing or the jitter axes; band must be ordered).
  * SLOW: an optimized filter-robust gate beats a nominal one on the band metric.
"""
import numpy as np
import pytest
import torch

import gradpulse as gp
from gradpulse import ParametricCouplerProfile
from gradpulse.parametric import ParametricCZOptimizer, DEVICE

SIG, ALPHA, LO, HI, NF = 0.4, 1.0, 1e-3, 5.0, 96


def _opt(precision="double"):
    return gp.ParametricCZOptimizer(n_channels=3, activation="sigmoid",
                                    precision=precision)


# ---- correctness gates (fast: one toggling-frame build, no optimization) ----

def test_inloop_estimator_matches_scorer_to_machine_precision():
    """_filter_dephasing_infidelity (in-loop) == filter_function_fidelity['infidelity']
    (scorer) for the same pulse/band: they share _filter_band_F_used, so the two are
    the identical estimator, not merely close."""
    opt = _opt()
    n, dt = 80, 1.0
    x = opt._warm_start(n, mode="parametric_cz").unsqueeze(0)
    inloop = float(opt._filter_dephasing_infidelity(x, dt, SIG, ALPHA, LO, HI, NF)[0])
    scorer = opt.filter_function_fidelity(x[0], dt=dt, sigma_mhz=SIG, alpha=ALPHA,
                                          f_low_mhz=LO, f_high_mhz=HI, n_freq=NF)
    assert abs(inloop - scorer["infidelity"]) < 1e-9


def test_inloop_infidelity_is_differentiable():
    """The in-loop infidelity carries gradients (the scorer path is under no_grad;
    this is the check that the in-loop path actually backprops)."""
    opt = _opt()
    n, dt = 80, 1.0
    x = opt._warm_start(n, mode="parametric_cz").unsqueeze(0).clone().requires_grad_(True)
    infid = opt._filter_dephasing_infidelity(x, dt, SIG, ALPHA, LO, HI, NF)
    assert infid.requires_grad
    g = torch.autograd.grad(infid.sum(), x)[0]
    assert torch.isfinite(g).all() and float(g.norm()) > 0.0


def test_inloop_infidelity_scales_with_sigma_squared():
    """First order in the noise: the infidelity is sigma_rad^2 * <F>_band, so doubling
    sigma quadruples it (same pulse, same band)."""
    opt = _opt()
    n, dt = 80, 1.0
    x = opt._warm_start(n, mode="parametric_cz").unsqueeze(0)
    a = float(opt._filter_dephasing_infidelity(x, dt, SIG, ALPHA, LO, HI, NF)[0])
    b = float(opt._filter_dephasing_infidelity(x, dt, 2 * SIG, ALPHA, LO, HI, NF)[0])
    assert abs(b / a - 4.0) < 1e-6


# ---- standalone-objective guards (fast: raise before any compute) ----

def test_filter_objective_requires_process_fidelity():
    opt = _opt()
    with pytest.raises(ValueError, match="use_process_fidelity"):
        opt.optimize_multi_seed(n_seeds=1, iterations=1, n_slices=10,
                                use_process_fidelity=False,
                                robust_filter_sigma_mhz=SIG, lbfgs_polish=False)


def test_filter_objective_not_combinable_with_dephasing():
    opt = _opt()
    with pytest.raises(ValueError, match="double-counts|standalone"):
        opt.optimize_multi_seed(n_seeds=1, iterations=1, n_slices=10,
                                robust_filter_sigma_mhz=SIG,
                                robust_dephasing_sigma_mhz=SIG, lbfgs_polish=False)


def test_filter_objective_not_combinable_with_jitter():
    opt = _opt()
    with pytest.raises(ValueError, match="standalone|jitter"):
        opt.optimize_multi_seed(n_seeds=1, iterations=1, n_slices=10,
                                robust_filter_sigma_mhz=SIG,
                                robust_g_jitter=0.1, lbfgs_polish=False)


def test_filter_band_must_be_ordered():
    opt = _opt()
    with pytest.raises(ValueError, match="f_low < f_high"):
        opt.optimize_multi_seed(n_seeds=1, iterations=1, n_slices=10,
                                robust_filter_sigma_mhz=SIG,
                                robust_filter_band_mhz=(5.0, 1.0), lbfgs_polish=False)


# ---- end-to-end (slow): optimizing the filter objective hardens the band ----

@pytest.mark.slow
def test_filter_robust_gate_beats_nominal_on_band():
    """A filter-robust pulse scores higher on the full-band filter fidelity than a
    nominal pulse optimized from the same seed, at negligible nominal cost. Same seed
    so the comparison is deterministic (only the objective differs)."""
    prof = ParametricCouplerProfile(t1_ns_q1=4000.0, t1_ns_q2=4000.0,
                                    t2_ns_q1=3000.0, t2_ns_q2=3000.0)
    opt = ParametricCZOptimizer(prof, precision="double")
    n, dt = 90, 1.0
    common = dict(n_seeds=2, iterations=120, n_slices=n, dt_ns=dt, lr=0.02,
                  lbfgs_iters=30)
    u_nom = opt.optimize_multi_seed(
        label="nom", rng=torch.Generator(device=DEVICE).manual_seed(0), **common
    )["best_raw_param"]
    u_ff = opt.optimize_multi_seed(
        label="ff", rng=torch.Generator(device=DEVICE).manual_seed(0),
        robust_filter_sigma_mhz=SIG, robust_filter_alpha=ALPHA,
        robust_filter_band_mhz=(LO, HI), robust_filter_n_freq=NF, **common
    )["best_raw_param"]

    def band_fproc(u):
        return opt.filter_function_fidelity(u, dt=dt, sigma_mhz=SIG, alpha=ALPHA,
                                            f_low_mhz=LO, f_high_mhz=HI,
                                            n_freq=400)["F_proc"]

    gain = band_fproc(u_ff) - band_fproc(u_nom)
    assert gain > 1e-4, f"filter objective did not harden the band (gain={gain:.2e})"
    # And it did not wreck the nominal gate (negligible cost).
    cost = opt.error_budget(u_nom, dt=dt)["F_proc"] - opt.error_budget(u_ff, dt=dt)["F_proc"]
    assert cost < 5e-3
