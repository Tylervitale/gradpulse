"""Filter-function (analytic) dephasing-sensitivity tests.

The filter function is the cheap, no-Monte-Carlo robustness tool: 1 - F = sigma^2 *
(band-weighted F), quasi-static limit sigma^2 * F(0). Two independent validations:

  * FAST: F(0) equals the leakage-inclusive error generator measured by a direct
    closed-system finite difference of the realized gate -- holds for ANY pulse, so
    it is the unit-level correctness check (the toggling-frame algebra is right).
  * SLOW: on an OPTIMIZED gate (F->1, the regime the tool is for) the analytic
    infidelity agrees with the trusted quasi_static_fidelity and colored_noise_fidelity
    Monte-Carlo drops to a few percent at small sigma.
"""
import math

import numpy as np
import pytest
import torch

import gradpulse as gp
from gradpulse.parametric import DEVICE


def _gate_unitary(opt, x, dt, n, d1=0.0, d2=0.0):
    """Closed-system full Hilbert-space gate unitary at static per-qubit detuning."""
    c = opt._smoothed_controls(x.unsqueeze(0), dt)
    u1, u2, u3 = c["u1_eff"][0], c["u2_eff"][0], c["u3"][0]
    U = torch.eye(opt._dim, dtype=opt.cdtype, device=DEVICE)
    for i in range(n):
        H = (opt._H_DRIFT + (u1[i] * opt.OMEGA_MAX) * opt._X1
             + (u2[i] * opt.OMEGA_MAX) * opt._X2 + (u3[i] * opt.G_MAX) * opt._COUPLING)
        if d1 or d2:
            H = H + d1 * opt._N_Q1 + d2 * opt._N_Q2
        U = torch.linalg.matrix_exp(-1j * H * dt) @ U
    return U


def test_filter_F0_matches_direct_error_generator():
    """F(0) per qubit == (1 - F_e)/delta^2 from a direct closed-system finite
    difference (leakage included). Holds for any pulse; the core correctness check."""
    opt = gp.ParametricCZOptimizer(n_channels=3, activation="sigmoid", precision="double")
    dt, n = 1.0, 60
    x = opt._warm_start(n, "parametric_cz").to(opt.rdtype)
    ci = opt._comp_idx
    U0 = _gate_unitary(opt, x, dt, n)
    delta = 1e-3
    Vsub = (U0.conj().t() @ _gate_unitary(opt, x, dt, n, d1=delta))[ci][:, ci]
    Fe = (abs(torch.trace(Vsub).item()) ** 2) / 16.0
    direct = (1.0 - Fe) / delta ** 2                    # ~ VarEig(G_1) incl. leakage
    F0_q1 = opt.filter_function(x, dt=dt, f_max_mhz=1.0, n_freq=2)["F_per_qubit"][0][0]
    assert abs(F0_q1 - direct) / direct < 0.03, f"filter {F0_q1} vs direct {direct}"


def test_filter_function_curve_is_nonnegative_and_even_structure():
    opt = gp.ParametricCZOptimizer(n_channels=3, activation="sigmoid")
    x = opt._warm_start(100, "parametric_cz").to(opt.rdtype)
    ff = opt.filter_function(x, dt=1.0, f_max_mhz=60.0, n_freq=128)
    assert np.all(ff["F"] >= -1e-9)
    assert ff["F0"] > 0.0
    assert len(ff["F_per_qubit"]) == 2


def test_filter_fidelity_quasi_static_is_cheaper_than_mc():
    """Smoke: the analytic call returns a sane fidelity with no Monte Carlo."""
    opt = gp.ParametricCZOptimizer(n_channels=3, activation="sigmoid")
    x = opt._warm_start(80, "parametric_cz").to(opt.rdtype)
    r = opt.filter_function_fidelity(x, dt=1.0, sigma_mhz=0.1, quasi_static=True)
    assert 0.0 <= r["F_proc"] <= 1.0
    assert r["infidelity"] >= 0.0


@pytest.mark.slow
def test_filter_matches_quasistatic_and_colored_mc_on_optimized_gate():
    opt = gp.ParametricCZOptimizer(n_channels=3, activation="sigmoid")
    res = opt.optimize_multi_seed(n_seeds=2, iterations=120, n_slices=120, dt_ns=1.0,
                                  lbfgs_polish=True)
    x = torch.tensor(res["best_raw_param"], device=DEVICE, dtype=opt.rdtype)
    # quasi-static limit, small sigma
    sig = 0.05
    fil = opt.filter_function_fidelity(x, dt=1.0, sigma_mhz=sig, quasi_static=True)
    qs = opt.quasi_static_fidelity(x, dt=1.0, sigma_mhz=sig, n_nodes=9,
                                   include_decoherence=False)
    drop_qs = qs["F_proc_nominal"] - qs["F_proc"]
    assert abs(fil["infidelity"] - drop_qs) / drop_qs < 0.1
    # 1/f band vs colored-noise Monte Carlo
    filb = opt.filter_function_fidelity(x, dt=1.0, sigma_mhz=sig, alpha=1.0,
                                        f_low_mhz=1e-3, f_high_mhz=5.0)
    cn = opt.colored_noise_fidelity(x, dt=1.0, sigma_mhz=sig, alpha=1.0, f_low_mhz=1e-3,
                                    f_high_mhz=5.0, n_traj=400, include_decoherence=False,
                                    seed=1)
    drop_cn = cn["F_proc_nominal"] - cn["F_proc"]
    assert abs(filb["infidelity"] - drop_cn) / drop_cn < 0.15
