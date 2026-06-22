"""Colored (1/f^alpha) frequency noise across the full band -- closing the gap
between quasi_static_fidelity (the slow limit) and the Markovian Lindblad T_phi
(the white/fast limit).

colored_noise_fidelity is a direct colored-noise Monte-Carlo: it synthesizes
per-qubit frequency trajectories with a 1/f^alpha PSD over [f_low, f_high] and
averages the gate's channel over them. Two checks anchor it:

  * SLOW LIMIT -- with f_high*T_gate << 1 each trajectory is ~constant over the
    gate, so the channel average reduces to the deterministic Gauss-Hermite
    quasi-static average. The two agree to Monte-Carlo accuracy.
  * INTERMEDIATE BAND -- extending the band to higher frequency does not increase
    the dephasing (fast noise motionally narrows), the physics neither the slow
    quasi-static model nor a fixed Lindblad rate captures on its own.

Run:  pytest tests/test_colored_noise.py
"""
import numpy as np
import pytest
import torch

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer
from gradpulse.parametric import DEVICE


@pytest.fixture(scope="module")
def short_cz():
    """A short, real CZ pulse + its eval optimizer (single precision, fast)."""
    prof = ParametricCouplerProfile()
    opt = ParametricCZOptimizer(prof, bandwidth_mhz=80.0, activation="sigmoid")
    res = opt.optimize_multi_seed(n_seeds=2, iterations=80, n_slices=50, dt_ns=1.0,
                                  use_process_fidelity=True, lbfgs_polish=False)
    return opt, res["best_raw_param"]


def test_colored_noise_slow_limit_matches_quasistatic(short_cz):
    """f_high*T << 1: colored Monte-Carlo == deterministic quasi-static average."""
    opt, x = short_cz
    u = torch.as_tensor(x, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    sigma = 0.3
    qs = opt.quasi_static_fidelity(u, dt=1.0, sigma_mhz=sigma, n_nodes=7,
                                   include_decoherence=False)
    slow = opt.colored_noise_fidelity(u, dt=1.0, sigma_mhz=sigma, alpha=1.0,
                                      f_low_mhz=2e-4, f_high_mhz=8e-4, n_traj=400,
                                      include_decoherence=False, seed=0)
    # Both isolate the coherent dephasing (diss off); agree to MC accuracy.
    assert abs((1 - slow["F_proc"]) - (1 - qs["F_proc"])) < 3e-3
    # Nominal (no-noise) fidelity is reported consistently.
    assert abs(slow["F_proc_nominal"] - qs["F_proc_nominal"]) < 1e-6


def test_colored_noise_band_does_not_increase_dephasing(short_cz):
    """Extending the band to higher frequency motionally narrows (does not worsen)
    the dephasing -- the intermediate-band behaviour, captured by direct simulation.
    Same seed for both bands so the comparison is correlated (low MC variance)."""
    opt, x = short_cz
    u = torch.as_tensor(x, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    sigma = 0.3
    slow = opt.colored_noise_fidelity(u, dt=1.0, sigma_mhz=sigma, f_low_mhz=2e-4,
                                      f_high_mhz=8e-4, n_traj=400,
                                      include_decoherence=False, seed=7)
    fast = opt.colored_noise_fidelity(u, dt=1.0, sigma_mhz=sigma, f_low_mhz=2e-4,
                                      f_high_mhz=80.0, n_traj=400,
                                      include_decoherence=False, seed=7)
    r_slow, r_fast = 1 - slow["F_proc"], 1 - fast["F_proc"]
    assert r_fast <= r_slow + 1.5e-3      # narrowing direction (generous for MC noise)
    assert fast["n_traj"] == 400 and fast["f_high_mhz"] == 80.0


def test_colored_noise_zero_sigma_is_noop(short_cz):
    """sigma=0 reproduces the nominal gate exactly (no noise injected)."""
    opt, x = short_cz
    u = torch.as_tensor(x, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    r = opt.colored_noise_fidelity(u, dt=1.0, sigma_mhz=0.0, n_traj=8,
                                   include_decoherence=False)
    assert abs(r["F_proc"] - r["F_proc_nominal"]) < 1e-9


def test_colored_noise_cross_qubit_correlation(short_cz):
    """Spatially-correlated cross-qubit noise is now modelled. Same noise power,
    three correlations: independent (rho=0), common-mode (+1), differential (-1).
    Correlation must change the dephasing impact -- a thing independent per-qubit
    draws cannot represent -- and the parameter must be echoed and bounded."""
    opt, x = short_cz
    u = torch.as_tensor(x, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    kw = dict(dt=1.0, sigma_mhz=0.5, alpha=1.0, f_high_mhz=2.0, n_traj=300,
              seed=3, include_decoherence=False)
    indep = opt.colored_noise_fidelity(u, correlation=0.0, **kw)
    common = opt.colored_noise_fidelity(u, correlation=1.0, **kw)
    differ = opt.colored_noise_fidelity(u, correlation=-1.0, **kw)
    # parameter echoed
    assert indep["correlation"] == 0.0 and common["correlation"] == 1.0
    assert differ["correlation"] == -1.0
    # default (no arg) reproduces the independent case exactly
    default = opt.colored_noise_fidelity(u, **kw)
    assert abs(default["F_proc"] - indep["F_proc"]) < 1e-12
    # correlation actually changes the result
    assert (abs(common["F_proc"] - indep["F_proc"]) > 1e-5
            or abs(differ["F_proc"] - indep["F_proc"]) > 1e-5)


def test_colored_noise_correlation_out_of_range_raises(short_cz):
    opt, x = short_cz
    u = torch.as_tensor(x, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    with pytest.raises(ValueError, match="correlation"):
        opt.colored_noise_fidelity(u, sigma_mhz=0.3, correlation=1.5)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
