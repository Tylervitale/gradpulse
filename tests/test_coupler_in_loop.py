"""Coupler-in-the-loop CZ opt-in (gradpulse.coupler_in_loop_cz).

The dispersive ParametricCZOptimizer eliminates the coupler under Schrieffer-Wolff,
so its coupler population is identically zero. ``coupler_in_loop_cz`` re-introduces
the coupler as a live transmon between the same two qubits and reports the leakage
that elimination cannot see, plus the SW small-parameter ``(gc/Delta)^2``. It is
built on the QuTiP-cross-checked MultiQubitOptimizer engine, so the load-bearing
test is the same apples-to-apples cross-check -- which holds at any fidelity level,
independent of how far a short run converges.
"""
import math

import pytest

import gradpulse as gp


def test_diagnostic_fields_and_sw_math():
    """The wrapper returns the coupler-specific diagnostics with correct SW math."""
    r = gp.coupler_in_loop_cz(coupler_freq_ghz=5.9, gc_mhz=95.0,
                              n_seeds=1, iterations=6, n_slices=80, verbose=False)
    for k in ("coupler_leakage", "sw_param", "J_eff_mhz", "coupler_freq_ghz", "optimizer"):
        assert k in r, f"missing {k}"
    # leakage is a physical population fraction
    assert 0.0 <= r["coupler_leakage"] <= 1.0
    # SW small-parameter (gc/Delta)^2 with Delta = f_q1 - f_coupler (default profile q1=4.85)
    delta_mhz = (4.85 - 5.9) * 1000.0
    assert math.isclose(r["sw_param"], (95.0 / delta_mhz) ** 2, rel_tol=1e-9)
    # J_eff = (gc^2/2)(1/D0 + 1/D1) with the default profile (q1=4.85, q2=5.05)
    d0 = (4.85 - 5.9) * 1000.0
    d1 = (5.05 - 5.9) * 1000.0
    j_expected = (95.0 ** 2 / 2.0) * (1.0 / d0 + 1.0 / d1)
    assert math.isclose(r["J_eff_mhz"], j_expected, rel_tol=1e-9)


def test_coupler_leakage_is_visible():
    """The explicit model exposes a coupler-population channel the pair model omits:
    on a generic (un-converged) pulse the coupler is excited, so leakage is nonzero."""
    r = gp.coupler_in_loop_cz(n_seeds=1, iterations=4, n_slices=80, verbose=False)
    assert r["coupler_leakage"] > 0.0   # the DOF the eliminated pair model cannot show


def test_qutip_cross_check_apples_to_apples():
    """The returned optimizer feeds straight into multiqubit_cross_check, and the
    independent QuTiP integrator reproduces F_proc to ~machine precision -- the
    invariant that holds regardless of how far the (slow) 27-dim run converged."""
    pytest.importorskip("qutip")
    from gradpulse.validate import multiqubit_cross_check
    r = gp.coupler_in_loop_cz(precision="double", n_seeds=1, iterations=6,
                              n_slices=80, verbose=False)
    xc = multiqubit_cross_check(r["optimizer"], r["best_waveform"], dt_ns=1.0)
    assert xc["delta"] < 1e-6, f"QuTiP disagrees: {xc}"


def test_accepts_pair_profile():
    """Hand it the pair profile you already use -- the qubit params flow through."""
    from gradpulse import ParametricCouplerProfile
    prof = ParametricCouplerProfile(freq_ghz_q1=4.70, freq_ghz_q2=4.90)
    r = gp.coupler_in_loop_cz(prof, n_seeds=1, iterations=4, n_slices=80, verbose=False)
    # SW param must reflect the profile's q1, not the default
    delta_mhz = (4.70 - r["coupler_freq_ghz"]) * 1000.0
    assert math.isclose(r["sw_param"], (95.0 / delta_mhz) ** 2, rel_tol=1e-9)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
