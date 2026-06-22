"""Schrieffer-Wolff coupler elimination -- the parametric model replaces the
physical transmon-COUPLER-transmon system with a direct effective exchange
J*(a1+ a2 + a1 a2+), assuming the coupler's own population/coherence are negligible
in the dispersive regime.

validate.coupler_elimination_cross_check builds the explicit 3-body system and the
2-body effective-exchange model and compares the single-excitation swap. The
elimination is validated two ways:

  * the effective model reproduces the full 3-body swap with the coupler only lightly
    populated (the eliminated degree of freedom), and
  * the residual error is the elimination's small parameter (gc/Delta)^2: doubling the
    coupler detuning QUARTERS both the coupler population and the trajectory error --
    the signature of a correct leading-order reduction, not an assumed one.

Run:  pytest tests/test_coupler_elimination.py
"""
import pytest


def test_coupler_elimination_dispersive():
    """In the dispersive regime the 2-body effective exchange reproduces the explicit
    3-body swap, with the coupler (the eliminated DOF) only lightly populated."""
    pytest.importorskip("qutip")
    from gradpulse import validate
    r = validate.coupler_elimination_cross_check(coupler_detuning_mhz=1200.0, gc_mhz=80.0)
    assert r["J_eff_mhz"] > 0.0
    assert r["max_traj_diff"] < 0.05       # effective model reproduces the swap
    assert r["max_coupler_pop"] < 0.05     # eliminated DOF stays lightly populated


def test_coupler_elimination_error_scales_as_sw_parameter():
    """The reduction is correct to leading order: doubling the coupler detuning
    (halving gc/Delta) quarters both the coupler population and the trajectory error,
    the (gc/Delta)^2 signature of Schrieffer-Wolff."""
    pytest.importorskip("qutip")
    from gradpulse import validate
    r1 = validate.coupler_elimination_cross_check(coupler_detuning_mhz=1200.0, gc_mhz=80.0)
    r2 = validate.coupler_elimination_cross_check(coupler_detuning_mhz=2400.0, gc_mhz=80.0)
    # Delta doubled -> (gc/Delta)^2 quarters; both residuals follow (~4x).
    assert 3.0 < r1["max_coupler_pop"] / r2["max_coupler_pop"] < 5.0
    assert 2.5 < r1["max_traj_diff"] / r2["max_traj_diff"] < 6.0
    # And the small parameter itself quarters, as the cross-check reports it.
    assert 3.5 < r1["sw_param"] / r2["sw_param"] < 4.5


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
