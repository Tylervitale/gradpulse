"""Hardware-in-the-loop scaffolding: model refinement from a measured fidelity.

The fast tests evaluate the committed reference CZ pulse (no optimization), so
they pin the HITL math deterministically. The closed-loop test runs a tiny
calibration to confirm a wrong model is pulled toward the truth.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from gradpulse import ParametricCouplerProfile
from gradpulse.hardware import (
    GateMeasurement, QuTiPDeviceBackend, SimulatedBackend, apply_coherence_scale,
    calibrate_to_hardware, infer_coherence_scale, predicted_f_avg,
    predicted_process_fidelity,
)

FIXTURE = Path(__file__).parent / "fixtures" / "reference_cz_pulse.json"


def _ref():
    meta = json.loads(FIXTURE.read_text())
    prof = ParametricCouplerProfile(**meta["profile"])
    wave = np.load(FIXTURE.parent / Path(meta["pulse_npy"]).name)
    return prof, wave, meta


def test_apply_coherence_scale_shortens_times():
    prof = ParametricCouplerProfile(t1_ns_q1=30000, t2_ns_q1=25000,
                                    t1_ns_q2=30000, t2_ns_q2=25000)
    hot = apply_coherence_scale(prof, 2.0)          # noisier => shorter T
    assert hot.t1_ns_q1 == pytest.approx(15000)
    assert hot.t2_ns_q2 == pytest.approx(12500)
    assert prof.t1_ns_q1 == 30000                   # replace() leaves original intact
    assert any("hardware feedback" in n for n in hot.notes)


def test_predicted_fidelity_matches_reference():
    prof, wave, meta = _ref()
    fp = predicted_process_fidelity(prof, wave, dt_ns=1.0)
    assert fp == pytest.approx(float(meta["grape_f"]), abs=1e-3)
    fa = predicted_f_avg(prof, wave, dt_ns=1.0)
    assert fa == pytest.approx((4.0 * fp + 1.0) / 5.0, abs=1e-9)


def test_infer_coherence_scale_identity():
    prof, wave, _ = _ref()
    f = predicted_f_avg(prof, wave, dt_ns=1.0)       # measured == model at diss=1
    assert infer_coherence_scale(prof, wave, f, dt_ns=1.0) == pytest.approx(1.0, abs=0.05)


def test_infer_coherence_scale_recovers_known():
    prof, wave, _ = _ref()
    f2 = predicted_f_avg(prof, wave, dt_ns=1.0, diss_scale=2.0)  # "measured" at 2x diss
    assert infer_coherence_scale(prof, wave, f2, dt_ns=1.0) == pytest.approx(2.0, rel=0.1)


def test_simulated_backend_returns_measurement():
    prof, wave, _ = _ref()
    colder = apply_coherence_scale(prof, 0.5)        # 2x longer T => better than model
    meas = SimulatedBackend(colder).measure_gate(wave, dt_ns=1.0)
    assert isinstance(meas, GateMeasurement)
    assert meas.source == "simulated_analytic"
    assert meas.f_avg > predicted_f_avg(prof, wave, dt_ns=1.0)


def test_calibrate_to_hardware_closes_gap():
    prof, _, _ = _ref()
    # Model starts optimistic (2x too-long coherence); truth = the reference profile.
    optimistic = apply_coherence_scale(prof, 0.5)
    backend = SimulatedBackend(prof)
    out = calibrate_to_hardware(
        optimistic, backend, rounds=2, dt_ns=1.0,
        opt_kwargs=dict(n_seeds=1, iterations=50, n_slices=80,
                        warm_start_mode="parametric_cz",
                        use_process_fidelity=True, lbfgs_polish=False),
    )
    h = out["history"]
    assert len(h) == 2
    assert h[0]["coherence_scale"] > 1.0             # detects the model is too optimistic
    assert abs(h[1]["gap"]) <= abs(h[0]["gap"]) + 1e-4   # gap does not grow


def test_qutip_backend_independent_and_correct():
    """The QuTiP 'device' is an INDEPENDENT integrator (different library). When the
    profile matches it agrees with gradpulse's own double-precision model -- the
    matched cross-check scheme -- and a noisier true device measures strictly lower.
    This parity is what makes a real, mismatched gap trustworthy.
    """
    pytest.importorskip("qutip")
    prof, wave, _ = _ref()
    own = predicted_f_avg(prof, wave, dt_ns=1.0)                       # gradpulse engine
    nominal = QuTiPDeviceBackend(prof).measure_gate(wave, dt_ns=1.0)   # QuTiP engine
    assert nominal.source == "qutip_independent"
    assert nominal.f_avg == pytest.approx(own, abs=1e-3)              # independent parity
    hot = QuTiPDeviceBackend(apply_coherence_scale(prof, 2.0)).measure_gate(
        wave, dt_ns=1.0)
    assert hot.f_avg < nominal.f_avg                                  # worse device => lower F


def test_calibrate_against_independent_engine_pulls_model_to_truth():
    """The full loop closing against an INDEPENDENT integrator (QuTiP), not
    gradpulse's own simulator. From a 2x-too-optimistic model, the loop detects the
    gap and pulls the model's effective coherence toward the device's truth -- a
    physically well-determined direction (the per-round gaps themselves sit near the
    optimizer noise floor at these fast settings, so we assert the direction, not
    strict per-round shrinkage). Tiny settings keep it fast.
    """
    pytest.importorskip("qutip")
    prof, _, _ = _ref()
    optimistic = apply_coherence_scale(prof, 0.5)     # model thinks coherence is 2x better
    out = calibrate_to_hardware(
        optimistic, QuTiPDeviceBackend(prof), rounds=2, dt_ns=1.0,
        opt_kwargs=dict(n_seeds=1, iterations=40, n_slices=60,
                        warm_start_mode="parametric_cz",
                        use_process_fidelity=True, lbfgs_polish=False),
    )
    h = out["history"]
    assert len(h) == 2
    assert h[0]["source"] == "qutip_independent"
    assert h[0]["coherence_scale"] > 1.2              # detects the optimistic model (~2x)
    refined = out["refined_profile"]                  # effective coherence pulled to truth
    assert abs(refined.t1_ns_q1 - prof.t1_ns_q1) < abs(optimistic.t1_ns_q1 - prof.t1_ns_q1)
    assert abs(h[1]["gap"]) < 1.5e-3                  # model now tracks the device closely


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
