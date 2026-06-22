"""Noisy closed-loop calibration: a finite-shot IRB estimator + fitter, so the HITL
loop sees realistic statistical measurement noise (not a clean number).
"""
import os

import numpy as np
import pytest

import gradpulse as gp
from gradpulse.hardware import simulate_noisy_irb, QuTiPDeviceBackend


def test_irb_estimator_unbiased_and_shot_scaling():
    """The fit is unbiased (mean -> true F_avg) and its spread shrinks ~1/sqrt(shots)."""
    true = 0.99

    def spread(shots, draws=300):
        rng = np.random.default_rng(1)
        e = np.array([simulate_noisy_irb(true, shots=shots, n_sequences=30, rng=rng)
                      for _ in range(draws)])
        return e.mean(), e.std()

    m_lo, s_lo = spread(200)
    m_hi, s_hi = spread(2000)
    assert abs(m_lo - true) < 3e-3                 # unbiased at low shots
    assert abs(m_hi - true) < 1e-3                 # tighter at high shots
    assert s_hi < s_lo                             # more shots -> less noise
    assert 2.0 < (s_lo / s_hi) < 5.0               # 10x shots -> ~sqrt(10) ~ 3.16x


def test_qutip_backend_emits_shot_noise():
    """QuTiPDeviceBackend(shots=...) returns a noisy RB fit that scatters around the
    exact QuTiP fidelity; the exact value is preserved in metadata for reference."""
    pytest.importorskip("qutip")
    wf = np.load(os.path.join(os.path.dirname(__file__), "fixtures",
                              "reference_cz_pulse.npy"))
    prof = gp.ParametricCouplerProfile()

    exact = QuTiPDeviceBackend(prof, shots=None).measure_gate(wf, dt_ns=1.0).f_avg

    be = QuTiPDeviceBackend(prof, shots=500, n_irb_sequences=40, rng_seed=3)
    draws = np.array([be.measure_gate(wf, dt_ns=1.0).f_avg for _ in range(6)])
    assert draws.std() > 1e-5                       # genuinely noisy, not a constant
    assert abs(draws.mean() - exact) < 8e-3         # scatters around the exact value

    m = be.measure_gate(wf, dt_ns=1.0).metadata
    assert m["shots"] == 500
    assert abs(m["f_avg_exact"] - exact) < 1e-9     # exact value retained
    assert "shotnoise" in be.measure_gate(wf, dt_ns=1.0).source
