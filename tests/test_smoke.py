"""Smoke test: the optimizer runs end-to-end and returns a structurally valid,
non-trivial CZ pulse.

Deliberately short (single seed, few iterations, brief pulse) so it gates CI in
seconds with or without a GPU. The headline fidelity claim (the 150 ns CZ's process
fidelity ~0.990, QuTiP-cross-checked) is demonstrated by examples/optimize_cz.py +
gradpulse.validate and pinned by tests/test_reproducibility.py; this test only guards
"it runs and produces a valid, better-than-trivial result."

Run:  pytest tests/        OR        python tests/test_smoke.py
"""
import numpy as np

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer

N_SLICES, N_CHANNELS = 60, 3


def run_short():
    profile = ParametricCouplerProfile(
        freq_ghz_q1=4.85, freq_ghz_q2=5.05,
        anharm_ghz_q1=-0.20, anharm_ghz_q2=-0.20,
        t1_ns_q1=30_000, t2_ns_q1=25_000,
        t1_ns_q2=30_000, t2_ns_q2=25_000,
        g_max_mhz=12.0, omega_max_mhz=50.0,
    )
    opt = ParametricCZOptimizer(profile, bandwidth_mhz=80.0, use_drag=False,
                                n_channels=N_CHANNELS, activation="sigmoid")
    return opt.optimize_multi_seed(
        n_seeds=1, iterations=25, n_slices=N_SLICES, dt_ns=1.0,
        warm_start_mode="parametric_cz", use_process_fidelity=True,
        lbfgs_polish=False,
    )


def test_returns_valid_pulse():
    result = run_short()
    f = float(result["best_fidelity"])
    wf = np.asarray(result["best_waveform"])
    assert 0.0 <= f <= 1.0, f"fidelity out of range: {f}"
    assert wf.shape == (N_SLICES, N_CHANNELS), f"bad waveform shape: {wf.shape}"
    assert wf.min() >= -1e-6 and wf.max() <= 1 + 1e-6, "waveform not in [0, 1]"


def test_beats_trivial():
    # Process fidelity of a random/identity map to CZ is ~0.25; a parametric-CZ
    # warm start with a few GRAPE steps must clear that comfortably.
    assert float(run_short()["best_fidelity"]) > 0.3


if __name__ == "__main__":
    test_returns_valid_pulse()
    test_beats_trivial()
    print("smoke tests passed")
