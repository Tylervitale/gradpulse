"""Sparse/Krylov evaluation path (MultiQubitOptimizer.process_fidelity_sparse).

Evaluation-only scaling beyond the dense ~4-qubit wall: propagate the 2**N
computational basis state vectors with scipy expm_multiply instead of building the
D x D propagator. Correctness = it must reproduce the dense closed-system metric to
integrator precision; it must also run at a register size and report a valid number.
"""
import math

import numpy as np
import pytest

pytest.importorskip("scipy", reason="needs scipy for expm_multiply")

from gradpulse import MultiQubitProfile, MultiQubitOptimizer


def _chain(n, n_levels=2, g=8.0):
    return MultiQubitProfile(
        n_qubits=n, freqs_ghz=[4.8 + 0.03 * i for i in range(n)],
        anharm_mhz=[-200] * n, t1_ns=[3e4] * n, t2_ns=[2e4] * n,
        couplings={(i, i + 1): g for i in range(n - 1)}, n_levels=n_levels)


def test_sparse_matches_dense_unitary():
    """Krylov state-propagation reproduces the dense unitary F_proc to ~machine
    precision (same metric, both exact per-slice exponentials)."""
    prof = _chain(2)
    opt = MultiQubitOptimizer(prof, target_gate="cz", target_qubits=(0, 1),
                              open_system=False, precision="double", verbose=False)
    r = opt.optimize(n_slices=40, iterations=25, n_seeds=1)
    f_dense = opt.process_fidelity(r["best_waveform"])
    f_sparse = opt.process_fidelity_sparse(r["best_waveform"])
    assert abs(f_dense - f_sparse) < 1e-6, (f_dense, f_sparse)


def test_sparse_runs_at_larger_register():
    """Evaluates at N=6 (D=64) and returns a valid fidelity -- the path is not
    limited to the small systems the dense Choi optimizer targets."""
    prof = _chain(6)
    opt = MultiQubitOptimizer(prof, target_gate="cz", target_qubits=(0, 1),
                              open_system=False, precision="double", verbose=False)
    rng = np.random.RandomState(0)
    wf = rng.rand(25, opt.n_channels)
    f = opt.process_fidelity_sparse(wf, dt_ns=1.0)
    assert math.isfinite(f) and 0.0 <= f <= 1.0


def test_sparse_matches_dense_with_freq_control():
    """Agreement also holds with the tunable-coupler frequency-control channel."""
    import gradpulse as gp
    opt = gp.tunable_coupler_cz(precision="double", verbose=False)
    # closed-system sparse vs a dense closed-system reference on the same model
    ref = MultiQubitOptimizer(opt.profile, target_gate="cz", target_qubits=(0, 2),
                              drive_qubits=[], tunable_edges=[],
                              freq_control_qubits=[0, 1, 2], delta_max_mhz=300.0,
                              open_system=False, precision="double", verbose=False)
    rng = np.random.RandomState(1)
    wf = rng.rand(20, ref.n_channels)
    assert abs(ref.process_fidelity(wf) - ref.process_fidelity_sparse(wf)) < 1e-6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
