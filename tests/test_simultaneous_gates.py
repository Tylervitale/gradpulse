"""Simultaneous / parallel multi-gate optimization (MultiQubitOptimizer).

A real scheduler runs several gates at once; the single-subset target only ever
asked for ONE gate + identity. These tests pin that a list of (gate, disjoint qubit
group) specs builds the correct tensor-product target, leaves cross-group couplings
FIXED (the crosstalk the optimizer must fight), and -- the load-bearing claim --
that the combined target is the SAME physics an independent QuTiP rebuild computes.
"""
import math

import numpy as np
import pytest
import torch

from gradpulse.multiqubit import MultiQubitProfile, MultiQubitOptimizer

CZ = np.diag([1, 1, 1, -1]).astype(complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)


def _line4(n_levels=2):
    return MultiQubitProfile(
        n_qubits=4, freqs_ghz=(4.9, 5.0, 5.15, 5.25), anharm_mhz=(-300,) * 4,
        t1_ns=(40_000,) * 4, t2_ns=(30_000,) * 4,
        couplings={(0, 1): 12.0, (1, 2): 4.0, (2, 3): 12.0}, n_levels=n_levels)


def test_combined_target_is_tensor_product():
    o = MultiQubitOptimizer(_line4(), target_gate=["cz", "cz"],
                            target_qubits=[(0, 1), (2, 3)], open_system=False, verbose=False)
    assert np.allclose(o.u_target.cpu().numpy(), np.kron(CZ, CZ))


def test_single_qubit_group_and_replicated_gate():
    p3 = MultiQubitProfile(n_qubits=3, n_levels=2, couplings={(0, 1): 12.0, (1, 2): 6.0})
    # CZ on (0,1) AND X on q2 (bare-int single-qubit group)
    o = MultiQubitOptimizer(p3, target_gate=["cz", "x"], target_qubits=[(0, 1), 2],
                            open_system=False, verbose=False)
    assert np.allclose(o.u_target.cpu().numpy(), np.kron(CZ, X))
    # one gate name replicated across groups
    o2 = MultiQubitOptimizer(_line4(), target_gate="cz", target_qubits=[(0, 1), (2, 3)],
                             open_system=False, verbose=False)
    assert np.allclose(o2.u_target.cpu().numpy(), np.kron(CZ, CZ))


def test_default_tunable_edges_are_intra_group():
    """Gate-mediating couplings are tunable; the cross-group (crosstalk) edge stays
    fixed/always-on so the optimizer cannot trivially switch it off."""
    o = MultiQubitOptimizer(_line4(), target_gate=["cz", "cz"],
                            target_qubits=[(0, 1), (2, 3)], open_system=False, verbose=False)
    assert set(o.tunable_edges) == {(0, 1), (2, 3)}
    assert (1, 2) in o.fixed_edges


def test_overlapping_groups_rejected():
    with pytest.raises(ValueError, match="disjoint|two gate groups"):
        MultiQubitOptimizer(_line4(), target_gate=["cz", "cz"],
                            target_qubits=[(0, 1), (1, 2)], open_system=False, verbose=False)


def test_backward_compatible_single_gate():
    p2 = MultiQubitProfile(n_qubits=2, freqs_ghz=(5.0, 5.1), anharm_mhz=(-300, -300),
                           t1_ns=(40_000, 40_000), t2_ns=(30_000, 30_000),
                           couplings={(0, 1): 12.0}, n_levels=2)
    o = MultiQubitOptimizer(p2, target_gate="cz", target_qubits=(0, 1),
                            open_system=False, verbose=False)
    assert np.allclose(o.u_target.cpu().numpy(), CZ)


@pytest.mark.slow
def test_simultaneous_target_matches_qutip():
    """The combined CZ x CZ channel matches an INDEPENDENT QuTiP rebuild."""
    qt = pytest.importorskip("qutip")
    from gradpulse import validate as V
    o = MultiQubitOptimizer(_line4(n_levels=2), target_gate=["cz", "cz"],
                            target_qubits=[(0, 1), (2, 3)], open_system=True, verbose=False)
    rng = np.random.RandomState(0)
    wf = 0.3 + 0.4 * rng.rand(24, o.n_channels)
    cc = V.multiqubit_cross_check(o, wf, dt_ns=1.0)
    assert cc["delta"] < 1e-5, f"torch/QuTiP disagree on combined target: {cc}"


@pytest.mark.slow
def test_optimizer_improves_both_gates():
    o = MultiQubitOptimizer(_line4(n_levels=2), target_gate=["cz", "cz"],
                            target_qubits=[(0, 1), (2, 3)], open_system=True, verbose=False)
    res = o.optimize(n_slices=40, dt_ns=1.0, iterations=40, n_seeds=1, lr=0.06, verbose=False)
    assert res["best_fidelity"] > 0.2     # from ~0.01 cold start; both gates moving
