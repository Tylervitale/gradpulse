"""Tests for the general N-qubit GRAPE optimizer (gradpulse.multiqubit).

Covers: register/operator construction, subset-target correctness (gate on a few
qubits, identity elsewhere), the open- and closed-system fidelity paths, that the
optimizer actually drives fidelity up with a coupled spectator present (i.e. it
optimizes *against* crosstalk), and -- the load-bearing one -- that an independent
QuTiP rebuild reproduces F_proc, so the model is real and not a self-consistent
artifact.
"""
import numpy as np
import pytest
import torch

from gradpulse import multiqubit as mq


def _prof3(g12=12.0, g23=6.0, n_levels=3):
    return mq.MultiQubitProfile(
        n_qubits=3, freqs_ghz=[5.00, 5.20, 5.40], anharm_mhz=[-300, -300, -300],
        t1_ns=[50000] * 3, t2_ns=[40000] * 3,
        couplings={(0, 1): g12, (1, 2): g23}, n_levels=n_levels)


def test_construction_dimensions():
    opt = mq.MultiQubitOptimizer(_prof3(), target_gate="cz", target_qubits=(0, 1),
                                 verbose=False)
    assert opt.D == 27 and opt._dcomp == 8
    assert len(opt._comp_idx) == 8
    # 3 drives + 1 tunable edge (0,1) [the only target-internal edge]
    assert opt.n_channels == 4
    assert opt.tunable_edges == [(0, 1)]
    assert (1, 2) in opt.fixed_edges          # spectator edge stays always-on


def test_target_cz_on_subset_is_identity_elsewhere():
    opt = mq.MultiQubitOptimizer(_prof3(), target_gate="cz", target_qubits=(0, 1),
                                 verbose=False)
    U = opt.u_target.cpu().numpy()
    assert np.allclose(U.conj().T @ U, np.eye(8))
    # CZ on (0,1) (x) I on q2 => -1 exactly on |11x>: comp states index 6,7
    diag = np.real(np.diag(U))
    expected = np.array([1, 1, 1, 1, 1, 1, -1, -1.0])
    assert np.allclose(diag, expected)
    assert np.allclose(U, np.diag(diag))      # purely diagonal


def test_target_on_nonadjacent_qubits_and_custom_unitary():
    # CZ between qubits 0 and 2 (skipping 1): -1 where q0=q2=1, any q1.
    opt = mq.MultiQubitOptimizer(_prof3(), target_gate="cz", target_qubits=(0, 2),
                                 verbose=False)
    diag = np.real(np.diag(opt.u_target.cpu().numpy()))
    # comp index = 4*q0 + 2*q1 + q2; q0=q2=1 -> idx in {5,7}
    minus = {i for i in range(8) if diag[i] < 0}
    assert minus == {5, 7}
    # a custom 4x4 (sqrt-CZ) is accepted and stays unitary
    sq = np.diag([1, 1, 1, 1j]).astype(complex)
    opt2 = mq.MultiQubitOptimizer(_prof3(), target_gate=sq, target_qubits=(0, 1),
                                  verbose=False)
    assert np.allclose(opt2.u_target.cpu().numpy().conj().T
                       @ opt2.u_target.cpu().numpy(), np.eye(8))


def test_non_unitary_target_rejected():
    bad = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=complex)
    with pytest.raises(ValueError):
        mq.MultiQubitOptimizer(_prof3(), target_gate=bad, target_qubits=(0, 1),
                               verbose=False)


def test_cost_estimate_flags_large_systems():
    small = mq.MultiQubitOptimizer(_prof3(), target_qubits=(0, 1), verbose=False)
    assert small.cost_estimate()["hilbert_dim"] == 27
    big = mq.MultiQubitProfile(
        n_qubits=5, freqs_ghz=[5.0, 5.1, 5.2, 5.3, 5.4], anharm_mhz=[-300] * 5,
        t1_ns=[40000] * 5, t2_ns=[30000] * 5,
        couplings={(0, 1): 10.0, (1, 2): 10.0, (2, 3): 10.0, (3, 4): 10.0}, n_levels=3)
    c = mq.MultiQubitOptimizer(big, target_qubits=(1, 2), open_system=True,
                               verbose=False).cost_estimate()
    assert c["hilbert_dim"] == 243 and "LARGE" in c["warning"]


def test_closed_system_optimization_improves_fidelity():
    # Closed system is fast; a short run must measurably raise F_proc from random.
    opt = mq.MultiQubitOptimizer(_prof3(), target_gate="cz", target_qubits=(0, 1),
                                 open_system=False, precision="single", verbose=False)
    res = opt.optimize(n_slices=80, dt_ns=1.0, iterations=120, n_seeds=1, lr=0.06)
    assert res["best_fidelity"] > 0.80
    assert res["best_waveform"].shape == (80, opt.n_channels)


def test_drag_is_derived_not_a_channel():
    """DRAG is a derived quadrature of the in-phase drive, so enabling it must NOT
    add an optimization channel, and no channel may be left dead (unused)."""
    base = mq.MultiQubitOptimizer(_prof3(), target_gate="cz", target_qubits=(0, 1),
                                  use_drag=False, open_system=False, verbose=False)
    drag = mq.MultiQubitOptimizer(_prof3(), target_gate="cz", target_qubits=(0, 1),
                                  use_drag=True, open_system=False, verbose=False)
    assert drag.n_channels == base.n_channels      # DRAG consumes no channel
    res = drag.optimize(n_slices=60, dt_ns=1.0, iterations=80, n_seeds=1, lr=0.06)
    assert res["best_fidelity"] > 0.80
    # every channel column is actually exercised -- no dead parameter
    assert (res["best_waveform"].std(axis=0) > 1e-4).all()


def test_optimizes_against_coupled_spectator():
    """With a strongly-coupled spectator, the optimizer should still reach a good
    gate -- it folds the crosstalk into the objective rather than ignoring it."""
    opt = mq.MultiQubitOptimizer(_prof3(g23=10.0), target_gate="cz",
                                 target_qubits=(0, 1), open_system=False,
                                 precision="single", verbose=False)
    res = opt.optimize(n_slices=100, dt_ns=1.0, iterations=150, n_seeds=2, lr=0.06)
    assert res["best_fidelity"] > 0.90


def test_open_nodiss_matches_closed_system():
    """Internal-consistency check between the two fidelity code paths: the open-system
    Choi fidelity with dissipation switched off must equal the closed-system unitary
    fidelity for the same pulse, and turning dissipation on may only lower it."""
    prof = _prof3()
    oc = mq.MultiQubitOptimizer(prof, target_gate="cz", target_qubits=(0, 1),
                                open_system=False, precision="double", verbose=False)
    res = oc.optimize(n_slices=40, dt_ns=1.0, iterations=50, n_seeds=1, lr=0.06)
    wf = res["best_waveform"]
    f_closed = oc.process_fidelity(wf, dt_ns=1.0)

    oo = mq.MultiQubitOptimizer(prof, target_gate="cz", target_qubits=(0, 1),
                                open_system=True, precision="double", verbose=False)
    f_open_nodiss = oo.process_fidelity(wf, dt_ns=1.0, diss_scale=0.0)
    f_open_diss = oo.process_fidelity(wf, dt_ns=1.0, diss_scale=1.0)
    assert abs(f_closed - f_open_nodiss) < 1e-9
    assert f_open_diss <= f_open_nodiss + 1e-9


def test_state_transfer_converges_to_exact_choi():
    """The memory-light state-transfer estimator is an unbiased estimate of the exact
    Choi F_proc; averaging over several seeds (deterministic variance reduction) it
    matches the exact value. This is what lets the open-system optimizer reach larger
    N -- O(n_states) propagated operators instead of 4**N."""
    opt = mq.MultiQubitOptimizer(_prof3(), target_gate="cz", target_qubits=(0, 1),
                                 open_system=True, precision="double", verbose=False)
    res = opt.optimize(n_slices=40, dt_ns=1.0, iterations=60, n_seeds=1, lr=0.06)
    wf = res["best_waveform"]
    f_exact = opt.process_fidelity(wf, dt_ns=1.0)
    # average the estimate over seeds to beat down the Monte-Carlo variance
    ests = [opt.state_transfer_fidelity(wf, dt_ns=1.0, n_states=512, seed=k)["F_proc"]
            for k in range(6)]
    assert abs(float(np.mean(ests)) - f_exact) < 6e-3


def test_state_transfer_optimize_produces_a_real_gate():
    """The state-transfer objective optimizes; the resulting pulse is a genuine gate
    when scored with the exact Choi fidelity."""
    opt = mq.MultiQubitOptimizer(_prof3(), target_gate="cz", target_qubits=(0, 1),
                                 open_system=True, precision="single", verbose=False)
    res = opt.optimize(n_slices=40, dt_ns=1.0, iterations=80, n_seeds=1, lr=0.06,
                       fidelity="state_transfer", n_states=64)
    assert res["fidelity_mode"] == "state_transfer"
    f_exact = opt.process_fidelity(res["best_waveform"], dt_ns=1.0)
    assert f_exact > 0.5


def test_state_transfer_fidelity_bad_mode_raises():
    opt = mq.MultiQubitOptimizer(_prof3(), target_gate="cz", target_qubits=(0, 1),
                                 open_system=True, verbose=False)
    with pytest.raises(ValueError, match="choi.*state_transfer|fidelity"):
        opt.optimize(n_slices=10, iterations=1, n_seeds=1, fidelity="nope")


@pytest.mark.slow
def test_qutip_cross_check_open_system():
    """Independent QuTiP rebuild reproduces F_proc -- the proof the N-qubit model
    is correct. Agreement must hold regardless of how good the pulse is, so a short
    optimization is enough."""
    pytest.importorskip("qutip")
    from gradpulse import validate as V
    opt = mq.MultiQubitOptimizer(_prof3(), target_gate="cz", target_qubits=(0, 1),
                                 open_system=True, precision="double", verbose=False)
    res = opt.optimize(n_slices=40, dt_ns=1.0, iterations=40, n_seeds=1, lr=0.06)
    xc = V.multiqubit_cross_check(opt, res["best_waveform"], dt_ns=1.0)
    assert xc["delta"] < 1e-5, f"torch/QuTiP disagree: {xc}"
