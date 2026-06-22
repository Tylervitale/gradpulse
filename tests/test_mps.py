"""MPS evaluator (gradpulse.mps) -- validate the chain-TEBD foundation against the
dense/sparse paths before trusting it past the dense wall.

Layer 1: second-order TEBD reproduces the EXACT closed-system process fidelity
         (MultiQubitOptimizer.process_fidelity_sparse) as the Trotter substep count
         grows -- proves the local-operator re-derivation + the TEBD splitting.
Layer 2: the UNtruncated MPS reproduces the full state vector to machine precision,
         and truncation error is tracked by the discarded Schmidt weight (the
         chi-convergence ship gate).

Deterministic, double precision, tiny systems. Run:  pytest tests/test_mps.py
"""
import numpy as np
import pytest

from gradpulse.multiqubit import MultiQubitProfile, MultiQubitOptimizer
from gradpulse.mps import ChainTEBD


def _chain_opt(N=3, use_drag=True):
    prof = MultiQubitProfile(
        n_qubits=N, freqs_ghz=[5.0 + 0.1 * q for q in range(N)],
        anharm_mhz=[-300.0] * N, t1_ns=[4e4] * N, t2_ns=[3e4] * N,
        couplings={(q, q + 1): 12.0 for q in range(N - 1)}, n_levels=3)
    return MultiQubitOptimizer(prof, target_gate="cz", target_qubits=(0, 1),
                               tunable_edges=[], open_system=False,
                               precision="double", use_drag=use_drag, verbose=False)


def _random_waveform(opt, Nt=10, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(0.5 + 0.3 * rng.uniform(-1, 1, (Nt, opt.n_channels)), 0.0, 1.0)


@pytest.mark.parametrize("use_drag", [False, True])
def test_tebd_converges_to_exact_sparse(use_drag):
    """Layer 1: TEBD F_proc -> exact sparse F_proc with 2nd-order (ratio ~1/4) Trotter."""
    pytest.importorskip("scipy")        # process_fidelity_sparse needs the [sparse] extra
    opt = _chain_opt(N=3, use_drag=use_drag)
    wf = _random_waveform(opt)
    teb = ChainTEBD(opt)
    f_exact = opt.process_fidelity_sparse(wf, dt_ns=1.0)
    errs = [abs(teb.process_fidelity_tebd(wf, dt_ns=1.0, substeps=s) - f_exact)
            for s in (1, 2, 4, 8)]
    # monotone decrease and clean second-order convergence (each doubling ~/4)
    assert errs[-1] < errs[0]
    assert errs[-1] < 1e-4
    for k in range(1, len(errs)):
        assert errs[k] < errs[k - 1]
    # last ratio close to the 0.25 second-order Trotter signature
    assert 0.18 < errs[-1] / errs[-2] < 0.32


def test_untruncated_mps_matches_full_statevector():
    """Layer 2: large-chi MPS == full-statevector TEBD (same scheme) to machine eps."""
    opt = _chain_opt(N=4, use_drag=True)
    teb = ChainTEBD(opt)
    N, d = opt.N, opt.d
    wf = _random_waveform(opt, Nt=12, seed=3)
    rng = np.random.default_rng(7)
    kets = [(lambda v: v / np.linalg.norm(v))(rng.normal(size=d) + 1j * rng.normal(size=d))
            for _ in range(N)]
    psi0 = kets[0]
    for q in range(1, N):
        psi0 = np.kron(psi0, kets[q])
    full = teb.evolve_statevector(psi0[None, :], wf, substeps=3)[0]
    full /= np.linalg.norm(full)
    mps, disc = teb.evolve_mps(kets, wf, substeps=3, chi_max=64)
    ov = abs(np.vdot(full, teb.mps_to_vector(mps))) ** 2
    assert abs(1.0 - ov) < 1e-10
    assert disc == 0.0                      # d^2=9 >= max Schmidt rank -> no truncation


def test_truncation_tracked_by_discarded_weight():
    """Layer 2: tighter chi -> more discarded weight and larger state error (the
    chi-convergence handle the ship gate reports)."""
    opt = _chain_opt(N=4, use_drag=True)
    teb = ChainTEBD(opt)
    N, d = opt.N, opt.d
    wf = _random_waveform(opt, Nt=12, seed=3)
    rng = np.random.default_rng(7)
    kets = [(lambda v: v / np.linalg.norm(v))(rng.normal(size=d) + 1j * rng.normal(size=d))
            for _ in range(N)]
    psi0 = kets[0]
    for q in range(1, N):
        psi0 = np.kron(psi0, kets[q])
    full = teb.evolve_statevector(psi0[None, :], wf, substeps=3)[0]
    full /= np.linalg.norm(full)
    errs, discs = [], []
    for chi in (9, 4, 2):
        mps, disc = teb.evolve_mps(kets, wf, substeps=3, chi_max=chi)
        errs.append(1.0 - abs(np.vdot(full, teb.mps_to_vector(mps))) ** 2)
        discs.append(disc)
    # tighter chi never helps: error and discarded weight both grow as chi shrinks
    assert errs[0] <= errs[1] <= errs[2]
    assert discs[0] <= discs[1] <= discs[2]
    assert discs[2] > 0.0


def test_trajectory_witness_matches_dense_open_system():
    """Layer 3: the trajectory-MPS restricted-ensemble witness reproduces the dense
    open-system (Lindblad) witness on the same ensemble, within statistical error.
    Short T1/T2 so decoherence (and jumps) are actually exercised."""
    import torch
    from gradpulse.parametric import DEVICE
    N, d = 3, 3
    prof = MultiQubitProfile(
        n_qubits=N, freqs_ghz=[5.0, 5.1, 5.2], anharm_mhz=[-300.0] * N,
        t1_ns=[2000.0] * N, t2_ns=[1500.0] * N,
        couplings={(0, 1): 12.0, (1, 2): 12.0}, n_levels=d)
    opt = MultiQubitOptimizer(prof, target_gate="cz", target_qubits=(0, 1),
                              tunable_edges=[], open_system=True,
                              precision="double", use_drag=False, verbose=False)
    teb = ChainTEBD(opt)
    rng = np.random.default_rng(1)
    Nt = 14
    wf = np.clip(0.5 + 0.25 * rng.uniform(-1, 1, (Nt, opt.n_channels)), 0.0, 1.0)
    # ensemble of computational product inputs (levels 0,1)
    ens = []
    for _ in range(2):
        kets = []
        for _q in range(N):
            c = rng.normal(size=2) + 1j * rng.normal(size=2)
            c /= np.linalg.norm(c)
            k = np.zeros(d, complex)
            k[0], k[1] = c
            kets.append(k)
        ens.append(kets)
    # dense restricted-ensemble witness via _propagate_choi
    D = d ** N
    rho0 = np.zeros((1, len(ens), D, D), complex)
    tgts = []
    for m, kets in enumerate(ens):
        psi = kets[0]
        for q in range(1, N):
            psi = np.kron(psi, kets[q])
        rho0[0, m] = np.outer(psi, psi.conj())
        tgts.append(teb._embed_target_vector(kets))
    x = torch.as_tensor(wf[None], dtype=opt.rdtype, device=DEVICE)
    r0 = torch.as_tensor(rho0, dtype=opt.cdtype, device=DEVICE)
    with torch.no_grad():
        out = opt._propagate_choi(x, 1.0, 1.0, r0).cpu().numpy()[0]
    w_dense = float(np.mean([np.real(tgts[m].conj() @ out[m] @ tgts[m])
                             for m in range(len(ens))]))
    res = teb.witness_open(ens, wf, dt_ns=1.0, substeps=2, chi_max=16,
                           n_traj=250, seed=7)
    assert 0.0 <= res["witness"] <= 1.0
    assert res["mean_jumps"] >= 0.0
    # agree within statistical error (a few sigma) or a small absolute floor
    assert abs(res["witness"] - w_dense) < max(5.0 * res["sem"], 4e-3)


def test_requires_chain_topology():
    """Non-nearest-neighbour coupling is rejected (honest scope, not silent fallback)."""
    prof = MultiQubitProfile(
        n_qubits=3, freqs_ghz=[5.0, 5.1, 5.2], anharm_mhz=[-300.0] * 3,
        t1_ns=[4e4] * 3, t2_ns=[3e4] * 3, couplings={(0, 2): 12.0}, n_levels=3)
    opt = MultiQubitOptimizer(prof, target_gate="cz", target_qubits=(0, 2),
                              tunable_edges=[], open_system=False,
                              precision="double", verbose=False)
    with pytest.raises(ValueError, match="chain"):
        ChainTEBD(opt)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
