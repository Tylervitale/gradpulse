"""gradpulse.rb - simulated leakage-aware interleaved randomized benchmarking.

`_process_fidelity` computes the *estimand* (the analytic average gate fidelity).
This module simulates the *estimator* a hardware experiment reports: it runs
randomized-benchmarking sequences of 2-qubit Cliffords -- compiled to the
device's native gates -- on the SIMULATED noisy gate, tracks both the survival
probability and the computational-subspace population, and fits the leakage-aware
decay model. The point is the bridge:

  * in the incoherent, leakage-free limit the interleaved-RB gate error recovers
    1 - F_avg from the analytic process fidelity (validated in tests/test_rb.py
    against an exact depolarizing channel), and
  * with real |2>-leakage the NAIVE single-exponential RB fit is biased, while
    the leakage-aware fit separates the leakage rate from the depolarizing decay
    -- which is exactly why a bare simulator fidelity and a hardware RB number are
    different estimators and must not be conflated.

Native gates are fixed superoperators computed ONCE (the noisy CZ from the
simulator, ideal single-qubit Cliffords), so a length-m sequence is m matrix
products of 81x81 superoperators, not m re-solved master equations.

Modeling choices (documented, not hidden):
  * Single-qubit Cliffords are treated as ideal (identity on the |2> levels):
    the parametric-coupler CZ is coupler-activated with near-quiet single-qubit
    drives, so single-qubit leakage is negligible and the benchmarked error is
    the CZ's. All leakage in the RB therefore comes from the CZ.
  * The 2-qubit Clifford group (order 11520) is enumerated by breadth-first
    search over {H1,H2,S1,S2,CZ}; each element is stored with the native-gate
    word that builds it, so its NOISY superoperator is the product of the native
    superoperators (mean 1.88 CZ per Clifford).
"""
from __future__ import annotations

import functools

import numpy as np

# Computational levels |00>,|01>,|10>,|11> in the 9-D (two-qutrit) space.
COMP_IDX = (0, 1, 3, 4)

# ---- 2-qubit Clifford group as native-gate words -------------------------
_S2 = 1.0 / np.sqrt(2.0)
_H = np.array([[_S2, _S2], [_S2, -_S2]], dtype=complex)
_S = np.array([[1.0, 0.0], [0.0, 1j]], dtype=complex)
_I2 = np.eye(2, dtype=complex)
_GENERATORS = {
    "H1": np.kron(_H, _I2),
    "H2": np.kron(_I2, _H),
    "S1": np.kron(_S, _I2),
    "S2": np.kron(_I2, _S),
    "CZ": np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex),
}


def _canon(U: np.ndarray):
    """Global-phase-fixed, rounded key for a 4x4 unitary (mod global phase).

    Divide out the phase of the FIRST entry (fixed scan order) whose magnitude
    clears 0.1 -- Clifford entries are 0 or >= 1/(2*sqrt2) ~ 0.354, so this is
    stable under float drift and picks the same reference for phase-equivalent
    and different-path copies of one element (argmax would be unstable on ties).
    """
    flat = U.reshape(-1)
    idx = int(np.argmax(np.abs(flat) > 0.1))
    phase = flat[idx] / abs(flat[idx])
    return tuple(np.round((U / phase).reshape(-1), 5).tolist())


class CliffordGroup:
    """The 11520-element 2-qubit Clifford group: ideal 4x4 unitaries, their
    native-gate words, and a canon->index map for inverse (recovery) lookup."""

    def __init__(self, unitaries, words, key_to_index):
        self.unitaries = unitaries          # list of 4x4 complex
        self.words = words                  # list of generator-name tuples
        self._key_to_index = key_to_index

    def __len__(self):
        return len(self.unitaries)

    def index_of(self, U: np.ndarray) -> int:
        """Group index of a unitary (mod global phase); raises if not a Clifford."""
        return self._key_to_index[_canon(U)]


@functools.lru_cache(maxsize=1)
def two_qubit_cliffords() -> CliffordGroup:
    """Enumerate the group once (BFS over the native generators); cached."""
    ident = np.eye(4, dtype=complex)
    key_to_index = {_canon(ident): 0}
    unitaries = [ident]
    words: list = [()]
    frontier = [0]
    while frontier:
        nxt = []
        for i in frontier:
            U, word = unitaries[i], words[i]
            for gname, G in _GENERATORS.items():
                V = G @ U
                k = _canon(V)
                if k not in key_to_index:
                    key_to_index[k] = len(unitaries)
                    unitaries.append(V)
                    words.append(word + (gname,))
                    nxt.append(key_to_index[k])
        frontier = nxt
    return CliffordGroup(unitaries, words, key_to_index)


# ---- Superoperators (9-D, column-stacking vec = reshape(-1)) --------------
def _embed4_in_9(U4: np.ndarray) -> np.ndarray:
    """Embed a 4x4 computational-subspace unitary in the 9-D space as identity
    on the non-computational (|2>) levels."""
    U9 = np.eye(9, dtype=complex)
    for a, ia in enumerate(COMP_IDX):
        for b, ib in enumerate(COMP_IDX):
            U9[ia, ib] = U4[a, b]
    return U9


def superop_from_unitary(U9: np.ndarray) -> np.ndarray:
    """81x81 superoperator of rho -> U9 rho U9^dag (vec = reshape(-1))."""
    return np.kron(U9, U9.conj())


def superop_from_basis_action(evolved_basis: np.ndarray) -> np.ndarray:
    """Assemble the superoperator from a channel's action on the 81 basis
    operators |a><b|. evolved_basis[k] is E(|a><b|) with k = a*9 + b (a 9x9
    matrix); its vec is column k of the superoperator."""
    return np.stack([evolved_basis[k].reshape(-1) for k in range(81)], axis=1)


def depolarizing_gate_superop(p: float, U4: np.ndarray | None = None,
                              leak: float = 0.0, leak_level: int = 8) -> np.ndarray:
    """Ideal gate U4 (default CZ) followed by a depolarizing channel on the
    computational subspace with parameter p:  D_p(rho) = p*rho + (1-p)*I4/4 * Tr.

    With leak == 0 the comp subspace is closed, so F_avg = p + (1-p)/4 exactly --
    the reference for the incoherent-limit test. With leak > 0 a fraction of the
    computational population is moved to a non-computational level each gate, a
    synthetic leak that lets the test demonstrate (and the leakage-aware fit
    correct) the bias leakage induces in naive RB. Used only by the tests."""
    if U4 is None:
        U4 = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)
    S_u = superop_from_unitary(_embed4_in_9(U4))
    evolved = np.zeros((81, 9, 9), dtype=complex)
    I4_over_d = np.zeros((9, 9), dtype=complex)
    for ia in COMP_IDX:
        I4_over_d[ia, ia] = 0.25
    for k in range(81):
        rho = np.zeros(81, dtype=complex); rho[k] = 1.0
        rho = (S_u @ rho).reshape(9, 9)                 # apply ideal gate first
        tr_comp = sum(rho[ia, ia] for ia in COMP_IDX)
        depol = p * rho + (1.0 - p) * tr_comp * I4_over_d
        evolved[k] = (1.0 - leak) * depol               # keep (1-leak) in comp...
        evolved[k][leak_level, leak_level] += leak * tr_comp   # ...leak the rest out
    return superop_from_basis_action(evolved)


# ---- Native-gate superoperators ------------------------------------------
def native_superops(cz_superop: np.ndarray) -> dict:
    """Superoperators for the native generators: ideal single-qubit Cliffords
    (embedded, leakage-free) and the supplied (noisy) CZ."""
    sup = {}
    for name in ("H1", "H2", "S1", "S2"):
        sup[name] = superop_from_unitary(_embed4_in_9(_GENERATORS[name]))
    sup["CZ"] = cz_superop
    return sup


def _clifford_superop(idx, group, native, cache):
    """Noisy superoperator of Clifford `idx` = product of its native-gate
    superoperators (later gate on the left). Cached per index."""
    s = cache.get(idx)
    if s is not None:
        return s
    s = np.eye(81, dtype=complex)
    for gname in group.words[idx]:
        s = native[gname] @ s
    cache[idx] = s
    return s


# ---- gate superoperator from the simulator -------------------------------
def gate_superoperator(opt, u_stack, dt: float = 1.0) -> np.ndarray:
    """81x81 superoperator of the simulated noisy gate for pulse `u_stack`.

    Evolves the 81 basis operators |a><b| through the optimizer's open-system
    simulator (one batched call) and assembles the superoperator. `u_stack` is
    the [1, n_slices, n_channels] control the optimizer consumes (raw param under
    its activation), matching simulate_gradient_batch.
    """
    import torch

    from gradpulse.parametric import DEVICE

    # Clifford superoperators here are built in the fixed 9-D (two-qutrit) space,
    # so this RB path requires the optimizer's default n_levels=3.
    if getattr(opt, "_dim", 9) != 9:
        raise NotImplementedError(
            "gate_superoperator (simulated RB) is implemented for n_levels=3 "
            f"(9-D) only; got n_levels={getattr(opt, 'n_levels', '?')}. Run RB "
            "at n_levels=3 and use n_levels>=4 only for truncation-convergence "
            "checks of F_proc/leakage.")
    basis = torch.zeros((1, 81, 9, 9), dtype=opt.cdtype, device=DEVICE)
    for a in range(9):
        for b in range(9):
            basis[0, a * 9 + b, a, b] = 1.0
    u = torch.as_tensor(u_stack, dtype=opt.rdtype, device=DEVICE)
    if u.dim() == 2:
        u = u.unsqueeze(0)
    with torch.no_grad():
        evolved = opt.simulate_gradient_batch(u, dt=dt, rho0=basis)[0]   # [81,9,9]
    return superop_from_basis_action(evolved.detach().cpu().numpy())


# ---- RB protocol ----------------------------------------------------------
def _rho0_vec():
    rho = np.zeros((9, 9), dtype=complex)
    rho[0, 0] = 1.0                                     # |00><00|
    return rho.reshape(-1)


def _survival_and_comp(vec):
    rho = vec.reshape(9, 9)
    survival = float(np.real(rho[0, 0]))               # <00|rho|00>
    comp = float(np.real(sum(rho[ia, ia] for ia in COMP_IDX)))
    return survival, comp


def _run_rb(group, native, lengths, n_sequences, rng, interleave=None):
    """Average survival and comp-subspace population vs sequence length.

    Each sequence: m uniformly-random Cliffords (optionally with `interleave`
    -- the noisy gate superoperator -- after each), then the recovery Clifford
    that inverts the IDEAL total (so a noiseless run returns to |00>). Returns
    arrays survival[len(lengths)], comp[len(lengths)].
    """
    n_cliff = len(group)
    cz_ideal = _embed4_in_9(_GENERATORS["CZ"]) if interleave is not None else None
    surv = np.zeros(len(lengths))
    comp = np.zeros(len(lengths))
    rho0 = _rho0_vec()
    cache: dict = {}
    for li, m in enumerate(lengths):
        s_acc = c_acc = 0.0
        for _ in range(n_sequences):
            vec = rho0.copy()
            ideal_total = np.eye(9, dtype=complex)     # ideal 9-D unitary so far
            for _ in range(m):
                idx = int(rng.integers(n_cliff))
                vec = _clifford_superop(idx, group, native, cache) @ vec
                ideal_total = _embed4_in_9(group.unitaries[idx]) @ ideal_total
                if interleave is not None:
                    vec = interleave @ vec
                    ideal_total = cz_ideal @ ideal_total
            # Recovery = inverse of the ideal total (a Clifford); apply its noisy
            # superoperator so a perfect run lands back on |00>.
            U_rec = ideal_total.conj().T
            rec_idx = group.index_of(U_rec[np.ix_(COMP_IDX, COMP_IDX)])
            vec = _clifford_superop(rec_idx, group, native, cache) @ vec
            s, c = _survival_and_comp(vec)
            s_acc += s
            c_acc += c
        surv[li] = s_acc / n_sequences
        comp[li] = c_acc / n_sequences
    return surv, comp


# ---- Decay fits -----------------------------------------------------------
def _fit_single_exp(m, y, alphas=None):
    """Least-squares fit y ~ A*alpha^m + B over a grid of alpha (linear in A,B
    for each alpha). Returns (alpha, A, B). Numpy-only (no scipy dependency)."""
    m = np.asarray(m, float)
    y = np.asarray(y, float)
    if alphas is None:
        alphas = np.linspace(0.50, 0.99999, 4000)
    best = None
    for a in alphas:
        X = np.stack([a ** m, np.ones_like(m)], axis=1)
        coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        err = float(np.sum((X @ coef - y) ** 2))
        if best is None or err < best[0]:
            best = (err, a, coef[0], coef[1])
    return best[1], best[2], best[3]


def _fit_leakage(m, comp):
    """Fit comp-subspace population comp ~ p_inf + (1-p_inf)*gamma^m. Returns
    (gamma, p_inf, L1, L2): leakage decay gamma=1-L1-L2, steady-state comp pop
    p_inf, leakage rate L1 and seepage L2 (Wood-Gambetta linearisation)."""
    gamma, A, B = _fit_single_exp(m, comp)
    p_inf = float(np.clip(B, 0.0, 1.0))
    L1 = (1.0 - gamma) * (1.0 - p_inf)                 # comp -> leakage per Clifford
    L2 = (1.0 - gamma) * p_inf                         # leakage -> comp (seepage)
    return gamma, p_inf, float(L1), float(L2)


def _r_from_alpha(alpha, d=4):
    return (d - 1.0) / d * (1.0 - alpha)


def interleaved_rb(cz_superop, lengths=(1, 2, 4, 8, 16, 24, 32),
                   n_sequences=40, seed=0, f_avg_analytic=None):
    """Simulated leakage-aware interleaved RB of the gate `cz_superop`.

    cz_superop : 81x81 noisy CZ superoperator (from gate_superoperator, or
                 depolarizing_gate_superop for the validation test).
    f_avg_analytic : optional analytic average gate fidelity (#1) to compare
                 against; the reported `bridge_gap` is r_irb_leakage_aware -
                 (1 - f_avg_analytic).

    Returns a dict of the reference/interleaved decays (naive single-exponential
    and leakage-aware), the extracted CZ error per gate both ways, the leakage
    rate, and -- if given -- the gap to the analytic estimand.
    """
    group = two_qubit_cliffords()
    native = native_superops(cz_superop)
    lengths = list(lengths)

    rng = np.random.default_rng(seed)
    ref_s, ref_c = _run_rb(group, native, lengths, n_sequences, rng)
    rng = np.random.default_rng(seed + 1)
    int_s, int_c = _run_rb(group, native, lengths, n_sequences, rng,
                           interleave=cz_superop)

    # Naive single-exponential RB (ignores leakage). The interleaved-RB gate
    # error is r = (d-1)/d * (1 - alpha_int/alpha_ref), d = 4.
    a_ref, _, _ = _fit_single_exp(lengths, ref_s)
    a_int, _, _ = _fit_single_exp(lengths, int_s)
    r_cz_naive = (3.0 / 4.0) * (1.0 - a_int / a_ref)

    # Leakage rate from the comp-subspace population decay.
    gamma_ref, p_inf, L1, L2 = _fit_leakage(lengths, ref_c)
    # Leakage-aware: condition the survival on remaining in the computational
    # subspace (S/L divides out the leakage decay), then fit the pure
    # depolarizing decay. Reduces to the naive fit when there is no leakage
    # (L == 1), and is well-conditioned where the 3-component fit is degenerate.
    ref_cond = np.asarray(ref_s) / np.clip(ref_c, 1e-9, None)
    int_cond = np.asarray(int_s) / np.clip(int_c, 1e-9, None)
    lam_ref, _, _ = _fit_single_exp(lengths, ref_cond)
    lam_int, _, _ = _fit_single_exp(lengths, int_cond)
    r_cz_leakage_aware = (3.0 / 4.0) * (1.0 - lam_int / lam_ref)

    out = {
        "lengths": lengths,
        "ref_survival": ref_s.tolist(), "int_survival": int_s.tolist(),
        "ref_comp_pop": ref_c.tolist(), "int_comp_pop": int_c.tolist(),
        "alpha_ref": a_ref, "alpha_int": a_int,
        "lambda_ref": lam_ref, "lambda_int": lam_int,
        "r_cz_naive": float(r_cz_naive),
        "r_cz_leakage_aware": float(r_cz_leakage_aware),
        "leakage_per_clifford_L1": L1, "seepage_per_clifford_L2": L2,
        "f_cz_irb": float(1.0 - r_cz_leakage_aware),
    }
    if f_avg_analytic is not None:
        out["r_analytic"] = float(1.0 - f_avg_analytic)
        out["bridge_gap"] = float(r_cz_leakage_aware - (1.0 - f_avg_analytic))
        out["naive_minus_aware"] = float(r_cz_naive - r_cz_leakage_aware)
    return out
