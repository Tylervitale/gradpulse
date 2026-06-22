"""gradpulse.diagnostics - channel diagnostics (Pauli transfer matrix, unitarity).

Stateless, numpy-only helpers that turn a computational-subspace superoperator
into the standard coherent-vs-incoherent error diagnostics. They are imported and
re-exported by ``gradpulse.parametric`` (so ``from gradpulse.parametric import
channel_unitarity`` keeps working) and used by ``ParametricCZOptimizer.error_budget``.
"""
from __future__ import annotations

import numpy as np

_PAULI_1Q = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _two_qubit_paulis() -> np.ndarray:
    """The 16 two-qubit Paulis P_a (x) P_b as a [16, 4, 4] array, identity first.

    Index 0 is I(x)I (required by the unitarity definition, which removes the
    identity row/column). The remaining 15 are the traceless generators.
    """
    order = ("I", "X", "Y", "Z")
    mats = [np.kron(_PAULI_1Q[a], _PAULI_1Q[b]) for a in order for b in order]
    return np.stack(mats, axis=0)


def pauli_transfer_matrix(comp_superop: np.ndarray) -> np.ndarray:
    """Pauli transfer matrix (PTM) of a 2-qubit (d=4) computational-subspace map.

    comp_superop : [16, 16] superoperator on vec(rho) (row-major reshape) of a
        4x4 operator; column m = vec(Phi(E_m)) for the matrix-unit basis
        E_m = |i><j|, m = i*4 + j. Returns the 16x16 real PTM R with
        R_{kl} = (1/d) Tr[P_k Phi(P_l)], P_0 = I(x)I -- the same vec convention
        on both sides, so this is an exact basis change.
    """
    P = _two_qubit_paulis().reshape(16, 16)          # rows = vec(P_k), row-major
    W = P.conj().T / 2.0                              # columns vec(P_l)/sqrt(d)
    R = W.conj().T @ np.asarray(comp_superop) @ W
    return R


def channel_unitarity(comp_superop: np.ndarray) -> float:
    """Unitarity u of a 2-qubit channel (Wallman et al., NJP 17, 113020 (2015)).

    u = ||R'||_F^2 / (d^2 - 1) with R' the PTM minus its identity row and column
    (the unital/traceless block), d = 4. u = 1 for a unitary channel; for a
    depolarizing channel with parameter p, u = p^2; lower u <=> more incoherent
    (stochastic) noise. Together with the average-gate infidelity r it separates
    coherent from incoherent error: r >= (d-1)/d (1 - sqrt(u)), with equality iff
    the noise is purely stochastic, so the gap is the coherent contribution.
    """
    R = pauli_transfer_matrix(comp_superop)
    Rp = R[1:, 1:]
    return float(np.sum(np.abs(Rp) ** 2) / (R.shape[0] - 1))
