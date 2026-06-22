"""Simulated leakage-aware interleaved RB (gradpulse.rb).

The estimator must (a) be built on the correct 2-qubit Clifford group, (b)
recover the analytic average gate fidelity in the incoherent, leakage-free limit
-- this is the bridge that makes the simulated number and a hardware RB number
the same estimator -- and (c) expose the bias leakage induces in naive RB while
the leakage-aware fit corrects it.

These tests run on exact, analytically-known channels (depolarizing +/- a
synthetic leak) so they are pure-numpy, fast, and deterministic-by-seed -- no
simulator needed. The end-to-end bridge on the *simulated* CZ (IRB recovers the
analytic F_avg to ~1e-5) is in examples/randomized_benchmarking.py.

Run:  pytest tests/        OR        python tests/test_rb.py
"""
import numpy as np
import pytest

from gradpulse.rb import (
    COMP_IDX,
    depolarizing_gate_superop,
    interleaved_rb,
    superop_from_unitary,
    two_qubit_cliffords,
)


def test_clifford_group_order_and_unitarity():
    g = two_qubit_cliffords()
    assert len(g) == 11520, f"2-qubit Clifford group order is 11520, got {len(g)}"
    # Spot-check a random sample is unitary and round-trips through index_of.
    rng = np.random.default_rng(0)
    for _ in range(50):
        i = int(rng.integers(len(g)))
        U = g.unitaries[i]
        assert np.allclose(U.conj().T @ U, np.eye(4), atol=1e-6)
        assert g.index_of(U) == i                       # canon lookup is consistent


def test_superop_convention():
    # superop_from_unitary applied to vec(rho) reproduces U rho U^dag.
    rng = np.random.default_rng(1)
    A = rng.standard_normal((9, 9)) + 1j * rng.standard_normal((9, 9))
    U, _ = np.linalg.qr(A)
    rho = rng.standard_normal((9, 9)) + 1j * rng.standard_normal((9, 9))
    rho = rho @ rho.conj().T
    out = (superop_from_unitary(U) @ rho.reshape(-1)).reshape(9, 9)
    assert np.allclose(out, U @ rho @ U.conj().T, atol=1e-10)


def test_irb_recovers_depolarizing_fidelity():
    # Incoherent, leakage-free limit: interleaved-RB gate error must recover the
    # analytic 1 - F_avg of an exact depolarizing channel (F_avg = p + (1-p)/4).
    p = 0.95
    true_r = (1.0 - p) * 3.0 / 4.0                       # = 1 - F_avg
    res = interleaved_rb(depolarizing_gate_superop(p),
                         lengths=(1, 2, 4, 8, 12, 16, 24), n_sequences=80, seed=1)
    assert res["leakage_per_clifford_L1"] < 1e-6, "leakage-free channel shows leakage"
    assert res["r_cz_leakage_aware"] == pytest.approx(true_r, abs=5e-3), \
        f"IRB {res['r_cz_leakage_aware']:.4f} != analytic {true_r:.4f}"
    # With no leakage the leakage-aware and naive fits must agree.
    assert abs(res["r_cz_naive"] - res["r_cz_leakage_aware"]) < 2e-3


def test_leakage_biases_naive_rb():
    # A gate that depolarizes (true comp error 0.015) AND leaks 1%/gate: naive RB
    # over-reports the error (leakage inflates the decay), the leakage-aware fit
    # corrects toward the truth, and the measured leakage matches the injection
    # scaled by the ~1.88 CZ per Clifford.
    p, leak = 0.98, 0.01
    true_r = (1.0 - p) * 3.0 / 4.0                       # 0.015
    res = interleaved_rb(depolarizing_gate_superop(p, leak=leak),
                         lengths=(1, 2, 4, 8, 12, 16, 24), n_sequences=80, seed=5)
    assert res["r_cz_naive"] > res["r_cz_leakage_aware"], "naive not biased above aware"
    # The leakage-aware estimate is closer to the truth than the naive one.
    assert abs(res["r_cz_leakage_aware"] - true_r) < abs(res["r_cz_naive"] - true_r)
    # L1 is per-Clifford ~ (CZ per Clifford ~1.88) * per-gate leak.
    assert res["leakage_per_clifford_L1"] == pytest.approx(1.88 * leak, rel=0.3)


def test_noiseless_identity_is_flat():
    # The ideal CZ (p=1, no leakage) gives ~zero RB error: sequence + recovery
    # returns to |00>, so survival stays ~1 and the extracted error is ~0.
    res = interleaved_rb(depolarizing_gate_superop(1.0),
                         lengths=(1, 2, 4, 8), n_sequences=20, seed=0)
    assert abs(res["r_cz_leakage_aware"]) < 5e-3
    # Population never leaves the computational subspace.
    assert max(1.0 - c for c in res["ref_comp_pop"]) < 1e-9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
