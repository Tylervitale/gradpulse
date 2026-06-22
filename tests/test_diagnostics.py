"""Channel diagnostics: error-budget decomposition, unitarity, and the shared
detuning-offset simulate primitive (plus the robustness sweep and quasi-static
dephasing analyses built on it).

Deterministic, fast checks of the analysis tools layered on the simulator -- not
convergence claims. The headline fidelity stays owned by examples/optimize_cz.py
+ gradpulse.validate.

Run:  pytest tests/test_diagnostics.py    OR    python tests/test_diagnostics.py
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer
from gradpulse.parametric import DEVICE, channel_unitarity, pauli_transfer_matrix


def _profile():
    return ParametricCouplerProfile(
        freq_ghz_q1=4.85, freq_ghz_q2=5.05,
        anharm_ghz_q1=-0.20, anharm_ghz_q2=-0.20,
        t1_ns_q1=30_000, t2_ns_q1=25_000,
        t1_ns_q2=30_000, t2_ns_q2=25_000,
        g_max_mhz=12.0, omega_max_mhz=50.0,
    )


def _depol_superop(p):
    """Computational-subspace superoperator of a depolarizing channel
    D_p(rho) = p*rho + (1-p)*Tr(rho)*I/4; column m = vec(D_p(|i><j|))."""
    S = np.zeros((16, 16), dtype=complex)
    I4 = np.eye(4)
    for i in range(4):
        for j in range(4):
            E = np.zeros((4, 4), dtype=complex); E[i, j] = 1.0
            out = p * E + (1.0 - p) * np.trace(E) * I4 / 4.0
            S[:, i * 4 + j] = out.reshape(-1)
    return S


def _short_pulse_opt():
    opt = ParametricCZOptimizer(_profile(), activation="sigmoid")
    res = opt.optimize_multi_seed(
        n_seeds=1, iterations=15, n_slices=60, dt_ns=1.0,
        use_process_fidelity=True, lbfgs_polish=False)
    return opt, res["best_raw_param"]


@pytest.fixture(scope="module")
def short_opt():
    """One short optimization shared across the (stateless) analysis tests."""
    return _short_pulse_opt()


def _reference_env_and_profile():
    """The committed 150 ns CZ envelope + its device profile."""
    fxdir = Path(__file__).parent / "fixtures"
    meta = json.loads((fxdir / "reference_cz_pulse.json").read_text())
    env = np.load(fxdir / meta["pulse_npy"])
    valid = {f.name for f in dataclasses.fields(ParametricCouplerProfile)}
    prof = ParametricCouplerProfile(
        **{k: v for k, v in meta["profile"].items() if k in valid})
    return env, prof, meta


# ---- unitarity ----------------------------------------------------------
def test_unitarity_of_identity_is_one():
    assert abs(channel_unitarity(np.eye(16, dtype=complex)) - 1.0) < 1e-12


@pytest.mark.parametrize("p", [0.95, 0.8, 0.5, 0.3])
def test_unitarity_of_depolarizing_is_p_squared(p):
    # A depolarizing channel of parameter p has unitarity exactly p^2.
    assert abs(channel_unitarity(_depol_superop(p)) - p * p) < 1e-9


def test_ptm_of_identity_is_identity():
    R = pauli_transfer_matrix(np.eye(16, dtype=complex))
    assert np.allclose(R, np.eye(16), atol=1e-10)


# ---- detuning-offset primitive -----------------------------------------
def test_detuning_offset_default_is_byte_identical():
    opt = ParametricCZOptimizer(_profile())
    torch.manual_seed(0)
    u = torch.rand((1, 50, 3), device=DEVICE)
    r0 = opt.simulate_gradient_batch(u, dt=1.0)
    r0d = opt.simulate_gradient_batch(u, dt=1.0, detuning_offset=0.0)
    assert torch.equal(r0, r0d)


def test_detuning_offset_changes_dynamics_and_is_per_qubit():
    opt = ParametricCZOptimizer(_profile())
    torch.manual_seed(1)
    u = torch.rand((1, 50, 3), device=DEVICE)
    r0 = opt.simulate_gradient_batch(u, dt=1.0)
    d = 2 * np.pi * 0.005    # 5 MHz in rad/ns
    r_common = opt.simulate_gradient_batch(u, dt=1.0, detuning_offset=d)
    r_q1only = opt.simulate_gradient_batch(u, dt=1.0, detuning_offset=(d, 0.0))
    assert not torch.allclose(r0, r_common)
    assert not torch.allclose(r_common, r_q1only)


# ---- error budget -------------------------------------------------------
def test_error_budget_is_additive_and_physical(short_opt):
    opt, raw = short_opt
    eb = opt.error_budget(raw, dt=1.0)
    # additive by construction (ablation split)
    assert abs(eb["r_control_leakage"] + eb["r_decoherence"] - eb["r_total"]) < 1e-12
    # decoherence only hurts: closed-system fidelity >= open-system fidelity
    assert eb["F_proc_closed"] >= eb["F_proc"] - 1e-9
    assert eb["r_decoherence"] > -1e-6
    # unitarity is a sensible (0, 1] number; leakage a small population
    assert 0.0 < eb["unitarity"] <= 1.0 + 1e-9
    assert 0.0 <= eb["leakage"] < 1.0


def test_error_budget_matches_direct_process_fidelity(short_opt):
    opt, raw = short_opt
    eb = opt.error_budget(raw, dt=1.0)
    u = torch.as_tensor(raw, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    f = float(opt._process_fidelity(opt.simulate_choi_batch(u, dt=1.0))[0])
    assert abs(eb["F_proc"] - f) < 1e-9


# ---- robustness sweep ---------------------------------------------------
def test_robustness_sweep_zero_point_is_nominal(short_opt):
    opt, raw = short_opt
    nominal = opt.error_budget(raw, dt=1.0)["F_avg"]
    sw = opt.robustness_sweep(raw, dt=1.0, amp_fracs=[0.0], freq_mhz=[0.0])
    for axis in ("amplitude", "frequency"):
        assert abs(sw[axis]["F_avg"][0] - nominal) < 1e-9


def test_robustness_sweep_detuning_degrades(short_opt):
    opt, raw = short_opt
    nominal = opt.error_budget(raw, dt=1.0)["F_avg"]
    # 5 MHz over a 60 ns gate (~0.3 phase-wraps, within the 1/T period) is a
    # large coherent Z error and clearly degrades any tuned two-qubit gate.
    sw = opt.robustness_sweep(raw, dt=1.0, amp_fracs=[0.0], freq_mhz=[0.0, 5.0])
    assert sw["frequency"]["F_avg"][1] < nominal - 1e-3


# ---- quasi-static dephasing --------------------------------------------
def test_quasi_static_zero_sigma_is_nominal(short_opt):
    opt, raw = short_opt
    q = opt.quasi_static_fidelity(raw, dt=1.0, sigma_mhz=0.0, n_nodes=5)
    # sigma=0 reproduces the nominal channel exactly (it IS nominal): the only
    # residual is float round-off from summing the n_nodes**2=25 Gauss-Hermite
    # terms, so bound it by the dtype's machine epsilon -- 0.0 in double, ~3e-8
    # in the default float32 -- not a fixed double-precision constant.
    tol = 100.0 * torch.finfo(opt.rdtype).eps
    assert abs(q["F_proc"] - q["F_proc_nominal"]) < tol


def test_quasi_static_degrades_and_is_deterministic(short_opt):
    opt, raw = short_opt
    q1 = opt.quasi_static_fidelity(raw, dt=1.0, sigma_mhz=1.0, n_nodes=5)
    q2 = opt.quasi_static_fidelity(raw, dt=1.0, sigma_mhz=1.0, n_nodes=5)
    # quasi-static dephasing can only reduce the averaged-channel fidelity
    assert q1["F_proc"] <= q1["F_proc_nominal"] + 1e-9
    assert q1["F_proc"] < q1["F_proc_nominal"]      # 1 MHz spread is non-trivial
    # deterministic (Gauss-Hermite quadrature, no RNG)
    assert q1["F_proc"] == q2["F_proc"]


# ---- realism knobs: finite temperature + static ZZ ----------------------
def test_cold_bath_has_no_thermal_operators():
    opt = ParametricCZOptimizer(_profile())          # n_thermal defaults to 0
    assert opt._L_TH_Q1 is None and opt._L_TH_Q2 is None


def test_finite_temperature_lowers_fidelity_and_excites():
    # Use the real (high-fidelity) reference pulse: on a tuned gate a thermal
    # bath unambiguously lowers fidelity and adds excited-state population.
    env, base, _ = _reference_env_and_profile()
    hot = dataclasses.replace(base, n_thermal_q1=0.05, n_thermal_q2=0.05)
    cold_opt = ParametricCZOptimizer(base, bandwidth_mhz=0.0, activation="clamp",
                                     precision="double")
    hot_opt = ParametricCZOptimizer(hot, bandwidth_mhz=0.0, activation="clamp",
                                    precision="double")
    assert hot_opt._L_TH_Q1 is not None
    u = torch.as_tensor(env, dtype=cold_opt.rdtype, device=DEVICE).unsqueeze(0)
    rc = cold_opt.simulate_choi_batch(u, dt=1.0)
    rh = hot_opt.simulate_choi_batch(u, dt=1.0)
    assert float(hot_opt._process_fidelity(rh)[0]) < \
        float(cold_opt._process_fidelity(rc)[0])
    assert float(hot_opt._leakage(rh)[0]) > float(cold_opt._leakage(rc)[0])


def test_static_zz_changes_dynamics():
    base = _profile()
    zz = dataclasses.replace(base, chi_zz_mhz=0.5)
    o0, o1 = ParametricCZOptimizer(base), ParametricCZOptimizer(zz)
    torch.manual_seed(0)
    u = torch.rand((1, 60, 3), device=DEVICE)
    assert not torch.allclose(o0.simulate_choi_batch(u, dt=1.0),
                              o1.simulate_choi_batch(u, dt=1.0))


# ---- robust ensemble optimization --------------------------------------
def test_robust_axes_run_change_result_and_restore_params():
    prof = _profile()
    common = dict(n_seeds=1, iterations=12, n_slices=50, dt_ns=1.0,
                  use_process_fidelity=True, lbfgs_polish=False)

    def _seed():
        return torch.Generator(device=DEVICE).manual_seed(7)

    base = ParametricCZOptimizer(prof, activation="sigmoid")
    rob = ParametricCZOptimizer(prof, activation="sigmoid")
    r0 = base.optimize_multi_seed(rng=_seed(), **common)
    r1 = rob.optimize_multi_seed(robust_amp_jitter=0.1, robust_freq_jitter_mhz=0.5,
                                 rng=_seed(), **common)
    # same seed, but the robust (perturbation-averaged) objective moves the pulse
    assert not np.allclose(r0["best_raw_param"], r1["best_raw_param"])
    # device/control params are restored exactly after the robust averaging loop
    fresh = ParametricCZOptimizer(prof, activation="sigmoid")
    assert rob.OMEGA_MAX == fresh.OMEGA_MAX
    assert rob.G_MAX == fresh.G_MAX
    assert rob.STARK_MAX == fresh.STARK_MAX


def test_finite_temperature_zz_cross_check(tmp_path):
    """QuTiP independently reproduces F_proc with thermal + static ZZ enabled,
    confirming gradpulse.validate mirrors both knobs."""
    pytest.importorskip("qutip")
    from dataclasses import asdict
    from gradpulse import validate
    env, base, meta0 = _reference_env_and_profile()
    prof = dataclasses.replace(base, chi_zz_mhz=0.2,
                               n_thermal_q1=0.04, n_thermal_q2=0.02)
    opt = ParametricCZOptimizer(prof, bandwidth_mhz=0.0, activation="clamp",
                                precision="double")
    u = torch.as_tensor(env, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    f = float(opt._process_fidelity(opt.simulate_choi_batch(u, dt=1.0))[0])
    np.save(tmp_path / "p.npy", env)
    (tmp_path / "p.json").write_text(json.dumps(
        dict(meta0, pulse_npy="p.npy", grape_f=f, profile=asdict(prof))))
    res = validate.cross_check(tmp_path / "p.json")
    assert res["status"] == "PASS" and abs(res["delta"]) < 1e-3


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
