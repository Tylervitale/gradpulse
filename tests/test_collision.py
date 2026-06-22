"""Resonant / frequency-collision regime (evolving near-resonant spectator).

Covers ParametricCZOptimizer.resonant_collision_fidelity and
CrossResonanceZXOptimizer.resonant_collision_fidelity -- the complement of the
frozen-spectator ``spectator_fidelity`` (static ZZ). Here the spectator is a third
transmon coupled by a TRANSVERSE exchange whose frequency can approach a gate
qubit's, so population coherently swaps into it during the gate. The checks:

  * J=0 (decoupled spectator) reproduces the bare gate -- validates the
    (n_levels**3)-D lift, partial trace and comp-index mapping.
  * the collision CRATERS the gate at resonance and RECOVERS the bare gate far
    off-resonant; spectator population-swap tracks the same way.
  * the PyTorch diagnostic agrees with an independent QuTiP simulation
    (validate.collision_cross_check / cr_collision_cross_check), the package's
    different-library / different-construction cross-check standard.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradpulse import (ParametricCouplerProfile, ParametricCZOptimizer,
                       CrossResonanceProfile, CrossResonanceZXOptimizer)
from gradpulse.parametric import DEVICE

FX = Path(__file__).parent / "fixtures"


# ============================ Parametric CZ ================================
@pytest.fixture(scope="module")
def cz_eval():
    meta = json.loads((FX / "reference_cz_pulse.json").read_text())
    env = np.load(FX / meta["pulse_npy"])
    valid = {f.name for f in dataclasses.fields(ParametricCouplerProfile)}
    prof = ParametricCouplerProfile(
        **{k: v for k, v in meta["profile"].items() if k in valid})
    opt = ParametricCZOptimizer(prof, bandwidth_mhz=0.0, activation="clamp",
                                precision="double")
    u = torch.as_tensor(env, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    return opt, u, env, prof


def test_cz_collision_j0_reproduces_bare_gate(cz_eval):
    """J=0 decouples the spectator -> exact bare-gate F_proc (validates the lift)."""
    opt, u, _, _ = cz_eval
    r = opt.resonant_collision_fidelity(u, dt=1.0, detuning_mhz=0.0, j_mhz=0.0)
    assert abs(r["f_proc"] - r["f_proc_isolated"]) < 1e-9
    assert r["spectator_leakage"] < 1e-9


def test_cz_collision_craters_and_recovers(cz_eval):
    """Fidelity craters at resonance, recovers the bare gate far off-resonant."""
    opt, u, _, _ = cz_eval
    sw = opt.resonant_collision_fidelity(
        u, dt=1.0, detuning_mhz=[2000.0, 200.0, 0.0], j_mhz=8.0, couples_to=2)
    f_far, f_mid, f_res = sw["f_proc"]
    lk_far, lk_mid, lk_res = sw["spectator_leakage"]
    # Monotone collapse toward resonance.
    assert f_far > f_mid > f_res
    # Resonance is a real collision: large fidelity loss + large population swap.
    assert sw["delta_r_collision"][2] > 0.1
    assert lk_res > 0.1
    # Far off-resonant recovers the bare gate (negligible added infidelity / swap).
    assert abs(sw["f_proc"][0] - sw["f_proc_isolated"]) < 2e-3
    assert lk_far < 1e-3
    assert lk_far < lk_mid < lk_res


def test_cz_collision_matches_qutip(cz_eval):
    """PyTorch diagnostic == independent QuTiP collision sim (machine precision)."""
    pytest.importorskip("qutip")
    from gradpulse import validate
    opt, u, env, prof = cz_eval
    pdict = {k: getattr(prof, k) for k in (
        "freq_ghz_q1", "freq_ghz_q2", "anharm_ghz_q1", "anharm_ghz_q2",
        "t1_ns_q1", "t1_ns_q2", "t2_ns_q1", "t2_ns_q2", "g_max_mhz",
        "omega_max_mhz", "chi_zz_mhz", "n_levels")}
    for d in (200.0, 50.0, 0.0):
        t = opt.resonant_collision_fidelity(u, dt=1.0, detuning_mhz=d,
                                            j_mhz=8.0, couples_to=2)
        q = validate.collision_cross_check(pdict, env, "cz", 1.0,
                                           detuning_mhz=d, j_mhz=8.0, couples_to=2)
        assert abs(t["f_proc"] - q["f_proc"]) < 1e-6
        assert abs(t["spectator_leakage"] - q["spectator_leakage"]) < 1e-6


def test_cz_collision_scalar_vs_sweep_api(cz_eval):
    """Scalar detuning returns floats; a list returns per-detuning lists."""
    opt, u, _, _ = cz_eval
    one = opt.resonant_collision_fidelity(u, dt=1.0, detuning_mhz=100.0, j_mhz=8.0)
    assert isinstance(one["f_proc"], float)
    many = opt.resonant_collision_fidelity(u, dt=1.0, detuning_mhz=[100.0, 100.0],
                                           j_mhz=8.0)
    assert isinstance(many["f_proc"], list) and len(many["f_proc"]) == 2
    # Same detuning twice -> identical, and equal to the scalar call.
    assert abs(many["f_proc"][0] - one["f_proc"]) < 1e-12


# ========================= Cross-resonance ZX ==============================
@pytest.fixture(scope="module")
def cr_eval():
    prof = CrossResonanceProfile()                       # default n_levels=4
    opt = CrossResonanceZXOptimizer(prof, precision="double")
    res = opt.optimize(n_slices=40, dt_ns=1.0, iterations=40)
    return opt, res["best_raw_param"], res["virtual_z"], res["best_waveform"], prof


def test_cr_collision_j0_reproduces_bare_gate(cr_eval):
    opt, x, vz, _, _ = cr_eval
    r = opt.resonant_collision_fidelity(x, dt=1.0, vz=vz, detuning_mhz=0.0, j_mhz=0.0)
    assert abs(r["f_proc"] - r["f_proc_isolated"]) < 1e-8
    assert r["spectator_leakage"] < 1e-8


def test_cr_collision_craters_and_recovers(cr_eval):
    opt, x, vz, _, _ = cr_eval
    sw = opt.resonant_collision_fidelity(
        x, dt=1.0, vz=vz, detuning_mhz=[2000.0, 0.0], j_mhz=8.0, couples_to="control")
    # Far off-resonant recovers the bare gate; resonance swaps real population.
    assert abs(sw["f_proc"][0] - sw["f_proc_isolated"]) < 2e-3
    assert sw["spectator_leakage"][0] < 1e-3
    assert sw["spectator_leakage"][1] > 0.05
    assert sw["f_proc"][1] < sw["f_proc"][0]


def test_cr_collision_matches_qutip(cr_eval):
    """PyTorch CR diagnostic == independent QuTiP collision sim (64-D, double)."""
    pytest.importorskip("qutip")
    from gradpulse import validate
    opt, x, vz, xw, prof = cr_eval
    pdict = dataclasses.asdict(prof)
    for d in (100.0, 0.0):
        t = opt.resonant_collision_fidelity(x, dt=1.0, vz=vz, detuning_mhz=d,
                                            j_mhz=8.0, couples_to="control")
        q = validate.cr_collision_cross_check(pdict, xw, vz, 1.0, detuning_mhz=d,
                                              j_mhz=8.0, couples_to="control")
        assert abs(t["f_proc"] - q["f_proc"]) < 1e-6
        assert abs(t["spectator_leakage"] - q["spectator_leakage"]) < 1e-6


def test_cr_collision_target_spectator(cr_eval):
    """A spectator on the target also collides (different base frequency)."""
    opt, x, vz, _, _ = cr_eval
    res = opt.resonant_collision_fidelity(x, dt=1.0, vz=vz, detuning_mhz=0.0,
                                          j_mhz=8.0, couples_to="target")
    assert res["couples_to"] == "target"
    assert res["spectator_leakage"] > 0.02
