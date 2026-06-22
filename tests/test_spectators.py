"""Spectator (always-on ZZ) crosstalk -- the dominant multi-qubit error this
single-pair model otherwise omits.

An off-resonant neighbour coupled by a static ZZ and frozen in |s> shifts a gate
qubit's frequency by zeta*s, so it is modelled exactly as an effective detuning.
These checks confirm the effective model on the committed CZ pulse and -- the
headline -- that it reproduces a genuine 3-transmon (27-D) QuTiP simulation.

Run:  pytest tests/test_spectators.py    OR    python tests/test_spectators.py
"""
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer
from gradpulse.parametric import DEVICE


def _reference_env_and_profile():
    fxdir = Path(__file__).parent / "fixtures"
    meta = json.loads((fxdir / "reference_cz_pulse.json").read_text())
    env = np.load(fxdir / meta["pulse_npy"])
    valid = {f.name for f in dataclasses.fields(ParametricCouplerProfile)}
    prof = ParametricCouplerProfile(
        **{k: v for k, v in meta["profile"].items() if k in valid})
    return env, prof


@pytest.fixture(scope="module")
def ref_eval():
    """Eval optimizer (no resmoothing) on the committed 150 ns CZ envelope."""
    env, prof = _reference_env_and_profile()
    opt = ParametricCZOptimizer(prof, bandwidth_mhz=0.0, activation="clamp",
                                precision="double")
    u = torch.as_tensor(env, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    nominal = float(opt._process_fidelity(opt.simulate_choi_batch(u, dt=1.0))[0])
    return opt, u, env, prof, nominal


def test_zeta_zero_is_noop(ref_eval):
    opt, u, _, _, nominal = ref_eval
    r = opt.spectator_fidelity(u, dt=1.0, zeta_mhz=0.0)
    assert r["n_evals"] == 1
    for k in ("f_proc_idle", "f_proc_excited", "f_proc_spectator_avg"):
        assert abs(r[k] - nominal) < 1e-9


def test_penalty_and_ordering(ref_eval):
    opt, u, _, _, nominal = ref_eval
    r = opt.spectator_fidelity(u, dt=1.0, zeta_mhz=0.3, spectator_pop=0.5)
    assert abs(r["f_proc_idle"] - nominal) < 1e-9
    # A neighbour degrades the gate; the unmeasured-neighbour average sits between
    # the idle and the (both-excited) raw worst case.
    assert r["f_proc_excited"] < r["f_proc_idle"]
    assert r["f_proc_excited"] <= r["f_proc_spectator_avg"] <= r["f_proc_idle"] + 1e-9
    # the neighbour adds a positive marginal infidelity (excludes the gate's own)
    assert r["delta_r_spectator"] > 0.0
    assert r["f_avg_idle"] > r["f_avg_spectator_avg"]
    assert r["n_evals"] == 4


def test_monotonic_in_zeta(ref_eval):
    opt, u, _, _, _ = ref_eval
    fs = [opt.spectator_fidelity(u, dt=1.0, zeta_mhz=z)["f_proc_excited"]
          for z in (0.05, 0.1, 0.2, 0.4)]
    assert all(fs[i] > fs[i + 1] for i in range(len(fs) - 1))


def test_single_neighbour_one_axis(ref_eval):
    opt, u, _, _, nominal = ref_eval
    # Neighbour only on q2: two channel evaluations, and it must degrade the gate.
    r = opt.spectator_fidelity(u, dt=1.0, zeta_mhz=(0.0, 0.3))
    assert r["n_evals"] == 2
    assert r["f_proc_excited"] < nominal


def test_spectator_reduction_qutip(ref_eval):
    """HEADLINE: a full 3-transmon (27-D) QuTiP simulation with an always-on ZZ to
    an idle |1> neighbour reproduces the 9-D effective detuning model exactly,
    confirming the reduction (and the PyTorch detuning path) is faithful."""
    pytest.importorskip("qutip")
    from gradpulse import validate
    opt, u, env, prof, _ = ref_eval
    pdict = dataclasses.asdict(prof)
    zeta_mhz = 0.3
    zeta_rad = 2.0 * math.pi * (zeta_mhz / 1000.0)
    for couples_to, det in ((1, (zeta_rad, 0.0)), (2, (0.0, zeta_rad))):
        f_eff = validate.qutip_f_proc(pdict, env, "cz", 1.0, detuning_offset=det)
        f_27 = validate.spectator_cross_check_3transmon(
            pdict, env, "cz", 1.0, zeta_mhz, couples_to=couples_to)
        assert abs(f_27 - f_eff) < 1e-6
        # PyTorch effective detuning also matches the independent QuTiP engine.
        pf = float(opt._process_fidelity(
            opt.simulate_choi_batch(u, dt=1.0, detuning_offset=det))[0])
        assert abs(pf - f_eff) < 1e-5


def test_cr_spectator_runs_and_penalizes():
    """Cross-resonance parity: the same spectator analysis on the CR architecture
    (which now shares the detuning_offset primitive)."""
    from gradpulse.crossresonance import (CrossResonanceProfile,
                                           CrossResonanceZXOptimizer)
    opt = CrossResonanceZXOptimizer(CrossResonanceProfile())
    res = opt.optimize(n_slices=80, dt_ns=1.0, iterations=60, n_seeds=1,
                       lr=0.06, seed0=0)
    x, vz = res["best_raw_param"], res["virtual_z"]
    # zeta=0 is an exact no-op (idle == excited, both undetuned).
    r0 = opt.spectator_fidelity(x, dt=1.0, vz=vz, zeta_mhz=0.0)
    assert abs(r0["f_proc_idle"] - r0["f_proc_excited"]) < 1e-9
    assert r0["f_proc_idle"] == pytest.approx(res["best_fidelity"], abs=2e-3)
    # A neighbour degrades it; averaged sits between idle and raw worst case.
    r = opt.spectator_fidelity(x, dt=1.0, vz=vz, zeta_mhz=0.3, spectator_pop=0.5)
    assert r["f_proc_excited"] < r["f_proc_idle"]
    assert r["f_proc_excited"] <= r["f_proc_spectator_avg"] <= r["f_proc_idle"] + 1e-9


def test_multi_spectator_additivity_qutip(ref_eval):
    """Multi-spectator: N frozen off-resonant neighbours on a gate qubit are exactly
    one detuning equal to the SUM of their ZZ rates. Validated to machine precision
    against an explicit 4-body (36-D) QuTiP sim with two spectators frozen in |1>."""
    pytest.importorskip("qutip")
    from gradpulse import validate
    opt, u, env, prof, _ = ref_eval
    pdict = dataclasses.asdict(prof)
    z1, z2 = 0.2, 0.3
    rad_sum = 2.0 * math.pi * ((z1 + z2) / 1000.0)
    # Explicit 4-body QuTiP (two |1> spectators on q1) == effective summed detuning.
    f_explicit = validate.spectator_cross_check_multi(
        pdict, env, "cz", 1.0, [(0, z1), (0, z2)])
    f_eff = validate.qutip_f_proc(pdict, env, "cz", 1.0, detuning_offset=(rad_sum, 0.0))
    assert abs(f_explicit - f_eff) < 1e-9
    # multi_spectator_fidelity with certain (pop=1) neighbours hits the summed detuning.
    eff = opt.multi_spectator_fidelity(u, [(0, z1, 1.0), (0, z2, 1.0)])
    assert abs(eff["f_proc_spectator_avg"] - f_eff) < 1e-6
    assert eff["n_neighbours"] == 2


def test_multi_spectator_reduces_to_single(ref_eval):
    """An unmeasured multi-neighbour ensemble degrades the gate and adds a positive
    marginal infidelity; with one neighbour it reproduces spectator_fidelity exactly."""
    opt, u, _, _, nominal = ref_eval
    r = opt.multi_spectator_fidelity(u, [(0, 0.2, 0.5), (1, 0.3, 0.5)])
    assert abs(r["f_proc_idle"] - nominal) < 1e-9
    assert r["f_proc_spectator_avg"] < r["f_proc_idle"]
    assert r["delta_r_spectator"] > 0.0
    one = opt.multi_spectator_fidelity(u, [(1, 0.3, 0.5)])          # neighbour on q2
    base = opt.spectator_fidelity(u, dt=1.0, zeta_mhz=(0.0, 0.3), spectator_pop=0.5)
    assert abs(one["f_proc_spectator_avg"] - base["f_proc_spectator_avg"]) < 1e-7


def test_cr_multi_spectator_parity():
    """Cross-resonance multi-spectator: parity with the parametric API and with the
    single-spectator CR method."""
    from gradpulse.crossresonance import (CrossResonanceProfile,
                                           CrossResonanceZXOptimizer)
    opt = CrossResonanceZXOptimizer(CrossResonanceProfile())
    res = opt.optimize(n_slices=80, dt_ns=1.0, iterations=60, n_seeds=1, lr=0.06, seed0=0)
    x, vz = res["best_raw_param"], res["virtual_z"]
    r = opt.multi_spectator_fidelity(x, [(0, 0.2, 0.5), (1, 0.3, 0.5)], vz=vz)
    assert r["f_proc_spectator_avg"] <= r["f_proc_idle"] + 1e-9
    assert r["n_neighbours"] == 2
    one = opt.multi_spectator_fidelity(x, [(1, 0.3, 0.5)], vz=vz)
    base = opt.spectator_fidelity(x, dt=1.0, vz=vz, zeta_mhz=(0.0, 0.3), spectator_pop=0.5)
    # single precision (complex64) -> ~1e-7 accumulation floor on the equality
    assert abs(one["f_proc_spectator_avg"] - base["f_proc_spectator_avg"]) < 1e-5


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
