"""Explicit lossy two-level-system (TLS) defect diagnostic.

Covers ParametricCZOptimizer.tls_defect_fidelity -- the gate pair next to ONE explicit
two-level defect that exchanges a real excitation with a gate qubit (vacuum-Rabi swap)
AND carries its own T1 relaxation. This is the coherent-quantum-bath physics that the
classical noise models (quasi_static / colored_noise_fidelity / white Markovian T_phi)
structurally cannot represent, and the lossy cousin of resonant_collision_fidelity.

  * g=0 (decoupled defect) reproduces the bare gate -- validates the 18-D lift, the
    partial trace over the TLS, and the comp-index mapping.
  * a RESONANT lossy TLS craters the gate AND excites the defect; the effect peaks on
    resonance and falls off either side.
  * the PyTorch diagnostic agrees with an independent QuTiP simulation
    (validate.tls_defect_cross_check) to machine precision -- the package's
    different-library cross-check standard.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer

FX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def cz_eval():
    meta = json.loads((FX / "reference_cz_pulse.json").read_text())
    env = np.load(FX / meta["pulse_npy"])
    valid = {f.name for f in dataclasses.fields(ParametricCouplerProfile)}
    prof = ParametricCouplerProfile(
        **{k: v for k, v in meta["profile"].items() if k in valid})
    opt = ParametricCZOptimizer(prof, bandwidth_mhz=0.0, activation="clamp",
                                precision="double")
    return opt, env, prof


def test_g0_reproduces_bare_gate(cz_eval):
    opt, env, _ = cz_eval
    r = opt.tls_defect_fidelity(env, dt=1.0, g_mhz=0.0, detuning_mhz=0.0, t1_tls_ns=500.0)
    assert abs(r["f_proc"] - r["f_proc_isolated"]) < 1e-9
    assert r["tls_excitation"] < 1e-9


def test_resonant_lossy_tls_degrades_and_excites(cz_eval):
    opt, env, _ = cz_eval
    r = opt.tls_defect_fidelity(env, dt=1.0, g_mhz=2.0, t1_tls_ns=400.0, detuning_mhz=0.0)
    assert r["delta_r_tls"] > 1e-3          # a resonant defect noticeably degrades the gate
    assert r["tls_excitation"] > 1e-3       # a real excitation swaps into the defect
    assert r["f_avg"] < r["f_avg_isolated"]


def test_resonance_peak_in_detuning_sweep(cz_eval):
    opt, env, _ = cz_eval
    sw = opt.tls_defect_fidelity(env, dt=1.0, g_mhz=2.0, t1_tls_ns=400.0,
                                 detuning_mhz=[-200.0, 0.0, 200.0])
    dr = sw["delta_r_tls"]
    assert dr[1] > dr[0] and dr[1] > dr[2]      # added infidelity peaks ON resonance
    assert dr[0] < 5e-3 and dr[2] < 5e-3        # far off-resonant is small


def test_matches_qutip_cross_check(cz_eval):
    pytest.importorskip("qutip")
    from gradpulse import validate
    opt, env, prof = cz_eval
    pdict = {k: getattr(prof, k) for k in (
        "freq_ghz_q1", "freq_ghz_q2", "anharm_ghz_q1", "anharm_ghz_q2",
        "t1_ns_q1", "t1_ns_q2", "t2_ns_q1", "t2_ns_q2", "g_max_mhz",
        "omega_max_mhz", "chi_zz_mhz", "n_levels")}
    for det, g in [(0.0, 2.0), (60.0, 2.0)]:
        t = opt.tls_defect_fidelity(env, dt=1.0, g_mhz=g, t1_tls_ns=500.0,
                                    detuning_mhz=det, couples_to=1)
        q = validate.tls_defect_cross_check(pdict, env, "cz", 1.0, detuning_mhz=det,
                                            g_mhz=g, t1_tls_ns=500.0, couples_to=1)
        assert abs(t["f_proc"] - q["f_proc"]) < 1e-6
        assert abs(t["tls_excitation"] - q["tls_excitation"]) < 1e-6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
