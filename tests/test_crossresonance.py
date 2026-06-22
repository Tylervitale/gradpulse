"""Cross-resonance ZX(pi/2) architecture: target, convergence, error budget,
DRAG, and an independent QuTiP cross-check.

Optimizations here are deliberately short (one seed, modest iterations) -- enough
to exercise the machinery and the cross-check, not to reproduce a headline number.
"""
import json
from dataclasses import asdict

import numpy as np
import pytest

from gradpulse.crossresonance import (
    CrossResonanceProfile, CrossResonanceZXOptimizer, zx90_target,
)


@pytest.fixture(scope="module")
def cr_opt():
    return CrossResonanceZXOptimizer(CrossResonanceProfile(),
                                     use_drag=True, use_target_cancel=True)


@pytest.fixture(scope="module")
def cr_result(cr_opt):
    # Short by design: enough to exercise the machinery + cross-check, not a headline.
    return cr_opt.optimize(n_slices=120, dt_ns=1.0, iterations=130,
                           n_seeds=1, lr=0.06, seed0=0)


def test_zx90_target_unitary_and_entangling():
    U = zx90_target()
    assert np.abs(U.conj().T @ U - np.eye(4)).max() < 1e-12
    # Conditional rotation: control-0 block != control-1 block (entangling).
    assert not np.allclose(U[:2, :2], U[2:, 2:])


def test_optimizer_converges(cr_result):
    assert cr_result["best_fidelity"] > 0.95
    assert cr_result["best_leakage"] < 1e-2
    assert cr_result["best_waveform"].shape[1] == 2          # control I + target I
    assert np.abs(cr_result["best_waveform"]).max() <= 1.0 + 1e-6


def test_error_budget_consistency(cr_opt, cr_result):
    eb = cr_opt.error_budget(cr_result["best_raw_param"], dt=1.0,
                             vz=cr_result["virtual_z"])
    assert eb["f_proc"] == pytest.approx(cr_result["best_fidelity"], abs=2e-3)
    assert 0.0 <= eb["unitarity"] <= 1.0
    assert eb["r_decoherence"] >= -1e-9
    assert eb["r_coherent"] + eb["r_decoherence"] == pytest.approx(eb["r_total"], abs=1e-6)


def test_drag_quadrature_is_active(cr_result):
    # The derived DRAG quadrature must actually enter the dynamics: simulating the
    # same physical pulse with DRAG on vs off gives measurably different leakage.
    # (The QuTiP cross-check validates that the DRAG-on dynamics are *correct*; the
    # example reports the on-vs-off leakage, which is regime-dependent.)
    import torch
    from gradpulse.crossresonance import DEVICE
    prof = CrossResonanceProfile()
    on = CrossResonanceZXOptimizer(prof, use_drag=True, use_target_cancel=True)
    off = CrossResonanceZXOptimizer(prof, use_drag=False, use_target_cancel=True)
    raw = torch.as_tensor(cr_result["best_raw_param"], dtype=on.rdtype,
                          device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        leak_on = float(on._leakage(on.simulate_choi_batch(raw, dt=1.0)))
        leak_off = float(off._leakage(off.simulate_choi_batch(raw, dt=1.0)))
    assert abs(leak_on - leak_off) > 1e-9, "use_drag has no effect on the dynamics"


def test_qutip_cross_check(tmp_path, cr_opt, cr_result):
    pytest.importorskip("qutip")
    from gradpulse.validate import cross_check
    np.save(tmp_path / "zx.npy", cr_result["best_waveform"])
    meta = {
        "architecture": "cross_resonance", "pulse_npy": "zx.npy", "pulse_dt_ns": 1.0,
        "n_channels": int(cr_result["best_waveform"].shape[1]),
        "bandwidth_mhz": cr_opt.bandwidth_mhz, "use_drag": cr_opt.use_drag,
        "use_target_cancel": cr_opt.use_target_cancel,
        "virtual_z": cr_result["virtual_z"],
        "grape_f": float(cr_result["best_fidelity"]),
        "profile": asdict(cr_opt.profile),
    }
    (tmp_path / "zx.json").write_text(json.dumps(meta))
    res = cross_check(tmp_path / "zx.json")
    assert res["architecture"] == "cross_resonance"
    assert abs(res["delta"]) < 1e-3, f"QuTiP cross-check delta too large: {res['delta']}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
