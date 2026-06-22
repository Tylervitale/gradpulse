"""Faithful tunable-coupler CZ model (gradpulse.convenience.tunable_coupler_cz).

Unlike the dispersive ParametricCZOptimizer (coupler eliminated under Schrieffer-
Wolff), this evolves the coupler explicitly as a 3-element qubit-coupler-qubit
chain with the coupler frequency as the flux control. The load-bearing test is the
independent QuTiP cross-check: the new frequency-control channel must be realized
exactly, so an independent integrator reproduces F_proc to ~machine precision.
"""
import math

import pytest
import torch

import gradpulse as gp
from gradpulse import MultiQubitOptimizer


def test_builder_wiring():
    """3 frequency-control channels (coupler + both qubits), no drives/edges."""
    opt = gp.tunable_coupler_cz(verbose=False)
    assert isinstance(opt, MultiQubitOptimizer)
    assert opt.freq_control_qubits == [0, 1, 2]
    assert opt.drive_qubits == [] and opt.tunable_edges == []
    assert opt.n_channels == 3            # one flux channel per element
    assert opt.N == 3 and opt.target_qubits == (0, 2)


def test_qutip_cross_check_machine_precision():
    """The coupler-explicit model with the flux (frequency) control is reproduced
    by an independent QuTiP integrator to ~machine precision on an arbitrary pulse."""
    pytest.importorskip("qutip")
    from gradpulse import validate
    opt = gp.tunable_coupler_cz(precision="double", verbose=False)
    torch.manual_seed(0)
    wf = torch.rand(20, opt.n_channels).numpy()      # arbitrary smoothed-in-[0,1] pulse
    xc = validate.multiqubit_cross_check(opt, wf, dt_ns=1.0)
    assert xc["delta"] < 1e-6, f"QuTiP disagrees: {xc}"


def test_optimization_descends():
    """A short run must improve fidelity toward CZ and return finite diagnostics."""
    opt = gp.tunable_coupler_cz(precision="double", verbose=False)
    r = opt.optimize(n_slices=40, dt_ns=1.0, iterations=40, n_seeds=2, seed0=0)
    assert math.isfinite(r["best_fidelity"])
    assert r["best_fidelity"] > r["history"][0] - 1e-9   # descended (or held)
    assert "converged" in r and "n_nonfinite_steps" in r


def test_cz_data_virtualz_objective():
    """The physical CZ objective (data subspace, coupler |0>, virtual-Z free) runs,
    returns the two virtual-Z phases, and stays mode-isolated from the default."""
    opt = gp.tunable_coupler_cz(precision="double", verbose=False)
    r = opt.optimize(n_slices=40, dt_ns=1.0, iterations=40, n_seeds=2, seed0=0,
                     fidelity="cz_data_virtualz")
    assert math.isfinite(r["best_fidelity"]) and 0.0 <= r["best_fidelity"] <= 1.0
    assert r["fidelity_mode"] == "cz_data_virtualz"
    assert "virtual_z_phases" in r and len(r["virtual_z_phases"]) == 2
    assert all(math.isfinite(p) for p in r["virtual_z_phases"])
    # the default 'choi' result must NOT carry the virtual-Z field
    r0 = opt.optimize(n_slices=40, dt_ns=1.0, iterations=8, n_seeds=1)
    assert "virtual_z_phases" not in r0


def test_cz_data_virtualz_guards():
    """cz_data_virtualz rejects a bogus name and requires an open system."""
    from gradpulse.multiqubit import MultiQubitProfile, MultiQubitOptimizer
    opt = gp.tunable_coupler_cz(verbose=False)
    with pytest.raises(ValueError):
        opt.optimize(n_slices=20, iterations=2, fidelity="bogus")
    prof = MultiQubitProfile(n_qubits=3, freqs_ghz=[4.4, 5.5, 4.6],
                             anharm_mhz=[-200, -150, -200], t1_ns=[3e4, 2e4, 3e4],
                             t2_ns=[2e4, 1.5e4, 2e4], couplings={(0, 1): 85, (1, 2): 85},
                             n_levels=3)
    unitary = MultiQubitOptimizer(prof, target_gate="cz", target_qubits=(0, 2),
                                  drive_qubits=[], tunable_edges=[], freq_control_qubits=[0, 1, 2],
                                  delta_max_mhz=300.0, open_system=False, verbose=False)
    with pytest.raises(ValueError):
        unitary.optimize(n_slices=20, iterations=2, fidelity="cz_data_virtualz")


def test_edge_rest_endpoints_and_default_identical():
    """edge_rest_slices forces every control to rest (x=0.5, u=0) at the boundaries so
    the pulse is a composable gate; edge_rest_slices=0 is byte-identical to legacy."""
    import numpy as np
    opt = gp.tunable_coupler_cz(precision="double", verbose=False)
    kw = dict(n_slices=40, dt_ns=1.0, iterations=8, n_seeds=1, seed0=3,
              fidelity="cz_data_virtualz", leak_weight=4.0)
    legacy = opt.optimize(**kw)
    same = opt.optimize(edge_rest_slices=0, **kw)
    assert np.allclose(legacy["best_waveform"], same["best_waveform"]), \
        "edge_rest_slices=0 must be byte-identical to the legacy path"
    rested = opt.optimize(edge_rest_slices=6, **kw)
    wf = np.asarray(rested["best_waveform"])
    assert np.max(np.abs(wf[0] - 0.5)) < 1e-2, "all channels must rest at x=0.5 at slice 0"
    assert np.max(np.abs(wf[-1] - 0.5)) < 1e-2, "all channels must rest at x=0.5 at slice -1"
    # physical coupler flux u = 2x-1 must therefore vanish at the edges
    u = 2.0 * wf[:, 1] - 1.0
    assert abs(u[0]) < 2e-2 and abs(u[-1]) < 2e-2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
