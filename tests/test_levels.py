"""Customizability: Fock-level truncation (n_levels), custom target unitary, and
the counter-rotating (beyond-RWA) validity check.

These cover the two knobs a quantum scientist reaches for when stress-testing a
two-qubit-gate model:

  * ``n_levels`` -- raise the per-transmon truncation (3 -> 4+) to test whether
    the reported fidelity/leakage are converged. They are for the near-quiet CZ
    (~2e-4) but NOT for the strong-drive cross-resonance gate, whose 3-level number
    is overstated by ~3% (|2>->|3> leakage the qutrit model cannot see) -- the knob
    catches that. Every operator, the computational indices, and the QuTiP
    cross-check rebuild from it, so a 4-level model stays independently validated.
  * a custom 4x4 ``target_gate`` -- optimize toward an arbitrary two-qubit
    unitary without editing the source (validated unitary so a typo can't define
    a non-physical target).
  * ``CrossResonanceZXOptimizer.counter_rotating_fidelity`` -- measure the RWA
    error of a CR pulse by restoring the counter-rotating drive terms.

Run:  pytest tests/test_levels.py   OR   python tests/test_levels.py
"""
import math

import numpy as np
import pytest
import torch

from gradpulse import (CrossResonanceProfile, CrossResonanceZXOptimizer,
                       ParametricCouplerProfile, ParametricCZOptimizer)
from gradpulse.parametric import DEVICE


# ---- n_levels: parametric CZ ---------------------------------------------
def test_parametric_default_n_levels_is_3():
    """Parametric CZ default unchanged: qutrit pair (the near-quiet CZ is converged
    at 3). Contrast the CR default, which is 4 (test_cr_default_n_levels_is_4)."""
    opt = ParametricCZOptimizer(ParametricCouplerProfile(),
                                bandwidth_mhz=0.0, activation="clamp")
    assert opt.n_levels == 3 and opt._dim == 9
    assert opt._comp_idx.tolist() == [0, 1, 3, 4]


def test_cr_default_n_levels_is_4():
    """Cross-resonance default is 4 -- the converged truncation. The strong drive
    makes |2>->|3> leakage real, so 3 levels overstate; 4 retires that caveat."""
    opt = CrossResonanceZXOptimizer(CrossResonanceProfile())
    assert opt.n_levels == 4 and opt._dim == 16
    assert opt._comp_idx.tolist() == [0, 1, 4, 5]


def test_n_levels_4_operators_and_indices():
    """n_levels=4 rebuilds every operator at dimension 16, comp = [0,1,4,5]."""
    opt = ParametricCZOptimizer(ParametricCouplerProfile(n_levels=4),
                                bandwidth_mhz=0.0, activation="clamp")
    assert opt._dim == 16
    assert opt._comp_idx.tolist() == [0, 1, 4, 5]
    assert opt._H_DRIFT.shape == (16, 16)
    # A forward pass produces a valid (leakage-aware) fidelity and leakage.
    u = torch.rand(1, 24, 3, dtype=opt.rdtype, device=DEVICE) * 0.2 + 0.45
    rho = opt.simulate_choi_batch(u, dt=1.0)
    f = float(opt._process_fidelity(rho)[0])
    lk = float(opt._leakage(rho)[0])
    assert 0.0 <= f <= 1.0 and 0.0 <= lk <= 1.0


def test_n_levels_below_3_rejected():
    with pytest.raises(ValueError):
        ParametricCZOptimizer(ParametricCouplerProfile(n_levels=2))


def test_truncation_convergence_same_pulse():
    """The SAME control evaluated at n_levels=3 and 4 gives the same fidelity and
    leakage when |3> is not populated -- i.e. the qutrit truncation is converged
    for the shipped operating point (this is exactly the check n_levels enables)."""
    p3 = ParametricCouplerProfile()
    p4 = ParametricCouplerProfile(n_levels=4)
    o3 = ParametricCZOptimizer(p3, bandwidth_mhz=0.0, activation="clamp", precision="double")
    o4 = ParametricCZOptimizer(p4, bandwidth_mhz=0.0, activation="clamp", precision="double")
    torch.manual_seed(0)
    # A gentle, low-leakage control (small drives) so |3> stays unpopulated.
    env = torch.rand(40, 3, dtype=o3.rdtype, device=DEVICE) * 0.1 + 0.45
    u = env.unsqueeze(0)
    f3 = float(o3._process_fidelity(o3.simulate_choi_batch(u, dt=1.0))[0])
    f4 = float(o4._process_fidelity(o4.simulate_choi_batch(u, dt=1.0))[0])
    lk3 = float(o3._leakage(o3.simulate_choi_batch(u, dt=1.0))[0])
    lk4 = float(o4._leakage(o4.simulate_choi_batch(u, dt=1.0))[0])
    assert abs(f3 - f4) < 5e-3
    assert abs(lk3 - lk4) < 5e-3


# ---- custom target unitary -----------------------------------------------
def test_custom_target_matches_named_cz():
    CZ = np.diag([1, 1, 1, -1]).astype(complex)
    o_named = ParametricCZOptimizer(ParametricCouplerProfile(), target_gate="cz")
    o_custom = ParametricCZOptimizer(ParametricCouplerProfile(), target_gate=CZ)
    assert o_custom.target_gate == "custom"
    assert torch.allclose(o_custom.u_target_4x4, o_named.u_target_4x4)


def test_custom_target_arbitrary_unitary_runs():
    """An arbitrary (non-named) two-qubit unitary is a valid target."""
    # A B-gate-ish entangler: exp(-i pi/4 (XX + YY)) on the comp subspace.
    th = math.pi / 4
    U = np.array([[1, 0, 0, 0],
                  [0, math.cos(th), -1j * math.sin(th), 0],
                  [0, -1j * math.sin(th), math.cos(th), 0],
                  [0, 0, 0, 1]], dtype=complex)
    opt = ParametricCZOptimizer(ParametricCouplerProfile(), target_gate=U,
                                bandwidth_mhz=0.0, activation="clamp")
    u = torch.rand(1, 20, 3, dtype=opt.rdtype, device=DEVICE) * 0.2 + 0.45
    f = float(opt._process_fidelity(opt.simulate_choi_batch(u, dt=1.0))[0])
    assert 0.0 <= f <= 1.0


def test_custom_target_validation():
    with pytest.raises(ValueError):     # not unitary
        ParametricCZOptimizer(ParametricCouplerProfile(),
                              target_gate=np.ones((4, 4)))
    with pytest.raises(ValueError):     # wrong shape
        ParametricCZOptimizer(ParametricCouplerProfile(),
                              target_gate=np.eye(3))


# ---- n_levels: cross-resonance + the 4-level leakage check ----------------
def test_cr_n_levels_4_optimizes_and_reports_leakage():
    """The named ask: a 4-level leakage check on the strong-drive CR gate."""
    opt = CrossResonanceZXOptimizer(CrossResonanceProfile(n_levels=4))
    assert opt._dim == 16 and opt._comp_idx.tolist() == [0, 1, 4, 5]
    res = opt.optimize(n_slices=60, dt_ns=1.0, iterations=50, n_seeds=1,
                       lr=0.06, seed0=0)
    assert 0.0 <= res["best_fidelity"] <= 1.0
    assert 0.0 <= res["best_leakage"] <= 1.0


def test_cr_truncation_not_free_3level_overstates():
    """The strong-drive CR gate is NOT converged at 3 levels -- the honest physics
    the n_levels knob exists to expose.

    A CR pulse optimized in the 3-level model is blind to |2>->|3> leakage on the
    strongly-driven control, so re-scoring it at n_levels=4 reveals real leakage and
    LOWERS the fidelity. Measured ~3% overstatement at the default operating point
    (omega_max=60 MHz; F_proc 0.899 -> 0.869, |3>-ish leakage ~9e-3). This is why CR
    leakage/fidelity claims must be made at n_levels>=4 (see CrossResonanceProfile.
    n_levels). Contrast test_truncation_convergence_same_pulse, where the near-quiet
    CZ stays converged at 3 levels to ~2e-4."""
    o3 = CrossResonanceZXOptimizer(CrossResonanceProfile(n_levels=3), use_drag=True)
    res = o3.optimize(n_slices=80, dt_ns=1.0, iterations=120, n_seeds=1, lr=0.06)
    x, vz = res["best_raw_param"], res["virtual_z"]
    o4 = CrossResonanceZXOptimizer(CrossResonanceProfile(n_levels=4), use_drag=True)
    xt = torch.tensor(x, device=DEVICE, dtype=o4.rdtype).unsqueeze(0)
    vzt = torch.tensor(vz, device=DEVICE, dtype=o4.rdtype)
    rho4 = o4.simulate_choi_batch(xt, dt=1.0)
    f4 = float(o4._process_fidelity(rho4, vzt)[0])
    lk4 = float(o4._leakage(rho4)[0])
    # The 3-level number is an overstatement: resolving |3> can only reveal MORE
    # error for the same pulse, so the 4-level score is clearly lower (not noise).
    assert f4 < res["best_fidelity"] - 2e-3
    # ...and the newly-resolved |3> channel shows up as strictly more total leakage.
    assert lk4 > res["best_leakage"]


def test_cr_converged_at_default_4_levels():
    """The CR default n_levels=4 IS converged -- the other half of the truncation
    story. Re-scoring a 4-level-optimal pulse at 5 moves F_proc by ~1e-4 (below the
    decoherence floor; measured 1.6e-4 at the full 180 ns config), in sharp contrast
    to the ~3% a 3->4 re-score exposes above. Pins the claim so it cannot drift."""
    o4 = CrossResonanceZXOptimizer(CrossResonanceProfile(n_levels=4), use_drag=True,
                                   use_target_cancel=True)
    res = o4.optimize(n_slices=80, dt_ns=1.0, iterations=120, n_seeds=1, lr=0.06)
    o5 = CrossResonanceZXOptimizer(CrossResonanceProfile(n_levels=5), use_drag=True,
                                   use_target_cancel=True)
    xt = torch.tensor(res["best_raw_param"], device=DEVICE, dtype=o5.rdtype).unsqueeze(0)
    vzt = torch.tensor(res["virtual_z"], device=DEVICE, dtype=o5.rdtype)
    f5 = float(o5._process_fidelity(o5.simulate_choi_batch(xt, dt=1.0), vzt)[0])
    # Converged at 4: 4->5 shift is tiny (generous bound for the short test opt),
    # an order of magnitude below the ~3e-2 the 3->4 re-score reveals.
    assert abs(res["best_fidelity"] - f5) < 3e-3


# ---- counter-rotating (beyond-RWA) validity check -------------------------
def test_counter_rotating_reference_reproduces_nominal():
    """The RWA reference of counter_rotating_fidelity (sub-stepped, no
    counter-rotating term) reproduces the nominal closed-system fidelity, and the
    measured RWA error is small for a moderate drive."""
    opt = CrossResonanceZXOptimizer(CrossResonanceProfile(omega_max_mhz=60.0),
                                    use_drag=True)
    res = opt.optimize(n_slices=60, dt_ns=1.0, iterations=80, n_seeds=1, lr=0.06)
    cr = opt.counter_rotating_fidelity(res["best_raw_param"], vz=res["virtual_z"],
                                       substeps=100)
    # RWA reference ~ nominal (differ only by the small decoherence over the gate)
    assert abs(cr["f_proc_rwa"] - res["best_fidelity"]) < 5e-3
    # the omitted physics is small and well-defined at 2*f_target
    assert abs(cr["delta_r_counter_rot"]) < 5e-3
    assert cr["omega_d_ghz"] == pytest.approx(4.85, abs=1e-9)
    for k in ("f_proc_counter_rot", "f_avg_rwa", "f_avg_counter_rot", "substeps"):
        assert k in cr


def test_counter_rotating_error_grows_with_drive():
    """RWA error increases with drive strength -- the whole reason it matters for
    the strong-drive CR gate (vs the near-quiet coupler-activated CZ)."""
    def delta(omega_mhz):
        o = CrossResonanceZXOptimizer(CrossResonanceProfile(omega_max_mhz=omega_mhz),
                                      use_drag=True)
        r = o.optimize(n_slices=60, dt_ns=1.0, iterations=60, n_seeds=1, lr=0.06)
        cr = o.counter_rotating_fidelity(r["best_raw_param"], vz=r["virtual_z"],
                                         substeps=100)
        return abs(cr["delta_r_counter_rot"])
    assert delta(90.0) > delta(20.0)


def test_counter_rotating_qutip_independent_cross_check():
    """The beyond-RWA number is held to the same independent-solver bar as the rest
    of the package. counter_rotating_fidelity restores the 2*omega_d term via fixed
    torch sub-stepping; validate.cr_counter_rotating_cross_check rebuilds the SAME
    time-dependent Hamiltonian and integrates it with QuTiP's ADAPTIVE propagator
    (different library AND different integration scheme). The beyond-RWA F_proc shift
    agreeing confirms both the implementation and that the sub-stepping is converged."""
    pytest.importorskip("qutip")
    from dataclasses import asdict
    from gradpulse import validate
    prof = CrossResonanceProfile(n_levels=3)   # qutrit keeps the QuTiP propagator cheap
    opt = CrossResonanceZXOptimizer(prof, bandwidth_mhz=60.0, use_drag=True,
                                    use_target_cancel=True)
    res = opt.optimize(n_slices=32, dt_ns=1.0, iterations=50, n_seeds=1, lr=0.06)
    cr_t = opt.counter_rotating_fidelity(res["best_raw_param"], vz=res["virtual_z"],
                                         substeps=150)
    cr_q = validate.cr_counter_rotating_cross_check(
        asdict(prof), res["best_waveform"], res["virtual_z"], 1.0, use_drag=True)
    # Independent QuTiP adaptive integrator reproduces the beyond-RWA shift...
    assert abs(cr_t["delta_r_counter_rot"] - cr_q["delta_r_counter_rot"]) < 2e-4
    # ...and the RWA reference fidelities agree (same physics, wholly different solver).
    assert abs(cr_t["f_proc_rwa"] - cr_q["f_proc_rwa"]) < 2e-4
    assert cr_q["omega_d_ghz"] == pytest.approx(cr_t["omega_d_ghz"], abs=1e-9)


def test_counter_rotating_substep_independent_double_precision():
    """Regression guard for a single-precision round-off bug in
    counter_rotating_fidelity. Its RWA reference is mathematically substep-INDEPENDENT:
    the per-slice Hamiltonian is piecewise-constant, so matrix_exp is exact and composing
    n_sub of them re-makes the same slice propagator. A single-precision substep loop
    instead accumulates round-off over ~10^4-10^5 propagators, which drifts f_rwa by
    ~1e-3 and turns the small (~7e-6) beyond-RWA delta into sign-flipping noise (the
    earlier QuTiP cross-check test missed this -- its 32-slice qutrit gate has ~zero
    effect). The method now integrates and contracts in double internally, so f_rwa is
    fixed and the delta is stable regardless of the optimizer's (here single) precision."""
    prof = CrossResonanceProfile(n_levels=3)
    opt = CrossResonanceZXOptimizer(prof, bandwidth_mhz=60.0, use_drag=True,
                                    use_target_cancel=True)        # default: SINGLE precision
    assert opt.cdtype == torch.complex64                           # exercise the bug case
    res = opt.optimize(n_slices=80, dt_ns=1.0, iterations=50, n_seeds=1, lr=0.06)
    x, vz = res["best_raw_param"], res["virtual_z"]
    # substeps=200 is the method default and its documented adequate range is ~150-300;
    # 100 under-resolves the ~10 GHz counter-rotating oscillation, leaving a ~1e-5 midpoint
    # discretization residual that is pulse- (hence platform-) dependent. Comparing the
    # converged operating point (200) against a finer grid (500) isolates the double-precision
    # stability we are guarding, not the integrator's truncation error.
    lo = opt.counter_rotating_fidelity(x, vz=vz, substeps=200)
    hi = opt.counter_rotating_fidelity(x, vz=vz, substeps=500)
    # The RWA reference is substep-exact: the single-precision bug drifted it ~1e-3.
    assert abs(lo["f_proc_rwa"] - hi["f_proc_rwa"]) < 1e-7
    # The measured beyond-RWA shift is stable/converged, not round-off noise.
    assert abs(lo["delta_r_counter_rot"] - hi["delta_r_counter_rot"]) < 1e-5
    # ...and stays a small, sane infidelity (the RWA pulse is not over-fit to the approx).
    assert -1e-4 < hi["delta_r_counter_rot"] < 1e-2


def test_refine_beyond_rwa_improves_counter_rotating_fidelity():
    """refine_beyond_rwa is the OPTIMIZATION counterpart of counter_rotating_fidelity
    (which only diagnoses): starting from an RWA-optimized pulse, descending on the
    beyond-RWA process fidelity -- counter-rotating terms inside the gradient loop --
    must not decrease, and removes the residual rather than only measuring it."""
    opt = CrossResonanceZXOptimizer(CrossResonanceProfile())
    res = opt.optimize(n_slices=24, dt_ns=1.0, iterations=40, n_seeds=1, lr=0.04)
    ref = opt.refine_beyond_rwa(res["best_raw_param"], vz=res["virtual_z"],
                                dt_ns=1.0, iterations=6, substeps=12, lr=0.01)
    # the beyond-RWA objective improves (gradient through the full Hamiltonian is correct)
    assert ref["f_proc_after"] >= ref["f_proc_before"] - 1e-6
    assert ref["delta_removed"] == pytest.approx(
        ref["f_proc_after"] - ref["f_proc_before"], abs=1e-9)
    assert ref["best_raw_param"].shape == res["best_raw_param"].shape


# ---- QuTiP cross-check survives the customization --------------------------
def test_qutip_cross_check_at_n_levels_4():
    """A 4-level model stays independently validated: the QuTiP integrator (a
    different library) agrees with our simulator at n_levels=4."""
    pytest.importorskip("qutip")
    from gradpulse import validate
    prof = ParametricCouplerProfile(n_levels=4)
    opt = ParametricCZOptimizer(prof, bandwidth_mhz=0.0, activation="clamp",
                                precision="double")
    torch.manual_seed(0)
    env = torch.rand(24, 3, dtype=opt.rdtype, device=DEVICE) * 0.2 + 0.45
    f_torch = float(opt._process_fidelity(
        opt.simulate_choi_batch(env.unsqueeze(0), dt=1.0))[0])
    f_qutip = validate.qutip_f_proc(prof, env.cpu().numpy(), "cz", 1.0)
    assert abs(f_torch - f_qutip) < 1e-3


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
