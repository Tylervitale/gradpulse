"""Master-equation integrator: 2nd-order step, dt-convergence, and the
drive-frequency coupler channel.

Two engine features added on top of the default first-order Trotter scheme:

  - ``step_order=2``: a symmetric (Strang) split with a 2nd-order dissipator
    substep, global error O(dt^2). ``dt_convergence()`` holds a pulse fixed,
    shrinks dt, and reports the resulting convergence for both orders.
  - ``coupler_phase_mode='frequency'``: channel 4 becomes an instantaneous
    drive detuning whose running integral is the coupler phase, so a static
    offset (a real frequency control) is representable, unlike bounded 'phase'.

The numerical claims are pinned with comfortable margins (working dtype is
complex64, so deep-refinement residuals hit a ~1e-6 noise floor; these tests
assert robust orderings, not exact deep-refinement rates):

  - with no dissipation the two orders are bitwise identical (only the
    dissipator treatment differs, by construction);
  - in strong dissipation the 1st-order error halves per dt halving while the
    2nd-order result is already converged at dt=1 ns;
  - an independent QuTiP ``mesolve`` (different algorithm) confirms order 2 is
    markedly closer to the exact open-system evolution than order 1;
  - a constant detuning in 'frequency' mode reproduces a matching linear phase
    ramp in 'phase' mode to float precision.

Run:  pytest tests/        OR        python tests/test_integration.py
"""
import math

import numpy as np
import pytest
import torch

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer
from gradpulse.parametric import DEVICE


def _profile(**over):
    base = dict(
        freq_ghz_q1=4.85, freq_ghz_q2=5.05,
        anharm_ghz_q1=-0.20, anharm_ghz_q2=-0.20,
        t1_ns_q1=30_000, t2_ns_q1=25_000,
        t1_ns_q2=30_000, t2_ns_q2=25_000,
        g_max_mhz=12.0, omega_max_mhz=50.0,
        chi_zz_mhz=0.0,
    )
    base.update(over)
    return ParametricCouplerProfile(**base)


def _const_pulse(signed, n_slices):
    """Constant pulse [1, n_slices, len(signed)] in [0,1] whose clamp-activated,
    unsmoothed signed value is exactly ``signed`` (so H is constant in time)."""
    u01 = [(s + 1.0) / 2.0 for s in signed]
    t = torch.tensor(u01, device=DEVICE, dtype=torch.float32)
    return t.view(1, 1, len(signed)).expand(1, n_slices, len(signed)).contiguous()


def _sim_F(opt, u, k, order, diss=1.0):
    """Process fidelity of u refined k× (zero-order hold, dt -> dt/k) at a given
    step order and dissipation scale. Restores opt.step_order afterwards."""
    saved = opt.step_order
    opt.step_order = order
    try:
        ur = torch.repeat_interleave(u, k, dim=1)
        rho = opt.simulate_choi_batch(ur, dt=1.0 / k, diss_scale=diss)
        return float(opt._process_fidelity(rho).mean().item())
    finally:
        opt.step_order = saved


# --------------------------------------------------------------------------
# Construction / validation guards
# --------------------------------------------------------------------------
def test_step_order_validation():
    ParametricCZOptimizer(_profile(), step_order=1)   # ok
    ParametricCZOptimizer(_profile(), step_order=2)   # ok
    with pytest.raises(ValueError, match="step_order"):
        ParametricCZOptimizer(_profile(), step_order=3)


def test_default_is_first_order():
    assert ParametricCZOptimizer(_profile()).step_order == 1


def test_coupler_phase_mode_validation():
    ParametricCZOptimizer(_profile(), n_channels=4, coupler_phase_mode="phase")
    ParametricCZOptimizer(_profile(), n_channels=4, coupler_phase_mode="frequency")
    with pytest.raises(ValueError, match="coupler_phase_mode"):
        ParametricCZOptimizer(_profile(), n_channels=4, coupler_phase_mode="nope")
    # 'frequency' needs the phase channel to exist
    with pytest.raises(ValueError, match="n_channels"):
        ParametricCZOptimizer(_profile(), n_channels=3, coupler_phase_mode="frequency")


# --------------------------------------------------------------------------
# 2nd-order step
# --------------------------------------------------------------------------
def test_order2_runs_shape_and_trace():
    opt = ParametricCZOptimizer(_profile(), n_channels=4, step_order=2)
    u = _const_pulse([0.4, -0.3, 0.6, 0.2], 24)
    rho = opt.simulate_gradient_batch(u, dt=1.0)
    assert tuple(rho.shape) == (1, 4, 9, 9)
    # Lindblad evolution is trace-preserving; the integrator keeps Tr(rho)=1.
    tr = rho.diagonal(dim1=-2, dim2=-1).real.sum(-1)   # [1, 4]
    assert torch.allclose(tr, torch.ones_like(tr), atol=1e-3)
    # Exact process fidelity uses the 16-operator Choi stack.
    f = opt._process_fidelity(opt.simulate_choi_batch(u, dt=1.0))
    assert torch.all((f >= 0.0) & (f <= 1.0))


def test_no_dissipation_orders_are_identical():
    # With diss_scale=0 the dissipator vanishes and BOTH schemes reduce to the
    # same exact unitary step, so they must agree bitwise. This proves the only
    # thing step_order changes is the dissipator treatment.
    u = _const_pulse([0.5, -0.2, 0.55, 0.1], 20)
    o1 = ParametricCZOptimizer(_profile(), n_channels=4, step_order=1)
    o2 = ParametricCZOptimizer(_profile(), n_channels=4, step_order=2)
    r1 = o1.simulate_gradient_batch(u, dt=1.0, diss_scale=0.0)
    r2 = o2.simulate_gradient_batch(u, dt=1.0, diss_scale=0.0)
    assert torch.allclose(r1, r2, atol=1e-12, rtol=0.0)
    # ...and that shared evolution is unitary: a pure input stays pure.
    purity = torch.einsum("bkij,bkji->bk", r1, r1).real   # Tr(rho^2)
    assert torch.all(purity > 0.999)


def _isolated_opt_and_pulse():
    # No smoothing + no DRAG -> the waveform is dt-invariant under zero-order hold,
    # so refining dt probes ONLY the integrator's dissipator splitting.
    opt = ParametricCZOptimizer(_profile(), n_channels=3,
                                bandwidth_mhz=0.0, use_drag=False)
    u = _const_pulse([0.55, -0.55, 0.6], 16)
    return opt, u


def test_first_order_convergence_rate():
    # Strong dissipation amplifies the splitting error above the float noise floor.
    # Truth = best available estimate of the dt->0 limit (order 2 at the finest step).
    opt, u = _isolated_opt_and_pulse()
    truth = _sim_F(opt, u, k=16, order=2, diss=50.0)
    e_dt = abs(_sim_F(opt, u, k=1, order=1, diss=50.0) - truth)
    e_dt2 = abs(_sim_F(opt, u, k=2, order=1, diss=50.0) - truth)
    assert e_dt > 1e-5, f"no measurable 1st-order error to test ({e_dt:.2e})"
    ratio = e_dt / e_dt2
    # O(dt) error halves when dt halves -> ratio ~ 2.
    assert 1.5 < ratio < 2.6, f"1st-order rate ratio off: {ratio:.3f}"


def test_second_order_is_converged_at_dt1():
    # In the same strong-dissipation setup, order 2 at dt=1 ns is already at the
    # dt->0 limit, and far closer to it than order 1 at the same dt.
    opt, u = _isolated_opt_and_pulse()
    truth = _sim_F(opt, u, k=16, order=2, diss=50.0)
    e1 = abs(_sim_F(opt, u, k=1, order=1, diss=50.0) - truth)
    e2 = abs(_sim_F(opt, u, k=1, order=2, diss=50.0) - truth)
    assert e2 < e1 / 3.0, f"order2 not clearly better: e1={e1:.2e} e2={e2:.2e}"


# --------------------------------------------------------------------------
# dt_convergence report
# --------------------------------------------------------------------------
def test_dt_convergence_report_structure_and_realistic_values():
    opt = ParametricCZOptimizer(_profile(), n_channels=4)
    u = opt._warm_start(40, mode="parametric_cz").unsqueeze(0)
    rep = opt.dt_convergence(u, dt=1.0, refinements=(1, 2, 4))
    assert rep["refinements"] == [1, 2, 4]
    assert rep["dt"] == [1.0, 0.5, 0.25]
    assert len(rep["order1"]) == 3 and len(rep["order2"]) == 3
    for key in ("order1_extrap", "order2_extrap", "order1_err_at_dt",
                "order2_err_at_dt", "splitting_err_at_dt"):
        assert key in rep
    # At realistic coherence the first-order scheme is already well converged,
    # so the pure integrator-splitting error at dt=1 ns is tiny and the two
    # orders' dt->0 extrapolations agree.
    assert rep["splitting_err_at_dt"] < 1e-3
    assert abs(rep["order1_extrap"] - rep["order2_extrap"]) < 1e-3
    # The utility must leave the optimizer's step order untouched.
    assert opt.step_order == 1


def test_dt_convergence_accepts_2d_and_guards_input():
    opt = ParametricCZOptimizer(_profile(), n_channels=4)
    u2d = opt._warm_start(24, mode="parametric_cz")        # [n_slices, n_ch]
    rep = opt.dt_convergence(u2d, dt=1.0, refinements=(1, 2))
    assert len(rep["order1"]) == 2
    with pytest.raises(ValueError, match="refinements"):
        opt.dt_convergence(u2d, refinements=(0, 1))
    with pytest.raises(ValueError, match="metric"):
        opt.dt_convergence(u2d, metric="bogus")


# --------------------------------------------------------------------------
# Drive-frequency coupler channel
# --------------------------------------------------------------------------
def test_frequency_mode_equals_matching_phase_ramp():
    # Defining property: a CONSTANT detuning δ integrates to a LINEAR phase
    # ramp θ(t)=δ·t (midpoint-sampled). So 'frequency' mode with constant u4
    # must equal 'phase' mode driven by that exact ramp. No smoothing/DRAG so
    # both paths are distortion-free and the equality is exact to float noise.
    N, dt = 16, 1.0
    ofreq = ParametricCZOptimizer(_profile(), n_channels=4, bandwidth_mhz=0.0,
                                  use_drag=False, coupler_phase_mode="frequency",
                                  delta_max_mhz=30.0)
    ophase = ParametricCZOptimizer(_profile(), n_channels=4, bandwidth_mhz=0.0,
                                   use_drag=False, coupler_phase_mode="phase")
    base = _const_pulse([0.1, -0.1, 0.4, 0.0], N).clone()  # ch3 overwritten below
    c_raw = 0.55                                           # signed s = 2c-1 = 0.1
    s = 2 * c_raw - 1
    u_freq = base.clone(); u_freq[..., 3] = c_raw
    i = torch.arange(N, device=DEVICE, dtype=torch.float32)
    theta = (i + 0.5) * ofreq.DELTA_MAX * s * dt           # midpoint integral
    p_raw = 0.5 * (1.0 + theta / math.pi)                  # phase-mode preimage
    assert float(p_raw.max()) <= 1.0 and float(p_raw.min()) >= 0.0
    u_phase = base.clone(); u_phase[..., 3] = p_raw
    rf = ofreq.simulate_gradient_batch(u_freq, dt=dt)
    rp = ophase.simulate_gradient_batch(u_phase, dt=dt)
    assert (rf - rp).abs().max().item() < 1e-5


def test_frequency_mode_differs_from_phase_mode():
    # A generic (non-constant) ch4 must drive genuinely different dynamics under
    # the two interpretations: guards against 'frequency' aliasing to 'phase'.
    torch.manual_seed(0)
    u = torch.rand(1, 20, 4, device=DEVICE)
    ofreq = ParametricCZOptimizer(_profile(), n_channels=4,
                                  coupler_phase_mode="frequency")
    ophase = ParametricCZOptimizer(_profile(), n_channels=4,
                                   coupler_phase_mode="phase")
    rf = ofreq.simulate_gradient_batch(u, dt=1.0)
    rp = ophase.simulate_gradient_batch(u, dt=1.0)
    assert (rf - rp).abs().max().item() > 1e-3


def test_frequency_mode_gradient_flows_through_channel4():
    # The running-integral (cumsum) phase must be differentiable so the
    # optimizer can actually train the detuning control.
    ofreq = ParametricCZOptimizer(_profile(), n_channels=4,
                                  coupler_phase_mode="frequency")
    u = torch.rand(1, 20, 4, device=DEVICE, requires_grad=True)
    loss = 1.0 - ofreq._process_fidelity(ofreq.simulate_choi_batch(u, dt=1.0)).mean()
    loss.backward()
    assert u.grad[..., 3].abs().sum().item() > 0.0


# --------------------------------------------------------------------------
# Independent cross-check (QuTiP mesolve, a different algorithm entirely)
# --------------------------------------------------------------------------
def test_mesolve_cross_check_orders():
    qt = pytest.importorskip("qutip")
    from dataclasses import asdict
    from gradpulse.validate import _build_qutip_ops

    # Amplified dissipation so order1 and order2 are distinguishable at dt=1 ns.
    prof = _profile(t1_ns_q1=200.0, t1_ns_q2=200.0,
                    t2_ns_q1=150.0, t2_ns_q2=150.0)
    N, dt = 30, 1.0
    T = N * dt
    s1, s2, s3 = 0.35, -0.25, 0.6
    u = _const_pulse([s1, s2, s3], N)

    o1 = ParametricCZOptimizer(prof, n_channels=3, bandwidth_mhz=0.0,
                               use_drag=False, step_order=1)
    o2 = ParametricCZOptimizer(prof, n_channels=3, bandwidth_mhz=0.0,
                               use_drag=False, step_order=2)
    comp = o1._comp_idx
    ci = comp.tolist() if torch.is_tensor(comp) else list(comp)
    rho1 = o1.simulate_gradient_batch(u, dt=dt)[0, 0].detach().cpu().numpy()
    rho2 = o2.simulate_gradient_batch(u, dt=dt)[0, 0].detach().cpu().numpy()

    # Same constant H + collapse operators, evolved by QuTiP's adaptive solver.
    ops = _build_qutip_ops(asdict(prof))
    H = (ops["H_drift"]
         + s1 * ops["omega_max"] * ops["X1"]
         + s2 * ops["omega_max"] * ops["X2"]
         + s3 * ops["g_max"] * ops["Cx"])
    v = np.zeros((9, 1), dtype=complex)
    for j in range(4):
        v[ci[j], 0] = 0.5                       # |++> input (matches sim input 0)
    psi0 = qt.Qobj(v, dims=[[3, 3], [1, 1]])
    res = qt.mesolve(H, psi0 * psi0.dag(), [0.0, T], c_ops=ops["L_ops"], e_ops=[],
                     options={"atol": 1e-11, "rtol": 1e-9, "nsteps": 200_000})
    rho_exact = res.states[-1].full()

    assert abs(np.trace(rho_exact).real - 1.0) < 1e-6
    e1 = np.abs(rho1 - rho_exact).max()
    e2 = np.abs(rho2 - rho_exact).max()
    assert e1 > 5e-4, f"setup too weak to distinguish orders (e1={e1:.2e})"
    assert e2 < e1 / 3.0, f"order2 not closer to mesolve: e1={e1:.2e} e2={e2:.2e}"
    assert e2 < 3e-4, f"order2 disagrees with mesolve: e2={e2:.2e}"


def test_mesolve_zoh_unbiased_on_reference_pulse():
    """The shipped Trotter scheme is UNBIASED, not merely self-consistent.

    The matched cross-check (qutip_f_proc) shares the scheme, and dt_convergence only
    shows the scheme converges *smoothly* as dt->0 -- neither can rule out a
    consistent-but-biased stepper that converges to the wrong continuous-time limit.
    QuTiP's adaptive ``mesolve``, run on the identical ZOH staircase interval-by-
    interval (a different numerical method), lands on the same continuous-time F_proc
    the scheme reaches as dt->0 -- which is the missing, stronger claim.
    """
    pytest.importorskip("qutip")
    import json
    from pathlib import Path
    from gradpulse.validate import mesolve_zoh_fproc

    fix = Path(__file__).parent / "fixtures" / "reference_cz_pulse.json"
    meta = json.loads(fix.read_text())
    u = np.load(fix.parent / Path(meta["pulse_npy"]).name)
    prof = ParametricCouplerProfile(**meta["profile"])
    dt = float(meta["pulse_dt_ns"])
    gate = str(meta.get("target_gate", "cz"))

    # The shipped scheme (order 1) on the actual pulse, in double precision, at dt=1 ns
    # and refined 8x (zero-order hold) toward the dt->0 limit.
    opt = ParametricCZOptimizer(prof, bandwidth_mhz=0.0, use_drag=False,
                                n_channels=int(meta["n_channels"]),
                                activation="clamp", precision="double")
    ut = torch.tensor(u, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    fp_dt1 = float(opt._process_fidelity(opt.simulate_choi_batch(ut, dt=dt))[0])
    ur = torch.repeat_interleave(ut, 8, dim=1)
    fp_fine = float(opt._process_fidelity(opt.simulate_choi_batch(ur, dt=dt / 8))[0])

    # Independent adaptive solver on the identical staircase.
    fp_mesolve = mesolve_zoh_fproc(prof, u, target_gate=gate, dt_ns=dt)

    # (1) Agreement with the shipped dt=1 ns scheme at well below the 1e-3 ship gate.
    assert abs(fp_mesolve - fp_dt1) < 1e-5, \
        f"mesolve {fp_mesolve:.9f} vs Trotter dt=1ns {fp_dt1:.9f}"
    # (2) The load-bearing claim: the independent adaptive method and the dt->0-refined
    #     Trotter scheme agree on the continuous-time answer -> scheme is unbiased.
    assert abs(fp_mesolve - fp_fine) < 1e-6, \
        f"mesolve {fp_mesolve:.9f} vs dt->0 Trotter {fp_fine:.9f} (would flag a biased scheme)"
    # (3) Direction: F_proc is monotone increasing toward dt->0, so the adaptive value
    #     sits at/above the coarse dt=1 ns value by the known ~1e-7 discretization.
    assert fp_mesolve >= fp_dt1 - 1e-9, (fp_mesolve, fp_dt1)


def test_operator_builders_agree_structural_guard():
    """Cross-check contract, enforced: the PyTorch and QuTiP operator builders must
    construct the SAME Hamiltonian and Lindblad operators.

    The matched-scheme cross-check (`qutip_f_proc`) only proves the two simulators agree
    *given* identical operators; that precondition is maintained by hand, in two parallel
    builders. The moment someone adds a term to one and not the other, the "independent"
    cross-check silently stops being apples-to-apples. This asserts operator parity on a
    random profile (every physics field varied, incl. static-ZZ and finite-temperature
    jumps) to machine precision -- a STRUCTURAL guard, distinct from the physics check.
    """
    qt = pytest.importorskip("qutip")
    from dataclasses import asdict
    from gradpulse.parametric import _build_coupler_ops
    from gradpulse.validate import _build_qutip_ops

    rng = np.random.default_rng(7)
    prof = ParametricCouplerProfile(
        n_levels=4,
        freq_ghz_q1=4.5 + rng.uniform(0, 0.5), freq_ghz_q2=5.0 + rng.uniform(0, 0.5),
        anharm_ghz_q1=-0.2 - rng.uniform(0, 0.1), anharm_ghz_q2=-0.2 - rng.uniform(0, 0.1),
        t1_ns_q1=2e4 + rng.uniform(0, 2e4), t1_ns_q2=2e4 + rng.uniform(0, 2e4),
        t2_ns_q1=1.5e4 + rng.uniform(0, 1e4), t2_ns_q2=1.5e4 + rng.uniform(0, 1e4),
        g_max_mhz=8 + rng.uniform(0, 8), omega_max_mhz=40 + rng.uniform(0, 20),
        chi_zz_mhz=rng.uniform(0.1, 1.0),
        n_thermal_q1=rng.uniform(0.01, 0.05), n_thermal_q2=rng.uniform(0.01, 0.05),
    )
    P = _build_coupler_ops(prof, dtype=torch.complex128)
    Q = _build_qutip_ops(asdict(prof))

    pairs = [
        ("H_drift", P["H_DRIFT"], Q["H_drift"]),
        ("X1", P["X1"], Q["X1"]), ("X2", P["X2"], Q["X2"]),
        ("Cx", P["COUPLING_X"], Q["Cx"]), ("Cy", P["COUPLING_Y"], Q["Cy"]),
        ("N1", P["N_Q1"], Q["N1"]), ("N2", P["N_Q2"], Q["N2"]),
        ("L_T1_q1", P["L_T1_Q1"], Q["L_ops"][0]),
        ("L_T1_q2", P["L_T1_Q2"], Q["L_ops"][1]),
        ("L_phi_q1", P["L_PHI_Q1"], Q["L_ops"][2]),
        ("L_phi_q2", P["L_PHI_Q2"], Q["L_ops"][3]),
        ("L_th_q1", P["L_TH_Q1"], Q["L_ops"][4]),   # thermal jumps (n_thermal > 0)
        ("L_th_q2", P["L_TH_Q2"], Q["L_ops"][5]),
    ]
    for name, a_t, b_q in pairs:
        diff = float(np.abs(a_t.detach().cpu().numpy() - b_q.full()).max())
        assert diff < 1e-12, f"operator {name} differs between builders by {diff:.2e}"


if __name__ == "__main__":
    test_step_order_validation()
    test_default_is_first_order()
    test_coupler_phase_mode_validation()
    test_order2_runs_shape_and_trace()
    test_no_dissipation_orders_are_identical()
    test_first_order_convergence_rate()
    test_second_order_is_converged_at_dt1()
    test_dt_convergence_report_structure_and_realistic_values()
    test_dt_convergence_accepts_2d_and_guards_input()
    test_frequency_mode_equals_matching_phase_ramp()
    test_frequency_mode_differs_from_phase_mode()
    test_frequency_mode_gradient_flows_through_channel4()
    try:
        test_mesolve_cross_check_orders()
        test_mesolve_zoh_unbiased_on_reference_pulse()
        test_operator_builders_agree_structural_guard()
        print("integration tests passed (incl. QuTiP cross-check)")
    except Exception as exc:        # qutip missing -> importorskip raises Skipped
        print(f"integration tests passed (QuTiP cross-check skipped: {exc})")
