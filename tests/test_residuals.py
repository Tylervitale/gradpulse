"""The three honest residuals from the integration-features work, now resolved
into real, tested capabilities.

  1. Raw-parameter surfacing -- optimize_multi_seed returns 'best_raw_param',
     the exact optimizer parameter the simulator consumed. Feeding it back
     through simulate_gradient_batch / dt_convergence reproduces the reported
     fidelity bit-for-bit, with no re-smoothing distortion. This also fixes a
     latent bug: the L-BFGS polish used to save smoothed_waveform(sigmoid(x)),
     double-applying the sigmoid; it now returns the raw param so the saved
     waveform matches what was actually evaluated.

  2. Configurable double precision (precision='double') -- complex128/float64
     drops the integrator noise floor far below complex64's ~1e-6, so under deep
     dt refinement the first-order Euler error keeps halving cleanly (ratio ~0.5)
     where single precision floors. precision='single' is byte-for-byte the
     original behavior.

  3. Coupling rolloff with detuning (coupler_g_linewidth_mhz) -- a phenomenological
     single-pole model g_eff(delta) = g_max / sqrt(1 + (delta/kappa)^2), active
     only in coupler_phase_mode='frequency' and off by default, so the optimizer
     pays for detuning the parametric drive off resonance instead of getting peak
     coupling at any detuning.

Numbers are pinned with comfortable margins; the precision and rolloff checks use
constant pulses + an isolated integrator (no smoothing/DRAG) so they are fully
deterministic.

Run:  pytest tests/        OR        python tests/test_residuals.py
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


def _const_pulse(signed, n_slices, dtype=torch.float32):
    """Constant pulse [1, n_slices, len(signed)] in [0,1] whose clamp-activated,
    unsmoothed signed value is exactly ``signed`` (so H is constant in time)."""
    u01 = [(s + 1.0) / 2.0 for s in signed]
    t = torch.tensor(u01, device=DEVICE, dtype=dtype)
    return t.view(1, 1, len(signed)).expand(1, n_slices, len(signed)).contiguous()


# ==========================================================================
# 1. Raw-parameter surfacing  (+ L-BFGS double-sigmoid fix)
# ==========================================================================
def _quick_optimize(activation):
    opt = ParametricCZOptimizer(_profile(), n_channels=4, activation=activation)
    res = opt.optimize_multi_seed(n_seeds=2, iterations=12, n_slices=32,
                                  lbfgs_polish=True, lbfgs_iters=10)
    return opt, res


def test_best_raw_param_present_and_shaped():
    opt, res = _quick_optimize("sigmoid")
    assert "best_raw_param" in res
    raw = res["best_raw_param"]
    assert raw.shape == (32, opt.n_channels)
    # Distinct from the smoothed envelope that best_waveform reports.
    assert res["best_waveform"].shape == raw.shape


@pytest.mark.parametrize("activation", ["sigmoid", "clamp"])
def test_raw_param_roundtrips_to_reported_fidelity(activation):
    # The whole point: feed best_raw_param straight back into the simulator and
    # recover best_fidelity exactly -- no second activation, no re-smoothing.
    opt, res = _quick_optimize(activation)
    raw = torch.as_tensor(res["best_raw_param"], device=DEVICE,
                          dtype=opt.rdtype).unsqueeze(0)
    rho = opt.simulate_choi_batch(raw, dt=1.0)
    f = float(opt._process_fidelity(rho).item())
    assert abs(f - res["best_fidelity"]) < 1e-5, (
        f"round-trip gap {abs(f - res['best_fidelity']):.2e} "
        f"(polished={res['lbfgs_polished']})")


def test_best_waveform_is_smoothed_raw_no_double_activation():
    # best_waveform must equal smoothed_waveform(best_raw_param): a SINGLE
    # activation+smooth of the raw param. Under the old L-BFGS bug it was
    # smoothed_waveform(sigmoid(x)) -- a second sigmoid -- which this catches
    # whenever the polish improves on Adam (sigmoid activation).
    opt, res = _quick_optimize("sigmoid")
    raw = torch.as_tensor(res["best_raw_param"], device=DEVICE,
                          dtype=opt.rdtype)
    wf_from_raw = opt.smoothed_waveform(raw, dt=1.0).cpu().numpy()
    assert np.allclose(res["best_waveform"], wf_from_raw, atol=1e-6, rtol=0.0)


def test_raw_param_feeds_dt_convergence_exactly():
    # best_raw_param is precisely dt_convergence's expected input, so its order-1
    # value at dt=1 ns reproduces best_fidelity (same pulse, dt, order, precision).
    opt, res = _quick_optimize("sigmoid")
    raw = torch.as_tensor(res["best_raw_param"], device=DEVICE, dtype=opt.rdtype)
    rep = opt.dt_convergence(raw, dt=1.0, refinements=(1, 2))
    assert abs(rep["order1"][0] - res["best_fidelity"]) < 1e-5


# ==========================================================================
# 2. Configurable double precision
# ==========================================================================
def test_precision_validation_and_dtypes():
    o1 = ParametricCZOptimizer(_profile(), precision="single")
    assert (o1.cdtype, o1.rdtype) == (torch.complex64, torch.float32)
    o2 = ParametricCZOptimizer(_profile(), precision="double")
    assert (o2.cdtype, o2.rdtype) == (torch.complex128, torch.float64)
    with pytest.raises(ValueError, match="precision"):
        ParametricCZOptimizer(_profile(), precision="quad")


def test_single_precision_is_byte_for_byte_default():
    # Default == explicit 'single': the precision plumbing must not perturb the
    # original complex64/float32 path at all.
    torch.manual_seed(1)
    u = torch.rand(1, 20, 4, device=DEVICE)
    rd = ParametricCZOptimizer(_profile(), n_channels=4).simulate_gradient_batch(u, dt=1.0)
    rs = ParametricCZOptimizer(_profile(), n_channels=4,
                               precision="single").simulate_gradient_batch(u, dt=1.0)
    assert torch.equal(rd, rs)


def _order1_diffs(precision, diss):
    """Successive |F(dt/2) - F(dt)| for the first-order step under deep dt
    refinement, isolated integrator, constant pulse (fully deterministic).

    Uses the cheap state-average metric: this probes the integrator's dt scaling,
    which is metric-independent, and the state average is the simplest linear
    functional of the evolved density matrices (the exact _process_fidelity would
    give identical refinement ratios but needs the 16-operator stack)."""
    o = ParametricCZOptimizer(_profile(), n_channels=3, bandwidth_mhz=0.0,
                              use_drag=False, precision=precision)
    o.step_order = 1
    u = _const_pulse([0.55, -0.55, 0.6], 16, dtype=o.rdtype)
    fids = []
    with torch.no_grad():
        for k in (16, 32, 64, 128):
            ur = torch.repeat_interleave(u, k, dim=1)
            rho = o.simulate_gradient_batch(ur, dt=1.0 / k, diss_scale=diss)
            fids.append(float(o._avg_state_fidelity(rho).mean().item()))
    return [abs(fids[i + 1] - fids[i]) for i in range(len(fids) - 1)]


def test_double_precision_resolves_fine_dt_where_single_floors():
    # Amplified dissipation puts the integrator in the regime where the O(dt)
    # Euler error dominates. In double precision the successive refinement
    # differences halve cleanly (ratio ~0.5); in single precision they cannot --
    # they sit on / grow off the ~1e-6 complex64 floor (ratios well above 0.5).
    diss = 50.0
    d_double = _order1_diffs("double", diss)
    d_single = _order1_diffs("single", diss)
    r_double = [d_double[i + 1] / d_double[i] for i in range(len(d_double) - 1)]
    r_single = [d_single[i + 1] / d_single[i] for i in range(len(d_single) - 1)]
    # Clean O(dt) halving in double: every refinement ratio sits near 0.5.
    assert all(0.4 < r < 0.6 for r in r_double), f"double not halving: {r_double}"
    # Single does NOT halve cleanly -- on the ~1e-6 complex64 floor its diffs are
    # round-off noise, so its ratios are erratic (some >1 as the diff regrows, some
    # <0.5), never the clean ~0.5 of true O(dt) convergence. Asserting "not clean
    # halving" is the robust contrast; the exact ratios depend on the float32
    # round-off realization and must not be pinned (an earlier `max(r_double) <
    # min(r_single)` over-fit one such realization and broke when batching the
    # matrix_exp shifted the float32 fingerprint -- same physics, different noise).
    assert not all(0.4 < r < 0.6 for r in r_single), (
        f"single unexpectedly halves cleanly (not flooring?): single={r_single}")
    # Physically: double's refinement differences keep shrinking toward 0; single's
    # finest difference stays stuck on the round-off floor, ABOVE where double
    # reaches -- i.e. single cannot resolve the fine-dt continuum that double does.
    assert d_double[-1] < d_double[0]
    assert d_single[-1] > d_double[-1]


def test_double_precision_runs_optimizer():
    # End-to-end: the double path actually optimizes (dtype plumbing is complete
    # through warm-start, smoother, DRAG, fidelity, and both step orders).
    # use_drag=True + step_order=2 exercises the DRAG and Strang branches in
    # complex128 -- a float32 tensor leaking into either would raise here.
    opt = ParametricCZOptimizer(_profile(), n_channels=4, activation="sigmoid",
                                use_drag=True, drag_order=2,
                                precision="double", step_order=2)
    res = opt.optimize_multi_seed(n_seeds=1, iterations=8, n_slices=24,
                                  lbfgs_polish=True, lbfgs_iters=5)
    assert 0.0 <= res["best_fidelity"] <= 1.0
    assert res["best_waveform"].dtype == np.float64
    assert res["best_raw_param"].dtype == np.float64


# ==========================================================================
# 3. Coupling rolloff with detuning  g_eff(delta) = g_max / sqrt(1 + (delta/kappa)^2)
# ==========================================================================
def test_g_linewidth_validation_and_default_off():
    assert ParametricCZOptimizer(_profile(), n_channels=4).G_LINEWIDTH is None
    o = ParametricCZOptimizer(_profile(), n_channels=4,
                              coupler_phase_mode="frequency",
                              coupler_g_linewidth_mhz=20.0)
    assert o.G_LINEWIDTH is not None
    assert math.isclose(o.G_LINEWIDTH, 2 * math.pi * 20.0 / 1000.0, rel_tol=1e-9)
    with pytest.raises(ValueError, match="coupler_g_linewidth_mhz"):
        ParametricCZOptimizer(_profile(), n_channels=4,
                              coupler_phase_mode="frequency",
                              coupler_g_linewidth_mhz=-5.0)


def _detuned_pulse():
    # Coupler envelope on, channel-4 pinned to max detuning (u4=1 -> delta=+DELTA_MAX).
    p = torch.full((1, 24, 4), 0.5, device=DEVICE)
    p[..., 2] = 0.9
    p[..., 3] = 1.0
    return p


def _sim_with_linewidth(lw, pulse):
    kw = dict(n_channels=4, coupler_phase_mode="frequency", delta_max_mhz=30.0)
    if lw is not None:
        kw["coupler_g_linewidth_mhz"] = lw
    return ParametricCZOptimizer(_profile(), **kw).simulate_gradient_batch(pulse, dt=1.0)


def test_g_rolloff_weakens_coupling_monotonically_in_linewidth():
    # Smaller linewidth kappa => stronger rolloff at a fixed detuning => the
    # dynamics deviate more from the peak-g (no-rolloff) baseline.
    pulse = _detuned_pulse()
    base = _sim_with_linewidth(None, pulse)            # peak g (legacy)
    d10 = (_sim_with_linewidth(10.0, pulse) - base).abs().max().item()
    d30 = (_sim_with_linewidth(30.0, pulse) - base).abs().max().item()
    assert d10 > d30 > 1e-4, f"not monotonic in linewidth: d10={d10:.2e} d30={d30:.2e}"


def test_g_rolloff_recovers_peak_at_large_linewidth():
    # kappa -> infinity => g_scale -> 1 => identical to the no-rolloff baseline.
    pulse = _detuned_pulse()
    base = _sim_with_linewidth(None, pulse)
    d_big = (_sim_with_linewidth(1e6, pulse) - base).abs().max().item()
    assert d_big < 1e-6, f"large-linewidth limit didn't recover peak g: {d_big:.2e}"


def test_g_rolloff_ignored_in_phase_mode():
    # 'phase' mode has no detuning, so a linewidth is a documented no-op there.
    torch.manual_seed(3)
    u = torch.rand(1, 20, 4, device=DEVICE)
    o_plain = ParametricCZOptimizer(_profile(), n_channels=4,
                                    coupler_phase_mode="phase")
    o_lw = ParametricCZOptimizer(_profile(), n_channels=4,
                                 coupler_phase_mode="phase",
                                 coupler_g_linewidth_mhz=10.0)
    assert torch.equal(o_plain.simulate_gradient_batch(u, dt=1.0),
                       o_lw.simulate_gradient_batch(u, dt=1.0))


def test_g_rolloff_gradient_flows_through_detuning():
    # The rolloff is differentiable in the detuning, so the optimizer can trade
    # coupling strength against drive frequency.
    o = ParametricCZOptimizer(_profile(), n_channels=4,
                              coupler_phase_mode="frequency",
                              coupler_g_linewidth_mhz=15.0)
    u = torch.rand(1, 20, 4, device=DEVICE, requires_grad=True)
    loss = 1.0 - o._process_fidelity(o.simulate_choi_batch(u, dt=1.0)).mean()
    loss.backward()
    assert u.grad[..., 3].abs().sum().item() > 0.0


if __name__ == "__main__":
    test_best_raw_param_present_and_shaped()
    test_raw_param_roundtrips_to_reported_fidelity("sigmoid")
    test_raw_param_roundtrips_to_reported_fidelity("clamp")
    test_best_waveform_is_smoothed_raw_no_double_activation()
    test_raw_param_feeds_dt_convergence_exactly()
    test_precision_validation_and_dtypes()
    test_single_precision_is_byte_for_byte_default()
    test_double_precision_resolves_fine_dt_where_single_floors()
    test_double_precision_runs_optimizer()
    test_g_linewidth_validation_and_default_off()
    test_g_rolloff_weakens_coupling_monotonically_in_linewidth()
    test_g_rolloff_recovers_peak_at_large_linewidth()
    test_g_rolloff_ignored_in_phase_mode()
    test_g_rolloff_gradient_flows_through_detuning()
    print("residual-resolution tests passed")
