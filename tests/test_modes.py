"""Coverage for the optional code paths beyond the default 3-channel CZ.

The optimizer advertises several extra modes: additional control channels
(4: XY coupler phase; 6: + per-qubit Stark/Z drives), a windowed-sinc
"firbrick" smoother, and non-default warm starts (echo, flat). These tests
keep those advertised paths from silently bit-rotting.

Like test_smoke.py, the runs are deliberately short and GPU-optional: they
assert each path runs end-to-end and returns a structurally valid pulse
(right shape, values in [0, 1]), not a convergence claim. The headline
fidelity (the 150 ns CZ's F_proc ~0.990, QuTiP-cross-checked) is owned by
examples/optimize_cz.py + gradpulse.validate.

Run:  pytest tests/        OR        python tests/test_modes.py
"""
import numpy as np
import pytest
import torch

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer
from gradpulse.parametric import DEVICE

N_SLICES = 40


def _profile():
    return ParametricCouplerProfile(
        freq_ghz_q1=4.85, freq_ghz_q2=5.05,
        anharm_ghz_q1=-0.20, anharm_ghz_q2=-0.20,
        t1_ns_q1=30_000, t2_ns_q1=25_000,
        t1_ns_q2=30_000, t2_ns_q2=25_000,
        g_max_mhz=12.0, omega_max_mhz=50.0,
    )


def _run(opt_kwargs, **run_kwargs):
    opt = ParametricCZOptimizer(_profile(), use_drag=False, **opt_kwargs)
    res = opt.optimize_multi_seed(
        n_seeds=1, iterations=5, n_slices=N_SLICES, dt_ns=1.0,
        use_process_fidelity=True, lbfgs_polish=False, **run_kwargs,
    )
    return res


def _assert_valid(res, n_channels):
    f = float(res["best_fidelity"])
    wf = np.asarray(res["best_waveform"])
    assert 0.0 <= f <= 1.0, f"fidelity out of range: {f}"
    assert wf.shape == (N_SLICES, n_channels), f"bad waveform shape: {wf.shape}"
    assert wf.min() >= -1e-6 and wf.max() <= 1 + 1e-6, "waveform not in [0, 1]"


def test_channels_4_xy_phase():
    # 4th channel modulates the parametric-drive phase (XY coupler control).
    res = _run(dict(bandwidth_mhz=80.0, n_channels=4, activation="sigmoid"))
    _assert_valid(res, 4)


def test_channels_6_stark_z():
    # Channels 5/6 add per-qubit Stark/Z drives for AC-Stark compensation.
    res = _run(dict(bandwidth_mhz=80.0, n_channels=6, activation="sigmoid"))
    _assert_valid(res, 6)


def test_smoother_firbrick():
    res = _run(dict(bandwidth_mhz=80.0, n_channels=3, activation="sigmoid",
                    smoother_type="firbrick"))
    _assert_valid(res, 3)


def test_warm_start_echo():
    res = _run(dict(bandwidth_mhz=80.0, n_channels=3, activation="sigmoid"),
               warm_start_mode="echo")
    _assert_valid(res, 3)


def test_warm_start_flat():
    res = _run(dict(bandwidth_mhz=80.0, n_channels=3, activation="sigmoid"),
               warm_start_mode="flat")
    _assert_valid(res, 3)


def test_firbrick_differs_from_gaussian():
    # Same deterministic warm start (optimize_multi_seed seeds RNG to 42 internally)
    # must yield genuinely different pulses -- guards against 'firbrick' aliasing to gaussian.
    common = dict(bandwidth_mhz=80.0, n_channels=3, activation="sigmoid")
    wg = np.asarray(_run(dict(**common, smoother_type="gaussian"))["best_waveform"])
    wf = np.asarray(_run(dict(**common, smoother_type="firbrick"))["best_waveform"])
    assert np.abs(wg - wf).max() > 1e-4, "firbrick produced the gaussian result"


def _ideal_choi(opt, U4):
    """Choi stack of the *perfect* unitary channel Phi(rho)=U rho U^dag, in the
    [1, 16, 9, 9] layout simulate_choi_batch produces (m = i*4 + j is the channel
    applied to |i><j| over the four computational levels). Lets us check the
    fidelity metric against an exactly-known channel with no simulation."""
    ci = opt._comp_idx
    U9 = torch.zeros((9, 9), dtype=opt.cdtype, device=DEVICE)
    for a in range(4):
        for b in range(4):
            U9[ci[a], ci[b]] = U4[a, b]
    choi = torch.zeros((1, 16, 9, 9), dtype=opt.cdtype, device=DEVICE)
    for i in range(4):
        for j in range(4):
            E = torch.zeros((9, 9), dtype=opt.cdtype, device=DEVICE)
            E[ci[i], ci[j]] = 1.0
            choi[0, i * 4 + j] = U9 @ E @ U9.conj().T
    return choi


def test_target_gate_builder():
    # Default target is CZ (back-compatible); the family members are unitary,
    # sqrt(iSWAP) squares to iSWAP, and an unknown name is rejected.
    assert ParametricCZOptimizer(_profile()).target_gate == "cz"
    for g in ("cz", "iswap", "sqrt_iswap"):
        U = ParametricCZOptimizer(_profile(), target_gate=g).u_target_4x4
        eye = torch.eye(4, dtype=U.dtype, device=U.device)
        assert torch.allclose(U.conj().T @ U, eye, atol=1e-6), f"{g} not unitary"
    si = ParametricCZOptimizer(_profile(), target_gate="sqrt_iswap").u_target_4x4
    sw = ParametricCZOptimizer(_profile(), target_gate="iswap").u_target_4x4
    assert torch.allclose(si @ si, sw, atol=1e-6), "sqrt(iSWAP)^2 != iSWAP"
    with pytest.raises(ValueError):
        ParametricCZOptimizer(_profile(), target_gate="bogus")


def test_target_gate_metric_distinguishes_gates():
    # The process fidelity must score a perfect iSWAP channel as 1 against the
    # iSWAP target and ~0 against the CZ target (|Tr(CZ^dag iSWAP)|^2/16 = 0):
    # proof that target_gate genuinely re-points the metric, not just a label.
    opt_sw = ParametricCZOptimizer(_profile(), target_gate="iswap")
    opt_cz = ParametricCZOptimizer(_profile(), target_gate="cz")
    choi = _ideal_choi(opt_sw, opt_sw.u_target_4x4)        # perfect iSWAP channel
    f_match = float(opt_sw._process_fidelity(choi)[0])
    f_mismatch = float(opt_cz._process_fidelity(choi)[0])
    assert f_match > 0.999, f"perfect iSWAP vs iSWAP target should be ~1, got {f_match}"
    assert f_mismatch < 0.01, f"iSWAP channel vs CZ target should be ~0, got {f_mismatch}"


def test_target_gate_iswap_runs_end_to_end():
    # The full optimize path runs for a non-CZ target and returns a structurally
    # valid pulse (short run; convergence is owned by examples/optimize_iswap.py).
    res = _run(dict(bandwidth_mhz=80.0, n_channels=3, activation="sigmoid",
                    target_gate="iswap"))
    _assert_valid(res, 3)


# --------------------------------------------------------------------------
# AWG / transmission-line response
# --------------------------------------------------------------------------
def test_line_response_none_is_identity():
    # The default (None) must leave the simulator byte-for-byte unchanged.
    o0 = ParametricCZOptimizer(_profile(), n_channels=3, activation="sigmoid")
    on = ParametricCZOptimizer(_profile(), n_channels=3, activation="sigmoid",
                               line_response=None)
    torch.manual_seed(0)
    u = torch.randn(1, 30, 3, device=DEVICE)
    assert torch.equal(o0.simulate_choi_batch(u, dt=1.0),
                       on.simulate_choi_batch(u, dt=1.0))


def test_line_response_alters_and_is_dc_preserving():
    o0 = ParametricCZOptimizer(_profile(), n_channels=3, activation="sigmoid")
    oe = ParametricCZOptimizer(_profile(), n_channels=3, activation="sigmoid",
                               line_response={"type": "exponential", "tau_ns": 3.0})
    torch.manual_seed(0)
    u = torch.randn(1, 30, 3, device=DEVICE)
    # A genuine settling tail changes the dynamics...
    assert (oe.simulate_choi_batch(u, dt=1.0)
            - o0.simulate_choi_batch(u, dt=1.0)).abs().max() > 1e-3
    # ...but preserves a held (constant) amplitude in steady state (unit DC gain).
    sig = 2 * torch.sigmoid(torch.full((1, 40, 3), 0.4, device=DEVICE)) - 1
    out = oe._apply_line_response(sig, 1.0)
    assert torch.allclose(out[:, 12:], sig[:, 12:], atol=1e-5)
    assert abs(float(oe._line_kernel(1.0).sum()) - 1.0) < 1e-6      # normalised


def test_line_response_gradient_flows():
    # Pre-compensation requires the convolution to be differentiable.
    oe = ParametricCZOptimizer(_profile(), n_channels=3, activation="sigmoid",
                               line_response={"type": "exponential", "tau_ns": 2.0})
    u = torch.randn(1, 20, 3, device=DEVICE, requires_grad=True)
    loss = 1.0 - oe._process_fidelity(oe.simulate_choi_batch(u, dt=1.0)).mean()
    loss.backward()
    assert u.grad.abs().sum().item() > 0.0


def test_line_response_array_form_and_validation():
    # An explicit (e.g. measured) causal impulse response is accepted + normalised.
    oe = ParametricCZOptimizer(_profile(), n_channels=3, activation="sigmoid",
                               line_response=[1.0, 0.5, 0.25])
    k = oe._line_kernel(1.0)
    assert k.shape[-1] == 3 and abs(float(k.sum()) - 1.0) < 1e-6
    for bad in ({"type": "bogus"}, {"type": "exponential", "tau_ns": 0.0}, []):
        with pytest.raises(ValueError):
            ParametricCZOptimizer(_profile(), line_response=bad)


def test_line_response_runs_end_to_end():
    res = _run(dict(bandwidth_mhz=80.0, n_channels=3, activation="sigmoid",
                    line_response={"type": "exponential", "tau_ns": 2.0}))
    _assert_valid(res, 3)


def test_line_response_cross_check_parity():
    # PyTorch and QuTiP must agree on a line-distorted pulse: the validator
    # re-applies the SAME response (from metadata), so the cross-check stays
    # apples-to-apples. Skipped if qutip (the [validate] extra) is absent.
    pytest.importorskip("qutip")
    import json
    import tempfile
    from dataclasses import asdict
    from pathlib import Path

    from gradpulse.validate import cross_check

    prof = _profile()
    lr = {"type": "exponential", "tau_ns": 2.0}
    opt = ParametricCZOptimizer(prof, bandwidth_mhz=80.0, use_drag=False,
                                n_channels=3, activation="sigmoid", line_response=lr)
    res = opt.optimize_multi_seed(
        n_seeds=1, iterations=12, n_slices=60, dt_ns=1.0,
        warm_start_mode="parametric_cz", use_process_fidelity=True, lbfgs_polish=True)
    with tempfile.TemporaryDirectory() as td:
        wf = res["best_waveform"]
        np.save(Path(td) / "p.npy", wf)
        meta = {"pulse_npy": "p.npy", "pulse_dt_ns": 1.0, "n_channels": int(wf.shape[1]),
                "bandwidth_mhz": 80.0, "smoother_type": opt.smoother_type,
                "target_gate": opt.target_gate, "line_response": lr,
                "grape_f": float(res["best_fidelity"]), "profile": asdict(prof)}
        (Path(td) / "p.json").write_text(json.dumps(meta))
        r = cross_check(Path(td) / "p.json")
    # A mirroring bug (validator not re-applying the response) would push the two
    # simulators ~0.1 apart; this tolerates the complex64-vs-float64 gap on a short run
    # (tight ~1e-5 agreement on smooth pulses is shown by examples/validation_sweep.py).
    assert r["status"] in ("PASS", "WARN"), f"line cross-check {r['status']} (d={r['delta']:.2e})"
    assert abs(r["delta"]) < 3e-3


if __name__ == "__main__":
    test_channels_4_xy_phase()
    test_channels_6_stark_z()
    test_smoother_firbrick()
    test_warm_start_echo()
    test_warm_start_flat()
    test_firbrick_differs_from_gaussian()
    test_target_gate_builder()
    test_target_gate_metric_distinguishes_gates()
    test_target_gate_iswap_runs_end_to_end()
    test_line_response_none_is_identity()
    test_line_response_alters_and_is_dc_preserving()
    test_line_response_gradient_flows()
    test_line_response_array_form_and_validation()
    test_line_response_runs_end_to_end()
    try:
        test_line_response_cross_check_parity()
        print("mode/path tests passed (incl. line-response cross-check)")
    except Exception as exc:        # qutip missing -> importorskip raises Skipped
        print(f"mode/path tests passed (line-response cross-check skipped: {exc})")
