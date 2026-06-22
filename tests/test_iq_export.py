"""Tests for the complete-I/Q export (DRAG quadrature baked in) and the
vendor-neutral OpenPulse 3.0 / OpenQASM 3 exporter.

The point of iq_waveform is that an exported pulse is COMPLETE: with DRAG the
simulator drives a derived quadrature (Y) tone that the real [0,1] envelope omits,
so an export built from that envelope is missing its imaginary part. These tests
pin (a) that iq_waveform carries exactly the in-phase + quadrature the simulator
applied, and (b) that the OpenPulse text export is both valid (it re-parses with an
independent parser) and lossless (samples survive), including the complex part.
"""
import numpy as np
import pytest
import torch

import gradpulse as gp
from gradpulse import openpulse_export as ope
from gradpulse.parametric import DEVICE


def _opt(use_drag, n_channels=3):
    return gp.ParametricCZOptimizer(use_drag=use_drag, drag_order=2,
                                    n_channels=n_channels, activation="sigmoid")


# ---- iq_waveform faithfulness -------------------------------------------------
def test_iq_drag_off_has_zero_drive_quadrature():
    opt = _opt(use_drag=False, n_channels=3)
    x = torch.randn(80, 3, device=DEVICE, dtype=opt.rdtype)
    iq = opt.iq_waveform(x, dt=1.0)["iq"]
    # q1, q2 are the two drive channels; coupler is ch2 (real in 3-channel mode).
    assert np.max(np.abs(iq[:, 0].imag)) < 1e-6
    assert np.max(np.abs(iq[:, 1].imag)) < 1e-6
    assert np.max(np.abs(iq[:, 2].imag)) < 1e-6   # 3-channel coupler is real


def test_iq_inphase_matches_smoothed_waveform_rescaled():
    """With DRAG off, real(drive) == signed-smoothed envelope * OMEGA_MAX."""
    opt = _opt(use_drag=False, n_channels=3)
    x = torch.randn(80, 3, device=DEVICE, dtype=opt.rdtype)
    iq = opt.iq_waveform(x, dt=1.0)["iq"]
    wf01 = opt.smoothed_waveform(x, dt=1.0).cpu().numpy()      # [n,3] in [0,1]
    signed = 2.0 * wf01 - 1.0
    assert np.allclose(iq[:, 0].real, signed[:, 0] * opt.OMEGA_MAX, atol=1e-5)
    assert np.allclose(iq[:, 1].real, signed[:, 1] * opt.OMEGA_MAX, atol=1e-5)


def test_iq_drag_quadrature_matches_analytic_motzoi():
    """With DRAG on, imag(q1) == -d/dt(in-phase)/alpha * OMEGA, the Motzoi tone."""
    opt = _opt(use_drag=True, n_channels=3)
    x = torch.randn(80, 3, device=DEVICE, dtype=opt.rdtype)
    dt = 1.0
    iq = opt.iq_waveform(x, dt=dt)["iq"]
    # Independent re-derivation from the signed smoothed in-phase envelope.
    wf01 = opt.smoothed_waveform(x, dt=dt).cpu().numpy()
    u1 = 2.0 * wf01[:, 0] - 1.0
    du1 = np.gradient(u1, dt)                                  # central difference
    v1_expected = -du1 * opt.OMEGA_MAX / opt._alpha1
    # Match the interior (edge derivative conventions differ slightly).
    assert np.max(np.abs(iq[:, 0].imag[2:-2] - v1_expected[2:-2])) < 5e-3
    assert np.max(np.abs(iq[:, 0].imag)) > 1e-3                # genuinely nonzero


def test_cr_iq_carries_drag_quadrature():
    cr = gp.CrossResonanceZXOptimizer(use_drag=True)
    x = torch.randn(150, cr.n_channels, device=DEVICE, dtype=cr.rdtype)
    iq = cr.iq_waveform(x, dt=1.0)
    assert "control_drive" in iq["labels"]
    assert np.max(np.abs(iq["iq"][:, 0].imag)) > 1e-3          # DRAG quadrature present


# ---- OpenPulse export ---------------------------------------------------------
pytestmark_parser = pytest.importorskip("openpulse", reason="needs an OpenPulse parser")


def test_openpulse_real_envelope_roundtrips():
    wf = np.random.RandomState(0).rand(48, 3)
    err = ope.verify_openpulse_roundtrip(wf, dt_ns=1.0, gate_name="grad_cz")
    assert err < 1e-9


def test_openpulse_complex_iq_roundtrips_and_parses():
    opt = _opt(use_drag=True, n_channels=3)
    x = torch.randn(64, 3, device=DEVICE, dtype=opt.rdtype)
    iq = opt.iq_waveform(x, dt=1.0)
    assert np.max(np.abs(iq["iq"].imag)) > 1e-3                # there IS a quadrature
    err = ope.verify_openpulse_roundtrip(iq, dt_ns=1.0, gate_name="grad_cz", qubits=(4, 5))
    assert err < 1e-9, f"complex I/Q did not survive export: {err}"


def test_openpulse_program_is_valid_openpulse3():
    """The emitted text parses with the INDEPENDENT openpulse parser."""
    from openpulse import parse
    wf = np.random.RandomState(1).rand(20, 2) + 0.1j * np.random.RandomState(2).rand(20, 2)
    program = ope.to_openpulse_program(wf, dt_ns=1.0, gate_name="grad_zx", qubits=(0, 1))
    tree = parse(program)                                      # raises if invalid
    assert any(type(s).__name__ == "CalibrationStatement" for s in tree.statements)
    assert 'defcalgrammar "openpulse"' in program


def test_openpulse_readiness_report_flags_iq():
    opt = _opt(use_drag=True, n_channels=3)
    x = torch.randn(40, 3, device=DEVICE, dtype=opt.rdtype)
    iq = opt.iq_waveform(x, dt=1.0)
    rep = ope.openpulse_readiness_report(iq, dt_ns=1.0, verbose=False)
    assert rep["roundtrip_faithful"]
    assert rep["carries_iq_quadrature"]
    assert rep["n_channels"] == 3


def test_qiskit_pulse_path_errors_cleanly_on_qiskit2():
    """qiskit.pulse was removed in Qiskit 2.0 -- the native path must fail with a
    helpful message pointing at the OpenPulse text exporter, not a cryptic error."""
    pytest.importorskip("qiskit")
    import qiskit
    if tuple(int(p) for p in qiskit.__version__.split(".")[:1]) < (2,):
        pytest.skip("qiskit < 2.0 still has qiskit.pulse")
    wf = np.random.rand(10, 2)
    with pytest.raises(ImportError, match="to_openpulse_program"):
        ope.to_qiskit_schedule(wf)
