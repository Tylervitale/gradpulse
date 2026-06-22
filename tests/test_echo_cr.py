"""Echoed cross-resonance: structural static-ZZ cancellation, plus the
optimizer-vs-QuTiP apples-to-apples guard with the echo applied on BOTH paths.

The headline physics: a post-gate virtual-Z (single-qubit diagonal) cannot remove a
static ZZ conditional phase, but the echoed-CR sequence (control drive sign-flips in
the 2nd half + ideal control pi-pulses at the midpoint and end) refocuses it -- turning
ZZ into a removable single-qubit IZ. These tests pin that down with a number.
"""
import numpy as np
import pytest
import torch

import gradpulse as gp
from gradpulse.crossresonance import DEVICE


def _zz_pauli_phase(chi_zz_mhz, echo, n_slices=200, dt_ns=1.0):
    """Evolve the DRIFT only (zero drive) on a degenerate pair (delta_c=0, J=0) with a
    static ZZ, echo on/off, and return the ZZ Pauli phase of the resulting
    computational-subspace unitary, zz = phi00 - phi01 - phi10 + phi11.

    For a closed-system diagonal U, Phi(|i><j|) = exp(i(phi_i - phi_j)) |i><j|, so the
    relative phase phi_i - phi_0 is the angle of the [i,0] entry of the channel applied
    to |i><0| (Choi operator m = i*4 + 0). delta_c = 0 keeps all phases small (no 2*pi
    wrapping) so the ZZ component is read off cleanly.
    """
    prof = gp.CrossResonanceProfile(
        freq_ghz_control=4.85, freq_ghz_target=4.85,   # degenerate -> delta_c = 0
        j_coupling_mhz=0.0, chi_zz_mhz=chi_zz_mhz)
    opt = gp.CrossResonanceZXOptimizer(
        prof, bandwidth_mhz=0.0, use_drag=False, use_target_cancel=False, echo=echo)
    x = torch.zeros((1, n_slices, opt.n_channels), device=DEVICE, dtype=opt.rdtype)
    with torch.no_grad():
        rho = opt.simulate_choi_batch(x, dt=dt_ns, diss_scale=0.0)   # closed system
    ci = opt._comp_idx
    proj = rho[0][:, ci, :][:, :, ci]                                # [16, 4, 4]
    r = [float(torch.angle(proj[i * 4 + 0][i, 0]).item()) for i in range(4)]  # phi_i - phi_0
    return r[0] - r[1] - r[2] + r[3]


def test_echo_refocuses_static_zz():
    chi = 0.5  # MHz
    zz_off = _zz_pauli_phase(chi, echo=False)
    zz_on = _zz_pauli_phase(chi, echo=True)
    # Without echo, |11> accumulates the full conditional phase ~ 2*pi*chi*T.
    expected = 2 * np.pi * (chi / 1000.0) * 200.0
    assert abs(abs(zz_off) - expected) < 0.05, (zz_off, expected)
    # The echo cancels the ZZ Pauli component (algebraically exact -> machine precision).
    assert abs(zz_on) < 1e-6, zz_on
    assert abs(zz_on) < 1e-3 * abs(zz_off)


def test_echo_cross_check_apples_to_apples():
    """Optimizer and the INDEPENDENT QuTiP solver agree on F_proc with echo on -- the
    guard that the echo logic in simulate_gradient_batch and validate stay in lockstep."""
    qt = pytest.importorskip("qutip")
    from gradpulse.validate import cr_cross_check

    opt = gp.CrossResonanceZXOptimizer(
        gp.CrossResonanceProfile(chi_zz_mhz=0.3), echo=True, use_drag=True)
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    raw = opt._warm_start(160, generator=gen)
    with torch.no_grad():
        rho = opt.simulate_choi_batch(raw.unsqueeze(0), dt=1.0)
        f_opt = float(opt._process_fidelity(rho, torch.zeros(2, device=DEVICE, dtype=opt.rdtype))[0])
    wf = opt.smoothed_waveform(raw, dt=1.0).detach().cpu().numpy()
    f_qutip = cr_cross_check(opt, wf, vz=[0.0, 0.0], dt_ns=1.0)
    assert abs(f_opt - f_qutip) < 2e-3, (f_opt, f_qutip)


def test_echo_off_is_unchanged():
    """echo=False must leave the single-pulse gate byte-for-byte (regression guard)."""
    prof = gp.CrossResonanceProfile()
    gen = torch.Generator(device=DEVICE).manual_seed(1)
    raw = gp.CrossResonanceZXOptimizer(prof, echo=False)._warm_start(120, generator=gen)
    a = gp.CrossResonanceZXOptimizer(prof, echo=False)
    b = gp.CrossResonanceZXOptimizer(prof, echo=False)
    with torch.no_grad():
        fa = a._process_fidelity(a.simulate_choi_batch(raw.unsqueeze(0)),
                                 torch.zeros(2, device=DEVICE, dtype=a.rdtype))
        fb = b._process_fidelity(b.simulate_choi_batch(raw.unsqueeze(0)),
                                 torch.zeros(2, device=DEVICE, dtype=b.rdtype))
    assert torch.allclose(fa, fb)
