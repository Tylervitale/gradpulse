"""Band-limited (Fourier / CRAB) spectral parameterization tests.

The promise is "band-limited by construction": every basis frequency is below the
cutoff, so the synthesized control has no out-of-band energy. These tests pin that
property on the basis directly (fast) and that optimize_spectral produces a
band-limited, fewer-parameter pulse that cross-checks against QuTiP (slow).
"""
import numpy as np
import pytest
import torch

import gradpulse as gp
from gradpulse.basis import FourierBasis
from gradpulse.parametric import DEVICE


def test_basis_is_band_limited_by_construction():
    n, dt, fmax = 128, 1.0, 60.0
    basis = FourierBasis(n, dt, f_max_mhz=fmax)
    # synthesize a random control and confirm ~zero spectral energy above f_max
    g = torch.Generator().manual_seed(0)
    coeffs = torch.randn(basis.n_basis, 1, generator=g)
    u = basis.synthesize(coeffs).squeeze(-1).numpy()
    ac = u - u.mean()
    p = np.abs(np.fft.rfft(ac)) ** 2
    freqs = np.fft.rfftfreq(n, d=dt) * 1000.0           # MHz
    oob = p[freqs > basis.f_max_mhz + 1e-6].sum() / p.sum()
    assert oob < 1e-12, f"basis leaked {oob} above f_max"


def test_basis_size_and_frequencies():
    basis = FourierBasis(120, 1.0, f_max_mhz=80.0)
    assert basis.n_basis == 2 * basis.n_harmonics + 1
    assert basis.frequencies_mhz.max() <= 80.0 + 1e-6
    assert basis.frequencies_mhz[0] == 0.0               # DC term first


def test_synthesize_shapes():
    basis = FourierBasis(64, 1.0, n_harmonics=6)
    coeffs = torch.zeros(3, basis.n_basis, 4)
    out = basis.synthesize(coeffs)
    assert tuple(out.shape) == (3, 64, 4)


def test_out_of_band_fraction_helper():
    opt = gp.ParametricCZOptimizer(n_channels=3, activation="sigmoid")
    # A pure 25 MHz tone sits exactly on an FFT bin (3/120 ns = 25 MHz) so there is
    # no spectral leakage: all energy above 20 MHz, none above 50 MHz.
    n = 120
    t = np.arange(n) * 1.0
    u = 0.5 + 0.3 * np.cos(2 * np.pi * (3.0 / n) * np.arange(n))   # 25 MHz, on-bin
    u = np.stack([u, u, u], axis=1)
    assert opt.out_of_band_fraction(u, dt_ns=1.0, f_max_mhz=50.0) < 1e-9
    assert opt.out_of_band_fraction(u, dt_ns=1.0, f_max_mhz=20.0) > 0.9


@pytest.mark.slow
def test_optimize_spectral_is_bandlimited_and_fewer_params():
    prof = gp.ParametricCouplerProfile(g_max_mhz=12.0, omega_max_mhz=50.0)
    opt = gp.ParametricCZOptimizer(prof, bandwidth_mhz=80.0, n_channels=3,
                                   activation="sigmoid")
    res = opt.optimize_spectral(n_slices=120, dt_ns=1.0, n_seeds=2, iterations=150,
                                f_max_mhz=80.0, verbose=False)
    assert res["n_params"] < res["n_params_piecewise"]   # far fewer parameters
    assert res["out_of_band_fraction"] < 5e-3            # genuinely band-limited
    assert res["best_fidelity"] > 0.9
    # QuTiP cross-check: validator consumes the envelope directly (no re-smoothing)
    pytest.importorskip("qutip")
    from gradpulse import validate as V
    fq = V.qutip_f_proc(prof, res["best_waveform"], "cz", 1.0)
    assert abs(fq - res["best_fidelity"]) < 1e-3
