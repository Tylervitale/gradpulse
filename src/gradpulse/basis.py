"""gradpulse.basis -- band-limited control parameterizations (CRAB / Fourier).

The default optimizer parameterizes the control as one free value per time slice
(piecewise-constant), then band-limits it AFTER the fact with a Gaussian/FIR smoother
and discourages out-of-band energy with an FFT penalty. That works, but the smoother
only *attenuates* content above the cutoff, and the penalty fights the optimizer.

A spectral (Fourier / CRAB-style) parameterization instead makes the control a sum of
sinusoids whose frequencies are ALL below the cutoff -- so the pulse is band-limited
*by construction*. Benefits a hardware-minded user cares about:
  * Fewer, better-conditioned parameters: ~2*f_max*T coefficients per channel instead
    of n_slices, so the optimization lives in a much smaller, smoother landscape.
  * No out-of-band energy to penalize: the basis cannot represent it, so the smoother
    and the anti-cheating FFT penalty become unnecessary.
  * Natively hardware-friendly: the emitted envelope already respects the AWG/line
    bandwidth without a post-hoc filter that the QuTiP cross-check must replicate.

``FourierBasis`` is the synthesis map: a fixed [n_slices, n_basis] real matrix whose
columns are a DC term plus cos/sin pairs at the harmonics of 1/T up to ``f_max``.
``ParametricCZOptimizer.optimize_spectral`` optimizes the coefficients and reports the
measured out-of-band energy, so the band-limiting is verified, not just asserted.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


class FourierBasis:
    """Band-limited synthesis basis: control(t) = Phi(t) @ coeffs, every basis
    frequency <= f_max, so the synthesized control is band-limited by construction.

    Columns of Phi: a constant (DC) term, then for each harmonic k = 1..K a
    cos(2*pi*k*f0*t) and a sin(2*pi*k*f0*t) pair, where f0 = 1/T is the fundamental
    (T = n_slices*dt) and K = floor(f_max / f0). So n_basis = 2*K + 1.

    Parameters
    ----------
    n_slices, dt_ns : the time grid (must match the simulator's).
    f_max_mhz : spectral cutoff (MHz). Every component sits at or below it. Default
        None uses the full slice-Nyquist 1/(2*dt); pass your AWG/line bandwidth to
        guarantee a hardware-respecting pulse.
    n_harmonics : override K directly (otherwise derived from f_max_mhz). Capped at
        the Nyquist limit so the basis never aliases.
    """

    def __init__(self, n_slices: int, dt_ns: float = 1.0,
                 f_max_mhz: Optional[float] = None, n_harmonics: Optional[int] = None,
                 dtype=torch.float32, device=None):
        self.n_slices = int(n_slices)
        self.dt_ns = float(dt_ns)
        T = self.n_slices * self.dt_ns                       # ns
        f0_cyc = 1.0 / T                                     # cycles/ns (fundamental)
        nyq_cyc = 0.5 / self.dt_ns                           # slice Nyquist (cycles/ns)
        if n_harmonics is not None:
            K = int(n_harmonics)
        else:
            fmax_cyc = (nyq_cyc if f_max_mhz is None else float(f_max_mhz) / 1000.0)
            K = int(math.floor(fmax_cyc / f0_cyc))
        K = max(1, min(K, int(math.floor(nyq_cyc / f0_cyc)) - 1))   # below Nyquist
        self.n_harmonics = K
        self.f_max_mhz = (1000.0 * K * f0_cyc)
        t = (np.arange(self.n_slices) + 0.5) * self.dt_ns    # slice-midpoint times
        cols = [np.ones(self.n_slices)]                      # DC
        freqs = [0.0]
        for k in range(1, K + 1):
            w = 2.0 * math.pi * (k * f0_cyc)
            cols.append(np.cos(w * t))
            cols.append(np.sin(w * t))
            freqs += [k * f0_cyc, k * f0_cyc]
        Phi = np.stack(cols, axis=1)                         # [n_slices, 2K+1]
        self._device = device
        self.Phi = torch.as_tensor(Phi, dtype=dtype, device=device)
        self.frequencies_mhz = np.asarray(freqs) * 1000.0
        self.n_basis = self.Phi.shape[1]

    def synthesize(self, coeffs: torch.Tensor) -> torch.Tensor:
        """coeffs [..., n_basis, n_channels] -> control [..., n_slices, n_channels]."""
        if coeffs.shape[-2] != self.n_basis:
            raise ValueError(f"coeffs second-to-last dim must be n_basis={self.n_basis}, "
                             f"got {coeffs.shape[-2]}")
        Phi = self.Phi.to(coeffs.dtype)
        return torch.einsum('tb,...bc->...tc', Phi, coeffs)

    def dc_coeff_for_level(self, level: float) -> float:
        """The DC coefficient that synthesizes a constant ``level`` (Phi[:,0]==1)."""
        return float(level)

    def __repr__(self):
        return (f"FourierBasis(n_slices={self.n_slices}, dt_ns={self.dt_ns}, "
                f"n_harmonics={self.n_harmonics}, f_max~{self.f_max_mhz:.1f} MHz, "
                f"n_basis={self.n_basis})")
