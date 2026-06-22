"""Analytic filter-function robustness: dephasing sensitivity to ANY noise PSD,
with NO Monte Carlo.

    python examples/filter_function.py

The filter function F(f) is the cheap, analytic complement to the (correct but
expensive) quasi_static / colored-noise Monte-Carlo sweeps: the dephasing infidelity
is 1 - F = sigma^2 * (band-weighted F), so you can overlay your device's measured
noise spectrum on F(f) without re-simulating. This script optimizes a CZ, prints the
filter curve, and confirms the analytic 1/f estimate agrees with the colored-noise
Monte Carlo (the validation that it is a faithful surrogate).
"""
import numpy as np
import torch

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer
from gradpulse.parametric import DEVICE

opt = ParametricCZOptimizer(ParametricCouplerProfile(), n_channels=3, activation="sigmoid")
res = opt.optimize_multi_seed(n_seeds=2, iterations=150, n_slices=120, dt_ns=1.0,
                              lbfgs_polish=True)
x = torch.tensor(res["best_raw_param"], device=DEVICE, dtype=opt.rdtype)
print(f"optimized F_proc (nominal) = {res['best_fidelity']:.6f}\n")

# The filter function F(f): the per-frequency dephasing sensitivity.
ff = opt.filter_function(x, dt=1.0, f_max_mhz=60.0, n_freq=200)
print(f"quasi-static value F(0) = {ff['F0']:.1f} ns^2   "
      f"(static dephasing infidelity = sigma_rad^2 * F0)")
peak_i = int(np.argmax(ff["F"]))
print(f"filter-function peak at  {ff['freq_mhz'][peak_i]:.1f} MHz\n")

# Analytic vs Monte-Carlo under 1/f noise (the validation).
print("1/f dephasing infidelity -- analytic filter function vs colored-noise MC:")
for sig in (0.2, 0.05):
    fil = opt.filter_function_fidelity(x, dt=1.0, sigma_mhz=sig, alpha=1.0,
                                       f_low_mhz=1e-3, f_high_mhz=5.0)
    cn = opt.colored_noise_fidelity(x, dt=1.0, sigma_mhz=sig, alpha=1.0, f_low_mhz=1e-3,
                                    f_high_mhz=5.0, n_traj=400, include_decoherence=False,
                                    seed=1)
    drop = cn["F_proc_nominal"] - cn["F_proc"]
    print(f"  sigma={sig:>4} MHz | filter={fil['infidelity']:.3e} | "
          f"MC={drop:.3e} | agree to {abs(fil['infidelity']-drop)/drop*100:.0f}%")
print("\n(The filter function is the no-Monte-Carlo robustness number; overlay your\n"
      " device's measured S(f) on ff['F'] to design for a real noise spectrum.)")
