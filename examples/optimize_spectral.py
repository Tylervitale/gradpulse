"""Band-limited (Fourier / CRAB) optimization: a pulse that respects the AWG
bandwidth BY CONSTRUCTION, with far fewer parameters than piecewise-constant.

    python examples/optimize_spectral.py

The control on each channel is a sum of sinusoids at harmonics of 1/T up to a
cutoff, so it cannot contain out-of-band energy -- no post-hoc smoother, no FFT
anti-cheating penalty, and ~6x fewer parameters. The reported out_of_band_fraction
MEASURES the residual band-limiting (only the [0,1] amplitude clamp can add any),
and the result cross-checks against an independent QuTiP simulation.
"""
import numpy as np

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer

profile = ParametricCouplerProfile(
    freq_ghz_q1=4.85, freq_ghz_q2=5.05, anharm_ghz_q1=-0.20, anharm_ghz_q2=-0.20,
    t1_ns_q1=30_000, t2_ns_q1=25_000, t1_ns_q2=30_000, t2_ns_q2=25_000,
    g_max_mhz=12.0, omega_max_mhz=50.0,
)
opt = ParametricCZOptimizer(profile, bandwidth_mhz=80.0, n_channels=3,
                            activation="sigmoid")

res = opt.optimize_spectral(n_slices=120, dt_ns=1.0, n_seeds=3, iterations=250,
                            f_max_mhz=80.0, lbfgs_polish=True)

print(f"\nbest process fidelity   : {res['best_fidelity']:.6f}")
print(f"parameters              : {res['n_params']} spectral coeffs "
      f"(vs {res['n_params_piecewise']} piecewise-constant)")
print(f"out-of-band energy frac : {res['out_of_band_fraction']:.2e}  "
      f"(band-limited by construction; only the [0,1] clamp can add any)")
print(f"max amplitude overshoot : {res['max_overshoot']:.2e}")
print(f"basis                   : {res['basis']}")

# bandwidth_mhz=0: a spectral pulse is already band-limited, so the QuTiP validator
# evolves the envelope as-is without double-filtering it.
import json
from dataclasses import asdict
np.save("spectral_pulse.npy", res["best_waveform"])
with open("spectral_pulse.json", "w") as f:
    json.dump({"pulse_npy": "spectral_pulse.npy", "pulse_dt_ns": 1.0,
               "n_channels": int(res["best_waveform"].shape[1]), "bandwidth_mhz": 0.0,
               "smoother_type": "gaussian", "target_gate": "cz",
               "grape_f": float(res["best_fidelity"]), "profile": asdict(profile)}, f, indent=2)
print("\nSaved spectral_pulse.npy + spectral_pulse.json")
print("Cross-check:  python -m gradpulse.validate --pulse spectral_pulse.json")
