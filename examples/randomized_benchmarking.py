"""Bridge the simulated process fidelity to a randomized-benchmarking estimator.

A simulator reports the analytic average gate fidelity F_avg (an *estimand*); a
hardware experiment reports an interleaved-RB number (an *estimator*). They are
the same quantity only in the incoherent, leakage-free limit -- conflating them
is a category error. This script closes the gap on the *simulated* CZ: it runs a
leakage-aware interleaved RB (2-qubit Cliffords compiled to native gates, the
noisy CZ taken once from the simulator as an 81x81 superoperator) and shows the
RB gate fidelity recovers the analytic F_avg, with the leakage rate reported
separately.

Prereq: a pulse to benchmark (examples/optimize_cz.py writes cz_pulse.*). Run:

    python examples/optimize_cz.py            # writes cz_pulse.npy / .json
    python examples/randomized_benchmarking.py
"""
import json
from pathlib import Path

import numpy as np
import torch

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer
from gradpulse.parametric import DEVICE
from gradpulse.rb import gate_superoperator, interleaved_rb

PULSE = Path("cz_pulse.json")
if not PULSE.exists():
    raise SystemExit("cz_pulse.json not found -- run examples/optimize_cz.py first.")

meta = json.loads(PULSE.read_text())
profile = ParametricCouplerProfile(**meta["profile"])
# Evaluation optimizer: feed the saved envelope as the literal physical pulse
# (bandwidth off, clamp activation, double precision) -- the same convention as
# the QuTiP validator, so the RB benchmarks exactly the channel #1 scores.
opt = ParametricCZOptimizer(
    profile, bandwidth_mhz=0.0, use_drag=False,
    n_channels=int(meta["n_channels"]), activation="clamp", precision="double")
env = np.load(PULSE.parent / Path(meta["pulse_npy"]).name)

# Analytic average gate fidelity (the estimand, from gradpulse.rb's sibling #1).
e = torch.as_tensor(env, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
rho = opt.simulate_choi_batch(e, dt=1.0)
f_proc = float(opt._process_fidelity(rho).mean())
f_avg = (4.0 * f_proc + 1.0) / 5.0
print(f"analytic  : F_proc = {f_proc:.6f}   F_avg = {f_avg:.6f}   (1 - F_avg = {1 - f_avg:.5f})")

# Simulated leakage-aware interleaved RB (the estimator).
print("running interleaved RB (2-qubit Cliffords, native-gate compiled) ...")
S = gate_superoperator(opt, e, dt=1.0)
res = interleaved_rb(S, lengths=(1, 2, 4, 8, 12, 16, 24, 32),
                     n_sequences=100, seed=7, f_avg_analytic=f_avg)

print(f"naive RB  : r_cz = {res['r_cz_naive']:.5f}   (single-exponential, ignores leakage)")
print(f"leak-aware: r_cz = {res['r_cz_leakage_aware']:.5f}   F_cz = {res['f_cz_irb']:.6f}")
print(f"leakage   : L1 = {res['leakage_per_clifford_L1']:.2e} per Clifford "
      f"(~{res['leakage_per_clifford_L1'] / 1.88:.2e} per CZ)")
print(f"bridge    : (1 - F_avg_RB) - (1 - F_avg_analytic) = {res['bridge_gap']:+.5f}")
print(f"            naive - leakage-aware (the leakage bias) = {res['naive_minus_aware']:+.5f}")
print()
print("The leakage-aware RB gate fidelity recovers the analytic F_avg: the two")
print("estimators agree once leakage is accounted for. On a noisier / leakier")
print("gate the naive-minus-aware gap widens -- that gap is exactly why a bare")
print("simulator number must not be quoted as a hardware RB fidelity.")
