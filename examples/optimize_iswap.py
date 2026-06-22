"""Optimize an iSWAP gate: the parametric coupler's *native* two-qubit gate.

This is the companion to ``optimize_cz.py`` and exists to demonstrate that the
target unitary is just a parameter: the same optimizer, simulator, fidelity
metric, and QuTiP cross-check produce a different gate by changing one argument
(``target_gate="iswap"``). Run:

    python examples/optimize_iswap.py
    python -m gradpulse.validate --pulse iswap_pulse.json

Why a different device profile than optimize_cz.py? The coupler is an exchange
interaction g(t)*(a1^dag a2 + a1 a2^dag); iSWAP is a full population swap in the
{|01>, |10>} subspace. That swap is resonant only when the two qubits are
DEGENERATE (equal frequency); at the 200 MHz detuning optimize_cz.py uses, a
static exchange coupler drives the swap only weakly (~g^2/Delta^2) and the
optimizer instead dumps population into |02>/|20>. So this example puts the pair
on resonance (freq_ghz_q1 == freq_ghz_q2), which is the standard configuration
for an exchange-activated iSWAP. CZ, by contrast, is synthesized indirectly via
the |11>-|02> avoided crossing and does not need degeneracy.

Use ``target_gate="sqrt_iswap"`` for the half-swap (a perfect entangler: two of
them plus single-qubit gates compile a CNOT).
"""
import json
from dataclasses import asdict

import numpy as np
import torch

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer
from gradpulse.parametric import DEVICE

# 1. Device profile. On-resonance pair so the exchange coupler resonantly swaps
#    |01> <-> |10>; for a native iSWAP the two frequencies should be (near-)degenerate.
profile = ParametricCouplerProfile(
    freq_ghz_q1=4.95, freq_ghz_q2=4.95,        # degenerate: resonant exchange
    anharm_ghz_q1=-0.20, anharm_ghz_q2=-0.20,
    t1_ns_q1=30_000, t2_ns_q1=25_000,
    t1_ns_q2=30_000, t2_ns_q2=25_000,
    g_max_mhz=12.0, omega_max_mhz=50.0,
)

# 2. Optimizer. Identical to optimize_cz.py except target_gate="iswap" -- only the
#    target unitary differs; the "parametric_cz" warm start suits this gate as-is.
opt = ParametricCZOptimizer(
    profile,
    bandwidth_mhz=80.0,
    use_drag=False,
    n_channels=3,            # q1 drive, q2 drive, coupler envelope
    activation="sigmoid",
    target_gate="iswap",     # <-- CZ becomes iSWAP by changing this one argument
)

# 3. Multi-seed GRAPE.
DT_NS = 1.0
N_SLICES = 150              # 150 ns gate
result = opt.optimize_multi_seed(
    n_seeds=4,
    iterations=250,
    n_slices=N_SLICES,
    dt_ns=DT_NS,
    warm_start_mode="parametric_cz",
    use_process_fidelity=True,
    lbfgs_polish=True,
)

waveform = result["best_waveform"]
print(f"target gate           : {opt.target_gate}")
print(f"best process fidelity : {result['best_fidelity']:.6f}")
print(f"per-seed fidelities   : {np.round(result['all_fidelities'], 5).tolist()}")

# 4. Save in the format gradpulse.validate consumes. target_gate is written into
#    the metadata so the QuTiP cross-check builds the matching target unitary.
np.save("iswap_pulse.npy", waveform)
meta = {
    "pulse_npy": "iswap_pulse.npy",
    "pulse_dt_ns": DT_NS,
    "n_channels": int(waveform.shape[1]),
    "bandwidth_mhz": opt.bandwidth_mhz,
    "smoother_type": opt.smoother_type,
    "target_gate": opt.target_gate,
    "line_response": opt.line_response,
    "grape_f": float(result["best_fidelity"]),
    "profile": asdict(profile),
}
with open("iswap_pulse.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved iswap_pulse.npy + iswap_pulse.json")
print("Cross-check against QuTiP with:")
print("    python -m gradpulse.validate --pulse iswap_pulse.json")

# 5. Leakage out of the computational subspace for the optimized pulse.
env = torch.as_tensor(waveform, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
rho = opt.simulate_choi_batch(env, dt=DT_NS)
print(f"\nleakage out of comp. subspace: {float(opt._leakage(rho).mean()):.2e}")
