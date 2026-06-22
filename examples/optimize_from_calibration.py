"""End-to-end on MEASURED device parameters (not the representative defaults).

This is the answer to "but your defaults aren't my qubits": load a real Braket
``standardized_gate_model_qpu_device_properties`` JSON, optimize a CZ against the
*measured* T1/T2 of a specific pair, and compare the optimized gate to that
device's *measured* native-CZ fidelity -- then cross-check the result with QuTiP.

    python examples/optimize_from_calibration.py

Loads the real **Rigetti Cepheus-1-108Q** calibration bundled in ``examples/data/``.
Point ``CALIBRATION`` at your own ``device.properties.standardized`` JSON and set
``PAIR`` to your assigned qubits to run it on any device's numbers -- the loader is
device-agnostic, so switching hardware is just changing the path and the pair.

Produces ``cz_calibrated.npy`` / ``cz_calibrated.json``; cross-check with:

    python -m gradpulse.validate --pulse cz_calibrated.json
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer
from gradpulse.parametric import DEVICE

# A real device-properties export (Rigetti Cepheus-1-108Q). Swap in your own and
# set PAIR to your qubits -- any standardized Braket calibration JSON works.
CALIBRATION = Path(__file__).resolve().parent / "data" / "rigetti_cepheus_calibration.json"
PAIR = (46, 47)

# 1. Load the MEASURED T1/T2 + native-CZ fidelity for this pair; the standardized
#    schema carries no frequency/anharmonicity, so pass representative values for those.
profile = ParametricCouplerProfile.from_braket_calibration(
    str(CALIBRATION), PAIR,
    freq_ghz_q1=4.6838, freq_ghz_q2=4.7788,   # q46/q47 drive freqs (Braket device frames)
    anharm_ghz_q1=-0.33, anharm_ghz_q2=-0.33,  # not in the schema; representative
    g_max_mhz=12.0, omega_max_mhz=50.0,
)
print(f"Loaded calibration for qubits {PAIR} from {CALIBRATION.name}:")
print(f"  T1: q{PAIR[0]}={profile.t1_ns_q1/1000:.1f} us   q{PAIR[1]}={profile.t1_ns_q2/1000:.1f} us")
print(f"  T2: q{PAIR[0]}={profile.t2_ns_q1/1000:.1f} us   q{PAIR[1]}={profile.t2_ns_q2/1000:.1f} us")
print(f"  device's measured native-CZ fidelity: {profile.native_cz_fidelity:.5f}")
for note in profile.notes:
    print(f"  note: {note}")

# 2. Optimize a CZ against THIS pair's measured coherence.
opt = ParametricCZOptimizer(profile, bandwidth_mhz=80.0, use_drag=False,
                            n_channels=3, activation="sigmoid")
DT_NS, N_SLICES = 1.0, 150
result = opt.optimize_multi_seed(
    n_seeds=4, iterations=200, n_slices=N_SLICES, dt_ns=DT_NS,
    warm_start_mode="parametric_cz", use_process_fidelity=True, lbfgs_polish=True,
)
f_proc = result["best_fidelity"]
f_avg = (4.0 * f_proc + 1.0) / 5.0
print(f"\noptimized CZ:  F_proc = {f_proc:.5f}   F_avg = {f_avg:.5f}")

# 3. Honest comparison: the device's number is hardware interleaved-RB, ours is a
#    leakage-aware simulation -- NOT directly comparable, but should be in the same
#    neighbourhood on identical T1/T2 (see rb.py for why sim != hardware).
print(f"device measured native CZ (interleaved RB): {profile.native_cz_fidelity:.5f}")
print(f"simulated optimized CZ   (F_avg, this run): {f_avg:.5f}")
print("  (simulation vs hardware RB -- a sanity-check neighbourhood, not equality)")

# 4. Save for the QuTiP cross-check (carries the measured profile with it).
waveform = result["best_waveform"]
np.save("cz_calibrated.npy", waveform)
meta = {
    "pulse_npy": "cz_calibrated.npy",
    "pulse_dt_ns": DT_NS,
    "n_channels": int(waveform.shape[1]),
    "bandwidth_mhz": opt.bandwidth_mhz,
    "smoother_type": opt.smoother_type,
    "target_gate": opt.target_gate,
    "line_response": opt.line_response,
    "grape_f": float(f_proc),
    "profile": asdict(profile),
}
Path("cz_calibrated.json").write_text(json.dumps(meta, indent=2))
print("\nSaved cz_calibrated.npy + cz_calibrated.json")
print("Cross-check against QuTiP with:")
print("    python -m gradpulse.validate --pulse cz_calibrated.json")
