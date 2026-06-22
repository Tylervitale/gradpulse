"""The honest hardware-realizable Level-B ceiling -- and why a GO needs the MEASURED coupler.

Cepheus has FIXED-FREQUENCY transmons + a tunable coupler: during a CZ it can play only the
COUPLER FLUX (+ post-hoc virtual-Z via shift_phase). It has NO qubit-frequency control. But the
default `tunable_coupler_cz` model gives every element a frequency channel (`freq_control_qubits=
[0,1,2]`), so the optimizer can also tune the QUBITS into the |11>-|02> resonance -- a control
Cepheus lacks. Measured handicap (validation, fresh 120-iter run): dropping the qubit-frequency
channels and re-optimizing virtual-Z costs ~+0.156 F_avg (full 0.845 -> coupler-only 0.690). So
the model's headline 0.97 was NOT hardware-transferable; the coupler+virtual-Z number is.

This script optimizes the FAITHFUL hardware-realizable control set: coupler flux SHAPE (full
freedom) + virtual-Z, qubit-frequency channels DISABLED (`freq_control_qubits=[1]`). Its fidelity
is what a transferred Level-B pulse can deliver in-model.

Result with REPRESENTATIVE coupler params (freq/anharm/J guessed -- Braket does not expose them):
F_avg ~0.69, leak ~0.5%. That is a NO-GO in-model. The crucial context: Cepheus's NATIVE CZ is
ALSO coupler-flux-only and reaches ~0.994 -- so coupler-only IS GO-capable on the real device.
The cap here is the WRONG (representative) coupler params, not the control set or gradpulse.

=> Path to a GO: measure the coupler (frequency, anharmonicity, exchange J) on-device
   (avoided-crossing / spectroscopy sweeps), rebuild the model, re-optimize coupler-only, then
   closed-loop-calibrate the coupler SHAPE on-device (examples/cepheus/cepheus_closed_loop_cal.py drives
   that loop). Representative-param tuning of (peak, virtual-Z) alone cannot reach it.

Run:  python examples/cepheus/cepheus_coupler_only_ceiling.py
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from gradpulse.multiqubit import MultiQubitProfile, MultiQubitOptimizer

# q16/q25 freqs + T1/T2 MEASURED; coupler freq/anharm/J REPRESENTATIVE (Braket does not expose).
MODEL = dict(freqs_ghz=(4.654, 6.80, 4.806), anharm_mhz=(-220.0, -180.0, -220.0),
             gc=100.0, t1_ns=(45196.8, 20000.0, 29784.9), t2_ns=(16681.5, 15000.0, 13017.7))
prof = MultiQubitProfile(n_qubits=3, freqs_ghz=list(MODEL["freqs_ghz"]),
                         anharm_mhz=list(MODEL["anharm_mhz"]), t1_ns=list(MODEL["t1_ns"]),
                         t2_ns=list(MODEL["t2_ns"]),
                         couplings={(0, 1): MODEL["gc"], (1, 2): MODEL["gc"]}, n_levels=3)

# COUPLER-ONLY control set -- the hardware-realizable model for a fixed-frequency-qubit device.
opt = MultiQubitOptimizer(prof, target_gate="cz", target_qubits=(0, 2), drive_qubits=[],
                          tunable_edges=[], freq_control_qubits=[1], delta_max_mhz=300.0,
                          open_system=True, precision="double", verbose=False)
print(f"control channels: {opt.n_channels} (coupler-only; qubit-frequency control disabled)")

_shape = np.load(os.path.join(HERE, "cepheus_cz_shape.npy"))          # native flat-top, normalized 0->1->0
WARM = [np.clip(0.5 + 0.5 * A * _shape, 0, 1).reshape(-1, 1) for A in (-0.5, -0.65, -0.8, -0.95)]

r = opt.optimize(n_slices=96, dt_ns=1.0, iterations=300, n_seeds=len(WARM), warm_start=WARM,
                 leak_weight=4.0, fidelity="cz_data_virtualz", edge_rest_slices=8, verbose=True)
vz = tuple(round(float(np.degrees(p)), 1) for p in r["virtual_z_phases"])
print(f"\nCOUPLER-ONLY (hardware-realizable) ceiling, REPRESENTATIVE coupler params:")
print(f"  F_proc={r['best_fidelity']:.4f}  F_avg={r['F_avg']:.4f}  leak={r['leakage']:.2e}  "
      f"virtual-Z(q16,q25)={vz} deg")
print("  Native Cepheus CZ (coupler-only too) reaches ~0.994 -> coupler-only IS GO-capable on the")
print("  real device; this cap is the representative coupler params. GO needs the MEASURED coupler.")
