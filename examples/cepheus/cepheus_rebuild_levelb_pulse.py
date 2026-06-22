"""Design a high-fidelity tunable-coupler CZ for Cepheus pair 16-25 (capability demo).

Inputs that are MEASURED (free, from the device's Braket calibration):
  - q16/q25 transition frequencies (4.654 / 4.806 GHz) and T1/T2,
  - the native CZ flat-top SHAPE (examples/cepheus_cz_shape.npy) used as the warm-start.
Inputs that are REPRESENTATIVE (Braket does NOT expose them -- they need on-device RUN_SWEEPS):
  - coupler frequency / anharmonicity / qubit-coupler exchange J, and qubit anharmonicity.

The optimizer uses the PHYSICAL CZ objective (`fidelity="cz_data_virtualz"`): the gate on the
two data qubits with the coupler idle in |0>, single-qubit virtual-Z free -- exactly how the
device realizes its native CZ (a coupler flux pulse plus `shift_phase` frame updates). The
default strict CZ(x)I target instead penalizes coupler-excited inputs that never occur, which
caps a tunable-coupler CZ well below its true fidelity.

HONEST SCOPE: F~0.95 shows gradpulse CAN design a high-fidelity tunable-coupler CZ. It is a
REPRESENTATIVE-coupler number (sensitive to the unmeasured coupler params), NOT a Cepheus-
specific fidelity, and an OPEN-LOOP pulse -- below the device's closed-loop-tuned native CZ
until on-device calibration. The canary already proved a gradpulse pulse RUNS on Cepheus; this
is the in-model design capability behind it.
"""
import os
# 27-D open-system model -- cap BLAS/OMP threads BEFORE torch import (PC-safe).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
HERE = os.path.dirname(os.path.abspath(__file__))

import json
import numpy as np
import gradpulse as gp

MEAS = dict(
    freqs_ghz=(4.654, 6.80, 4.806),       # q16, coupler, q25: QUBITS MEASURED; coupler REPRESENTATIVE
    anharm_mhz=(-220.0, -180.0, -220.0),  # representative (not in Braket)
    g_qubit_coupler_mhz=100.0,            # representative qubit-coupler exchange
    t1_ns=(45196.8, 20000.0, 29784.9),    # q16, coupler(default), q25  [qubit T1 MEASURED]
    t2_ns=(16681.5, 15000.0, 13017.7),    # q16, coupler(default), q25  [qubit T2 MEASURED]
)
N_SLICES, DT_NS, ITERS = 96, 1.0, 400     # 96 ns matches the device's native CZ duration

# Warm-start the coupler-flux channel with the device's OWN native CZ shape, across depths.
_shape = np.load(os.path.join(HERE, "cepheus_cz_shape.npy"))        # normalized 0 -> 1 -> 0 flat-top
_L = len(_shape)
def _warm(A):
    env = np.full((_L, 3), 0.5)                          # qubit-Z channels at rest...
    env[:, 1] = np.clip(0.5 + 0.5 * A * _shape, 0.0, 1.0)  # ...coupler tuned down (A<0)
    return env
WARM = [_warm(A) for A in (-0.5, -0.65, -0.8, -0.95)]

print(f"designing tunable-coupler CZ @ {N_SLICES} ns, measured qubit freqs "
      f"{MEAS['freqs_ghz'][0]}/{MEAS['freqs_ghz'][2]} GHz, representative coupler "
      f"{MEAS['freqs_ghz'][1]} GHz; physical objective (data CZ + virtual-Z)")

opt = gp.tunable_coupler_cz(verbose=False, **MEAS)
# edge_rest_slices ramps every control to rest at the boundaries so the pulse is a valid
# COMPOSABLE gate (chainable in RB); without it the optimizer drifts the endpoints off rest.
r = opt.optimize(n_slices=N_SLICES, dt_ns=DT_NS, iterations=ITERS, n_seeds=len(WARM),
                 warm_start=WARM, leak_weight=4.0, fidelity="cz_data_virtualz",
                 edge_rest_slices=8, verbose=True)

F = r["best_fidelity"]
vz = tuple(round(float(np.degrees(p)), 1) for p in r["virtual_z_phases"])
print(f"\nF_proc (data-subspace CZ, virtual-Z optimized) = {F:.4f}  F_avg={r['F_avg']:.4f}  "
      f"leak={r['leakage']:.2e}")
print(f"virtual-Z to apply on (q16, q25) = {vz} deg  (free frame shifts, like the native CZ)")

# Save the PHYSICAL flux u=2x-1 (rest=0), not the raw [0,1] envelope (0.5=rest) --
# build_bench_cz_pulse_sequence plays samples as a rest=0 flux activation, so the raw
# envelope would play rest at a large DC offset (a different, wrong pulse).
coupler_x = np.asarray(r["best_waveform"])[:, 1]         # [0,1] envelope, 0.5 = rest
coupler = 2.0 * coupler_x - 1.0                           # physical coupler-flux activation, rest=0
np.save(os.path.join(HERE, "levelb_flux_tunable_measured.npy"), coupler)
meta = {"F_proc": float(F), "F_avg": float(r["F_avg"]), "leakage": float(r["leakage"]),
        "virtual_z_phases_rad": [float(p) for p in r["virtual_z_phases"]],
        "n_slices": N_SLICES, "dt_ns": DT_NS, "objective": "cz_data_virtualz",
        "waveform_convention": "physical coupler flux u=2x-1, rest=0 (bipolar); feed directly "
                               "to build_bench_cz_pulse_sequence",
        "measured_inputs": "qubit freqs + T1/T2 + native CZ shape",
        "representative_inputs": "coupler freq/anharm/J, qubit anharm (need RUN_SWEEPS)",
        "scope": "capability demo; representative coupler; open-loop -- on-device cal needed for a fair benchmark"}
with open(os.path.join(HERE, "levelb_flux_tunable_measured.meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"saved coupler activation ({coupler.shape}) + metadata -> examples/cepheus/levelb_flux_tunable_measured.*")
