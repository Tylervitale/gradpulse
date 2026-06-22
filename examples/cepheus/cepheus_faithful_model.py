"""Faithful Cepheus (16,25) CZ model: MEASURED qubits + COMPLETE Rigetti-prototype coupler set.

The strongest in-model GO-capability test we can build without a Cepheus-specific coupler sweep.
Combines what we genuinely have for the q16-q25 edge (Braket-measured qubit freqs + T1/T2) with a
COMPLETE, self-consistent coupler parameter set from Rigetti's 2026 adiabatic-CZ paper:

  g_1c=96.2 MHz, g_2c=83.9 MHz (asymmetric qubit-coupler), g_12=3.96 MHz (direct qubit-qubit),
  qubit anharm -227/-221 MHz, coupler anharm -178 bare (-112 measured),
  tunable coupler idles at 2.644 GHz bare, tunes UP ~978 MHz to 3.622 max (3.51 pulse-end).

HONEST PROVENANCE (do NOT overclaim): these are Rigetti PROTOTYPE adiabatic-CZ parameters, NOT
confirmed Cepheus-1-108Q q16-q25 values. So this is "Cepheus qubits + prototype-like coupler" -- a
faithful PLAUSIBILITY test, not a device-specific prediction. The sensitivity sweep
(cepheus_coupler_sensitivity.json) shows coupling g is HIGH-impact (g=60->0.65 vs g=100->0.92), so
"prototype-like" != "Cepheus-exact"; the Cepheus (16,25) g + coupler idle frequency still need
RUN_SWEEPS/Rigetti to make this a true prediction. (Verified free: those Hamiltonian params are NOT
in device.gate_calibrations -- the coupler is element 140 on a baseband flux frame, freq=0, no drive
frame -- so Braket cannot supply them; they are physics behind characterization sweeps.)

Result (free, in-model): coupler anharm -178 bare -> F_avg 0.940, F_proc 0.925, leak 0.4% -- a clean,
low-leakage GO-capable gate. The faithful idle->operating swing (2.644->3.622) also retires an earlier
MISLEADING "idle 2.54 -> 0.698" test that gave the coupler too little flux range to reach the gate.

Run:  python examples/cepheus/cepheus_faithful_model.py
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "3")
HERE = os.path.dirname(os.path.abspath(__file__))
import json
import numpy as np
from gradpulse.multiqubit import MultiQubitProfile, MultiQubitOptimizer

QF = (4.654, 4.806)                                   # Cepheus MEASURED q16/q25 (bare-approx)
T1 = (45196.8, 20000.0, 29784.9)                      # q16, coupler(est), q25
T2 = (16681.5, 15000.0, 13017.7)
COUPLER_IDLE, DMAX = 2.644, 978.0                     # bare idle; swing to 3.622 GHz max
G1C, G2C, G12 = 96.2, 83.9, 3.96                      # prototype couplings (MHz)
ANH_Q1, ANH_Q2 = -227.0, -221.0                       # prototype qubit anharmonicities

_shape = np.load(os.path.join(HERE, "cepheus_cz_shape.npy"))
WARM = [np.clip(0.5 + 0.5 * A * _shape, 0, 1).reshape(-1, 1) for A in (0.85, 0.6, -0.85)]


def run(anh_c, tag):
    prof = MultiQubitProfile(n_qubits=3, freqs_ghz=[QF[0], COUPLER_IDLE, QF[1]],
                             anharm_mhz=[ANH_Q1, anh_c, ANH_Q2], t1_ns=list(T1), t2_ns=list(T2),
                             couplings={(0, 1): G1C, (1, 2): G2C, (0, 2): G12}, n_levels=3)
    opt = MultiQubitOptimizer(prof, target_gate="cz", target_qubits=(0, 2), drive_qubits=[],
                              tunable_edges=[], freq_control_qubits=[1], delta_max_mhz=DMAX,
                              open_system=True, precision="double", verbose=False)
    r = opt.optimize(n_slices=96, dt_ns=1.0, iterations=200, n_seeds=len(WARM), warm_start=WARM,
                     leak_weight=4.0, fidelity="cz_data_virtualz", edge_rest_slices=8, verbose=False)
    print(f"  coupler anharm {anh_c:.0f} MHz ({tag}): F_avg={r['F_avg']:.4f}  "
          f"F_proc={r['best_fidelity']:.4f}  leak={r['leakage']:.4f}", flush=True)
    return {"anh_c": anh_c, "tag": tag, "F_avg": float(r["F_avg"]), "leak": float(r["leakage"])}


if __name__ == "__main__":
    print("FAITHFUL model: Cepheus q16/q25 + FULL prototype coupler set "
          "(g1c=96.2, g2c=83.9, g12=3.96)")
    print(f"  coupler {COUPLER_IDLE}->3.622 GHz (dmax {DMAX}), qubit anharm {ANH_Q1}/{ANH_Q2}\n")
    out = [run(-178.0, "bare"), run(-112.0, "measured")]
    json.dump(out, open(os.path.join(HERE, "cepheus_faithful_prototype.json"), "w"), indent=2)
    print("\nPROTOTYPE params, not Cepheus-confirmed; g is high-sensitivity (see sensitivity sweep).")
