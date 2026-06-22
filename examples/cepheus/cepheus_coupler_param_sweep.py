"""Path 2 (free): is a coupler-ONLY CZ achievable in-model, and at what coupler params?

Cepheus's native CZ is coupler-flux-only and reaches ~0.994, so coupler-only IS GO-capable on the
real device -- but with our REPRESENTATIVE coupler (6.80 GHz, J=100) the coupler-only optimum caps
at F_avg ~0.69. This sweeps the dominant unmeasured knob -- the coupler frequency -- to find the
regime where coupler-only becomes high-fidelity. The peak location is the TARGET for the on-device
coupler characterization (Path 2's measurement); if no regime works in-model, the model can't
capture native's mechanism and the on-device model-free loop (Path 1) is the only route.

Coupler-only control set (freq_control_qubits=[1]); qubits fixed at the MEASURED 4.654/4.806 GHz.
Run:  python examples/cepheus/cepheus_coupler_param_sweep.py
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
HERE = os.path.dirname(os.path.abspath(__file__))
import json
import numpy as np
from gradpulse.multiqubit import MultiQubitProfile, MultiQubitOptimizer

QF = (4.654, 4.806)                 # MEASURED qubit freqs (fixed); coupler freq is the sweep knob
GC, ANH_C, ANH_Q = 100.0, -180.0, -220.0
T1 = (45196.8, 20000.0, 29784.9)
T2 = (16681.5, 15000.0, 13017.7)
COUPLER_FREQS = [5.8, 6.2, 6.6, 7.0, 7.4, 7.8, 8.2]      # GHz, spanning representative 6.80

_shape = np.load(os.path.join(HERE, "cepheus_cz_shape.npy"))
WARM = [np.clip(0.5 + 0.5 * A * _shape, 0, 1).reshape(-1, 1) for A in (-0.6, -0.85)]

results = []
for fc in COUPLER_FREQS:
    prof = MultiQubitProfile(n_qubits=3, freqs_ghz=[QF[0], fc, QF[1]],
                             anharm_mhz=[ANH_Q, ANH_C, ANH_Q], t1_ns=list(T1), t2_ns=list(T2),
                             couplings={(0, 1): GC, (1, 2): GC}, n_levels=3)
    opt = MultiQubitOptimizer(prof, target_gate="cz", target_qubits=(0, 2), drive_qubits=[],
                              tunable_edges=[], freq_control_qubits=[1], delta_max_mhz=300.0,
                              open_system=True, precision="double", verbose=False)
    r = opt.optimize(n_slices=96, dt_ns=1.0, iterations=150, n_seeds=len(WARM), warm_start=WARM,
                     leak_weight=4.0, fidelity="cz_data_virtualz", edge_rest_slices=8, verbose=False)
    results.append({"coupler_ghz": fc, "F_avg": float(r["F_avg"]),
                    "F_proc": float(r["best_fidelity"]), "leak": float(r["leakage"])})
    print(f"  coupler {fc:.2f} GHz: F_avg={r['F_avg']:.4f}  F_proc={r['best_fidelity']:.4f}  "
          f"leak={r['leakage']:.2e}", flush=True)
    with open(os.path.join(HERE, "cepheus_coupler_param_sweep.json"), "w") as f:
        json.dump(results, f, indent=2)

best = max(results, key=lambda d: d["F_avg"])
print(f"\nBEST coupler-only regime: {best['coupler_ghz']:.2f} GHz -> F_avg={best['F_avg']:.4f}")
if best["F_avg"] > 0.9:
    print("  => coupler-only IS achievable in-model. The on-device coupler measurement should pin")
    print(f"     the coupler near {best['coupler_ghz']:.2f} GHz; then re-optimize coupler-only for a")
    print("     genuine model-designed Cepheus CZ (Path 2).")
else:
    print("  => even the best in-model coupler-only caps below 0.9; the model does not capture")
    print("     native's mechanism. On-device model-free closed loop (Path 1) is the route.")
