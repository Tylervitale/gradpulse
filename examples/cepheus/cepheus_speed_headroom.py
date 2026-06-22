"""Is there SPEED headroom to beat native? (free, in-model)

Native CZ is coherence-limited at 96 ns -> error floor ~ linear in gate time. The one physical
lever to BEAT it is a FASTER gate (shorter -> less decoherence), bounded by leakage (faster ->
stronger drive -> more |11>-|02> leakage). This sweeps the coupler-only gate DURATION at a working
coupler regime (5.8 GHz, where the param sweep showed coupler-only reaches ~0.89) and optimizes the
open-system CZ at each, reporting F_avg + leakage. If F_avg PEAKS below 96 ns, the speed lever is
real (a shorter gate wins); if it monotonically worsens as you shorten, native's duration is near
the model's speed limit.

HONEST: representative coupler (the absolute number is not Cepheus-specific and the model caps below
native's 0.994); this measures the RELATIVE duration dependence -- whether faster helps at all.
Run:  python examples/cepheus/cepheus_speed_headroom.py
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
HERE = os.path.dirname(os.path.abspath(__file__))
import json
import numpy as np
from gradpulse.multiqubit import MultiQubitProfile, MultiQubitOptimizer

# Working coupler regime (5.8 GHz) + measured qubit freqs/coherence.
prof = MultiQubitProfile(n_qubits=3, freqs_ghz=[4.654, 5.80, 4.806],
                         anharm_mhz=[-220.0, -180.0, -220.0], t1_ns=[45196.8, 20000.0, 29784.9],
                         t2_ns=[16681.5, 15000.0, 13017.7], couplings={(0, 1): 100.0, (1, 2): 100.0},
                         n_levels=3)
_shape = np.load(os.path.join(HERE, "cepheus_cz_shape.npy")).ravel()        # native flat-top, resampled per dur

DURATIONS = [40, 52, 64, 76, 88, 96]            # ns (1 ns/slice); native is 96
results = []
for ns in DURATIONS:
    opt = MultiQubitOptimizer(prof, target_gate="cz", target_qubits=(0, 2), drive_qubits=[],
                              tunable_edges=[], freq_control_qubits=[1], delta_max_mhz=300.0,
                              open_system=True, precision="double", verbose=False)
    warm = [np.clip(0.5 + 0.5 * A * _shape, 0, 1).reshape(-1, 1) for A in (-0.6, -0.85)]
    r = opt.optimize(n_slices=ns, dt_ns=1.0, iterations=200, n_seeds=len(warm), warm_start=warm,
                     leak_weight=4.0, fidelity="cz_data_virtualz", edge_rest_slices=6, verbose=False)
    results.append({"ns": ns, "F_avg": float(r["F_avg"]), "F_proc": float(r["best_fidelity"]),
                    "leak": float(r["leakage"])})
    print(f"  {ns:3d} ns: F_avg={r['F_avg']:.4f}  F_proc={r['best_fidelity']:.4f}  "
          f"leak={r['leakage']:.2e}", flush=True)
    with open(os.path.join(HERE, "cepheus_speed_headroom.json"), "w") as f:
        json.dump(results, f, indent=2)

best = max(results, key=lambda d: d["F_avg"])
print(f"\nBEST duration: {best['ns']} ns -> F_avg={best['F_avg']:.4f}")
if best["ns"] < 96:
    print(f"  => SPEED LEVER IS REAL: a {best['ns']} ns gate beats the 96 ns one in-model "
          f"({best['F_avg']:.4f} vs {[r['F_avg'] for r in results if r['ns']==96][0]:.4f}). "
          "On a device with measured coupler params, this is the route to BEAT native.")
else:
    print("  => no speed headroom in-model: 96 ns is already near the model's optimum; beating "
          "native would need better coherence or a different coupler regime, not a faster pulse.")
