"""Optimize TWO gates at once on disjoint qubit pairs, under one shared crosstalk
budget -- the parallel-gate problem a real scheduler faces.

    python examples/simultaneous_gates.py

The N-qubit optimizer already puts identity-on-spectators in the objective; here the
target is a list of gates on disjoint groups (CZ on (0,1) AND CZ on (2,3)). The
gate-mediating couplings (0,1) and (2,3) are tunable controls; the cross-group edge
(1,2) is left FIXED/always-on -- exactly the crosstalk the optimizer must fight. The
combined target is cross-checked against an independent QuTiP rebuild.
"""
import numpy as np

from gradpulse.multiqubit import MultiQubitProfile, MultiQubitOptimizer

# 4 qubits in a line; the (1,2) edge is the crosstalk channel between the two gates.
profile = MultiQubitProfile(
    n_qubits=4, freqs_ghz=(4.9, 5.0, 5.15, 5.25), anharm_mhz=(-300,) * 4,
    t1_ns=(40_000,) * 4, t2_ns=(30_000,) * 4,
    couplings={(0, 1): 12.0, (1, 2): 4.0, (2, 3): 12.0}, n_levels=2,
)
opt = MultiQubitOptimizer(profile, target_gate=["cz", "cz"],
                          target_qubits=[(0, 1), (2, 3)], open_system=True)
print(f"tunable (gate) edges : {opt.tunable_edges}")
print(f"fixed crosstalk edges: {opt.fixed_edges}\n")

res = opt.optimize(n_slices=60, dt_ns=1.0, iterations=120, n_seeds=2, lr=0.06)
print(f"\nsimultaneous CZ x CZ  F_proc = {res['best_fidelity']:.5f}")

# Independent QuTiP cross-check of the combined-gate channel.
try:
    from gradpulse import validate as V
    cc = V.multiqubit_cross_check(opt, res["best_waveform"], dt_ns=1.0)
    print(f"QuTiP cross-check: f_torch={cc['f_torch']:.6f}  f_qutip={cc['f_qutip']:.6f}  "
          f"delta={cc['delta']:.2e}")
except ImportError:
    print("(install gradpulse[validate] for the QuTiP cross-check)")
