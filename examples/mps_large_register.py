"""Score a pulse on a register too big for the dense open-system simulator, using the
evaluation-only MPS evaluator (gradpulse.mps.ChainTEBD).

The dense MultiQubitOptimizer is exact but exponential: the open-system process fidelity
evolves a 4**N Choi-operator stack on an n_levels**N Hilbert space, which caps it at
~4 qubits. When a gate keeps entanglement LOW (a local gate on a chain, spectators
nearly idle), a matrix product state compresses each evolved input to bounded bond
dimension chi, so a single trajectory costs O(N * chi**2 * d) instead of O(d**N) -- and
N=6, 8, ... become reachable.

What you get, stated honestly
-----------------------------
* It is a RESTRICTED-ENSEMBLE fidelity WITNESS (mean input-output fidelity over a finite
  product-state ensemble), evolved with quantum-trajectory unraveling over PURE MPS so
  positivity is automatic. It is NOT the exact process fidelity -- the product ensemble
  is not a 2-design, so it does not map through F_avg=(d*F_proc+1)/(d+1) and typically
  OVER-estimates. Treat it as a witness, with its statistical (trajectory) error bar.
* chi-CONVERGENCE IS THE SHIP GATE. The witness is only trustworthy once the discarded
  Schmidt weight has plateaued near zero at the chi you used. This script sweeps chi and
  prints both, so you can SEE the convergence (or see that you must raise chi).

Run:  python examples/mps_large_register.py
"""
import time
import warnings

import numpy as np

from gradpulse import ChainTEBD
from gradpulse.multiqubit import MultiQubitProfile, MultiQubitOptimizer

warnings.filterwarnings("ignore")


def main():
    N = 6                                    # past the dense open-system ~4-qubit wall
    prof = MultiQubitProfile(
        n_qubits=N, freqs_ghz=[5.0 + 0.1 * q for q in range(N)],
        anharm_mhz=[-300.0] * N, t1_ns=[3000.0] * N, t2_ns=[2500.0] * N,
        couplings={(q, q + 1): 10.0 for q in range(N - 1)}, n_levels=3)
    opt = MultiQubitOptimizer(prof, target_gate="cz", target_qubits=(0, 1),
                              tunable_edges=[], open_system=True,
                              precision="double", use_drag=False, verbose=False)
    print(f"N={N}, n_levels={opt.d}: Hilbert dim D={opt.D}. The dense open-system Choi "
          f"stack is {4 ** N} operators on {opt.D}x{opt.D} -- intractable.")
    print("The MPS witness below evolves each input as a bounded-chi pure MPS instead.\n")

    teb = ChainTEBD(opt)
    rng = np.random.default_rng(0)
    Nt = 16
    waveform = np.clip(0.5 + 0.2 * rng.uniform(-1, 1, (Nt, opt.n_channels)), 0.0, 1.0)

    # a small ensemble of computational product inputs (levels |0>, |1>)
    ensemble = []
    for _ in range(2):
        kets = []
        for _q in range(N):
            c = rng.normal(size=2) + 1j * rng.normal(size=2)
            c /= np.linalg.norm(c)
            k = np.zeros(opt.d, dtype=complex)
            k[0], k[1] = c
            kets.append(k)
        ensemble.append(kets)

    print("chi-convergence (the ship gate): trust the witness only once max_discarded")
    print("has plateaued near zero.\n")
    print(f"  {'chi':>4}  {'witness':>9}  {'sem':>9}  {'max_discarded':>13}  {'time':>6}")
    for chi in (2, 4, 8, 16):
        t0 = time.time()
        r = teb.witness_open(ensemble, waveform, dt_ns=1.0, substeps=1,
                             chi_max=chi, n_traj=80, seed=1)
        print(f"  {chi:>4}  {r['witness']:>9.5f}  {r['sem']:>9.1e}  "
              f"{r['max_discarded']:>13.1e}  {time.time() - t0:>5.1f}s")

    print("\nWhen max_discarded is tiny and the witness no longer moves with chi, it is "
          "converged.\nThe value is a restricted-ensemble witness (random pulse here, so "
          "it is low) --\nnot the exact process fidelity. See the module docstring for the "
          "honest scope.")


if __name__ == "__main__":
    main()
