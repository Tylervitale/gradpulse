"""Optimize a two-qubit gate inside a larger register -- crosstalk in the loop.

The pair optimizers (parametric CZ, cross-resonance ZX) optimize one isolated pair;
a spectator is only *scored* afterwards. gradpulse.multiqubit lifts GRAPE to an
arbitrary N-qubit register, so a coupled neighbour is part of the optimization
objective: the target is the gate on the chosen pair AND identity on every other
qubit, so the optimizer is rewarded for not disturbing the spectators.

This demo: a CZ on qubits (0,1) of a 3-transmon chain 0-1-2 where qubit 2 is
exchange-coupled to qubit 1 (an always-on crosstalk edge). We optimize against it,
then independently cross-check the result in QuTiP.

Run:  python -m examples.optimize_multiqubit
"""
import time

try:
    from gradpulse.multiqubit import MultiQubitProfile, MultiQubitOptimizer
except ImportError:
    from multiqubit import MultiQubitProfile, MultiQubitOptimizer


def main():
    # 3-transmon chain 0-1-2; gate on (0,1), spectator q2 coupled to q1 at 6 MHz.
    prof = MultiQubitProfile(
        n_qubits=3,
        freqs_ghz=[5.00, 5.20, 5.40],
        anharm_mhz=[-300, -300, -300],
        t1_ns=[50_000, 50_000, 50_000],
        t2_ns=[40_000, 40_000, 40_000],
        couplings={(0, 1): 12.0, (1, 2): 6.0},   # (1,2) is the always-on spectator edge
        n_levels=3,
    )

    # target_qubits=(0,1): CZ there, identity on q2. The (1,2) edge is NOT a tunable
    # channel, so it is crosstalk the optimizer must actively suppress.
    opt = MultiQubitOptimizer(prof, target_gate="cz", target_qubits=(0, 1),
                              open_system=True, precision="double")

    t = time.time()
    res = opt.optimize(n_slices=120, dt_ns=1.0, iterations=250, n_seeds=2, lr=0.05,
                       verbose=True)
    print(f"\nCZ(0,1) with coupled spectator q2:")
    print(f"  F_proc = {res['best_fidelity']:.5f}   F_avg = {res['F_avg']:.5f}   "
          f"leakage = {res['leakage']:.2e}   ({time.time() - t:.0f}s)")
    print(f"  waveform {res['best_waveform'].shape}  (drives q0,q1,q2 + tunable edge 0-1)")

    # Independent confirmation in QuTiP (a different library + integrator).
    try:
        from gradpulse import validate as V
    except ImportError:
        import validate as V
    xc = V.multiqubit_cross_check(opt, res["best_waveform"], dt_ns=1.0)
    print(f"\nQuTiP cross-check:  f_torch={xc['f_torch']:.6f}  "
          f"f_qutip={xc['f_qutip']:.6f}  delta={xc['delta']:.2e}")
    print("  (the general-N model is independently validated, like the pair gates.)")

    print("\nScaling note: this is EXACT density-matrix simulation -- cost is "
          "exponential in N.\nopen_system=True is practical to ~4 qubits; "
          "open_system=False (unitary) reaches more.\nThe constructor prints the "
          "Hilbert dimension and cost so you see what you ask for.")


if __name__ == "__main__":
    main()
