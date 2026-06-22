"""Coupler-in-the-loop CZ -- the leakage the dispersive pair model cannot see.

The parametric pair optimizer (gradpulse.optimize_cz) eliminates the tunable coupler
under Schrieffer-Wolff: it is fast and accurate to O((gc/Delta)^2), but its only
leakage channel is the qubits' own |2> states -- the coupler's population is
identically zero because the coupler is gone from the model.

`coupler_in_loop_cz` is the opt-in for tunable-coupler devices (Rigetti Cepheus,
Google Sycamore, ...): it takes the SAME two qubits (just hand it your pair profile)
and re-introduces the coupler as a live transmon between them, then optimizes the
flux-activated CZ on the explicit 3-element chain. You stay in the pair workflow but
get the number the pair model structurally cannot produce -- how much population sits
in the coupler -- plus the Schrieffer-Wolff small-parameter (gc/Delta)^2 telling you
whether the pair model's elimination is even in its regime of validity for the device.

It is built on the QuTiP-cross-checked MultiQubitOptimizer engine, so the result is
independently validated exactly like the pair gates (shown below).

Run:  python -m examples.coupler_in_loop_cz
"""
import time

import gradpulse as gp
from gradpulse import ParametricCouplerProfile
from gradpulse.validate import multiqubit_cross_check


def main():
    # The pair profile you already use for optimize_cz -- same two qubits.
    prof = ParametricCouplerProfile(freq_ghz_q1=4.85, freq_ghz_q2=5.05)

    print("Optimizing CZ with the tunable coupler EXPLICITLY in the loop")
    print("(27-dim open-system model -- a few minutes; raise iterations for production)\n")
    t = time.time()
    r = gp.coupler_in_loop_cz(prof, coupler_freq_ghz=5.9, gc_mhz=95.0,
                              n_seeds=2, iterations=150, verbose=True)
    dt = time.time() - t

    print(f"\nExplicit-coupler CZ:")
    print(f"  F_proc          = {r['best_fidelity']:.5f}   ({dt:.0f}s)")
    print(f"  coupler leakage = {r['coupler_leakage']:.3e}   "
          f"<- INVISIBLE to the eliminated pair model (its coupler pop is 0)")
    print(f"  sw_param        = {r['sw_param']:.3e}   = (gc/Delta)^2, the SW small parameter")
    print(f"  J_eff           = {r['J_eff_mhz']:.2f} MHz   (static exchange the coupler mediates)")

    # Independent confirmation in QuTiP -- the same gate the cross-check applies to the
    # pair models; the coupler-in-loop result is validated to ~machine precision too.
    xc = multiqubit_cross_check(r["optimizer"], r["best_waveform"], dt_ns=1.0)
    print(f"\nQuTiP cross-check:  f_torch={xc['f_torch']:.6f}  "
          f"f_qutip={xc['f_qutip']:.6f}  delta={xc['delta']:.2e}")

    print("\nInterpretation: the pair model is the right tool when sw_param is small and you")
    print("only need the gate; reach for this when you need the coupler-leakage budget that")
    print("the elimination throws away. The rigorous residual of the elimination ITSELF is")
    print("gradpulse.validate.coupler_elimination_cross_check (the O((gc/Delta)^2) check).")


if __name__ == "__main__":
    main()
