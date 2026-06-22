"""Robust ensemble optimization: optimize *against* a miscalibration ensemble so
calibration tolerance is built into the pulse, not just characterized afterwards.

`robustness_sweep.py` *measures* how a tuned pulse degrades under amplitude /
drive-frequency miscalibration; this script trains a pulse whose loss is averaged
over +/- amplitude perturbations each optimizer step, then compares its
calibration robustness against an ordinarily-tuned (nominal) pulse over the same
miscalibration range.

    python examples/optimize_robust.py                 # 160-iter training (a few min)
    python examples/optimize_robust.py --iterations 60 # quicker, same qualitative result
    python examples/optimize_robust.py --amp-jitter 0.10

What this honestly shows (and an important caveat). Robust optimization can only
flatten error that the *pulse shape* controls -- i.e. coherent / leakage error. The
bundled 150 ns CZ is **decoherence-limited**: its error budget (see
`opt.error_budget` / `examples/validation_checks.py`) is ~94% T1/T_phi floor and only
~6% coherent-plus-leakage, and that coherent part is already near its achievable
minimum. So there is little coherent error left to redistribute, and robust amplitude
optimization does *not* improve the worst case here -- a uniform rescaling of every
drive amplitude on a fixed-duration gate is a stiff coherent error with no flat
direction for a fixed pulse to exploit. The script prints the comparison so you can
see this directly rather than take it on faith.

Robust optimization is the right tool when *coherent / control* error dominates --
faster gates, stronger drives, or drive-activated schemes (e.g. cross-resonance),
and against coupling-rate or coherence-time uncertainty (`robust_g_jitter`,
`robust_t12_jitter`) where the pulse genuinely can trade one operating point against
another. L-BFGS polish is left off: the polish runs on nominal physics only, so it
would partly undo whatever robustness the Adam ensemble search built in.
"""
import argparse

import numpy as np

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer

profile = ParametricCouplerProfile(
    freq_ghz_q1=4.85, freq_ghz_q2=5.05,
    anharm_ghz_q1=-0.20, anharm_ghz_q2=-0.20,
    t1_ns_q1=30_000, t2_ns_q1=25_000,
    t1_ns_q2=30_000, t2_ns_q2=25_000,
    g_max_mhz=12.0, omega_max_mhz=50.0,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=160)
    ap.add_argument("--n-slices", type=int, default=150)
    ap.add_argument("--amp-jitter", type=float, default=0.10,
                    help="+/- fractional amplitude trained against AND swept")
    args = ap.parse_args()
    jit = args.amp_jitter

    common = dict(n_seeds=4, iterations=args.iterations, n_slices=args.n_slices,
                  dt_ns=1.0, warm_start_mode="parametric_cz",
                  use_process_fidelity=True, lbfgs_polish=False)

    print("[1/2] training NOMINAL pulse ...")
    nom = ParametricCZOptimizer(profile, bandwidth_mhz=80.0, activation="sigmoid")
    r_nom = nom.optimize_multi_seed(**common)

    print(f"\n[2/2] training ROBUST pulse (amp +/-{jit:.0%}) ...")
    rob = ParametricCZOptimizer(profile, bandwidth_mhz=80.0, activation="sigmoid")
    r_rob = rob.optimize_multi_seed(robust_amp_jitter=jit, **common)

    # Sweep both over the SAME (matched) amplitude range they were trained on, so
    # the comparison is fair (not an extrapolation beyond the training range).
    amp = np.linspace(-jit, jit, 21)
    print(f"\n  pulse      nominal F_avg    worst-case F_avg (amp +/-{jit:.0%})")
    rows = {}
    for label, opt, res in (("nominal", nom, r_nom), ("robust ", rob, r_rob)):
        sw = opt.robustness_sweep(res["best_raw_param"], amp_fracs=amp, freq_mhz=[0.0])
        fa = np.array(sw["amplitude"]["F_avg"])
        f_avg_nom = (4.0 * res["best_fidelity"] + 1.0) / 5.0
        rows[label.strip()] = (f_avg_nom, fa.min())
        print(f"  {label}    {f_avg_nom:.5f}          {fa.min():.5f}")

    # Why: the error budget shows how much error is even *available* to flatten.
    eb = nom.error_budget(r_nom["best_raw_param"], dt=1.0)
    coh_frac = eb["r_control_leakage"] / eb["r_total"]
    print(f"\n  error budget (nominal): {100*coh_frac:.0f}% of the infidelity is "
          f"coherent+leakage,\n  {100*(1-coh_frac):.0f}% is the T1/T_phi floor "
          f"(r_total={eb['r_total']:.1e}).")
    better = rows["robust"][1] > rows["nominal"][1] + 1e-4
    decoherence_limited = coh_frac < 0.3
    if better:
        print("  -> robust optimization improved the worst case for this setup "
              "(coherent error dominates and has a flatter direction to exploit).")
    elif decoherence_limited:
        print("  -> robust optimization did NOT improve the worst case: this gate is\n"
              "     decoherence-limited, so the coherent error robust opt could "
              "redistribute\n     is already near its floor (see the module docstring).")
    else:
        print("  -> robust optimization did NOT improve the worst case here even though\n"
              "     coherent error dominates -- a global amplitude rescale of a fixed-\n"
              "     duration pulse has no flat direction to exploit. It pays off against\n"
              "     coupling-rate / coherence-time spread (robust_g_jitter, "
              "robust_t12_jitter).")


if __name__ == "__main__":
    main()
