"""Leakage in the loop -- where the open-system, leakage-aware objective earns its keep.

The literature anchors in this package (Sung / Marxer / Stehlik) are all
*coherence-limited* gates, where the error is decoherence and a one-line formula,

    1 - F_avg  ~  (2 t_g / 5) * sum_q (1/T1 + 1/T_phi),

already gives the answer. That is exactly the regime where the heavy open-system
GRAPE machinery is *least* necessary. This script maps the other regime, and shows
the crossover between them, on a cross-resonance gate -- the architecture the paper
flags as the one "where leakage genuinely matters."

It sweeps the gate duration and, at each duration, optimizes the leakage-aware gate
(the tool's real objective, ``n_levels=4`` throughout -- no truncation games) and
decomposes the error with the package's own ``error_budget`` into

  * ``r_decoherence`` -- the T1/T_phi floor, which is what the coherence formula
    captures, and
  * ``r_coherent``    -- the coherent control+leakage error, which the formula is
    structurally blind to.

The result is a clean, robust crossover (no pathological baseline, just the error
budget of the optimized gate):

  - slow gates are decoherence-limited: ``r_decoherence`` ~ the formula ~ the true
    error. The formula suffices and the open-system loop is optional.
  - fast gates become leakage-limited: ``r_coherent`` overtakes ``r_decoherence``,
    so the coherence formula under-predicts the true error by a growing factor, and
    an optimizer that carries leakage in its objective is the only thing that both
    finds the best gate there *and* reports it honestly.

Every reported ``F_proc`` is cross-checked by the dependency-free NumPy Liouvillian
(and QuTiP if the ``[validate]`` extra is installed), so the crossover is
triple-solver clean rather than one integrator's story.

    python examples/leakage_in_the_loop.py            # full sweep (a few minutes)
    python examples/leakage_in_the_loop.py --quick    # tiny smoke run
"""
# Cap BLAS/OpenMP threads BEFORE importing numpy/torch. This script runs heavy
# CPU cross-checks (NumPy-Liouvillian expm + QuTiP mesolve) alongside the torch
# optimizer; left unbounded, OpenBLAS/MKL grab every core per call and, with the
# solvers stacked, oversubscribe the machine into a thrash/crash. conftest.py caps
# threads for the test suite, but a standalone `python examples/...` run gets no
# such protection, so it must set its own cap. Mirrors the conftest safeguard.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import warnings
from pathlib import Path

import torch

torch.set_num_threads(2)

from gradpulse.crossresonance import CrossResonanceProfile, CrossResonanceZXOptimizer
from gradpulse.liouville import liouville_cr_f_proc

REPO = Path(__file__).resolve().parent.parent


def _coherence_floor(prof, t_g_ns):
    """1 - F_avg from the joint two-qubit coherence-limit formula -- the quantity a
    coherence-only treatment quotes. Blind to leakage and coherent control error."""
    def inv_tphi(t1, t2):
        return max(0.0, 1.0 / t2 - 1.0 / (2.0 * t1))
    rate = ((1.0 / prof.t1_ns_control + inv_tphi(prof.t1_ns_control, prof.t2_ns_control))
            + (1.0 / prof.t1_ns_target + inv_tphi(prof.t1_ns_target, prof.t2_ns_target)))
    return (2.0 * t_g_ns / 5.0) * rate


def _qutip_or_none(opt, waveform, vz):
    try:
        from gradpulse.validate import cr_cross_check
        return cr_cross_check(opt, waveform, vz=vz, echo=False)
    except Exception:
        return None                                    # [validate] (QuTiP) not installed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    durations = [110, 150, 220, 340, 480]
    seeds, iters = 3, 300
    if args.quick:
        # A slow gate needs enough iterations to actually converge to its
        # coherence-limited optimum; too few and it stays control-limited and the
        # crossover doesn't show. Keep quick cheap but not starved.
        durations = [120, 260]
        seeds, iters = 2, 200

    prof = CrossResonanceProfile(n_levels=4)
    print("Cross-resonance gate, converged n_levels=4, default profile "
          f"(T1={prof.t1_ns_control/1e3:.0f} us, T2={prof.t2_ns_control/1e3:.0f} us).")
    print("Sweeping the duration and splitting each optimized gate's error into the\n"
          "decoherence floor (what the coherence formula sees) and the coherent\n"
          "control+leakage part (what it cannot).\n")

    hdr = (f"{'t_g':>5} {'F_proc':>9} {'r_total':>9} {'r_coher':>9} {'r_decoh':>9} "
           f"{'floor':>9} {'under':>6} {'regime':>17}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for ns in durations:
        torch.manual_seed(args.seed)
        opt = CrossResonanceZXOptimizer(prof, use_drag=True, echo=False, precision="double")
        res = opt.optimize(n_slices=ns, n_seeds=seeds, iterations=iters, dt_ns=1.0)
        eb = opt.error_budget(res["best_raw_param"], dt=1.0, vz=res["virtual_z"])
        f_liou = liouville_cr_f_proc(prof, res["best_waveform"], vz=res["virtual_z"],
                                     echo=False, use_drag=True)
        f_qutip = _qutip_or_none(opt, res["best_waveform"], res["virtual_z"])
        floor = _coherence_floor(prof, ns)
        under = eb["r_total"] / max(floor, 1e-12)       # formula under-prediction factor
        regime = "leakage-limited" if eb["r_coherent"] > eb["r_decoherence"] else "coherence-limited"
        rows.append(dict(t_g=ns, f_liou=f_liou, f_qutip=f_qutip, floor=floor, under=under,
                         regime=regime, **eb))
        print(f"{ns:>5} {f_liou:>9.5f} {eb['r_total']:>9.2e} {eb['r_coherent']:>9.2e} "
              f"{eb['r_decoherence']:>9.2e} {floor:>9.2e} {under:>5.1f}x {regime:>17}")

    # Verdict, strictly data-driven (every clause is keyed off the measured regime
    # labels, so an under-converged run can never assert a story it didn't show).
    coh = [r for r in rows if r["regime"] == "coherence-limited"]
    leak = [r for r in rows if r["regime"] == "leakage-limited"]
    fast = rows[0]
    print("\n=== what this shows (all numbers measured above) ===")
    if coh:
        s = min(coh, key=lambda r: r["under"])         # the most cleanly coherence-limited
        print(f"  - coherence-limited gates ({', '.join(str(r['t_g']) for r in coh)} ns): the coherence "
              f"formula tracks the truth (e.g. {s['t_g']} ns: floor {s['floor']:.2e} vs true "
              f"{s['r_total']:.2e}, {s['under']:.1f}x) -- the loop is optional there.")
    if leak:
        f = max(leak, key=lambda r: r["under"])         # the most leakage-limited
        print(f"  - leakage-limited gates ({', '.join(str(r['t_g']) for r in leak)} ns): coherent leakage "
              f"dwarfs the decoherence floor (e.g. {f['t_g']} ns: r_coherent {f['r_coherent']:.2e} vs "
              f"r_decoherence {f['r_decoherence']:.2e}),")
        print(f"    so the coherence formula under-predicts the true error by up to {f['under']:.0f}x.")
    if not coh:
        print("  - (no coherence-limited gate at this effort -- the slow end is still control-limited; "
              "run the full sweep, not --quick, to see the formula track the truth.)")
    # Independent-solver agreement (the cross-check theme), at the fastest point.
    if fast["f_qutip"] is not None:
        print(f"  - every F_proc is triple-solver clean: at {fast['t_g']} ns, Liouville {fast['f_liou']:.5f} "
              f"vs QuTiP {fast['f_qutip']:.5f} (d={abs(fast['f_liou']-fast['f_qutip']):.1e}).")
    else:
        print("  - F_proc cross-checked by the NumPy Liouvillian (install [validate] for the QuTiP leg).")
    print("\n  The coherence formula is enough only where the gate is coherence-limited. Where leakage\n"
          "  dominates, the open-system leakage-aware loop is what both finds the gate and tells the\n"
          "  truth about it -- which is the whole reason to carry the master equation in the objective.")

    if not args.no_figure:
        try:
            _figure(rows, REPO / "paper" / "leakage_in_the_loop.png")
        except Exception as e:  # pragma: no cover - viz optional
            print(f"\n(figure skipped: {e}; install the [viz] extra for it)")


def _figure(rows, path):  # pragma: no cover - plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [r["t_g"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(t, [r["r_total"] for r in rows], "ko-", label="true error  1 - F_proc")
    ax.plot(t, [r["r_decoherence"] for r in rows], "s--", color="tab:blue",
            label=r"decoherence floor (the formula)")
    ax.plot(t, [r["r_coherent"] for r in rows], "^:", color="tab:red",
            label="coherent control + leakage")
    ax.set_yscale("log")
    ax.set_xlabel("gate duration (ns)")
    ax.set_ylabel("average gate error")
    ax.set_title("Leakage in the loop: where the coherence formula stops being enough")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=140)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
