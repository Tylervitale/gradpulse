"""Decoherence in the loop vs optimise-coherent-then-multiply -- the claim, measured.

This package argues against folding decoherence in *after* optimisation as a fixed
``F ~= F_coherent * exp(-t_g/T)`` budget. This script demonstrates the difference
instead of asserting it: on a deliberately decoherence-pressured device (a few-
microsecond T1/T2, so leakage and decoherence genuinely compete over the gate
duration) it sweeps the duration and runs both recipes via ``gradpulse.headtohead``.

Output: a per-duration table and a summary that reports, all measured from the
sweep, (i) whether the in-loop optimiser picks a shorter gate, (ii) the delivered-
fidelity gap over the multiply recipe, and (iii) that gap split into a pulse-shaping
part and a duration-selection part. With the [viz] extra it also writes
``paper/decoherence_in_the_loop.png``.

    python examples/decoherence_in_the_loop.py            # full sweep (minutes)
    python examples/decoherence_in_the_loop.py --quick    # tiny smoke run
    pip install -e .[viz] && python examples/decoherence_in_the_loop.py   # + figure
"""
import argparse
from pathlib import Path

from gradpulse import ParametricCouplerProfile
from gradpulse.headtohead import run_head_to_head

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke run")
    ap.add_argument("--no-figure", action="store_true", help="skip the matplotlib figure")
    args = ap.parse_args()

    # A few-microsecond-coherence pair: decoherence and leakage genuinely compete,
    # so the gate duration is a real tradeoff -- unlike the long-lived headline CZ,
    # where decoherence is a near-fixed cost and the two recipes converge.
    prof = ParametricCouplerProfile(
        t1_ns_q1=4000.0, t1_ns_q2=4000.0, t2_ns_q1=3000.0, t2_ns_q2=3000.0,
        notes=["decoherence-pressured regime for the in-loop head-to-head"],
    )

    durations = [60.0, 90.0, 120.0, 160.0, 200.0, 260.0]
    kw = dict(n_seeds=3, iterations=200, lbfgs_iters=60, lr=0.02)
    if args.quick:
        durations = [90.0, 160.0]
        kw = dict(n_seeds=2, iterations=60, lbfgs_iters=20, lr=0.02)

    print(f"Device: T1={prof.t1_ns_q1/1e3:.1f} us, T2={prof.t2_ns_q1/1e3:.1f} us "
          f"(decoherence-pressured)\nSweeping {len(durations)} durations...\n")
    out = run_head_to_head(prof, durations, **kw)
    s = out["summary"]

    print("\n=== summary (all numbers measured from the sweep) ===")
    print(f"  in-loop optimum         : {s['inloop_best_f_avg']:.5f} F_avg "
          f"at {s['inloop_best_duration_ns']:.0f} ns")
    print(f"  multiply recipe picks   : {s['multiply_chosen_duration_ns']:.0f} ns "
          f"(predicting {s['multiply_predicted_f_avg']:.5f}), "
          f"delivers {s['multiply_delivered_f_avg']:.5f}")
    print(f"  in-loop picks shorter?  : {s['picks_shorter_gate']}")
    print(f"  delivered gain (in-loop - multiply) : "
          f"{s['delivered_fidelity_gain_vs_multiply']:+.2e} F_avg")
    print(f"     = pulse-shaping {s['pulse_shaping_gain']:+.2e} "
          f"+ duration-selection {s['duration_selection_loss_of_multiply']:+.2e}")
    print(f"  multiplicative budget over-prediction at its duration : "
          f"{s['multiplicative_overprediction']:+.2e}")

    if not args.no_figure:
        try:
            _figure(out, REPO / "paper" / "decoherence_in_the_loop.png")
        except Exception as e:  # pragma: no cover - viz optional
            print(f"\n(figure skipped: {e}; install the [viz] extra for it)")


def _figure(out, path):  # pragma: no cover - plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = out["rows"]
    t = [r["t_g_ns"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(t, [1 - r["f_inloop"] for r in rows], "o-", label="in-loop (true)")
    ax.plot(t, [1 - r["f_delivered"] for r in rows], "s--", label="coherent, delivered")
    ax.plot(t, [1 - r["f_predicted"] for r in rows], "^:", label="coherent x budget (predicted)")
    ax.set_yscale("log")
    ax.set_xlabel("gate duration (ns)")
    ax.set_ylabel("average gate error  1 - F_avg")
    ax.set_title("Decoherence in the loop vs optimise-coherent-then-multiply")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=140)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
