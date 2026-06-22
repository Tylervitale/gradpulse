"""Generate the gradpulse summary figure: a six-panel composite that conveys the
whole package, not just one gate.

Top row -- the worked, cross-checked 150 ns CZ gate (architecture #1):
  (a) the optimized control envelope (two qubit drives + the coupler),
  (b) the |11> population/leakage dynamics -- the transient |11>->|02> excursion
      that builds the conditional phase, and the residual leakage at the gate end,
  (c) the error budget: how the infidelity splits into removable control/leakage
      vs the T1/T_phi decoherence floor (this gate is decoherence-limited).

Bottom row -- what makes gradpulse gradpulse (the methodology, on the same pulse):
  (d) the TRIPLE-SOLVER validation: the same F_proc from three independent solvers
      (PyTorch optimizer, QuTiP, a pure-NumPy Liouvillian) agreeing far inside the
      1e-3 ship gate -- the package's central claim,
  (e) the amplitude-miscalibration robustness bowl (the calibration tolerance an
      experimentalist asks for first),
  (f) multi-qubit crosstalk: a near-resonant spectator swept toward a frequency
      collision -- the gate craters as population resonantly swaps into it (the
      3-transmon physics the N-qubit architecture handles).

Behind the optional [viz] extra so the core install never pulls in matplotlib:
    pip install -e .[viz]
    python examples/make_paper_figure.py                 # writes paper/figure.png
    python examples/make_paper_figure.py --out /tmp/f.png --dpi 200

Everything is computed from the committed reference pulse (tests/fixtures/),
evaluated in the exact configuration gradpulse.validate uses (band-limit off, clamp
activation, double precision) -- so the figure is the validated gate itself,
recomputed from the saved envelope, with no re-optimization. Panels (d)/(f) need the
[validate] extra (QuTiP); without it the figure still builds and those panels note
that QuTiP was unavailable.
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer, liouville_f_proc
from gradpulse.parametric import DEVICE

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"


def _load_reference():
    """The committed 150 ns CZ envelope + its device profile (the shipped gate)."""
    meta = json.loads((FIXTURES / "reference_cz_pulse.json").read_text())
    env = np.load(FIXTURES / meta["pulse_npy"])            # (n_slices, 3), in [0,1]
    valid = {f.name for f in dataclasses.fields(ParametricCouplerProfile)}
    prof = ParametricCouplerProfile(
        **{k: v for k, v in meta["profile"].items() if k in valid})
    return env, prof, meta


def _population_trajectory(opt, env, dt=1.0):
    """Per-slice populations starting from |11>, by chaining the open-system step.

    In the evaluation configuration (band-limit off => identity smoother, clamp
    activation, no DRAG, no line response) slice i's Hamiltonian depends only on
    env[i], so feeding the density matrix forward one slice at a time reproduces
    the full evolution exactly. Returns (t_ns, pops, ci) with pops[k] the diagonal
    populations of all 9 two-qutrit levels after slice k (rows = time).
    """
    n = env.shape[0]
    ci = [int(x) for x in opt._comp_idx]     # indices of |00>,|01>,|10>,|11>
    u = torch.as_tensor(env, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    rho = torch.zeros((1, 1, 9, 9), dtype=opt.cdtype, device=DEVICE)
    rho[0, 0, ci[3], ci[3]] = 1.0            # start in |11>
    pops = []
    with torch.no_grad():
        for i in range(n):
            rho = opt.simulate_gradient_batch(u[:, i:i + 1], dt=dt, rho0=rho)
            pops.append(torch.diagonal(rho[0, 0]).real.cpu().numpy().copy())
    t = np.arange(1, n + 1) * dt
    return t, np.array(pops), ci


def _triple_solver(prof, env):
    """F_proc of the reference pulse from three independent solvers, and the
    pairwise agreement deltas. QuTiP is optional; returns what is available."""
    opt = ParametricCZOptimizer(prof, bandwidth_mhz=0.0, activation="clamp",
                                precision="double")
    u = torch.as_tensor(env, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        f_torch = float(opt._process_fidelity(opt.simulate_choi_batch(u, dt=1.0))[0])
    f_liou = liouville_f_proc(prof, env, "cz", 1.0)
    out = {"PyTorch": f_torch, "NumPy\nLiouvillian": f_liou}
    deltas = [("PyTorch vs\nLiouvillian", abs(f_torch - f_liou))]
    try:
        from gradpulse.validate import qutip_f_proc
        f_qutip = qutip_f_proc(prof, env, "cz", 1.0)
        out["QuTiP"] = f_qutip
        deltas.insert(0, ("PyTorch vs\nQuTiP", abs(f_torch - f_qutip)))
    except Exception:
        pass
    return out, deltas, f_torch


def make_figure(out_path: Path, dpi: int = 150):
    import matplotlib
    matplotlib.use("Agg")                                  # headless: no display needed
    import matplotlib.pyplot as plt

    env, prof, meta = _load_reference()
    # Evaluation optimizer: identical to gradpulse.validate (band-limit off so the
    # saved envelope is fed verbatim, clamp activation, double precision).
    opt = ParametricCZOptimizer(prof, bandwidth_mhz=0.0, activation="clamp",
                                precision="double")
    n_slices = env.shape[0]
    dt = 1.0
    t_ctrl = (np.arange(n_slices) + 0.5) * dt              # slice midpoints (ns)

    u = torch.as_tensor(env, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    choi = opt.simulate_choi_batch(u, dt=dt)
    f_proc = float(opt._process_fidelity(choi)[0])
    leak_avg = float(opt._leakage(choi)[0])
    eb = opt.error_budget(env, dt=dt)

    # (b) population trajectory from |11>.
    t_pop, pops, ci = _population_trajectory(opt, env, dt=dt)
    p11 = pops[:, ci[3]]
    p_leak = 1.0 - pops[:, ci].sum(axis=1)
    noncomp = [i for i in range(9) if i not in ci]
    lead = noncomp[int(np.argmax(pops[:, noncomp].max(axis=0)))]   # dominant leakage level

    # (d) triple-solver agreement.
    solver_f, deltas, _ = _triple_solver(prof, env)

    # (e) amplitude-miscalibration robustness bowl.
    amp = np.linspace(-0.10, 0.10, 41)
    sweep = opt.robustness_sweep(env, dt=dt, amp_fracs=amp, freq_mhz=[0.0])
    amp_pct = 100.0 * np.array(sweep["amplitude"]["x"])
    f_avg = np.array(sweep["amplitude"]["F_avg"])
    f_avg0 = f_avg[len(f_avg) // 2]

    # (f) resonant-collision sweep (a near-resonant exchange-coupled spectator).
    det_list = [2000.0, 600.0, 300.0, 150.0, 75.0, 30.0, 0.0]
    coll = opt.resonant_collision_fidelity(env, dt=dt, detuning_mhz=det_list,
                                           j_mhz=8.0, couples_to=2)
    det = np.array(coll["detuning_mhz"])
    f_coll = np.array(coll["f_proc"])
    leak_coll = np.array(coll["spectator_leakage"])
    order = np.argsort(det)
    det, f_coll, leak_coll = det[order], f_coll[order], leak_coll[order]

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
    (axA, axB, axC), (axD, axE, axF) = axes

    # --- (a) control envelope (signed amplitude 2u-1 in [-1, 1]) ---
    signed = 2.0 * env - 1.0
    labels = ["drive q1", "drive q2", "coupler"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for c in range(env.shape[1]):
        axA.plot(t_ctrl, signed[:, c], color=colors[c], lw=1.6, label=labels[c])
    axA.axhline(0.0, color="0.7", lw=0.6, zorder=0)
    axA.set_xlabel("time (ns)"); axA.set_ylabel("control amplitude (norm.)")
    axA.set_title("(a) optimized CZ envelope")
    axA.set_xlim(0, n_slices * dt)
    axA.legend(frameon=False, fontsize=8, loc="upper right")

    # --- (b) population / leakage dynamics from |11> ---
    axB.plot(t_pop, p11, color="#d62728", lw=1.8, label=r"$P(|11\rangle)$")
    axB.plot(t_pop, pops[:, lead], color="#9467bd", lw=1.5,
             label=r"$P(|02\rangle)$ (leakage path)")
    axB.set_xlabel("time (ns)"); axB.set_ylabel("population")
    axB.set_title(r"(b) dynamics from $|11\rangle$")
    axB.set_xlim(0, n_slices * dt); axB.set_ylim(-0.03, 1.24)
    axB.annotate(rf"residual leak $={p_leak[-1]:.1e}$", xy=(0.04, 0.06),
                 xycoords="axes fraction", ha="left", fontsize=7.5, color="0.3")
    axB.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2,
               handlelength=1.4, columnspacing=1.1, borderaxespad=0.2)

    # --- (c) error budget: removable control/leakage vs the decoherence floor ---
    eb_labels = ["total", "control\n+leakage", "decoherence\nfloor"]
    eb_vals = [eb["r_total"], eb["r_control_leakage"], eb["r_decoherence"]]
    eb_colors = ["#444", "#1f77b4", "#d62728"]
    bars = axC.bar(eb_labels, eb_vals, color=eb_colors)
    axC.bar_label(bars, fmt="%.1e", fontsize=7.5, padding=2)
    axC.set_ylabel("infidelity contribution")
    axC.set_title("(c) error budget (decoherence-limited)")
    axC.set_ylim(0, max(eb_vals) * 1.25)
    axC.tick_params(axis="x", labelsize=8)
    frac = 100.0 * eb["r_decoherence"] / eb["r_total"]
    axC.annotate(f"{frac:.0f}% is the T1/T$_\\phi$ floor", xy=(0.5, 0.86),
                 xycoords="axes fraction", ha="center", fontsize=7.5, color="0.3")

    # --- (d) triple-solver validation: agreement deltas vs the 1e-3 ship gate ---
    dnames = [d[0] for d in deltas]
    dvals = [max(d[1], 1e-16) for d in deltas]             # floor 0 for the log axis
    ypos = np.arange(len(dnames))
    axD.barh(ypos, dvals, color="#2ca02c", height=0.5, zorder=3)
    axD.set_yticks(ypos); axD.set_yticklabels(dnames, fontsize=7.5)
    axD.set_xscale("log"); axD.set_xlim(1e-16, 1e-2)
    axD.set_ylim(-0.7, len(dnames) - 0.3)
    axD.axvline(1e-3, color="#d62728", lw=1.4, ls="--", zorder=4)
    axD.text(1e-3, -0.62, r"ship gate $10^{-3}$", color="#d62728", fontsize=7,
             ha="center", va="bottom")
    for y, v in zip(ypos, dvals):
        axD.text(v * 1.8, y, f"{v:.0e}", va="center", fontsize=7.5, color="0.2")
    axD.set_xlabel(r"$|\Delta F_\mathrm{proc}|$  between solvers")
    axD.set_title("(d) triple-solver agreement")
    axD.annotate(f"all agree on $F_\\mathrm{{proc}}={f_proc:.5f}$", xy=(0.5, -0.30),
                 xycoords="axes fraction", ha="center", fontsize=7.5, color="0.3")
    axD.invert_yaxis()

    # --- (e) amplitude-miscalibration robustness ---
    axE.plot(amp_pct, f_avg, color="#1f77b4", lw=1.8)
    axE.axvline(0.0, color="0.7", lw=0.6, zorder=0)
    within = np.abs(f_avg - f_avg0) <= 1e-3
    if within.any():
        lo, hi = amp_pct[within].min(), amp_pct[within].max()
        axE.axvspan(lo, hi, color="#1f77b4", alpha=0.10)
        axE.annotate(rf"$\Delta F_\mathrm{{avg}}\leq10^{{-3}}$" + f"\nfor {lo:.0f} to {hi:+.0f}%",
                     xy=(0.5, 0.10), xycoords="axes fraction", ha="center",
                     fontsize=7.5, color="#1f77b4")
    axE.set_xlabel("amplitude miscalibration (%)"); axE.set_ylabel(r"$F_\mathrm{avg}$")
    axE.set_title("(e) calibration robustness")
    axE.set_xlim(amp_pct.min(), amp_pct.max())

    # --- (f) multi-qubit crosstalk: resonant frequency collision ---
    axF.plot(det, f_coll, "o-", color="#1f77b4", lw=1.7, ms=4, label=r"$F_\mathrm{proc}$")
    axF.set_xlabel("spectator detuning from q2 (MHz)")
    axF.set_ylabel(r"$F_\mathrm{proc}$", color="#1f77b4")
    axF.tick_params(axis="y", labelcolor="#1f77b4")
    axF.set_title("(f) frequency-collision crosstalk")
    axF.set_xscale("symlog", linthresh=30.0)
    axft = axF.twinx()
    axft.spines["top"].set_visible(False)
    axft.plot(det, leak_coll, "s--", color="#d62728", lw=1.3, ms=3.5,
              label="spectator pop. swapped")
    axft.set_ylabel("population swapped into spectator", color="#d62728", fontsize=8)
    axft.tick_params(axis="y", labelcolor="#d62728")
    axF.annotate("J = 8 MHz", xy=(0.96, 0.55), xycoords="axes fraction",
                 ha="right", fontsize=7.5, color="0.3")
    lines = axF.get_lines() + axft.get_lines()
    axF.legend(lines, [ln.get_label() for ln in lines], frameon=False,
               fontsize=7, loc="center left")

    fig.suptitle(
        r"gradpulse: differentiable open-system GRAPE for superconducting qubits  "
        rf"$-$  CZ $F_\mathrm{{proc}}={f_proc:.5f}$, leakage $={leak_avg:.1e}$, 150 ns, "
        r"triple-solver validated", fontsize=10.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"  F_proc (recomputed from saved envelope): {f_proc:.6f}")
    print(f"  basis-averaged leakage (Choi):           {leak_avg:.3e}")
    print(f"  final leakage from |11> input:           {p_leak[-1]:.3e}")
    print(f"  error budget: decoherence floor          {100*eb['r_decoherence']/eb['r_total']:.0f}% of total")
    print(f"  triple-solver F_proc:                    "
          + ", ".join(f"{k.replace(chr(10),' ')}={v:.8f}" for k, v in solver_f.items()))
    print(f"  solver agreement deltas:                 "
          + ", ".join(f"{n.replace(chr(10),' ')}={v:.1e}" for n, v in deltas))
    print(f"  collision @0 MHz: F_proc={f_coll[np.argmin(np.abs(det))]:.4f}, "
          f"spectator swap={leak_coll[np.argmin(np.abs(det))]:.3f}")
    print(f"  amplitude tolerance (|dF_avg|<=1e-3):    "
          f"{amp_pct[within].min():.1f}% to {amp_pct[within].max():+.1f}%")
    print(f"  wrote {out_path}  ({dpi} dpi)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "paper" / "figure.png")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise SystemExit(
            "matplotlib is required for the paper figure. Install the viz extra:\n"
            "    pip install -e .[viz]")
    make_figure(args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
