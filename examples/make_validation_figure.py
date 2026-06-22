"""Generate the paper's validation figure: the triple-solver discipline as MEASURED numbers.

  (a) dt-convergence of the Trotter splitting error on the blessed CZ -- first-order toward
      the exact-generator (Liouvillian) value, so the headline gap IS the splitting error.
  (b) four-way solver agreement on the CZ: the matched QuTiP integrator reproduces the
      optimizer to machine precision (~1e-14); the exact-generator and adaptive-mesolve legs
      sit at the ~1e-7 splitting-error level.
  (c) independent-QuTiP cross-check agreement across the other architectures and channels:
      the cross-resonance gate, a 27-D ZZ spectator, a resonant collision, an 18-D TLS defect.

Everything is recomputed from the committed reference pulse (plus one short CR optimization
for panel c). Behind the [viz] extra:

    pip install -e .[viz] && python examples/make_validation_figure.py   # -> paper/validation_figure.png
"""
import argparse
import dataclasses
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer, liouville_f_proc
from gradpulse import validate
from gradpulse.parametric import DEVICE

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
FIX = REPO / "tests" / "fixtures"
_SUB = ("freq_ghz_q1", "freq_ghz_q2", "anharm_ghz_q1", "anharm_ghz_q2", "t1_ns_q1",
        "t1_ns_q2", "t2_ns_q1", "t2_ns_q2", "g_max_mhz", "omega_max_mhz", "chi_zz_mhz", "n_levels")


def _utf8_stdout():
    """Best-effort: let stdout/stderr carry non-ASCII (the pi glyph in the panel-c keys)
    so the diagnostic prints don't raise UnicodeEncodeError on a legacy Windows cp1252
    console. errors='replace' keeps it crash-proof even where UTF-8 can't be set."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _load_cz():
    meta = json.loads((FIX / "reference_cz_pulse.json").read_text())
    env = np.load(FIX / meta["pulse_npy"])
    valid = {f.name for f in dataclasses.fields(ParametricCouplerProfile)}
    prof = ParametricCouplerProfile(**{k: v for k, v in meta["profile"].items() if k in valid})
    return env, prof


def _dt_convergence(env, prof):
    opt = ParametricCZOptimizer(prof, bandwidth_mhz=0.0, activation="clamp", precision="double")
    f_liou = liouville_f_proc(prof, env, "cz", 1.0)
    envt = torch.as_tensor(env, dtype=opt.rdtype, device=DEVICE)
    dts, gaps = [], []
    for k in (1, 2, 4, 8, 16, 32):
        up = envt.repeat_interleave(k, dim=0).unsqueeze(0)
        with torch.no_grad():
            f = float(opt._process_fidelity(opt.simulate_choi_batch(up, dt=1.0 / k))[0])
        dts.append(1.0 / k); gaps.append(abs(f - f_liou))
    return np.array(dts), np.array(gaps)


def _four_way(env, prof):
    opt = ParametricCZOptimizer(prof, bandwidth_mhz=0.0, activation="clamp", precision="double")
    u = torch.as_tensor(env, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        f_t = float(opt._process_fidelity(opt.simulate_choi_batch(u, dt=1.0))[0])
    f_l = liouville_f_proc(prof, env, "cz", 1.0)
    f_q = validate.qutip_f_proc(prof, env, "cz", 1.0)
    f_m = validate.mesolve_zoh_fproc(prof, env, "cz", 1.0)
    return {"Trotter vs\nQuTiP-matched": abs(f_t - f_q),
            "Trotter vs\nadaptive mesolve": abs(f_t - f_m),
            "Trotter vs\nLiouville": abs(f_t - f_l)}


def _cross_arch(env, prof):
    out = {}
    pfull = dataclasses.asdict(prof)
    psub = {k: getattr(prof, k) for k in _SUB}
    opt = ParametricCZOptimizer(prof, bandwidth_mhz=0.0, activation="clamp", precision="double")
    u = torch.as_tensor(env, dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    zr = 2 * math.pi * (0.3 / 1000.0)
    f_eff = validate.qutip_f_proc(pfull, env, "cz", 1.0, detuning_offset=(0.0, zr))
    f27 = validate.spectator_cross_check_3transmon(pfull, env, "cz", 1.0, 0.3, couples_to=2)
    out["ZZ spectator\n(27-D)"] = abs(f27 - f_eff)
    t = opt.resonant_collision_fidelity(u, dt=1.0, detuning_mhz=50.0, j_mhz=8.0, couples_to=2)
    q = validate.collision_cross_check(psub, env, "cz", 1.0, detuning_mhz=50.0, j_mhz=8.0, couples_to=2)
    out["collision"] = abs(t["f_proc"] - q["f_proc"])
    t = opt.tls_defect_fidelity(env, dt=1.0, g_mhz=2.0, t1_tls_ns=500.0, detuning_mhz=0.0, couples_to=1)
    q = validate.tls_defect_cross_check(psub, env, "cz", 1.0, detuning_mhz=0.0, g_mhz=2.0, t1_tls_ns=500.0, couples_to=1)
    out["TLS defect\n(18-D)"] = abs(t["f_proc"] - q["f_proc"])
    from gradpulse.crossresonance import CrossResonanceProfile, CrossResonanceZXOptimizer
    cprof = CrossResonanceProfile()
    # Double precision so the bar is the scheme/model agreement (independent QuTiP vs the
    # optimizer), not single-precision torch error -- matching the CZ/channel panels.
    copt = CrossResonanceZXOptimizer(cprof, bandwidth_mhz=60.0, use_drag=True,
                                     use_target_cancel=True, precision="double")
    cres = copt.optimize(n_slices=120, dt_ns=1.0, iterations=150, n_seeds=2, lr=0.05)
    fq = validate.cr_cross_check(copt, cres["best_waveform"], vz=cres["virtual_z"], echo=copt.echo, dt_ns=1.0)
    out["CR ZX(π/2)"] = abs(cres["best_fidelity"] - fq)
    return out


def make_figure(out_path: Path, dpi: int = 150):
    _utf8_stdout()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env, prof = _load_cz()
    dts, gaps = _dt_convergence(env, prof)
    b = _four_way(env, prof)
    c = _cross_arch(env, prof)

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.5, 3.9))

    axA.loglog(dts, gaps, "o-", color="#1f77b4", lw=1.8, ms=5, zorder=3)
    ref = gaps[-1] * (dts / dts[-1])                       # slope-1 guide through the finest point
    axA.loglog(dts, ref, "--", color="0.6", lw=1.2, label="first-order (slope 1)")
    axA.set_xlabel("integration step $dt$ (ns)")
    axA.set_ylabel(r"$|\Delta F_\mathrm{proc}|$ vs exact generator")
    axA.set_title("(a) Trotter splitting error vs $dt$")
    axA.legend(frameon=False, fontsize=7.5, loc="upper left")

    for ax, data, color, lim, xlabel, title in (
        (axB, b, "#2ca02c", (1e-16, 1e-4), r"$|\Delta F_\mathrm{proc}|$", "(b) CZ four-way agreement"),
        (axC, c, "#9467bd", (1e-16, 1e-3), r"$|\Delta F_\mathrm{proc}|$ vs independent QuTiP",
         "(c) cross-architecture / channel")):
        names = list(data.keys()); vals = [max(v, 1e-16) for v in data.values()]
        y = np.arange(len(names))
        ax.barh(y, vals, color=color, height=0.55, zorder=3)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xscale("log"); ax.set_xlim(*lim)
        ax.axvline(1e-3, color="#d62728", lw=1.2, ls="--", zorder=2)
        ax.text(1e-3, len(names) - 0.4, "ship gate", color="#d62728", fontsize=6.5, ha="center", va="bottom")
        for yi, v in zip(y, vals):
            ax.text(v * 1.7, yi, f"{v:.0e}", va="center", fontsize=7, color="0.2")
        ax.set_xlabel(xlabel); ax.set_title(title); ax.invert_yaxis()

    fig.suptitle("gradpulse triple-solver validation, as measured", fontsize=10.5, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print("panel (a) dt-convergence:", [(round(d, 4), f"{g:.2e}") for d, g in zip(dts, gaps)])
    print("panel (b) four-way:", {k.replace(chr(10), ' '): f"{v:.2e}" for k, v in b.items()})
    print("panel (c) cross-arch:", {k.replace(chr(10), ' '): f"{v:.2e}" for k, v in c.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "paper" / "validation_figure.png")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()
    make_figure(args.out, args.dpi)
