"""Sweep the QuTiP cross-check across operating points.

The headline cross-check (``python -m gradpulse.validate``) confirms PyTorch ==
QuTiP at ONE operating point. This script shows that agreement is not cherry-
picked: it re-runs the same independent cross-check across a grid of durations,
coupling rates, channel counts, and coherence times, and reports
|F_proc(PyTorch) - F_proc(QuTiP)| for each. A short optimization (not a full
search) is enough -- the cross-check tests whether the two simulators agree on
the SAME pulse and physics, which is independent of how optimal that pulse is.

DRAG is deliberately excluded: with ``use_drag=True`` the saved envelope does not
carry the derived quadrature correction, so a saved-waveform cross-check would not
be apples-to-apples (a documented limitation of the envelope format, not a
simulator disagreement). Every axis swept here changes the Hamiltonian or the
dissipator that BOTH simulators must reproduce.

Run:  python examples/validation_sweep.py
"""
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer
from gradpulse.validate import cross_check

BASE = dict(
    freq_ghz_q1=4.85, freq_ghz_q2=5.05, anharm_ghz_q1=-0.20, anharm_ghz_q2=-0.20,
    t1_ns_q1=30_000, t2_ns_q1=25_000, t1_ns_q2=30_000, t2_ns_q2=25_000,
    g_max_mhz=12.0, omega_max_mhz=50.0,
)

# Profile-level keys a config may override (everything else is an optimizer arg).
_PROF_KEYS = (
    "freq_ghz_q1", "freq_ghz_q2", "anharm_ghz_q1", "anharm_ghz_q2",
    "t1_ns_q1", "t2_ns_q1", "t1_ns_q2", "t2_ns_q2", "g_max_mhz", "omega_max_mhz",
)

# Each config overrides the baseline on one cross-checkable axis.
CONFIGS = [
    {"label": "baseline 150ns / 3ch / g12"},
    {"label": "short 100 ns", "n_slices": 100},
    {"label": "long 200 ns", "n_slices": 200},
    {"label": "weak coupling g=8", "g_max_mhz": 8.0},
    {"label": "strong coupling g=16", "g_max_mhz": 16.0},
    {"label": "4 channels (XY phase)", "n_channels": 4},
    {"label": "6 channels (+Stark/Z)", "n_channels": 6},
    {"label": "low coherence T1=5us", "t1_ns_q1": 5_000, "t2_ns_q1": 4_000,
     "t1_ns_q2": 5_000, "t2_ns_q2": 4_000},
]


def run_one(cfg, tmpdir):
    prof_over = {k: cfg[k] for k in _PROF_KEYS if k in cfg}
    profile = ParametricCouplerProfile(**{**BASE, **prof_over})
    n_slices = cfg.get("n_slices", 150)
    n_channels = cfg.get("n_channels", 3)
    bandwidth = cfg.get("bandwidth_mhz", 80.0)
    opt = ParametricCZOptimizer(profile, bandwidth_mhz=bandwidth, use_drag=False,
                                n_channels=n_channels, activation="sigmoid")
    # Short search: a non-trivial pulse is enough; agreement is what we test.
    res = opt.optimize_multi_seed(
        n_seeds=1, iterations=40, n_slices=n_slices, dt_ns=1.0,
        warm_start_mode="parametric_cz", use_process_fidelity=True,
        lbfgs_polish=False,
    )
    wf = res["best_waveform"]
    np.save(Path(tmpdir) / "p.npy", wf)
    meta = {
        "pulse_npy": "p.npy", "pulse_dt_ns": 1.0, "n_channels": int(wf.shape[1]),
        "bandwidth_mhz": bandwidth, "smoother_type": opt.smoother_type,
        "target_gate": opt.target_gate, "grape_f": float(res["best_fidelity"]),
        "profile": asdict(profile),
    }
    js = Path(tmpdir) / "p.json"
    js.write_text(json.dumps(meta))
    return cross_check(js)   # prints the per-config detail and returns its dict


def main():
    results = []
    with tempfile.TemporaryDirectory() as td:
        for cfg in CONFIGS:
            print("=" * 68)
            print(f" CONFIG: {cfg['label']}")
            print("=" * 68)
            r = run_one(cfg, td)
            r["label"] = cfg["label"]
            results.append(r)

    print("\n" + "=" * 72)
    print(f" {'config':30s} {'F(torch)':>9s} {'F(qutip)':>9s} {'|dF|':>9s}  status")
    print("-" * 72)
    worst = 0.0
    for r in results:
        worst = max(worst, abs(r["delta"]))
        print(f" {r['label']:30s} {r['reported']:9.5f} {r['F_proc']:9.5f} "
              f"{abs(r['delta']):9.2e}  {r['status']}")
    print("-" * 72)
    print(f" max |dF| across all {len(results)} configs: {worst:.2e}   (PASS gate 1e-3)")
    assert worst < 1e-3, f"a config exceeded the cross-check tolerance: {worst:.2e}"
    print(" ALL CONFIGS AGREE within +/-1e-3 -- the cross-check holds across the grid.")


if __name__ == "__main__":
    main()
