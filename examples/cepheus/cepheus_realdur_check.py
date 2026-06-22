"""Exact (not rescaled) before/after for the gate-time fix on the >=1% target pairs.

Imports the rewired sweep's own optimize_fidelity (real per-pair t_g) and recomputes the
GRAPE floor for the target pairs at their REAL active duration, vs the flat-60 ns baseline
already in cepheus_grape_sweep_results.json. Confirms the fix end-to-end and prints the
honest shift. ~12 optimizations, a few minutes.
"""
import os
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
warnings.filterwarnings("ignore")

import importlib
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sweep = importlib.import_module("cepheus_grape_sweep_all")  # the rewired functions
import torch  # noqa: E402

torch.set_num_threads(6)
from gradpulse.literature import f_avg  # noqa: E402
from gradpulse.profiles import ParametricCouplerProfile  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
cal = json.load(open(os.path.join(HERE, "cepheus_calibration_snapshot.json")))["pairs"]
dur = json.load(open(os.path.join(HERE, "cepheus_cz_active_ns.json")))
old = json.load(open(os.path.join(HERE, "cepheus_grape_sweep_results.json")))  # flat 60 ns
TARGETS = ["1-2", "34-35", "26-35", "78-87", "48-57", "86-87", "40-41"]

_fcoh = {}


def fcoh(tg):
    k = round(tg)
    if k not in _fcoh:
        _fcoh[k] = f_avg(sweep.optimize_fidelity(ParametricCouplerProfile(), 0.0, tg))
    return _fcoh[k]


print(f"{'pair':>7} | t_g | meas  | GRAPE @60ns -> @real  | verdict", flush=True)
print("-" * 70, flush=True)
for key in TARGETS:
    p = cal[key]
    meas = p["measured_err"]
    tg = dur[key]
    prof = ParametricCouplerProfile(t1_ns_q1=p["t1a"], t2_ns_q1=p["t2a"],
                                    t1_ns_q2=p["t1b"], t2_ns_q2=p["t2b"])
    ff = sweep.optimize_fidelity(prof, 1.0, tg)
    dec = fcoh(tg) - f_avg(ff)
    r_real = dec / meas
    r_old = old[key]["grape_ratio"]
    verdict = "coherence-limited" if 0.8 <= r_real <= 1.25 else "lower bound (control err)"
    print(f"{key:>7} | {tg:3.0f} | {meas*100:4.2f}% | {r_old:.2f}x -> {r_real:.2f}x "
          f"(GRAPE {dec*100:.2f}%) | {verdict}", flush=True)
