"""Is the >1x over-prediction on 40-41 / 78-87 under-converged f_full (not real)?

Physics: gradpulse's floor uses IDLE T1/T2, which is *better* than gate-effective coherence,
so the floor should UNDER-predict (ratio <= 1). A ratio >1 can therefore only come from f_full
(the open-system optimum) being under-optimized -- the optimizer leaving fidelity on the table
inflates dec_err = f_avg(f_coh) - f_avg(f_full). The sweep used 2 seeds x 400 iters.

Test: re-run ONLY f_full at higher convergence (reuse the stored per-duration f_coh, which
isolates the f_full effect). If f_full climbs and the ratio falls toward/below 1.0, the
over-shoot was optimization, not physics -- and the whole sweep should use more seeds/iters.
"""
import os
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
warnings.filterwarnings("ignore")

import json

import torch

torch.set_num_threads(6)
from gradpulse.literature import f_avg
from gradpulse.parametric import ParametricCZOptimizer
from gradpulse.profiles import ParametricCouplerProfile

HERE = os.path.dirname(os.path.abspath(__file__))
cal = json.load(open(os.path.join(HERE, "cepheus_calibration_snapshot.json")))["pairs"]
dur = json.load(open(os.path.join(HERE, "cepheus_cz_active_ns.json")))
realdur = json.load(open(os.path.join(HERE, "cepheus_grape_sweep_realdur.json")))

PAIRS = ["40-41", "78-87"]
# separate the two knobs cheaply: more SEEDS (basin diversity) vs more ITERS (depth),
# each > the sweep's (2, 400). If either drops the ratio toward 1.0, that knob was the issue.
LEVELS = [(16, 400), (2, 1200)]


def ffull(prof, tg, seeds, iters):
    o = ParametricCZOptimizer(prof, bandwidth_mhz=200.0, use_drag=True, drag_order=2,
                              n_channels=4, precision="double")
    nsl = max(1, int(round(tg)))
    return o.optimize_multi_seed(n_slices=nsl, dt_ns=tg / nsl, n_seeds=seeds,
                                 iterations=iters, diss_scale=1.0)["best_fidelity"]


for key in PAIRS:
    p = cal[key]
    meas = p["measured_err"]
    tg = dur[key]
    fcoh = realdur[f"_f_coh_{int(round(tg))}"]          # reuse stored coherent ceiling
    ff0 = realdur[key]["f_full"]                         # sweep's (2,400) f_full
    r0 = realdur[key]["grape_ratio"]
    fa_coh = f_avg(fcoh)
    prof = ParametricCouplerProfile(t1_ns_q1=p["t1a"], t2_ns_q1=p["t2a"],
                                    t1_ns_q2=p["t1b"], t2_ns_q2=p["t2b"])
    print(f"\n=== {key}  ({tg:.0f} ns, measured {meas*100:.2f}%) ===", flush=True)
    print(f"  sweep (2,400): f_full={ff0:.7f} -> GRAPE {(fa_coh-f_avg(ff0))*100:.3f}% ({r0:.2f}x)",
          flush=True)
    for seeds, iters in LEVELS:
        ff = ffull(prof, tg, seeds, iters)
        dec = fa_coh - f_avg(ff)
        flag = "  <-- f_full improved" if ff > ff0 + 1e-6 else ""
        print(f"  ({seeds:2d},{iters:4d}):  f_full={ff:.7f} -> GRAPE {dec*100:.3f}% "
              f"({dec/meas:.2f}x){flag}", flush=True)
