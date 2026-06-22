"""Triple-check: re-validate on a FULLY CONSISTENT current calibration.

The headline sigma-validation paired snapshot-era gradpulse floors with live std-errors. This
removes that caveat: pull T1/T2 + measured CZ error + standardError for a set of coherence-limited
pairs ALL from one current cal, recompute the GRAPE floor at the CURRENT T1/T2, and report sigma.
(f_coh is coherent-only -> independent of the cal, so the cached per-duration value is reused;
only f_full, which uses T1/T2, is recomputed.) If these land within ~1-2 sigma on internally
consistent data, the snapshot/live mix was not biasing the result.
"""
import os
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
warnings.filterwarnings("ignore")

import json

import sys

import numpy as np
import torch

torch.set_num_threads(6)
try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows cp1252 can't encode 'σ'
except Exception:
    pass
from gradpulse.literature import f_avg
from gradpulse.parametric import ParametricCZOptimizer
from gradpulse.profiles import ParametricCouplerProfile

HERE = os.path.dirname(os.path.abspath(__file__))
ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"
PAIRS = ["2-3", "73-82", "78-87", "28-29", "52-53", "40-41"]
dur = json.load(open(os.path.join(HERE, "cepheus_cz_active_ns.json")))
realdur = json.load(open(os.path.join(HERE, "cepheus_grape_sweep_realdur.json")))


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def f_full(prof, tg):
    o = ParametricCZOptimizer(prof, bandwidth_mhz=200.0, use_drag=True, drag_order=2,
                              n_channels=4, precision="double")
    nsl = max(1, int(round(tg)))
    return o.optimize_multi_seed(n_slices=nsl, dt_ns=tg / nsl, n_seeds=2,
                                 iterations=400, diss_scale=1.0)["best_fidelity"]


def main():
    load_env()
    import boto3
    from braket.aws import AwsDevice, AwsSession
    dev = AwsDevice(ARN, aws_session=AwsSession(
        boto_session=boto3.Session(region_name="us-west-1")))
    std = json.loads(dev.properties.json())["standardized"]
    oneq, twoq = std["oneQubitProperties"], std["twoQubitProperties"]

    def cz(k):
        for kk in (k, "-".join(reversed(k.split("-")))):
            if kk in twoq:
                g = [x for x in twoq[kk]["twoQubitGateFidelity"] if x.get("gateName") == "CZ"][0]
                return 1 - float(g["fidelity"]), float(g.get("standardError") or 0)
        return None, None

    def t1t2(q):
        p = oneq[q]
        return p["T1"]["value"] * 1e9, p["T2"]["value"] * 1e9

    print(f"{'pair':>7} | t_g | measured ± err (CURRENT cal) | gradpulse | σ", flush=True)
    print("-" * 70, flush=True)
    sigmas = []
    for k in PAIRS:
        tg = dur[k]
        fcoh = realdur[f"_f_coh_{int(round(tg))}"]      # cal-independent (coherent-only)
        meas, se = cz(k)
        qa, qb = k.split("-")
        t1a, t2a = t1t2(qa)
        t1b, t2b = t1t2(qb)
        prof = ParametricCouplerProfile(t1_ns_q1=t1a, t2_ns_q1=t2a, t1_ns_q2=t1b, t2_ns_q2=t2b)
        dec = f_avg(fcoh) - f_avg(f_full(prof, tg))
        s = abs(dec - meas) / se if se else float("nan")
        sigmas.append(s)
        print(f"{k:>7} | {tg:3.0f} | {meas*100:5.2f}% ± {se*100:.2f}%          | "
              f"{dec*100:5.2f}%   | {s:.2f}σ", flush=True)
    sigmas = np.array(sigmas)
    print(f"\nCONSISTENT-cal check: median {np.median(sigmas):.2f}σ | max {np.nanmax(sigmas):.2f}σ "
          f"| {(sigmas <= 2).sum()}/{len(sigmas)} within 2σ", flush=True)


if __name__ == "__main__":
    main()
