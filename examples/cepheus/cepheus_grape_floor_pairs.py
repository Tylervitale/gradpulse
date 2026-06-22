"""Is the multi-pair sweep's "under-prediction" an analytic-floor ARTIFACT or a real
non-coherence-limited gate? Re-run the FULLER GRAPE floor (not the analytic lower bound)
on a spread of pairs and compare to measured.

  * borderline pairs (~0.8x analytic) should jump to ~1.0x with the GRAPE floor
    -> the under-prediction was just the analytic lower bound (FIXABLE: use GRAPE);
  * genuinely non-coherence-limited pairs stay well below 1.0x even with GRAPE
    -> real control/calibration error; gradpulse correctly a LOWER BOUND (not a bug).
"""
import os
import pathlib
import warnings

warnings.filterwarnings("ignore")
from gradpulse.profiles import ParametricCouplerProfile
from gradpulse.parametric import ParametricCZOptimizer
from gradpulse.literature import f_avg, analytic_coherence_limit_epg

GATE_NS = 60.0
# pair -> measured CZ fidelity (from the sweep); a spread incl. the 1%+ targets.
TARGETS = ["16-25", "34-35", "1-2", "40-41", "80-89", "30-31"]


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def grape_floor(prof, tg_ns):
    opt = ParametricCZOptimizer(prof, bandwidth_mhz=200.0, use_drag=True, drag_order=2,
                                n_channels=4, precision="double")
    nsl = int(round(tg_ns)); dt = tg_ns / nsl
    fc = opt.optimize_multi_seed(n_slices=nsl, dt_ns=dt, n_seeds=2, iterations=400,
                                 diss_scale=0.0)["best_fidelity"]
    ff = opt.optimize_multi_seed(n_slices=nsl, dt_ns=dt, n_seeds=2, iterations=400,
                                 diss_scale=1.0)["best_fidelity"]
    return f_avg(fc) - f_avg(ff)


def main():
    load_env()
    import boto3
    import json
    from braket.aws import AwsDevice, AwsSession
    dev = AwsDevice("arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q",
                    aws_session=AwsSession(boto_session=boto3.Session(region_name="us-west-1")))
    std = json.loads(dev.properties.json())["standardized"]
    oneq, twoq = std["oneQubitProperties"], std["twoQubitProperties"]

    def t1t2(q):
        p = oneq[q]
        return p["T1"]["value"] * 1e9, p["T2"]["value"] * 1e9

    print(f"pair     | measured | analytic (ratio) | GRAPE (ratio) | verdict")
    print("-" * 78)
    for key in TARGETS:
        if key not in twoq:
            continue
        meas = 1.0 - float(twoq[key]["twoQubitGateFidelity"][0]["fidelity"])
        qa, qb = key.split("-")
        t1a, t2a = t1t2(qa); t1b, t2b = t1t2(qb)
        prof = ParametricCouplerProfile(t1_ns_q1=t1a, t2_ns_q1=t2a,
                                        t1_ns_q2=t1b, t2_ns_q2=t2b)
        ana = analytic_coherence_limit_epg(prof, GATE_NS)
        gr = grape_floor(prof, GATE_NS)
        rg = gr / meas
        verdict = ("coherence-limited (GRAPE matches)" if 0.8 <= rg <= 1.25
                   else "NON-coherence-limited (real control error; lower bound)")
        print(f"{key:>8} | {meas*100:6.2f}% | {ana*100:5.2f}% ({ana/meas:.2f}x) | "
              f"{gr*100:5.2f}% ({rg:.2f}x) | {verdict}")


if __name__ == "__main__":
    main()
