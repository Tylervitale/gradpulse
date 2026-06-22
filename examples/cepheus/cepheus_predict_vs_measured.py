"""FREE validation: does gradpulse's coherence-floor prediction match Cepheus's MEASURED
CZ error -- across ALL coupled pairs (not just the best one)?

For every pair Cepheus publishes a measured CZ fidelity (its own interleaved RB) and both
qubits' T1/T2. We feed gradpulse's analytic coherence floor (2*tg/5 * sum(1/T1 + 1/Tphi))
each pair's live T1/T2 and compare to the measured error -- spanning the full ~0.4% to >2%
range. No circuits, no spend; uses only the published calibration.

Expectation (the honest one): COHERENCE-LIMITED pairs land near ratio 1.0 (gradpulse's
floor == measured); pairs with extra coherent/control error sit ABOVE the floor (ratio < 1,
gradpulse correctly a LOWER BOUND -- the Kandala lesson). So this both validates the model
where it applies and flags where the gate is NOT coherence-limited.
"""
import os
import pathlib

import numpy as np

from gradpulse.literature import analytic_coherence_limit_epg
from gradpulse.profiles import ParametricCouplerProfile

GATE_NS = 60.0   # Cepheus CZ ~60 ns (API does not expose per-pair duration); floor ~ tg


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    load_env()
    import boto3
    from braket.aws import AwsDevice, AwsSession
    import json
    dev = AwsDevice("arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q",
                    aws_session=AwsSession(boto_session=boto3.Session(region_name="us-west-1")))
    std = json.loads(dev.properties.json())["standardized"]
    oneq = std["oneQubitProperties"]
    twoq = std["twoQubitProperties"]

    def t1t2(q):
        p = oneq[q]
        return p["T1"]["value"] * 1e9, p["T2"]["value"] * 1e9   # s -> ns

    rows = []
    for key, val in twoq.items():
        fids = [g for g in val.get("twoQubitGateFidelity", []) if g.get("gateName") == "CZ"]
        if not fids:
            continue
        qa, qb = key.split("-")
        if qa not in oneq or qb not in oneq:
            continue
        meas_err = 1.0 - float(fids[0]["fidelity"])
        if meas_err <= 0:
            continue
        t1a, t2a = t1t2(qa); t1b, t2b = t1t2(qb)
        prof = ParametricCouplerProfile(t1_ns_q1=t1a, t2_ns_q1=t2a,
                                        t1_ns_q2=t1b, t2_ns_q2=t2b)
        pred = analytic_coherence_limit_epg(prof, GATE_NS)
        rows.append((key, meas_err, pred, pred / meas_err))

    rows.sort(key=lambda r: r[1])
    ratios = np.array([r[3] for r in rows])
    meas = np.array([r[1] for r in rows])
    coh_limited = (ratios >= 0.8) & (ratios <= 1.2)

    print(f"Cepheus-1-108Q: gradpulse coherence-floor prediction vs MEASURED CZ error")
    print(f"  {len(rows)} coupled pairs, measured error {meas.min()*100:.2f}% .. "
          f"{meas.max()*100:.2f}%, gate {GATE_NS:.0f} ns\n")
    print(f"  ratio = predicted / measured  (1.0 = gradpulse nails it; <1 = gate exceeds its")
    print(f"          coherence floor, i.e. NOT coherence-limited -> gradpulse a lower bound)\n")
    print(f"  median ratio:               {np.median(ratios):.2f}x")
    print(f"  pairs within 0.8-1.2x:      {coh_limited.sum()}/{len(rows)} "
          f"({100*coh_limited.mean():.0f}%) -- coherence-limited, gradpulse MATCHES")
    print(f"  median ratio of those:      {np.median(ratios[coh_limited]):.2f}x")
    print(f"  median measured err (limited): {np.median(meas[coh_limited])*100:.2f}%\n")

    print("  examples spanning the range (measured | gradpulse | ratio):")
    idxs = sorted(set([0, len(rows)//4, len(rows)//2, 3*len(rows)//4, len(rows)-1]))
    for i in idxs:
        k, m, p, r = rows[i]
        tag = "coh-limited" if 0.8 <= r <= 1.2 else "above floor (not coh-limited)"
        print(f"    pair {k:>7}: {m*100:5.2f}% | {p*100:5.2f}% | {r:.2f}x  {tag}")
    # a few coherence-limited ones with measured >= 1% (the user's target regime)
    hi = [r for r in rows if r[1] >= 0.01 and 0.8 <= r[3] <= 1.2]
    print(f"\n  coherence-limited pairs with measured error >= 1.0% "
          f"({len(hi)} found):")
    for k, m, p, r in hi[:6]:
        print(f"    pair {k:>7}: measured {m*100:5.2f}% | gradpulse {p*100:5.2f}% | {r:.2f}x")


if __name__ == "__main__":
    main()
