"""The statistically correct gradpulse-vs-Cepheus validation: is the prediction within the
MEASUREMENT's own error bar?

Quoting "gradpulse is 1.9% off measured" is misleading -- the measured CZ error is itself only
known to +/-12-42% (the device's published interleaved-RB standardError). The right metric is the
sigma-distance |gradpulse - measured| / standardError: how many measurement error bars apart they
are. <=1 sigma = indistinguishable from the measurement; you cannot validate tighter than that.

The PRIMARY result is the UNSELECTED one-sided lower bound over all 160 pairs (floor <= measured,
within the RB error bar, on ~150/160; median floor 0.66x measured) -- nothing selected on the
prediction. The sigma metric is a REFINEMENT on the saturation subset (the pairs where the floor is
within 0.8-1.25x of measured): it gauges how tight the bound is where the gate is coherence-limited,
NOT an independent accuracy test, since that subset is *defined by* the floor saturating measured.
The full unselected scatter is examples/cepheus_lowerbound_scatter.py.

Note: gradpulse floors are from the swept snapshot (cepheus_grape_sweep_realdur.json); the
standardError is read from the current calibration as a representative measurement-precision
magnitude (it is set by the device's RB design -- n_seeds/shots -- not by the day's drift).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


SIDECAR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "cepheus_cz_std_errors.json")


def _symmetrize(out):
    """CZ is symmetric: mirror every 'a-b' key to 'b-a' so either ordering resolves."""
    for key in list(out):
        out["-".join(reversed(key.split("-")))] = out[key]
    return out


def pull_std_errors(refresh=False):
    """CZ interleaved-RB standardError per pair, pinned for reproducibility.

    Default reads the dated sidecar (cepheus_cz_std_errors.json, no AWS) so the
    sigma refinement is reproducible and does not drift with the day's
    recalibration. ``refresh=True`` re-pulls from the live device and rewrites
    the sidecar. The standardError is a measurement-precision magnitude (set by
    the device's RB design -- n_seeds/shots), not a per-pulse quantity, so a
    pinned snapshot is the honest scale to divide by.
    """
    if not refresh and os.path.exists(SIDECAR):
        raw = json.load(open(SIDECAR))["cz_standard_error"]
        return _symmetrize({k: v for k, v in raw.items() if 0 < v < 1.0})
    load_env()
    import boto3
    from braket.aws import AwsDevice, AwsSession
    dev = AwsDevice(ARN, aws_session=AwsSession(
        boto_session=boto3.Session(region_name="us-west-1")))
    twoq = json.loads(dev.properties.json())["standardized"]["twoQubitProperties"]
    out = {}
    for key, val in twoq.items():
        cz = [g for g in val.get("twoQubitGateFidelity", []) if g.get("gateName") == "CZ"]
        if cz:
            se = float(cz[0].get("standardError") or 0)
            if 0 < se < 1.0:                                 # 1.0 == placeholder, drop
                out[key] = se
    json.dump({"_meta": {"source": ARN, "note": "CZ interleaved-RB standardError "
                         "per pair; pinned for reproducible sigma refinement"},
               "cz_standard_error": out}, open(SIDECAR, "w"), indent=1)
    return _symmetrize(out)


def main():
    R = json.load(open(os.path.join(HERE, "cepheus_grape_sweep_realdur.json")))
    g = {k: v for k, v in R.items() if not k.startswith("_f_coh") and "grape_ratio" in v}

    # --- PRIMARY: unselected one-sided lower bound over ALL pairs (no AWS, no selection) ---
    ratio = np.array([v["grape_ratio"] for v in g.values()])
    N = len(g)
    print(f"UNSELECTED one-sided lower bound over all {N} pairs (the falsifiable claim):")
    print(f"  floor <= measured (strict):                {(ratio <= 1.0).sum()}/{N}")
    print(f"  floor <= measured within widest RB bar (<=1.42x): {(ratio <= 1.42).sum()}/{N}")
    print(f"  exceeds measured by >2x:                   {(ratio > 2.0).sum()}/{N} "
          f"(incl. the flagged impossible-T2 entry)")
    print(f"  median floor / measured:                   {np.median(ratio):.2f}x\n")

    # --- REFINEMENT: sigma on the saturation subset, which is DEFINED by floor ~ measured ---
    refresh = "--refresh" in sys.argv
    try:
        se = pull_std_errors(refresh=refresh)
    except Exception as e:                                # no sidecar and AWS down -> headline still prints
        print(f"(saturation sigma needs the device std-errors; unavailable: {e})")
        return
    print(f"(std-errors from {'live device' if refresh else 'pinned sidecar'}; "
          f"sigma refinement is reproducible offline)")
    rows = [(k, v, abs(v["grape_floor"] - v["measured_err"]) / se[k])
            for k, v in g.items() if k in se]
    coh = np.array([s for k, v, s in rows if 0.8 <= v["grape_ratio"] <= 1.25])
    print(f"SATURATION subset ({len(coh)} pairs, floor within 0.8-1.25x of measured -- a "
          f"MODEL-DEFINED\nregime, NOT an independent test): sigma = |pred-meas| / std-err")
    print(f"  median {np.median(coh):.2f}sigma | mean {coh.mean():.2f}sigma | max {coh.max():.2f}sigma")
    for s in (1, 2, 3):
        print(f"  within {s} sigma: {(coh <= s).sum()}/{len(coh)} ({100*(coh<=s).mean():.0f}%)")


if __name__ == "__main__":
    main()
