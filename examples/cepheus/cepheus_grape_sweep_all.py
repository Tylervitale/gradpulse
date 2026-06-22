"""Full Cepheus GRAPE-floor sweep over ALL coupled pairs (the rigorous version).

cepheus_predict_vs_measured.py uses the cheap analytic floor (a ~20% lower bound) over
all 193 pairs. This script instead runs the FULLER GRAPE floor -- re-optimizing the CZ
with each pair's live T1/T2 *in the loop* -- so borderline pairs that merely looked
"under-predicted" under the analytic floor can be confirmed coherence-limited (ratio ~1x)
or flagged as genuinely control-limited (ratio stays < 1, gradpulse correctly a bound).

Methodology (matches the Sung/Marxer literature anchors and the 6-pair sample):
    dec_err = F_avg(f_coh) - F_avg(f_full)              # the GRAPE decoherence floor
    ratio   = dec_err / measured_err                    # 1.0 = gradpulse nails it
  where f_coh optimizes with diss_scale=0 (coherent-only) and f_full re-optimizes with
  diss_scale=1 (that pair's decoherence in the loop). f_coh is pair-INDEPENDENT given the
  gate time (coherent Hamiltonian = representative defaults; diss_scale=0 ignores T1/T2),
  but it DOES depend on the gate time via the slice count, so it is cached PER UNIQUE
  duration -- still far cheaper than re-optimizing it for every pair.

Gate time is the device's REAL per-pair CZ duration, read for free from the native gate
calibration (braket_bridge.cz_durations_from_native_calibration), NOT a hardcoded 60 ns.
Using idle T1/T2 over the true (often shorter) active duration makes the floor an honest
LOWER bound: the earlier flat-60 ns "~1.0x" matches were partly a gate-time coincidence
(60 ns happened to be the active-support median). See examples/cepheus/cepheus_grape_sweep_results.json
(the old flat-60 ns run) vs cepheus_grape_sweep_realdur.json (this one).

Crash-proof / resumable: the live calibration is pulled once and cached; results are
atomically saved after EVERY pair; re-running skips pairs already in the results file.
Safe to interrupt (Ctrl-C / reboot) and relaunch -- it picks up where it left off.

Run (6 threads authorized; 9-D parametric CZ is light enough -- NOT the 27-D case):
    python examples/cepheus/cepheus_grape_sweep_all.py
"""
import json
import os
import time

# 6 threads (user-authorized). MUST be set before torch import to take effect.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"

import warnings

warnings.filterwarnings("ignore")

import torch

torch.set_num_threads(6)

from gradpulse.braket_bridge import cz_durations_from_native_calibration
from gradpulse.literature import analytic_coherence_limit_epg, f_avg
from gradpulse.parametric import ParametricCZOptimizer
from gradpulse.profiles import ParametricCouplerProfile

ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"
# Real per-pair CZ gate time, read from the device's native gate calibration; FALLBACK_NS
# is used only if a pair has none. DURATION_MODE picks buffer/active/effective (see
# braket_bridge.cz_durations_from_native_calibration); 'active' is the coherence-relevant choice.
FALLBACK_NS = 60.0
DURATION_MODE = "active"
N_SEEDS = 2
ITERS = 400
# Skip dead/decoupled pairs: measured "CZ error" >= this is a broken coupler, not a real
# gate -- comparing a coherence floor to it is meaningless.
MAX_MEAS_ERR = 0.10
HERE = os.path.dirname(os.path.abspath(__file__))
CAL_PATH = os.path.join(HERE, "cepheus_calibration_snapshot.json")
DUR_PATH = os.path.join(HERE, "cepheus_cz_active_ns.json")
OUT_PATH = os.path.join(HERE, "cepheus_grape_sweep_realdur.json")


def load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def atomic_save(obj, path):
    """Write then os.replace -- the on-disk file is never half-written."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def pull_calibration():
    """Fetch (once) or load cached per-pair measured CZ error + both qubits' T1/T2."""
    if os.path.exists(CAL_PATH):
        return json.load(open(CAL_PATH, encoding="utf-8"))
    load_env()
    import boto3
    from braket.aws import AwsDevice, AwsSession
    dev = AwsDevice(ARN, aws_session=AwsSession(
        boto_session=boto3.Session(region_name="us-west-1")))
    std = json.loads(dev.properties.json())["standardized"]
    oneq, twoq = std["oneQubitProperties"], std["twoQubitProperties"]

    def t1t2(q):
        p = oneq[q]
        return p["T1"]["value"] * 1e9, p["T2"]["value"] * 1e9  # s -> ns

    pairs = {}
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
        t1a, t2a = t1t2(qa)
        t1b, t2b = t1t2(qb)
        pairs[key] = dict(measured_err=meas_err, t1a=t1a, t2a=t2a, t1b=t1b, t2b=t2b)
    snap = dict(arn=ARN, gate_ns=GATE_NS, n_seeds=N_SEEDS, iters=ITERS, pairs=pairs)
    atomic_save(snap, CAL_PATH)
    return snap


def pull_cz_durations():
    """Real per-pair CZ active durations (ns): cached, else parsed from the native cal.

    The standardized cal (pull_calibration) has no gate time; it lives in the native gate
    calibration (a separate ~1.4 MB JSON behind device.properties.pulse.nativeGateCalibrationsRef).
    Free to read -- no task, no spend.
    """
    if os.path.exists(DUR_PATH):
        return json.load(open(DUR_PATH, encoding="utf-8"))
    load_env()
    import urllib.request
    import boto3
    from braket.aws import AwsDevice, AwsSession
    dev = AwsDevice(ARN, aws_session=AwsSession(
        boto_session=boto3.Session(region_name="us-west-1")))
    ref = json.loads(dev.properties.json())["pulse"]["nativeGateCalibrationsRef"]
    cal = json.loads(urllib.request.urlopen(ref, timeout=120).read())
    dur = cz_durations_from_native_calibration(cal, mode=DURATION_MODE)
    atomic_save(dur, DUR_PATH)
    return dur


def symmetric_durations(durations):
    """CZ is symmetric, but the standardized cal and the native cal can order a pair's
    node ids differently ('25-16' vs '16-25'). Index by BOTH orderings so every pair
    finds its real duration instead of silently falling back."""
    d = {}
    for k, v in durations.items():
        parts = str(k).split("-")
        d[str(k)] = v
        if len(parts) == 2:
            d[f"{parts[1]}-{parts[0]}"] = v
    return d


def make_opt(prof):
    return ParametricCZOptimizer(prof, bandwidth_mhz=200.0, use_drag=True, drag_order=2,
                                 n_channels=4, precision="double")


def optimize_fidelity(prof, diss_scale, tg_ns):
    opt = make_opt(prof)
    nsl = max(1, int(round(tg_ns)))
    dt = tg_ns / nsl
    return opt.optimize_multi_seed(n_slices=nsl, dt_ns=dt, n_seeds=N_SEEDS,
                                   iterations=ITERS, diss_scale=diss_scale)["best_fidelity"]


def main():
    import numpy as np
    snap = pull_calibration()
    durations = symmetric_durations(pull_cz_durations())
    all_pairs = snap["pairs"]
    pairs = {k: v for k, v in all_pairs.items() if v["measured_err"] < MAX_MEAS_ERR}
    dropped = len(all_pairs) - len(pairs)
    dvals = [durations[k] for k in pairs if k in durations]
    print(f"calibration: {len(all_pairs)} coupled pairs; dropping {dropped} dead/decoupled "
          f"(measured err >= {MAX_MEAS_ERR*100:.0f}%); sweeping {len(pairs)} real gates",
          flush=True)
    print(f"real CZ durations ('{DURATION_MODE}'): {len(dvals)}/{len(pairs)} pairs have one "
          f"| median {np.median(dvals):.0f} ns, range {min(dvals):.0f}-{max(dvals):.0f} ns "
          f"(fallback {FALLBACK_NS:.0f} ns for the rest)\n", flush=True)

    results = {}
    if os.path.exists(OUT_PATH):
        results = json.load(open(OUT_PATH, encoding="utf-8"))

    # f_coh is pair-independent GIVEN the gate time, but DOES depend on it (slice count),
    # so cache it per unique duration under reserved keys "_f_coh_<ns>".
    fcoh_cache = {k[len("_f_coh_"):]: v for k, v in results.items()
                  if k.startswith("_f_coh_")}

    def f_coh_for(tg_ns):
        kk = str(int(round(tg_ns)))
        if kk not in fcoh_cache:
            t0 = time.time()
            fcoh_cache[kk] = optimize_fidelity(ParametricCouplerProfile(), 0.0, tg_ns)
            results[f"_f_coh_{kk}"] = fcoh_cache[kk]
            atomic_save(results, OUT_PATH)
            print(f"    f_coh({kk:>3} ns) F_avg={f_avg(fcoh_cache[kk]):.8f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        return fcoh_cache[kk]

    todo = sorted([k for k in pairs if k not in results],
                  key=lambda k: pairs[k]["measured_err"])
    done = len(pairs) - len(todo)
    print(f"{len(pairs)} pairs | {done} already done | {len(todo)} to go "
          f"(n_seeds={N_SEEDS}, iters={ITERS}, real per-pair t_g)\n", flush=True)

    t_start = time.time()
    for i, key in enumerate(todo):
        p = pairs[key]
        meas = p["measured_err"]
        tg = float(durations.get(key, FALLBACK_NS))
        try:
            prof = ParametricCouplerProfile(t1_ns_q1=p["t1a"], t2_ns_q1=p["t2a"],
                                            t1_ns_q2=p["t1b"], t2_ns_q2=p["t2b"])
            ana = analytic_coherence_limit_epg(prof, tg)
            ff = optimize_fidelity(prof, 1.0, tg)
            dec_err = f_avg(f_coh_for(tg)) - f_avg(ff)
            results[key] = dict(measured_err=meas, gate_ns=tg, analytic_floor=ana,
                                grape_floor=dec_err, analytic_ratio=ana / meas,
                                grape_ratio=dec_err / meas, f_full=ff,
                                t1a=p["t1a"], t2a=p["t2a"], t1b=p["t1b"], t2b=p["t2b"])
            atomic_save(results, OUT_PATH)
            el = time.time() - t_start
            eta = el / (i + 1) * (len(todo) - i - 1)
            print(f"[{done+i+1:3d}/{len(pairs)}] {key:>8}: {tg:3.0f}ns | meas {meas*100:5.2f}% "
                  f"| GRAPE {dec_err*100:5.2f}% ({dec_err/meas:4.2f}x) | "
                  f"ana {ana*100:5.2f}% ({ana/meas:4.2f}x) | ETA {eta/60:4.0f}m", flush=True)
        except Exception as e:  # one bad pair must not kill a multi-hour run
            results[key] = dict(error=repr(e), measured_err=meas, gate_ns=tg)
            atomic_save(results, OUT_PATH)
            print(f"[{done+i+1:3d}/{len(pairs)}] {key:>8}: ERROR {e!r}", flush=True)

    good = {k: v for k, v in results.items()
            if not k.startswith("_f_coh") and "grape_ratio" in v}
    gr = np.array([v["grape_ratio"] for v in good.values()])
    lim = (gr >= 0.8) & (gr <= 1.25)
    print(f"\nDONE: {len(good)}/{len(pairs)} pairs computed in "
          f"{(time.time()-t_start)/60:.0f}m -> {OUT_PATH}")
    print(f"  GRAPE-floor ratio (real per-pair t_g): median {np.median(gr):.2f}x | "
          f"{lim.sum()}/{len(gr)} coherence-limited (0.8-1.25x) | "
          f"median of those {np.median(gr[lim]):.2f}x")


if __name__ == "__main__":
    main()
