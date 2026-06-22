"""Run interleaved randomized benchmarking of a CZ on a real Amazon Braket QPU.

The one step gradpulse's bridge stops at: device.run() on silicon. Everything before it is
offline-verifiable and unit-tested (gradpulse.braket_bridge, gradpulse.rb); this adds the
submission, result collection, decay fit, and comparison to gradpulse's prediction.

Two meanings of "validate on hardware":
  Level A (default): benchmark the device's NATIVE CZ, pull its calibration, check
    gradpulse's coherence-limited prediction against the measured gate error -> validates
    the MODEL.
  Level B (--pulse --pulse-file): benchmark a gradpulse-DESIGNED pulse the same way ->
    tests the OPTIMIZER on silicon. The gradpulse activation waveform plays on the
    device's own CZ frame (--flux-frame-id) as a pulse gate inside the verbatim box,
    anchored to the device's calibrated CZ flux peak (--native-cal-file). Same cost as
    Level A. This is an OPEN-LOOP transfer: the pulse is optimized against a model and
    scaled to the device's flux full-scale, NOT closed-loop calibrated on the device, so
    expect it to start BELOW the device's native (tuned) CZ -- the canary survival shows
    where, and on-device calibration (the HITL hooks) closes the gap. Match the gradpulse
    architecture to the device's activation: for Cepheus's baseband tunable coupler use
    the coupler channel of tunable_coupler_cz (see levelb_pulse_benchmark_offline.py).

CLEAN Level-B benchmark (--combined) is a STAGED on-device calibration, because the open-loop
  transfer above is NOT a fair comparison to the device's closed-loop-tuned native CZ. To put
  gradpulse on the same footing, calibrate its two dominant scalars on-device, then benchmark:
    Stage 1  --cal-peak-sweep      (~$33): sweep the flux peak (the |11>-|02> entangling angle)
             vs ONE shared native reference; fit r_cz(peak) -> the calibrated --cz-peak. Pass
             --drive-frame-ids + --virtual-z (the pulse's MODEL phase estimate) so r_cz is not
             penalized by uncompensated single-qubit phase -> a cleaner GO/NO-GO: if the shape
             transfers, F at the best peak is decent; if not, you learn it here, not at $87.
    Stage 2  --cal-virtualz-sweep  (~$22-52): at that peak, sequential 1-D sweep of the single-
             qubit virtual-Z phases (phi0 then phi1, separable) -> the calibrated --virtual-z.
    Stage 3  --combined            (~$87): Level A (native) + Level B (gradpulse at the calibrated
             peak & virtual-Z) interleaved-RB'd against ONE shared reference, same session/drift
             -> the clean native-vs-designed number. ~$29 cheaper than running A and B separately.
  The Level-B pulse MUST be a composable gate (returns the coupler to rest): generate it with
  examples/cepheus/cepheus_rebuild_levelb_pulse.py (optimize edge_rest_slices>0) and pass its saved
  PHYSICAL flux (u=2x-1, rest 0) + virtual_z. A [0,1] envelope or a non-rest pulse is rejected/
  warned by build_bench_cz_pulse_sequence.

Cost (Cepheus-1-108Q, checked vs the AWS Braket pricing page 2026-06, not the repo's
  BRAKET_QPU_PRICING): $0.30 per task (one task/circuit) + $0.000425/shot. The default run
  (112 circuits x 500 shots) is ~$57, so it exceeds the default --max-cost 50 guard -- pass
  --max-cost 65 to authorize the spend. Two pre-flight canaries (length-1 + the deepest
  circuit, ~$0.34 each) run first and abort on implausible data; --canary-only runs just
  those (~$0.68) and stops. Default mode is OFFLINE: it prints the bill and checks every
  circuit returns to |00> before any spend.

Choosing --lengths (max length = MIN of two limits):
  * gate error -- the decay must reach deep enough to fit (a 1%-sized ladder under-resolves
    a 0.4% gate); and
  * T2 / circuit duration -- a sequence longer than ~T2/clifford-time floors at 1/d from
    idle decoherence regardless of gate quality (Cepheus: m=128 ~= 44us vs T2 ~14us floored
    at 0.24 on hardware).
  Use suggest_lengths(r_cz, t2_us=<device T2>): Cepheus (0.4%, 14us) -> max ~44; the default
  (max 32) is T2-safe for T2>=10us. The fit leaves the asymptote FREE by default because real
  readout is asymmetric (Cepheus's canary piled errors into '10'), which shifts the asymptote
  off 1/d -- fixing it (--fixed-asymptote) then gives a tight-but-WRONG r_cz; RB's decay rate
  (-> r_cz) is SPAM-robust regardless. Backed by cepheus_irb_resolution_study.py (MC over the
  real Clifford group with T2 + asymmetric readout, validated against the canary).

  ARN = arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q   (best pair: 16 25)
    python examples/cepheus/run_irb_on_braket.py                        # offline: verify + cost
    python examples/cepheus/run_irb_on_braket.py --submit --device-arn $ARN --qubits 16 25 \
        --canary-only                                           # ~$0.68: canaries, then STOP
    python examples/cepheus/run_irb_on_braket.py --submit --device-arn $ARN --qubits 16 25 \
        --max-cost 65                                           # ~$57 Level-A validation run
    # --- clean Level-B benchmark: stage 1 -> 2 -> 3 (PULSE = a composable rest-0 flux) ---
    # Frames are Cepheus(16,25)'s OWN native-CZ frames: flux=play frame, drives=shift_phase frames.
    PULSE=examples/cepheus/levelb_flux_tunable_measured.npy; FF=Transmon_140_flux_tx_cz
    DF="Transmon_16_charge_tx Transmon_25_charge_tx"
    python examples/run_irb_on_braket.py --submit --device-arn $ARN --qubits 16 25 \
        --cal-peak-sweep --pulse-file $PULSE --flux-frame-id $FF --drive-frame-ids $DF \
        --virtual-z 1.44 3.403 --lengths 1 2 4 8 --seeds 2 --max-cost 34   # Stage 1: peak -> --cz-peak P*
    python examples/run_irb_on_braket.py --submit --device-arn $ARN --qubits 16 25 \
        --cal-virtualz-sweep --pulse-file $PULSE --flux-frame-id $FF --drive-frame-ids $DF \
        --cz-peak P* --lengths 2 4 8 --seeds 2 --max-cost 53    # Stage 2: phases -> --virtual-z PHI0 PHI1
    python examples/run_irb_on_braket.py --submit --device-arn $ARN --qubits 16 25 \
        --combined --pulse-file $PULSE --flux-frame-id $FF --drive-frame-ids $DF \
        --cz-peak P* --virtual-z PHI0 PHI1 --max-cost 88        # Stage 3: clean A+B benchmark

Cepheus's native 2-qubit gate is CZ and gradpulse's headline architecture is the
parametric-coupler CZ, so the device gate is what the tool models. Re-optimize against the
device's real calibration before trusting an absolute number.
"""
import argparse
import os

import numpy as np

from gradpulse import braket_bridge as bb
from gradpulse.rb import _fit_single_exp        # numpy-only decay fit, reused


def load_env(path=".env"):
    """Load KEY=VALUE pairs from a .env into os.environ (this process only) so boto3
    picks up AWS creds/region. Robust to quotes; never prints values. No-op if absent."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _averaged_survival(per_seq, lengths):
    """Mean P(|00>) at each length over its seeds."""
    out = []
    for m in lengths:
        vals = [s["survival"] for s in per_seq if s["length"] == m]
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return np.array(out)


def _seed_table(seqs, lengths):
    """Per-length array of the seeds' survivals (for averaging + bootstrap)."""
    return {m: np.array([s["survival"] for s in seqs if s["length"] == m], float)
            for m in lengths}


def _fit_fixed_asymptote(lengths, y, alphas, b=0.25):
    """Single-exp fit y = A*alpha^m + b with the asymptote b FIXED (A solved by
    1-D least squares per grid alpha).

    Fixing b at the depolarizing value 1/d removes the alpha<->b degeneracy that
    makes the free-offset fit useless for a good (shallow-decay) gate: when the
    longest sequence has not pulled the survival well down toward 1/d, a free b
    trades off against alpha and both the point estimate and its variance blow up.
    Measured at Cepheus's r_CZ=0.4% (see cepheus_irb_resolution_study.py): free b at
    max length 32 gave sigma(r)=+-0.71%; fixing b (with long-enough lengths) gave
    +-0.025% -- the SAME budget, ~30x tighter."""
    m = np.asarray(lengths, float)
    yb = np.asarray(y, float) - b
    best = None
    for a in alphas:
        basis = a ** m
        denom = float(np.dot(basis, basis))
        A = float(np.dot(basis, yb) / denom) if denom > 0 else 0.0
        err = float(np.sum((A * basis - yb) ** 2))
        if best is None or err < best[0]:
            best = (err, a)
    return best[1]


def _r_cz_from_tables(ref_tab, int_tab, lengths, alphas, asymptote=0.25):
    ref_y = [ref_tab[m].mean() for m in lengths]
    int_y = [int_tab[m].mean() for m in lengths]
    if asymptote is None:                                  # legacy free-offset fit
        ar, _, _ = _fit_single_exp(lengths, ref_y, alphas)
        ai, _, _ = _fit_single_exp(lengths, int_y, alphas)
    else:
        ar = _fit_fixed_asymptote(lengths, ref_y, alphas, asymptote)
        ai = _fit_fixed_asymptote(lengths, int_y, alphas, asymptote)
    return 0.75 * (1.0 - ai / ar), ar, ai


def suggest_lengths(r_cz_estimate, t2_us=None, clifford_ns=190.0,
                    depth=0.05, max_points=8):
    """Geometric (doubling) Clifford-length ladder reaching the asymptote, for a FREE-
    asymptote fit. The longest sequence should pull the survival amplitude down to ~`depth`
    of its initial value -- i.e. essentially FLOORED -- so those points PIN the asymptote
    (which, under real asymmetric readout, is not 1/d and must be fit). Short ladders that
    stop before the floor leave the asymptote underconstrained and fit much noisier.

    The decay per Clifford combines gate error (avg Clifford ~1.88 CZ -> alpha_gate ~
    1 - (4/3)*1.88*r_cz) and, if t2_us is given, idle T2 over the per-Clifford duration
    (alpha *= exp(-clifford_ns/T2)). Shorter T2 -> the floor arrives at smaller m, so the
    ladder is shorter (Cepheus 0.4%/14us -> max ~128; a longer-T2 device needs deeper). T2
    sets WHERE the floor is, it does not forbid reaching it. Cost is per-circuit, so the
    long sequences are ~free."""
    alpha = max(1.0 - (4.0 / 3.0) * 1.88 * max(r_cz_estimate, 1e-5), 0.5)
    if t2_us:                                    # fold in idle T2 decay per Clifford
        alpha *= float(np.exp(-(clifford_ns * 1e-3) / t2_us))
    m_max = min(512, max(8, int(round(float(np.log(depth)) / float(np.log(alpha))))))
    ladder, m = [1], 2
    while m < m_max and len(ladder) < max_points - 1:
        ladder.append(m)
        m *= 2
    ladder.append(m_max)
    return ladder


def fit_irb(ref_seqs, int_seqs, lengths, n_boot=300, seed=0, asymptote=None):
    """Reference + interleaved survival decays -> CZ error per gate, WITH an error bar.

    r_CZ = (d-1)/d * (1 - alpha_int / alpha_ref), d = 4. The asymptote defaults to FREE
    (`asymptote=None`): real readout is asymmetric (Cepheus showed a '10'-biased canary),
    which shifts the true asymptote off 1/d, so FIXING it at 0.25 gives a tight-but-WRONG
    answer (study: 0.93% vs ~0.4% truth). Pass `asymptote=0.25` only for a symmetric-
    readout device whose decay is too shallow to pin a free offset (see _fit_fixed_asymptote).
    RB's decay rate -> r_CZ is SPAM-robust either way; only the asymptote handling differs.

    The error bar is a seed-bootstrap (resample the per-length seeds with replacement,
    refit, repeat). Without it the point estimate is uninterpretable on a coarse run --
    the statistical spread can exceed the gate error being measured, so you could not
    tell agreement from disagreement with gradpulse's prediction.
    """
    ref_tab, int_tab = _seed_table(ref_seqs, lengths), _seed_table(int_seqs, lengths)
    alphas = np.linspace(0.50, 0.99999, 4000)
    r_cz, a_ref, a_int = _r_cz_from_tables(ref_tab, int_tab, lengths, alphas, asymptote)

    rng = np.random.default_rng(seed)
    boot_alphas = np.linspace(0.50, 0.99999, 1000)        # coarser grid for speed
    boot = []
    for _ in range(int(n_boot)):
        rt = {m: rng.choice(ref_tab[m], size=ref_tab[m].size, replace=True)
              for m in lengths}
        it = {m: rng.choice(int_tab[m], size=int_tab[m].size, replace=True)
              for m in lengths}
        boot.append(_r_cz_from_tables(rt, it, lengths, boot_alphas, asymptote)[0])
    boot = np.asarray(boot, float)
    lo, hi = np.percentile(boot, [16, 84])                # 68% CI
    return {"alpha_ref": a_ref, "alpha_int": a_int, "r_cz": float(r_cz),
            "f_cz": float(1.0 - r_cz), "r_cz_std": float(boot.std()),
            "r_cz_ci68": [float(lo), float(hi)],
            "ref_survival": [float(ref_tab[m].mean()) for m in lengths],
            "int_survival": [float(int_tab[m].mean()) for m in lengths]}


def fit_resolved(r_cz, r_cz_std):
    """Is a short-ladder interleaved-RB fit trustworthy, or clamped on noise?

    A fit is UNRESOLVED when r_cz is statistically consistent with zero -- the exact failure
    that wasted the first $33 peak sweep: noisy short-ladder fits clamped r_cz->0, and selecting
    the minimum r_cz then crowned a destroyed gate as 'best'. Such fits must never be selected on.
    """
    return bool(r_cz > 3.0 * float(r_cz_std) and r_cz > 1e-4)


def select_best_peak(results):
    """Rank calibration peaks by MAX interleaved survival (robust; cannot clamp), NOT by min
    r_cz. Offline MC over the real Clifford group (cepheus_peak_cal_study.py) shows max-survival recovers
    the optimal peak 100% of the time vs ~50% for fit->argmin. Each result needs a 'surv_lmax'
    key. Returns the list sorted best-first."""
    return sorted(results, key=lambda d: d["surv_lmax"], reverse=True)


def load_flux_waveform(path):
    """Load a 1-D activation waveform from .npy or .json ([...] or {"flux":[...]})."""
    if path.endswith(".npy"):
        return np.load(path).ravel()
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("flux") or data.get("waveform") or data.get("samples")
    return np.asarray(data, dtype=float).ravel()


def build_levelb_bench(args, *, flux, device=None):
    """Construct the Level-B benchmarked-CZ PulseSequence from a gradpulse activation
    `flux`. With a real `device`, plays on the device's actual CZ frame anchored to its
    calibrated flux peak; without one (offline), uses synthetic frames so the circuit
    still builds + serializes. Returns (bench_pulse, peak)."""
    import json
    peak = args.cz_peak
    if peak is None and args.native_cal_file:
        with open(args.native_cal_file, encoding="utf-8") as f:
            cal = json.load(f)
        site = f"{args.qubits[0]}-{args.qubits[1]}"
        peak = bb.bench_cz_peak_from_native_calibration(cal, site)
    if peak is None:
        peak = 1.0                      # offline/no-anchor: shape only, unit full-scale

    if device is not None:
        frames = device.frames
        if not args.flux_frame_id or args.flux_frame_id not in frames:
            raise SystemExit(
                f"--flux-frame-id must name a real device frame; available include "
                f"{list(frames)[:6]}... Pick the CZ/flux frame for this pair.")
        flux_frame = frames[args.flux_frame_id]
        drive_frames = (tuple(frames[i] for i in args.drive_frame_ids)
                        if args.drive_frame_ids else None)
    else:
        flux_frame, *dfs = bb.synthetic_frames(3)      # coupler + 2 drives (offline)
        drive_frames = (dfs[0], dfs[1]) if args.drive_frame_ids else None

    bench = bb.build_bench_cz_pulse_sequence(
        flux, flux_frame, peak_amplitude=float(peak), drive_frames=drive_frames,
        virtual_z=tuple(args.virtual_z))
    return bench, float(peak)


def run_combined(args, lengths, ref, intl):
    """Level A + Level B in ONE interleaved-RB run against a SHARED reference.

    Interleaved RB compares an interleaved decay to a *reference* decay, and the SAME
    reference legitimately serves more than one gate-under-test. So we run the reference
    ONCE and the interleaved sequences TWICE -- once binding CZ_BENCH to the device's
    native CZ (bench=None) and once to the gradpulse pulse (bench=PulseSequence). That is
    the maximally controlled native-vs-pulse comparison: identical random Cliffords, one
    shared reference, same session/drift/calibration. Circuits = 56 ref + 56 native + 56
    pulse = 168, vs 224 (= paying for the reference twice) if A and B are run separately.
    Cost: $86.10 @ 500 shots + 3 pre-flight canaries (~$1.03) = ~$87.13 on Cepheus.
    """
    import json
    if not args.pulse_file:
        raise SystemExit("--combined needs --pulse-file (the Level-B activation waveform); "
                         "generate one with examples/cepheus/levelb_pulse_benchmark_offline.py --tunable.")

    # The same interleaved sequences are run two ways; copy so the native and pulse
    # survivals don't collide on one dict. "gates" is shared (read-only here).
    for s in ref:
        s["group"] = "ref"
    intl_native = [dict(s, group="native") for s in intl]
    intl_pulse = [dict(s, group="pulse") for s in intl]
    all_seqs = ref + intl_native + intl_pulse

    # Offline correctness gate: every circuit must return to |00> ideally. native/pulse do
    # not change the ideal (the pulse IS a CZ), so checking ref+intl once covers all 168.
    uniq = ref + intl
    bad = [s for s in uniq if abs(bb.ideal_survival_probability(s["gates"]) - 1.0) > 1e-9]
    print(f"COMBINED A+B: {len(all_seqs)} circuits = {len(ref)} shared reference "
          f"+ {len(intl_native)} native-CZ interleaved + {len(intl_pulse)} gradpulse-pulse "
          f"interleaved (shared reference -> {len(ref)} circuits saved vs running separately)")
    print(f"ideal return-to-|00> check: {'PASS' if not bad else f'FAIL ({len(bad)} bad)'}")
    if bad:
        raise SystemExit("aborting: some ideal circuits do not return to |00>.")

    cost = bb.estimate_experiment_cost(len(all_seqs), args.shots)
    n_canary = 3
    ccost = bb.estimate_experiment_cost(n_canary, args.canary_shots)
    total = cost.total_usd + ccost.total_usd
    print(f"cost @ {args.shots} shots: ${cost.total_usd:.2f} "
          f"(task ${cost.task_fee_usd:.2f} + shot ${cost.shot_fee_usd:.2f})")
    print(f"  + {n_canary} pre-flight canaries @ {args.canary_shots} shots: ${ccost.total_usd:.2f}")
    print(f"  = ${total:.2f} TOTAL  (pricing {cost.pricing_as_of})")

    flux = load_flux_waveform(args.pulse_file)

    if not args.submit:
        # Offline: serialize the pulse-interleaved circuit on synthetic frames, and run the
        # NATIVE pipeline (ref + native-interleaved) on the noiseless local sim (pulses can't
        # run there). Proves both circuit families build before any spend.
        bench, peak = build_levelb_bench(args, flux=flux, device=None)
        rep = bb.verify_levelb_offline(bench, qubits=tuple(args.qubits))
        print(f"Level-B pulse: {flux.size} samples, anchor peak {peak:g} (synthetic frame offline)")
        print(f"  pulse-interleaved serializes: verbatim={rep['verbatim_pragma_present']}, "
              f"play={rep['play_present']}, clifford_closes={rep['ideal_clifford_closes']}, "
              f"offline_ok={rep['offline_ok']}")
        from braket.devices import LocalSimulator
        sim = LocalSimulator()
        deepest = max(intl, key=lambda s: len(s["gates"]))
        ok = rep["offline_ok"]
        for tag, seq in (("shortest-ref", ref[0]), ("deepest-native", deepest)):
            rc = bb.to_braket_rb_circuit(seq["gates"], qubits=(0, 1), verbatim=False)
            sv = bb.survival_from_counts(sim.run(rc, shots=400).result().measurement_counts)
            ok &= sv > 0.999
            print(f"  native rehearsal {tag} (length {seq['length']}, {len(seq['gates'])} "
                  f"gates): survival {sv:.4f} ({'PASS' if sv > 0.999 else 'FAIL'})")
        print(f"\noffline COMBINED check: {'PASS' if ok else 'FAIL -- do NOT submit'}")
        print("DRY RUN -- nothing submitted. Re-run with --submit --device-arn <ARN> "
              "--flux-frame-id <id> [--native-cal-file <json> | --cz-peak <p>] --max-cost "
              f"{int(total) + 1} to execute on silicon.")
        return

    # ----------------------- submit -----------------------
    if not args.device_arn:
        raise SystemExit("--submit needs --device-arn <Braket QPU ARN>.")
    if not args.canary_only and cost.total_usd > args.max_cost:
        raise SystemExit(
            f"aborting: estimated ${cost.total_usd:.2f} exceeds --max-cost ${args.max_cost:.2f}. "
            f"Re-run with --max-cost {int(cost.total_usd) + 1} (or higher) to proceed.")

    import boto3
    from braket.aws import AwsDevice, AwsSession
    bucket = args.s3_bucket or os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        raise SystemExit("--submit needs an S3 bucket (--s3-bucket or $AWS_S3_BUCKET).")
    s3_folder = (bucket, args.s3_prefix)
    session = AwsSession(boto_session=boto3.Session(region_name=args.region))
    device = AwsDevice(args.device_arn, aws_session=session)
    status = getattr(device, "status", "UNKNOWN")
    print(f"device {args.device_arn}\n  region={args.region}  status={status}  "
          f"s3=s3://{bucket}/{args.s3_prefix}")
    if status != "ONLINE":
        raise SystemExit(f"aborting: device is {status}, not ONLINE. Submit during its window.")

    qubits = tuple(args.qubits)
    bench, peak = build_levelb_bench(args, flux=flux, device=device)
    rep = bb.verify_levelb_offline(bench, qubits=qubits)
    print(f"LEVEL B pulse ({flux.size} samples) on frame {args.flux_frame_id}, "
          f"anchored to device CZ flux peak {peak:g}")
    print(f"  offline serialization: {'PASS' if rep['offline_ok'] else 'FAIL'}")
    if not rep["offline_ok"]:
        raise SystemExit("aborting: Level-B circuit failed offline serialization.")
    print("  NOTE: Level B is OPEN-LOOP -- expect it BELOW the native CZ until on-device cal.")

    def _run(seq, shots, *, pulse=None, announce=False):
        # disable_qubit_rewiring=True is MANDATORY with a verbatim box (physical qubits).
        circ = bb.to_braket_rb_circuit(seq["gates"], qubits=qubits, bench_cz_pulse=pulse)
        task = device.run(circ, shots=shots, disable_qubit_rewiring=True,
                          s3_destination_folder=s3_folder)
        if announce:
            print(f"    task ARN: {task.id}  (retrievable if this process dies)")
        counts = task.result().measurement_counts
        return counts, bb.survival_from_counts(counts)

    # --- 3 canaries: (1) sanity on a native reference, (2) native at max depth,
    #     (3) the PULSE at max depth (the novel pulse_gate-in-verbatim-at-depth path). ---
    cc = bb.estimate_experiment_cost(1, args.canary_shots).total_usd
    deepest = max(intl, key=lambda s: len(s["gates"]))

    def _check(label, counts, surv, *, lo=0.2, hi_reject=0.90):
        keys = sorted(counts.keys())
        print(f"  {label}: counts {keys}  survival {surv:.3f}")
        if any(len(k) != 2 for k in keys):
            raise SystemExit(f"aborting: {label} returned non-2-bit counts -- qubit mapping/bit-order off.")
        if surv > hi_reject:
            raise SystemExit(f"aborting: {label} survival {surv:.3f} implausibly HIGH "
                             "-- verbatim box likely collapsed; the decay would be flat.")
        if not (lo < surv <= 1.0):
            raise SystemExit(f"aborting: {label} survival {surv:.3f} implausibly low ({lo}<s<=1 expected).")

    print(f"canary 1/3 (sanity: native reference, length {ref[0]['length']}) "
          f"@ {args.canary_shots} shots (~${cc:.2f}) ...")
    c, s1 = _run(ref[0], args.canary_shots, pulse=None, announce=True)
    _check("sanity", c, s1, hi_reject=1.01)        # shallow ref may survive ~high; only reject >1
    print(f"canary 2/3 (native CZ at max depth, length {deepest['length']}, "
          f"{len(deepest['gates'])} gates) @ {args.canary_shots} shots (~${cc:.2f}) ...")
    c, s2 = _run(deepest, args.canary_shots, pulse=None, announce=True)
    _check("native-depth", c, s2)
    print(f"canary 3/3 (gradpulse PULSE at max depth, length {deepest['length']}, "
          f"{len(deepest['gates'])} gates) @ {args.canary_shots} shots (~${cc:.2f}) ...")
    c, s3 = _run(deepest, args.canary_shots, pulse=bench, announce=True)
    _check("pulse-depth", c, s3)

    if args.canary_only:
        print(f"\nCANARY-ONLY: 3 canary tasks (~${3 * cc:.2f}). sanity {s1:.3f}, "
              f"native-depth {s2:.3f}, pulse-depth {s3:.3f} are sane. STOPPING before the "
              f"{len(all_seqs)}-circuit batch. Re-run without --canary-only to run it.")
        return

    # --- full run, saved incrementally so a late failure never loses paid-for data ---
    print(f"submitting {len(all_seqs)} circuits x {args.shots} shots (saving to {args.out}) ...")
    for i, s in enumerate(all_seqs):
        s["survival"] = _run(s, args.shots, pulse=(bench if s["group"] == "pulse" else None))[1]
        if (i + 1) % 10 == 0 or i + 1 == len(all_seqs):
            with open(args.out, "w") as f:
                json.dump([{k: v for k, v in t.items() if k != "gates"} for t in all_seqs],
                          f, indent=2)
            print(f"  {i + 1}/{len(all_seqs)} done (saved)")

    # --- two fits against the SHARED reference (alpha_ref identical by construction) ---
    asym = 0.25 if args.fixed_asymptote else None
    res_a = fit_irb(ref, intl_native, lengths, asymptote=asym)
    res_b = fit_irb(ref, intl_pulse, lengths, asymptote=asym)
    print(f"\nLEVEL A (native CZ):   r = {res_a['r_cz']:.4e} +/- {res_a['r_cz_std']:.1e}   "
          f"F = {res_a['f_cz']:.5f}   68% CI [{res_a['r_cz_ci68'][0]:.3e}, {res_a['r_cz_ci68'][1]:.3e}]")
    print(f"LEVEL B (gradpulse, OPEN-LOOP):  r = {res_b['r_cz']:.4e} +/- {res_b['r_cz_std']:.1e}   "
          f"F = {res_b['f_cz']:.5f}   68% CI [{res_b['r_cz_ci68'][0]:.3e}, {res_b['r_cz_ci68'][1]:.3e}]")
    print(f"  shared alpha_ref = {res_a['alpha_ref']:.5f} (identical for both fits -- the point "
          f"of one run); alpha_int native {res_a['alpha_int']:.5f} / pulse {res_b['alpha_int']:.5f}")
    print("Level A validates gradpulse's coherence-limited MODEL (compare against "
          "ParametricCouplerProfile.from_braket_calibration + error_budget). Level B is the "
          "OPEN-LOOP pulse transfer -- expected below native until on-device calibration.")


def run_cal_peak_sweep(args, lengths, ref, intl):
    """On-device PEAK-amplitude calibration of the Level-B pulse (the cheap closed-loop step).

    The single highest-leverage open-loop -> on-device knob is the flux peak: it sets the
    |11>-|02> entangling angle, so a wrong peak is a large coherent over/under-rotation. This
    runs interleaved RB of the gradpulse pulse at each peak in ``--peak-grid`` against ONE
    shared native reference (run once), and picks the peak with the highest interleaved
    survival -- the device-calibrated anchor to feed the combined A+B run's ``--cz-peak``.

    Selection is by MAX SURVIVAL, not the interleaved-RB fit. Offline MC over the real Clifford
    group (``cepheus_peak_cal_study.py``) shows max-survival recovers the optimal peak 100% of the time vs
    ~50% for fit->argmin, which on short-ladder noise clamps r_cz->0 and mislabels a BAD peak as
    best (exactly what wasted the first $33 sweep). The fit is still reported as the fidelity
    estimate, flagged UNRESOLVED when it is statistically consistent with zero. Run SHORT
    (``--lengths 1 2 4 8 --seeds 2``); the reference is amortized over all peaks.

    Pass ``--drive-frame-ids`` + ``--virtual-z`` (the pulse's MODEL phase estimate) so the bare
    single-qubit phase does not inflate r_cz at every peak -- otherwise the best-peak fidelity
    reads pessimistically and the GO/NO-GO is muddier. The peak that minimizes r_cz is found
    either way (the phase penalty is ~peak-independent); the phases are then refined in Stage 2.
    """
    import json
    if not args.pulse_file:
        raise SystemExit("--cal-peak-sweep needs --pulse-file (the Level-B activation waveform).")
    peaks = list(args.peak_grid)
    n_circ = len(ref) + len(peaks) * len(intl)
    cost = bb.estimate_experiment_cost(n_circ, args.shots)
    ccost = bb.estimate_experiment_cost(2, args.canary_shots)   # native + pulse canaries
    total = cost.total_usd + ccost.total_usd
    print(f"CAL PEAK SWEEP: {len(peaks)} peaks {peaks}")
    print(f"  {n_circ} circuits = {len(ref)} shared native reference "
          f"+ {len(peaks)}x{len(intl)} pulse-interleaved")
    print(f"cost @ {args.shots} shots: ${cost.total_usd:.2f} + 2 canaries (native + pulse) "
          f"${ccost.total_usd:.2f} = ${total:.2f} TOTAL  (pricing {cost.pricing_as_of})")
    if not args.drive_frame_ids or not any(args.virtual_z):
        print("  NOTE: no --drive-frame-ids/--virtual-z -> running the BARE entangling flux; "
              "uncompensated single-qubit phase inflates r_cz, so best-peak F is a LOWER BOUND "
              "(the peak* is still correct). Pass the model virtual-Z for a cleaner GO/NO-GO.")

    bad = [s for s in ref + intl if abs(bb.ideal_survival_probability(s["gates"]) - 1.0) > 1e-9]
    print(f"ideal return-to-|00> check: {'PASS' if not bad else f'FAIL ({len(bad)})'}")
    if bad:
        raise SystemExit("aborting: ideal circuits do not return to |00>.")

    flux = load_flux_waveform(args.pulse_file)

    if not args.submit:
        bench, _ = build_levelb_bench(args, flux=flux, device=None)
        rep = bb.verify_levelb_offline(bench, qubits=tuple(args.qubits))
        print(f"Level-B pulse {flux.size} samples; offline serialize verbatim="
              f"{rep['verbatim_pragma_present']} play={rep['play_present']} ok={rep['offline_ok']}")
        print(f"\nDRY RUN. Re-run with --submit --device-arn <ARN> --flux-frame-id <id> "
              f"--max-cost {int(total) + 1} to calibrate on silicon.")
        return

    if not args.device_arn:
        raise SystemExit("--submit needs --device-arn <Braket QPU ARN>.")
    if not args.canary_only and cost.total_usd > args.max_cost:
        raise SystemExit(f"aborting: ${cost.total_usd:.2f} exceeds --max-cost ${args.max_cost:.2f}. "
                         f"Re-run with --max-cost {int(cost.total_usd) + 1} to proceed.")
    import boto3
    from braket.aws import AwsDevice, AwsSession
    bucket = args.s3_bucket or os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        raise SystemExit("--submit needs an S3 bucket (--s3-bucket or $AWS_S3_BUCKET).")
    s3_folder = (bucket, args.s3_prefix)
    device = AwsDevice(args.device_arn,
                       aws_session=AwsSession(boto_session=boto3.Session(region_name=args.region)))
    if getattr(device, "status", "UNKNOWN") != "ONLINE":
        raise SystemExit(f"aborting: device is {getattr(device, 'status', '?')}, not ONLINE.")
    qubits = tuple(args.qubits)
    flux_frame = device.frames[args.flux_frame_id]
    drive_frames = (tuple(device.frames[i] for i in args.drive_frame_ids)
                    if args.drive_frame_ids else None)

    def _run(seq, shots, pulse=None):
        circ = bb.to_braket_rb_circuit(seq["gates"], qubits=qubits, bench_cz_pulse=pulse)
        task = device.run(circ, shots=shots, disable_qubit_rewiring=True,
                          s3_destination_folder=s3_folder)
        return bb.survival_from_counts(task.result().measurement_counts)

    def _canary(label, seq, pulse):
        circ = bb.to_braket_rb_circuit(seq["gates"], qubits=qubits, bench_cz_pulse=pulse)
        task = device.run(circ, shots=args.canary_shots, disable_qubit_rewiring=True,
                          s3_destination_folder=s3_folder)
        print(f"canary {label} (length {seq['length']}) @ {args.canary_shots} shots -> task {task.id}")
        counts = task.result().measurement_counts
        keys = sorted(str(k) for k in counts)
        surv = bb.survival_from_counts(counts)
        print(f"  keys={keys}  survival {surv:.3f}")
        if not set(keys) <= {"00", "01", "10", "11"}:
            raise SystemExit(f"aborting: canary {label} returned non-2-bit keys {keys} -- mapping wrong.")
        return surv

    # Canary 1: native reference -> the device returns sane data on this pair right now.
    s_nat = _canary("1/native-reference", ref[0], None)
    if not (0.2 < s_nat <= 1.01):
        raise SystemExit(f"aborting: native canary survival {s_nat:.3f} implausible -- diagnose first.")

    # Canary 2: proves the COMPOSABLE pulse AND the shift_phase on the real drive frames
    # PLAY on silicon (neither has run on hardware before) -- the real pre-flight for the sweep.
    canary_peak = peaks[len(peaks) // 2]
    cbench = bb.build_bench_cz_pulse_sequence(
        flux, flux_frame, peak_amplitude=float(canary_peak), drive_frames=drive_frames,
        virtual_z=tuple(args.virtual_z))
    print(f"canary 2 plays the gradpulse pulse + virtual-Z {tuple(args.virtual_z)} @ peak {canary_peak:.3f}")
    s_pulse = _canary("2/pulse+virtualz", intl[0], cbench)
    if s_pulse <= 0.10:
        raise SystemExit(f"aborting: pulse canary survival {s_pulse:.3f} at the floor -- the pulse may "
                         "not be playing (check --flux-frame-id / --drive-frame-ids / verbatim).")
    print("  (a modest value is EXPECTED pre-calibration: this canary checks the pulse PLAYS and "
          "returns sane 2-bit data, not its fidelity.)")

    if args.canary_only:
        print(f"\nCANARY-ONLY: both canaries done (~${ccost.total_usd:.2f}). Native data sane; the "
              "corrected pulse + virtual-Z play on silicon and map to 2-bit counts. Re-run WITHOUT "
              "--canary-only (--max-cost 34) to do the full peak sweep.")
        return

    # shared native reference (run ONCE, amortized over all peaks)
    print(f"reference: {len(ref)} native circuits @ {args.shots} shots ...")
    for s in ref:
        s["survival"] = _run(s, args.shots, pulse=None)

    asym = 0.25 if args.fixed_asymptote else None
    Lmax = max(lengths)
    ref_lmax = float(np.mean([s["survival"] for s in ref if int(s["length"]) == Lmax]))
    results = []
    for pk in peaks:
        bench = bb.build_bench_cz_pulse_sequence(
            flux, flux_frame, peak_amplitude=float(pk), drive_frames=drive_frames,
            virtual_z=tuple(args.virtual_z))
        intl_pk = [dict(s) for s in intl]
        for s in intl_pk:
            s["survival"] = _run(s, args.shots, pulse=bench)
        # RAW interleaved survival per length -- the ROBUST selector (see select_best_peak).
        # Keep the fit as the FIDELITY estimate but never select on it.
        by_len = {}
        for s in intl_pk:
            by_len.setdefault(int(s["length"]), []).append(float(s["survival"]))
        surv_by_len = {m: float(np.mean(v)) for m, v in by_len.items()}
        s_lmax = surv_by_len[Lmax]
        res = fit_irb(ref, intl_pk, lengths, asymptote=asym)
        resolved = fit_resolved(res["r_cz"], res["r_cz_std"])
        results.append({"peak": float(pk), "r_cz": res["r_cz"], "r_cz_std": res["r_cz_std"],
                        "f_cz": res["f_cz"], "surv_lmax": s_lmax,
                        "surv_by_len": surv_by_len, "fit_resolved": resolved})
        print(f"  peak {pk:.4f}: surv(L={Lmax})={s_lmax:.3f}  |  fit r_cz={res['r_cz']:.4e} "
              f"+/- {res['r_cz_std']:.1e}{'' if resolved else '  (UNRESOLVED)'}")
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # SELECT on max interleaved survival (robust, cannot clamp) -- NOT min-r_cz.
    ranked = select_best_peak(results)
    best = ranked[0]
    print(f"\nBEST peak (max survival @ L={Lmax}) = {best['peak']:.4f}  "
          f"survival {best['surv_lmax']:.3f}")
    if len(ranked) > 1:
        print(f"  runner-up: peak {ranked[1]['peak']:.4f}  survival {ranked[1]['surv_lmax']:.3f}")
    print(f"  native reference survival @ L={Lmax}: {ref_lmax:.3f}")
    # GO/NO-GO via the survival RATIO best/native at the same length (anchor: a known-bad gate
    # measured ratio ~0.4 at L=1; a near-ideal gate -> ratio ~1). Read against that anchor
    # rather than a single hard cutoff, since the absolute value depends on L.
    ratio = best["surv_lmax"] / ref_lmax if ref_lmax > 0 else 0.0
    ref1 = float(np.mean([s["survival"] for s in ref if int(s["length"]) == min(lengths)]))
    best1 = best["surv_by_len"].get(min(lengths), float("nan"))
    ratio1 = best1 / ref1 if ref1 > 0 else 0.0
    print(f"  survival ratio best/native: {ratio:.2f} @ L={Lmax},  {ratio1:.2f} @ L={min(lengths)}"
          f"  (anchor: the bad $33 gate was ~0.4)")
    if ratio1 < 0.7:
        print("  >>> NO-GO signal: even the best peak's gate is far below native (ratio < 0.7, "
              "like the bad $33 gate). The open-loop pulse does not transfer well at any peak in "
              "this grid -- do NOT fund the full run; Level A is the standing result.")
    else:
        print("  >>> GO signal: the best peak's gate approaches native (ratio >= 0.7). A fine "
              "grid + longer lengths around this peak can resolve a fidelity number worth running.")
    if best["fit_resolved"]:
        print(f"  fit at best peak: r_cz = {best['r_cz']:.4e}  F = {best['f_cz']:.5f}")
    else:
        print("  fit at best peak is UNRESOLVED (short ladder) -- re-run a FINE grid around this "
              "peak with longer lengths for a resolved fidelity number.")
    print(f"Feed --cz-peak {best['peak']:.4f} to Stage 2 (--cal-virtualz-sweep) then the combined "
          "A+B run -- it's the device-calibrated entangling anchor (the dominant open-loop scalar).")


def run_virtualz_sweep(args, lengths, ref, intl):
    """STAGE 2 on-device calibration: single-qubit VIRTUAL-Z phases of the Level-B pulse.

    After the peak is set (Stage 1), the gradpulse flux still leaves uncalibrated single-qubit
    phases -- on hardware the pulse realizes ``CZ . Z0(phi0_hw) (x) Z1(phi1_hw)``. Virtual-Z
    (free ``shift_phase`` frame updates, exactly what the device's native CZ applies) cancels them.

    JOINT 2-D grid, NOT sequential 1-D. The two phases are COUPLED: the gate fidelity is nearly
    flat in either phase alone but peaks at a specific ``(phi0,phi1)`` pair, so a sequential
    1-D-then-1-D sweep gets stuck near the start. A closed-loop rehearsal over the real
    cz_data_virtualz landscape (examples/cepheus_closed_loop_cal.py) confirmed this: sequential
    recovered ~5% of the gap to the optimum, the joint 2-D grid ~91%. Selection is by MAX
    interleaved survival (the RB fit is biased for low-fidelity gates -- it over-rates near-
    depolarized points -- so argmin-r_cz can crown a bad phase). Cost is ``len(grid)^2`` interleaved
    blocks, so keep ``--vz-grid`` modest (e.g. 7-9 points). Run SHORT (``--lengths 2 4 8 --seeds
    2``); the native reference is amortized over every phase. Needs ``--cz-peak`` (from Stage 1)
    and, on ``--submit``, ``--drive-frame-ids`` (the q0/q1 drive frames the phases shift).
    """
    import json
    import numpy as np
    if not args.pulse_file:
        raise SystemExit("--cal-virtualz-sweep needs --pulse-file (the Level-B activation waveform).")
    if args.cz_peak is None and not args.native_cal_file:
        raise SystemExit("--cal-virtualz-sweep needs --cz-peak (the Stage-1 calibrated peak) "
                         "or --native-cal-file to anchor it.")
    grid = list(args.vz_grid)
    n_pts = len(grid) ** 2                      # JOINT 2-D (phi0, phi1) grid
    n_circ = len(ref) + n_pts * len(intl)
    cost = bb.estimate_experiment_cost(n_circ, args.shots)
    ccost = bb.estimate_experiment_cost(1, args.canary_shots)
    total = cost.total_usd + ccost.total_usd
    print(f"CAL VIRTUAL-Z SWEEP (joint 2-D): {len(grid)}x{len(grid)} = {n_pts} (phi0,phi1) points "
          f"over {[round(g, 3) for g in grid]} rad")
    print(f"  {n_circ} circuits = {len(ref)} shared native reference + {n_pts}x{len(intl)} "
          f"pulse-interleaved")
    print(f"cost @ {args.shots} shots: ${cost.total_usd:.2f} + 1 canary ${ccost.total_usd:.2f} "
          f"= ${total:.2f} TOTAL  (pricing {cost.pricing_as_of})")

    bad = [s for s in ref + intl if abs(bb.ideal_survival_probability(s["gates"]) - 1.0) > 1e-9]
    print(f"ideal return-to-|00> check: {'PASS' if not bad else f'FAIL ({len(bad)})'}")
    if bad:
        raise SystemExit("aborting: ideal circuits do not return to |00>.")

    flux = load_flux_waveform(args.pulse_file)

    if not args.submit:
        # Offline: the bench pulse must build + serialize at the grid phases (virtual-Z is a
        # shift_phase on the drive frames); pulses can't run on the local sim, so fidelity is a
        # QPU-only number, but every circuit family is proven to construct here for free.
        bench0, _ = build_levelb_bench(args, flux=flux, device=None)   # default phases
        rep = bb.verify_levelb_offline(bench0, qubits=tuple(args.qubits))
        # exercise two NONZERO (phi0, phi1) to prove both virtual-Z frame shifts serialize
        fr, d0, d1 = bb.synthetic_frames(3)
        probe = bb.build_bench_cz_pulse_sequence(
            flux, fr, peak_amplitude=float(args.cz_peak or 1.0), drive_frames=(d0, d1),
            virtual_z=(0.5, 1.0))
        sh = str(probe.to_ir()).count("shift_phase")
        print(f"Level-B pulse {flux.size} samples; offline serialize verbatim="
              f"{rep['verbatim_pragma_present']} play={rep['play_present']} ok={rep['offline_ok']}")
        print(f"  virtual-Z plumbing: shift_phase ops for 2 nonzero phases = {sh} "
              f"({'PASS' if sh == 2 else 'FAIL'})")
        if sh != 2:
            raise SystemExit("aborting: virtual-Z frame shifts did not serialize.")
        print(f"\nDRY RUN. Re-run with --submit --device-arn <ARN> --flux-frame-id <id> "
              f"--drive-frame-ids <F0> <F1> --cz-peak <p> --max-cost {int(total) + 1}.")
        return

    if not args.device_arn:
        raise SystemExit("--submit needs --device-arn <Braket QPU ARN>.")
    if not args.drive_frame_ids:
        raise SystemExit("--cal-virtualz-sweep --submit needs --drive-frame-ids <F0> <F1> "
                         "(virtual-Z shifts those frames; without them it is a no-op).")
    if cost.total_usd > args.max_cost:
        raise SystemExit(f"aborting: ${cost.total_usd:.2f} exceeds --max-cost ${args.max_cost:.2f}. "
                         f"Re-run with --max-cost {int(cost.total_usd) + 1} to proceed.")
    import boto3
    from braket.aws import AwsDevice, AwsSession
    bucket = args.s3_bucket or os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        raise SystemExit("--submit needs an S3 bucket (--s3-bucket or $AWS_S3_BUCKET).")
    s3_folder = (bucket, args.s3_prefix)
    device = AwsDevice(args.device_arn,
                       aws_session=AwsSession(boto_session=boto3.Session(region_name=args.region)))
    if getattr(device, "status", "UNKNOWN") != "ONLINE":
        raise SystemExit(f"aborting: device is {getattr(device, 'status', '?')}, not ONLINE.")
    qubits = tuple(args.qubits)
    flux_frame = device.frames[args.flux_frame_id]
    drive_frames = tuple(device.frames[i] for i in args.drive_frame_ids)
    peak = float(args.cz_peak) if args.cz_peak is not None else \
        bb.bench_cz_peak_from_native_calibration(json.load(open(args.native_cal_file)),
                                                 f"{qubits[0]}-{qubits[1]}")

    def _run(seq, shots, pulse=None):
        circ = bb.to_braket_rb_circuit(seq["gates"], qubits=qubits, bench_cz_pulse=pulse)
        task = device.run(circ, shots=shots, disable_qubit_rewiring=True,
                          s3_destination_folder=s3_folder)
        return bb.survival_from_counts(task.result().measurement_counts)

    print(f"canary (native reference, length {ref[0]['length']}) @ {args.canary_shots} shots ...")
    csurv = _run(ref[0], args.canary_shots, pulse=None)
    print(f"  survival {csurv:.3f}")
    if not (0.2 < csurv <= 1.01):
        raise SystemExit(f"aborting: canary survival {csurv:.3f} implausible -- diagnose first.")

    print(f"reference: {len(ref)} native circuits @ {args.shots} shots ...")
    for s in ref:
        s["survival"] = _run(s, args.shots, pulse=None)

    asym = 0.25 if args.fixed_asymptote else None
    Lmax = max(lengths)

    def _meas_at(phi0, phi1):
        bench = bb.build_bench_cz_pulse_sequence(
            flux, flux_frame, peak_amplitude=peak, drive_frames=drive_frames,
            virtual_z=(float(phi0), float(phi1)))
        intl_v = [dict(s) for s in intl]
        for s in intl_v:
            s["survival"] = _run(s, args.shots, pulse=bench)
        res = fit_irb(ref, intl_v, lengths, asymptote=asym)
        res["surv_lmax"] = float(np.mean([s["survival"] for s in intl_v
                                          if int(s["length"]) == Lmax]))
        return res

    # SELECT virtual-Z by MAX interleaved survival over a JOINT 2-D grid (the phases are coupled;
    # sequential 1-D gets stuck -- rehearsal-validated). Survival, not min(fit r_cz): the RB fit is
    # biased for low-fidelity gates (over-rates near-depolarized points), so argmin-r_cz can crown a
    # bad phase. r_cz is recorded as a diagnostic.
    results = []
    print(f"joint 2-D virtual-Z grid ({len(grid)}x{len(grid)} points):")
    for phi0 in grid:
        for phi1 in grid:
            res = _meas_at(phi0, phi1)
            results.append({"phi0": float(phi0), "phi1": float(phi1), "r_cz": res["r_cz"],
                            "r_cz_std": res["r_cz_std"], "f_cz": res["f_cz"],
                            "surv_lmax": res["surv_lmax"]})
            print(f"  ({phi0:.2f},{phi1:.2f}): surv(L={Lmax})={res['surv_lmax']:.3f}  "
                  f"r_cz={res['r_cz']:.4e}")
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)
    best = max(results, key=lambda d: d["surv_lmax"])
    print(f"\nBEST virtual-Z (max survival) = ({best['phi0']:.4f}, {best['phi1']:.4f}) rad  ->  "
          f"surv(L={Lmax})={best['surv_lmax']:.3f}  r_cz = {best['r_cz']:.4e}")
    print(f"Feed --virtual-z {best['phi0']:.4f} {best['phi1']:.4f} (with --cz-peak from Stage 1) "
          "to the combined A+B run -- the device-calibrated Level-B gate.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submit", action="store_true",
                    help="actually run on the device (costs money + needs AWS creds)")
    ap.add_argument("--canary-only", action="store_true",
                    help="submit ONLY the two canary tasks (length-1 + deepest, ~$0.68) and "
                         "STOP before the full batch -- spend the minimum to confirm the "
                         "device returns sane data AND accepts the batch's max depth first")
    ap.add_argument("--device-arn", default=None, help="Braket QPU device ARN")
    ap.add_argument("--qubits", type=int, nargs=2, default=[0, 1],
                    metavar=("Q0", "Q1"), help="the two physical device qubits")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128],
                    help="Clifford sequence lengths. GO LONG: the longest sequences floor at "
                         "the asymptote, and because real readout is asymmetric the asymptote "
                         "is NOT 1/d and must be FIT -- the floored long points are what pin "
                         "it (a short ladder leaves it underconstrained, ~7x noisier). Default "
                         "reaches m=128. T2 sets WHERE the floor is (Cepheus ~m=44), not a cap; "
                         "go ~2-3x past it. Tune via suggest_lengths(r_cz, t2_us=<device T2>).")
    ap.add_argument("--fixed-asymptote", action="store_true",
                    help="fix the RB asymptote at 1/d=0.25 instead of fitting it free "
                         "(default). Only for a SYMMETRIC-readout device whose decay is too "
                         "shallow to pin a free offset; on a device with asymmetric readout "
                         "(e.g. Cepheus) this gives a tight-but-biased result -- leave it off.")
    ap.add_argument("--buffer-bench-cz", action="store_true",
                    help="wrap each benchmarked CZ in barriers so it is never abutted to a "
                         "Clifford's own CZ. Addresses the back-to-back-flux context bias that "
                         "inflated a real Level-A run ~2.5x above the isolated-gate error "
                         "(RESULTS.md S10). Offline-verified to remove the adjacency; whether it "
                         "recovers the isolated ~0.5% on silicon is UNVERIFIED (non-Markovian "
                         "effect; possible flux-predistortion residual) -- needs a hardware run.")
    ap.add_argument("--seeds", type=int, default=7, help="random sequences per length")
    ap.add_argument("--shots", type=int, default=500)
    ap.add_argument("--canary-shots", type=int, default=100,
                    help="shots for the single pre-flight canary task (keeps it ~$0.34, "
                         "not a full-shots ~$0.73)")
    ap.add_argument("--max-cost", type=float, default=50.0,
                    help="abort --submit if the estimate exceeds this (guards against "
                         "an accidental big spend; raise it deliberately to proceed)")
    ap.add_argument("--out", default="irb_results.json",
                    help="incremental results file (written as the run proceeds, so a "
                         "late failure never loses the circuits already paid for)")
    ap.add_argument("--region", default="us-west-1",
                    help="AWS region of the device (Rigetti Cepheus is us-west-1)")
    ap.add_argument("--s3-bucket", default=None,
                    help="S3 bucket for results (default: $AWS_S3_BUCKET); must be in "
                         "the same region as the device")
    ap.add_argument("--s3-prefix", default="gradpulse-irb",
                    help="S3 key prefix for result objects")
    ap.add_argument("--pulse", action="store_true",
                    help="Level B: benchmark a gradpulse-DESIGNED pulse instead of the "
                         "native CZ. Needs --pulse-file; on --submit also reads the "
                         "device's CZ frame + calibrated flux peak (--native-cal-file / "
                         "--flux-frame-id). Open-loop transfer -- see the module docstring.")
    ap.add_argument("--pulse-file", default=None,
                    help="Level-B activation waveform: .npy/.json 1-D samples of the gate "
                         "ACTIVATION channel (e.g. the coupler channel of a "
                         "tunable_coupler_cz best_waveform). See "
                         "examples/cepheus/levelb_pulse_benchmark_offline.py to generate + inspect one free.")
    ap.add_argument("--native-cal-file", default=None,
                    help="downloaded native-gate-calibration JSON; used to anchor the "
                         "Level-B pulse to the device's own CZ flux peak for this pair")
    ap.add_argument("--flux-frame-id", default=None,
                    help="device Frame id the native CZ plays on (the Level-B pulse plays "
                         "on the same frame). Look it up in device.frames.")
    ap.add_argument("--drive-frame-ids", nargs=2, default=None,
                    metavar=("F0", "F1"), help="optional drive Frame ids for the "
                    "single-qubit virtual-Z corrections (--virtual-z)")
    ap.add_argument("--virtual-z", type=float, nargs=2, default=[0.0, 0.0],
                    metavar=("PHI0", "PHI1"),
                    help="virtual-Z phases (rad) on the drive frames; refine on-device")
    ap.add_argument("--cz-peak", type=float, default=None,
                    help="override the Level-B anchor flux peak (else read from --native-cal-file)")
    ap.add_argument("--combined", action="store_true",
                    help="Level A + Level B in ONE run against a SHARED reference: the native "
                         "CZ and the gradpulse pulse are both interleaved-RB'd against the same "
                         "reference (56 ref + 56 native + 56 pulse = 168 circuits, ~$86 @ 500 "
                         "shots + 3 canaries). Same controlled comparison, ~$29 cheaper than two "
                         "separate runs. Needs --pulse-file (+ device frame/peak on --submit).")
    ap.add_argument("--cal-peak-sweep", action="store_true",
                    help="On-device PEAK-amplitude calibration of the Level-B pulse: interleaved "
                         "RB of the gradpulse pulse at each --peak-grid value against ONE shared "
                         "native reference; fit r_cz(peak) and report the minimum (the device-"
                         "calibrated anchor). Run SHORT (--lengths 1 2 4 8 --seeds 2). Needs "
                         "--pulse-file (+ device frame on --submit).")
    ap.add_argument("--peak-grid", type=float, nargs="+",
                    default=[0.06, 0.09, 0.11, 0.13, 0.15, 0.17],
                    help="flux peak amplitudes to sweep for --cal-peak-sweep. Default brackets "
                         "the LOW basin: the first $33 sweep [0.15..0.33] found 0.18-0.33 badly "
                         "over-rotate (the gradpulse shape packs more entangling integral per "
                         "unit flux than the native raised-cosine), so the optimum is <=0.15. "
                         "0.17 confirms the over-rotation onset; 0.06-0.15 brackets the basin so "
                         "the max-survival selector has the optimum inside the grid.")
    ap.add_argument("--cal-virtualz-sweep", action="store_true",
                    help="STAGE 2 on-device calibration: sweep the Level-B pulse's single-qubit "
                         "virtual-Z phases (sequential 1-D: phi0 then phi1) at the Stage-1 peak, "
                         "minimizing r_cz against ONE shared native reference. Reports the "
                         "calibrated (phi0,phi1) for the combined run. Run SHORT (--lengths 2 4 8 "
                         "--seeds 2). Needs --pulse-file + --cz-peak (+ device frames on --submit).")
    ap.add_argument("--vz-grid", type=float, nargs="+",
                    default=[0.0, 0.785, 1.571, 2.356, 3.142, 3.927, 4.712, 5.498],
                    help="virtual-Z phases (rad) to sweep per axis for --cal-virtualz-sweep "
                         "(default: 8 points over [0,2pi); refine with a finer grid around the "
                         "found optimum). Swept absolutely, not assuming the model estimate.")
    args = ap.parse_args()
    load_env()                       # make AWS creds/region from .env visible to boto3

    lengths = args.lengths
    ref = bb.native_rb_sequences(lengths, args.seeds, seed=0, interleaved=False)
    intl = bb.native_rb_sequences(lengths, args.seeds, seed=0, interleaved=True)

    if args.combined:                # Level A + Level B, shared reference, one run
        run_combined(args, lengths, ref, intl)
        return

    if args.cal_peak_sweep:          # Stage 1: on-device peak calibration of the Level-B pulse
        run_cal_peak_sweep(args, lengths, ref, intl)
        return

    if args.cal_virtualz_sweep:      # Stage 2: on-device virtual-Z calibration
        run_virtualz_sweep(args, lengths, ref, intl)
        return

    all_seqs = ref + intl

    # --- offline correctness gate: every circuit must return to |00> ideally ---
    bad = [s for s in all_seqs if abs(bb.ideal_survival_probability(s["gates"]) - 1.0) > 1e-9]
    print(f"circuits: {len(all_seqs)} ({len(ref)} reference + {len(intl)} interleaved)")
    print(f"ideal return-to-|00> check: {'PASS' if not bad else f'FAIL ({len(bad)} bad)'}")
    if bad:
        raise SystemExit("aborting: some ideal circuits do not return to |00>.")

    cost = bb.estimate_experiment_cost(len(all_seqs), args.shots)
    print(f"cost @ {args.shots} shots: ${cost.total_usd:.2f} "
          f"(task ${cost.task_fee_usd:.2f} + shot ${cost.shot_fee_usd:.2f}), "
          f"pricing {cost.pricing_as_of}")

    if not args.submit and args.pulse:
        # Level B offline: the local simulator runs GATES, not pulse programs, so there is no
        # off-device survival number; checkable for free is that the pulse/circuit serialize.
        if not args.pulse_file:
            raise SystemExit("--pulse needs --pulse-file (the activation waveform); "
                             "generate one with examples/cepheus/levelb_pulse_benchmark_offline.py.")
        flux = load_flux_waveform(args.pulse_file)
        bench, peak = build_levelb_bench(args, flux=flux, device=None)
        rep = bb.verify_levelb_offline(bench, qubits=tuple(args.qubits))
        print(f"Level-B pulse: {flux.size} samples, anchor peak {peak:g}")
        print(f"  bench pulse serializes: {rep['bench_pulse_openpulse_chars']} chars OpenPulse")
        print(f"  RB circuit serializes:  {rep['circuit_openqasm_chars']} chars OpenQASM3 "
              f"(verbatim pragma={rep['verbatim_pragma_present']}, play={rep['play_present']})")
        print(f"  ideal Clifford closes:  {rep['ideal_clifford_closes']}")
        print(f"offline Level-B check: {'PASS' if rep['offline_ok'] else 'FAIL'}")
        print("\nDRY RUN -- the gate FIDELITY needs the QPU (pulses don't run on the local "
              "sim). Re-run with --submit --device-arn <ARN> --flux-frame-id <id> "
              "[--native-cal-file <json>] to benchmark it on silicon.")
        return

    if not args.submit:
        # Free, full rehearsal of generate->circuit->run->parse on the noiseless local
        # simulator. Survival must be ~1.0; if it is, only the device ARN differs for the QPU.
        demo = bb.to_braket_rb_circuit(intl[0]["gates"], qubits=tuple(args.qubits),
                                       buffer_bench_cz=args.buffer_bench_cz)
        from braket.devices import LocalSimulator
        sim = LocalSimulator()
        # Rehearse BOTH the shortest and the DEEPEST circuit -- the deep one (thousands of
        # native gates in one verbatim box) is the real pipeline/depth stress test.
        deepest = max(all_seqs, key=lambda s: len(s["gates"]))
        ok = True
        for tag, seq in (("shortest", intl[0]), ("deepest", deepest)):
            rc = bb.to_braket_rb_circuit(seq["gates"], qubits=(0, 1), verbatim=False,
                                         buffer_bench_cz=args.buffer_bench_cz)
            sv = bb.survival_from_counts(sim.run(rc, shots=400).result().measurement_counts)
            vc = bb.to_braket_rb_circuit(seq["gates"], qubits=tuple(args.qubits),
                                         buffer_bench_cz=args.buffer_bench_cz)  # verbatim build
            ok &= sv > 0.999
            print(f"  rehearsal {tag} (length {seq['length']}, {len(seq['gates'])} gates, "
                  f"{len(vc.instructions)} verbatim instr): survival {sv:.4f} "
                  f"({'PASS' if sv > 0.999 else 'FAIL'})")
        print(f"sample circuit builds: {demo.qubit_count} qubits, verbatim box")
        print(f"local-simulator noiseless rehearsal: "
              f"{'PASS' if ok else 'FAIL -- pipeline bug, do NOT submit'}")
        print("\nDRY RUN -- nothing submitted. Re-run with --submit --device-arn <ARN> "
              "to execute on silicon.")
        return

    # ----------------------- the device.run() step -------------------------
    if args.pulse and not args.pulse_file:
        raise SystemExit("--pulse needs --pulse-file (the activation waveform).")
    if not args.device_arn:
        raise SystemExit("--submit needs --device-arn <Braket QPU ARN>.")
    if not args.canary_only and cost.total_usd > args.max_cost:
        raise SystemExit(
            f"aborting: estimated ${cost.total_usd:.2f} exceeds --max-cost "
            f"${args.max_cost:.2f}. This guard prevents an accidental big spend; "
            f"re-run with --max-cost {cost.total_usd:.0f} (or higher) to proceed.")

    import json
    import boto3
    from braket.aws import AwsDevice, AwsSession
    bucket = args.s3_bucket or os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        raise SystemExit("--submit needs an S3 bucket (--s3-bucket or $AWS_S3_BUCKET).")
    s3_folder = (bucket, args.s3_prefix)
    session = AwsSession(boto_session=boto3.Session(region_name=args.region))
    device = AwsDevice(args.device_arn, aws_session=session)
    status = getattr(device, "status", "UNKNOWN")
    print(f"device {args.device_arn}\n  region={args.region}  status={status}  "
          f"s3=s3://{bucket}/{args.s3_prefix}")
    if status != "ONLINE":
        raise SystemExit(f"aborting: device is {status}, not ONLINE. Submit during its "
                         "availability window so tasks don't sit/fail in the queue.")

    qubits = tuple(args.qubits)

    # Level B: build the pulse on the device's real frame, anchored to its own CZ flux peak.
    # Bound only to interleaved CZ_BENCH markers; reference circuits are unaffected.
    bench_cz_pulse = None
    if args.pulse:
        flux = load_flux_waveform(args.pulse_file)
        bench_cz_pulse, peak = build_levelb_bench(args, flux=flux, device=device)
        rep = bb.verify_levelb_offline(bench_cz_pulse, qubits=qubits)
        print(f"LEVEL B: benchmarking a gradpulse pulse ({flux.size} samples) on frame "
              f"{args.flux_frame_id}, anchored to flux peak {peak:g}")
        print(f"  offline serialization check: {'PASS' if rep['offline_ok'] else 'FAIL'} "
              f"(verbatim={rep['verbatim_pragma_present']}, play={rep['play_present']})")
        if not rep["offline_ok"]:
            raise SystemExit("aborting: Level-B circuit failed offline serialization.")
        print("  NOTE: open-loop transfer -- expect this BELOW the native CZ until "
              "on-device calibration; the canary survival shows where it starts.")

    def _run(seq, shots, announce=False):
        # disable_qubit_rewiring=True is MANDATORY with a verbatim box -- the circuit
        # uses physical qubits and Rigetti must not remap them.
        circ = bb.to_braket_rb_circuit(seq["gates"], qubits=qubits,
                                       bench_cz_pulse=bench_cz_pulse,
                                       buffer_bench_cz=args.buffer_bench_cz)
        task = device.run(circ, shots=shots, disable_qubit_rewiring=True,
                          s3_destination_folder=s3_folder)
        if announce:
            # Printed before blocking: if this process dies the task still runs/charges and
            # is retrievable later with AwsQuantumTask(arn).result().
            print(f"  task ARN: {task.id}  (retrievable if this process dies)")
        counts = task.result().measurement_counts
        return counts, bb.survival_from_counts(counts)

    # --- CANARY: one throwaway circuit to confirm sane data before the full spend ---
    canary_cost = bb.estimate_experiment_cost(1, args.canary_shots).total_usd
    print(f"canary: 1 reference circuit @ {args.canary_shots} shots "
          f"(~${canary_cost:.2f}) to confirm the device returns sane data ...")
    counts, csurv = _run(ref[0], args.canary_shots, announce=True)
    print(f"canary counts keys: {sorted(counts.keys())}  (expect 2-bit '00'..'11'; "
          "if longer, your qubit mapping/bit-order is off -- stop and fix)")
    print(f"canary survival (length {ref[0]['length']}): {csurv:.3f}")
    if not (0.2 < csurv <= 1.0):
        raise SystemExit(
            f"aborting: canary survival {csurv:.3f} is implausible (wrong/poor qubit "
            "pair, bad calibration window, or a wiring bug). Diagnose before spending "
            f"the full budget -- this canary cost ~${canary_cost:.2f}, a full bad run "
            "costs the lot.")

    # --- DEPTH CANARY: the length-1 canary doesn't prove the device accepts the deepest
    #     circuit (thousands of native gates in ONE verbatim box) -- catch a depth rejection
    #     or a compiler that collapses the verbatim box (-> survival ~1.0) for ~$0.34, not
    #     mid-batch. A LOW survival here is EXPECTED: it pins the asymptote for the free fit. ---
    deepest = max(all_seqs, key=lambda s: len(s["gates"]))
    dcost = bb.estimate_experiment_cost(1, args.canary_shots).total_usd
    print(f"depth canary: deepest circuit (length {deepest['length']}, "
          f"{len(deepest['gates'])} native gates) @ {args.canary_shots} shots "
          f"(~${dcost:.2f}) to confirm the device accepts the batch's max depth ...")
    dcounts, dsurv = _run(deepest, args.canary_shots, announce=True)
    print(f"depth-canary counts keys: {sorted(dcounts.keys())}")
    print(f"depth-canary survival (length {deepest['length']}): {dsurv:.3f}")
    if sorted(dcounts.keys()) and any(len(k) != 2 for k in dcounts.keys()):
        raise SystemExit("aborting: depth-canary returned non-2-bit counts -- qubit "
                         "mapping/bit-order is off at depth.")
    if dsurv > 0.90:
        raise SystemExit(
            f"aborting: depth-canary survival {dsurv:.3f} is implausibly HIGH for a "
            f"length-{deepest['length']} circuit -- the compiler likely collapsed the "
            "verbatim box (cancelled the sequence against its recovery). The decay would "
            "be flat and the run worthless. Do NOT submit the batch.")
    if dsurv <= 0.30:
        print(f"  (survival {dsurv:.3f} near the floor is EXPECTED -- the deepest sequences "
              "are fully decayed, which is what pins the asymptote for the free fit.)")

    if args.canary_only:
        print(f"\nCANARY-ONLY: 2 canary tasks submitted (~${canary_cost + dcost:.2f} total), "
              f"length-1 survival {csurv:.3f} and depth survival {dsurv:.3f} are sane. "
              "STOPPING before the full batch, as requested.\nRe-run without --canary-only "
              "(add --max-cost for the full estimate) to run the experiment.")
        return

    # --- full run at equal shots, saved incrementally so a late failure never loses
    #     paid-for data (the canary is a throwaway; every dataset point gets full shots) -
    print(f"submitting {len(all_seqs)} circuits x {args.shots} shots (saving to {args.out}) ...")
    for i, s in enumerate(all_seqs):
        s["survival"] = _run(s, args.shots)[1]
        if (i + 1) % 10 == 0 or i + 1 == len(all_seqs):
            with open(args.out, "w") as f:
                json.dump([{k: v for k, v in t.items() if k != "gates"}
                           for t in all_seqs], f, indent=2)
            print(f"  {i + 1}/{len(all_seqs)} done (saved)")

    res = fit_irb([s for s in all_seqs if not s["interleaved"]],
                  [s for s in all_seqs if s["interleaved"]], lengths,
                  asymptote=0.25 if args.fixed_asymptote else None)
    ci = res["r_cz_ci68"]
    print(f"\nMEASURED CZ:  r = {res['r_cz']:.4e} +/- {res['r_cz_std']:.1e}   "
          f"F = {res['f_cz']:.5f}")
    print(f"  68% CI on r_cz: [{ci[0]:.3e}, {ci[1]:.3e}] (seed bootstrap)")
    print(f"  alpha_ref={res['alpha_ref']:.5f}  alpha_int={res['alpha_int']:.5f}")
    print("Compare against gradpulse's prediction for this pair's calibration "
          "(ParametricCouplerProfile.from_braket_calibration + error_budget).")


if __name__ == "__main__":
    main()
