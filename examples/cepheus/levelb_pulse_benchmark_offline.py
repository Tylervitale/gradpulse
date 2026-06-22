"""Level B, offline: turn a gradpulse-designed CZ into a Braket benchmarked-gate pulse
and verify everything that does NOT need a QPU.

This is the free half of "close the loop." It builds the exact PulseSequence that
``run_irb_on_braket.py --pulse`` would play on silicon and checks that it serializes
(OpenPulse 3.0 / OpenQASM 3), that the verbatim pragma + ``play`` survive (so the random
Cliffords stay exact and the gradpulse pulse is really in the program), and that the
Clifford structure still closes -- without spending a cent. The gate FIDELITY is the
other half, and it needs the device: Braket's local simulator runs gates, not pulses.

It also SAVES the activation waveform to a .npy you can feed straight to::

    run_irb_on_braket.py --submit --device-arn <ARN> --qubits 16 25 \
        --pulse --pulse-file <that .npy> --native-cal-file <native_cal.json> \
        --flux-frame-id <cz_frame_id> --canary-only

Architecture matching matters -- a gradpulse pulse only transfers if its activation
mechanism matches the device's:
  default     parametric ``optimize_cz`` (fast, 9-D). Its coupler channel is a PARAMETRIC
              coupler drive -> matches a parametrically-activated coupler.
  --tunable   ``tunable_coupler_cz`` (27-D, heavier, thread-capped). Its coupler channel
              is a BASEBAND flux tuning the coupler through |11>-|02> -> matches Cepheus /
              Sycamore-style baseband tunable couplers.

Open-loop caveat: the saved pulse is optimized against a model and, on hardware, scaled
to the device's calibrated CZ flux peak -- it is NOT closed-loop calibrated, so on silicon
it starts BELOW the device's tuned native CZ. That gap is what on-device calibration
closes; this script only proves the pulse is well-formed and submittable.
"""
import os
# The tunable-coupler model is a 27-D open system; cap BLAS/OMP threads BEFORE importing
# torch so a CPU run stays bounded.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))

import argparse

import numpy as np

from gradpulse import braket_bridge as bb


def parametric_flux():
    """Fast parametric-coupler CZ; return (flux_waveform, label)."""
    import gradpulse as gp
    r = gp.optimize_cz(n_seeds=2, iterations=120)
    wf = np.asarray(r["best_waveform"])             # [n_slices, n_channels]
    # channels: q1 drive, q2 drive, coupler[, phase][, stark] -- coupler activates the CZ
    coupler = wf[:, 2] if wf.shape[1] > 2 else wf[:, -1]
    return coupler, "parametric (matches a parametrically-activated coupler)"


def tunable_flux(iterations):
    """Cepheus-matched explicit tunable-coupler CZ (baseband flux). Heavier (27-D)."""
    import gradpulse as gp
    opt = gp.tunable_coupler_cz(verbose=False)
    r = opt.optimize(n_slices=120, dt_ns=1.0, iterations=iterations, n_seeds=1, verbose=False)
    wf = np.asarray(r["best_waveform"])             # [n_slices, n_channels]
    # freq_control_qubits=[0,1,2] -> channel 1 is the COUPLER flux (the activation);
    # channels 0 and 2 are the qubit detunings (the single-qubit virtual-Z corrections).
    return wf[:, 1], "tunable-coupler baseband flux (matches Cepheus/Sycamore)"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tunable", action="store_true",
                    help="use the 27-D tunable_coupler_cz baseband flux (Cepheus-matched) "
                         "instead of the fast parametric pulse (heavier; thread-capped)")
    ap.add_argument("--iterations", type=int, default=30,
                    help="tunable-coupler optimization iterations (--tunable only)")
    ap.add_argument("--peak", type=float, default=0.30,
                    help="anchor flux peak -- a stand-in for the device's calibrated CZ "
                         "peak (on hardware: bench_cz_peak_from_native_calibration)")
    ap.add_argument("--out", default=os.path.join(HERE, "levelb_flux.npy"),
                    help="save the activation waveform here (-> examples/cepheus/run_irb_on_braket.py --pulse-file)")
    args = ap.parse_args()

    flux, label = (tunable_flux(args.iterations) if args.tunable else parametric_flux())
    print(f"\ngradpulse activation waveform: {flux.size} samples -- {label}")

    # Synthetic device frames stand in for device.frames offline (coupler + 2 drives).
    flux_frame, d0, d1 = bb.synthetic_frames(3)
    bench = bb.build_bench_cz_pulse_sequence(flux, flux_frame, peak_amplitude=args.peak,
                                             drive_frames=(d0, d1))
    rep = bb.verify_levelb_offline(bench, qubits=(16, 25))
    print(f"  bench pulse:     {rep['bench_pulse_openpulse_chars']} chars OpenPulse 3.0")
    print(f"  RB circuit:      {rep['circuit_openqasm_chars']} chars OpenQASM 3 "
          f"(verbatim pragma={rep['verbatim_pragma_present']}, play={rep['play_present']})")
    print(f"  Clifford closes: {rep['ideal_clifford_closes']}")
    print(f"OFFLINE LEVEL-B CHECK: {'PASS' if rep['offline_ok'] else 'FAIL'}")

    np.save(args.out, np.asarray(flux))
    print(f"\nsaved activation waveform -> {args.out}")
    print("On silicon (needs AWS creds + ~$57; OPEN-LOOP transfer, starts below native):")
    print("  python examples/cepheus/run_irb_on_braket.py --submit --device-arn <ARN> \\")
    print(f"      --qubits 16 25 --pulse --pulse-file {args.out} \\")
    print("      --native-cal-file <native_cal.json> --flux-frame-id <cz_frame_id> --canary-only")
    print("The gate FIDELITY needs the QPU -- pulses do not run on the local simulator.")


if __name__ == "__main__":
    main()
