"""Export an optimized gate to an Amazon Braket pulse-level PulseSequence (offline).

Runs the FULL offline-verifiable export path (no AWS account, no credentials, no
submission): optimize a CZ, export its envelope to a braket.pulse PulseSequence
(round-trip verified), serialize to OpenPulse 3.0, and produce an offline cost estimate
for an interleaved-RB run.

The path stops at device.run(): executing on a QPU is out of scope for this package --
the irreducible simulation != hardware boundary.

Run:  python -m examples.braket_export        (needs the optional [braket] extra)
"""
import numpy as np

try:
    from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer
    from gradpulse import braket_bridge as bb
    from gradpulse.hardware import BraketPulseBackend
except ImportError:  # running from the repo root without install
    from parametric import ParametricCouplerProfile, ParametricCZOptimizer
    import braket_bridge as bb
    from hardware import BraketPulseBackend


def main():
    # 1. Optimize a CZ (small, fast settings -- this is a demo, not a benchmark).
    opt = ParametricCZOptimizer(ParametricCouplerProfile(), bandwidth_mhz=80.0,
                                n_channels=3)
    res = opt.optimize_multi_seed(dt_ns=1.0, n_seeds=2, iterations=150, n_slices=150,
                                  use_process_fidelity=True, lbfgs_polish=False)
    waveform = res["best_waveform"]          # [n_slices, n_channels] physical envelope
    print(f"optimized CZ: F_proc(model) ~ {res['best_fidelity']:.5f}, "
          f"shape {waveform.shape}\n")

    # 2. Export + inspect the OpenPulse program (fully offline).
    frames = bb.synthetic_frames(waveform.shape[1])   # device.frames on real hardware
    seq = bb.build_gate_pulse_sequence(waveform, frames)
    print("OpenPulse 3.0 program (first 12 lines):")
    print("\n".join(seq.to_ir().splitlines()[:12]), "\n   ...\n")

    # 3. The honest readiness report: what's validated offline + what needs silicon.
    bb.hardware_readiness_report(waveform, budget_usd=50.0, n_shots=500)

    # 4. The backend builds everything and stops exactly at the credential wall.
    print("\nBraketPulseBackend (offline -- no device):")
    be = BraketPulseBackend(shots=500)
    try:
        be.measure_gate(waveform)
    except RuntimeError as e:
        print("  refused to submit (as it must):", str(e).split(". This is")[0] + ".")

    print("\nTo actually close sim != hardware: re-optimize against the device's real "
          "calibration\n(ParametricCouplerProfile.from_braket_calibration), pass a live "
          "AwsDevice + a\nClifford->native compiler to BraketPulseBackend, and run it "
          "with your credentials.")


if __name__ == "__main__":
    main()
