"""Hardware-in-the-loop calibration: close the sim<->device gap from measured RB.

A simulated fidelity is not a hardware number. This loop is the bridge: optimize a
gate on the current model, "measure" it on a backend, infer how far the model's
noise was off, fold the correction into the device profile, and re-optimize. Over a
few rounds the model's predicted fidelity converges to the measured one.

    python examples/hardware_in_the_loop.py          # Part 2 needs the [validate] extra

Two backends, two levels of rigour:

  * Part 1 -- SimulatedBackend: the "device" is gradpulse's own simulator with worse
    coherence. Fast, no extra deps; shows the mechanics.
  * Part 2 -- QuTiPDeviceBackend: the SAME loop, but the "device" is measured by an
    INDEPENDENT engine (QuTiP), not gradpulse's own simulator. The model still
    converges to the device's true coherence -- now confirmed by different code, so
    the closure is not an artefact of the engine that proposed the pulse. (An
    out-of-model *coherent* error, like a static ZZ the model lacks, is only
    partially captured by the first-order coherence rescale -- documented in
    gradpulse.hardware, not demonstrated here, since attributing a coherent error to
    decoherence is exactly what the scaffold is honest about not fully doing.)

To run against a REAL device, implement gradpulse.hardware.HardwareBackend
(one method: submit the waveform, run interleaved RB, return the GateMeasurement)
and pass it to calibrate_to_hardware -- see gradpulse.hardware.BraketBackendTemplate.
"""
from gradpulse import ParametricCouplerProfile
from gradpulse.hardware import (
    QuTiPDeviceBackend, SimulatedBackend, apply_coherence_scale,
    calibrate_to_hardware,
)


def _print_history(out, true_device):
    print("\n round | F_model | F_hw    |   gap    | coherence_scale | T1_q1 (us)")
    print(" ------+---------+---------+----------+-----------------+-----------")
    for h in out["history"]:
        print(f"   {h['round']:2d}  | {h['f_model_avg']:.5f} | {h['f_hardware_avg']:.5f} | "
              f"{h['gap']:+.5f} |      {h['coherence_scale']:.3f}      | "
              f"{h['t1_ns_q1']/1000:6.1f}")
    final = out["refined_profile"]
    print(f"\nrefined model T1_q1 = {final.t1_ns_q1/1000:.1f} us "
          f"(true {true_device.t1_ns_q1/1000:.1f} us); "
          f"T2_q1 = {final.t2_ns_q1/1000:.1f} us (true {true_device.t2_ns_q1/1000:.1f} us)")


# The (unknown-to-us) TRUE device: representative coherence we are trying to match.
true_device = ParametricCouplerProfile(
    freq_ghz_q1=4.85, freq_ghz_q2=5.05, anharm_ghz_q1=-0.33, anharm_ghz_q2=-0.33,
    t1_ns_q1=28_000, t2_ns_q1=20_000, t1_ns_q2=32_000, t2_ns_q2=22_000,
    g_max_mhz=12.0, omega_max_mhz=50.0,
)

# Our starting MODEL is optimistic: it assumes ~2x longer coherence than reality
# (a common situation before you have measured your assigned qubits).
model_guess = apply_coherence_scale(true_device, 0.5)

OPT = dict(n_seeds=2, iterations=150, n_slices=150,
           warm_start_mode="parametric_cz", use_process_fidelity=True,
           lbfgs_polish=False)

# Part 1: same-engine stand-in (fast, no QuTiP); use_irb=True routes through rb.py instead.
print("=" * 72)
print("Part 1 - SimulatedBackend (gradpulse's own simulator as the 'device')")
print("=" * 72)
out1 = calibrate_to_hardware(model_guess, SimulatedBackend(true_device),
                             rounds=2, dt_ns=1.0, opt_kwargs=OPT, verbose=True)
_print_history(out1, true_device)
print("The gap shrinks as the model's effective coherence is pulled toward the device's.")

# Part 2: the SAME loop, measured by an INDEPENDENT engine (QuTiP) -- see module docstring.
print("\n" + "=" * 72)
print("Part 2 - QuTiPDeviceBackend (same loop, measured by an INDEPENDENT engine)")
print("=" * 72)
try:
    backend = QuTiPDeviceBackend(true_device, target_gate="cz")
    out2 = calibrate_to_hardware(model_guess, backend, rounds=2, dt_ns=1.0,
                                 opt_kwargs=dict(OPT, n_seeds=1, n_slices=120),
                                 verbose=True)
    _print_history(out2, true_device)
    print("Same convergence to the device's true coherence -- now confirmed by an")
    print("independent integrator, so it is not an artefact of gradpulse's own simulator.")
    print("(An out-of-model *coherent* error, e.g. a static ZZ the model lacks, is only")
    print(" partially captured by a T1/T2 rescale -- the first-order limit hardware.py notes.)")
except ImportError:
    print("QuTiP not installed - skipping the independent-engine loop.")
    print("Install it with:  pip install -e \".[validate]\"")
