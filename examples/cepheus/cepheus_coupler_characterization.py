"""Coupler swap-spectroscopy: the measurement that makes Level-B PREDICTIVE (free prep).

To fix the "can't predict the on-device number" caveat we need the unmeasured coupler params.
The standard, Braket-pulse-accessible probe is SWAP SPECTROSCOPY of the |11>-|02> avoided crossing:

  prep |11>  (RX(pi) on both qubits)  ->  coupler flux pulse (amplitude A, duration t)  ->  measure

As the coupler flux tunes |11> toward resonance with |02>, population swaps 11<->02 and back, so
P|11|(A, t) oscillates -- a "chevron". Crucially this needs only the |11| population (both qubits
read 1), so BINARY readout suffices (no leakage detection). Two reductions of the chevron give the
coupler operating point:
  * vs duration at the resonant amplitude: P|11| = cos^2(g_eff * t) -> the swap rate g_eff,
  * vs amplitude: the deepest swap locates the flux where |11>-|02> is resonant.
g_eff(flux) + the resonance then pin the coupler frequency / exchange J in the tunable_coupler
model (from_calibration), which Level A showed makes gradpulse predict measured CZ error to 0.42 sigma.

This file (a) builds the on-device spectroscopy circuit (offline-verifiable, ready to --submit on
Cepheus), and (b) validates the swap-rate FIT on synthetic data so the analysis is trustworthy
before any spend. The measurement itself is the only paid part.

Run (free: offline build + synthetic fit):  python examples/cepheus_coupler_characterization.py
"""
import os
for _v in ("OMP_NUM_THREADS",):
    os.environ.setdefault(_v, "2")
import numpy as np

import gradpulse.braket_bridge as bb


def flat_top_flux(n_slices, ramp=6):
    """A coupler flux pulse: cosine ramps to a flat plateau (rest 0 at the endpoints, so it sits in
    a verbatim box cleanly). Unit peak; build_bench scales it to the swept amplitude."""
    w = np.ones(int(n_slices))
    r = min(int(ramp), n_slices // 2)
    if r > 0:
        edge = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, r))
        w[:r], w[-r:] = edge, edge[::-1]
    return w


def build_swap_spec_circuit(flux_frame, *, amp, n_slices, qubits, verbatim=True):
    """Swap-spectroscopy circuit: prep |11> (RX(pi) on both) -> coupler flux(amp, n_slices) ->
    measure. The flux plays as a pulse_gate inside the verbatim box, exactly like the Level-B bench
    pulse; the RX(pi) prep is native. Returns a braket Circuit (offline-serializable)."""
    flux = flat_top_flux(n_slices)
    bench = bb.build_bench_cz_pulse_sequence(flux, flux_frame, peak_amplitude=float(amp),
                                             drive_frames=None, virtual_z=(0.0, 0.0))
    gates = [("rx", (0,), np.pi), ("rx", (1,), np.pi), ("CZ_BENCH", (0, 1))]   # prep |11> then flux
    return bb.to_braket_rb_circuit(gates, qubits=qubits, bench_cz_pulse=bench, verbatim=verbatim)


def p11_from_counts(counts):
    tot = sum(counts.values())
    return (counts.get("11", 0) or 0) / tot if tot else 0.0


# ---- analysis: swap rate g_eff from P|11|(t) at the resonant amplitude --------------------------
def synthetic_chevron(durations_ns, amps, g_res_mhz, detuning_slope_mhz, shots=2000, seed=0):
    """Faithful 2-level |11>-|02> avoided-crossing chevron + binomial readout noise. Detuning
    Delta(amp) = slope*(amp - amp_res) (linear near resonance), resonant amp = amps midpoint.
    P|11|(t) = 1 - [4g^2/(Delta^2+4g^2)] sin^2(sqrt(Delta^2+4g^2) t/2). Returns P|11|[amp, dur]."""
    rng = np.random.default_rng(seed)
    amp_res = float(np.median(amps))
    g = 2 * np.pi * g_res_mhz / 1000.0                       # rad/ns
    out = np.empty((len(amps), len(durations_ns)))
    for i, a in enumerate(amps):
        Delta = 2 * np.pi * detuning_slope_mhz * (a - amp_res) / 1000.0
        W = np.sqrt(Delta ** 2 + 4 * g ** 2)
        for j, t in enumerate(durations_ns):
            p11 = 1.0 - (4 * g ** 2 / (Delta ** 2 + 4 * g ** 2)) * np.sin(W * t / 2) ** 2
            out[i, j] = rng.binomial(shots, min(max(p11, 0), 1)) / shots
        # ^ ideal P|11|; binomial shot noise
    return out, amp_res


def fit_swap_rate(durations_ns, p11):
    """Fit g_eff (MHz) from a resonant-amplitude P|11|(t) = cos^2(g t) = 0.5(1+cos(2 g t)) trace.
    FFT locates the dominant frequency (seed); a cos^2 least-squares then refines it."""
    t = np.asarray(durations_ns, float)
    y = np.asarray(p11, float)
    dt = np.mean(np.diff(t))
    freqs = np.fft.rfftfreq(len(t), d=dt)                    # cycles/ns
    amp = np.abs(np.fft.rfft(y - y.mean()))
    f_peak = freqs[1:][np.argmax(amp[1:])] if len(freqs) > 1 else 0.0
    g_seed = np.pi * f_peak * 1000.0 / (2 * np.pi)           # MHz; P|11| oscillates at 2g
    try:
        from scipy.optimize import curve_fit
        # P|11| = off + amp2*cos(2*(2pi*g/1000)*t); fit off, amp2, g(MHz)
        def model(tt, off, a2, g):
            return off + a2 * np.cos(2 * (2 * np.pi * g / 1000.0) * tt)
        p0 = [float(y.mean()), float((y.max() - y.min()) / 2), max(g_seed, 0.5)]
        popt, _ = curve_fit(model, t, y, p0=p0, maxfev=20000)
        return float(abs(popt[2]))
    except Exception:
        return float(g_seed)


def main():
    qubits = (16, 25)
    amps = np.linspace(0.16, 0.30, 8)                        # flux amplitudes to sweep
    durations = list(range(4, 100, 4))                       # ns
    n_circ = len(amps) * len(durations)
    cost = bb.estimate_experiment_cost(n_circ, 2000)
    print(f"SWAP-SPECTROSCOPY chevron: {len(amps)} amps x {len(durations)} durations = {n_circ} "
          f"circuits @ 2000 shots ~= ${cost.total_usd:.2f}  (pricing {cost.pricing_as_of})")

    # (a) offline: the on-device circuit builds + serializes (synthetic frame; real frame on submit)
    from braket.circuits.serialization import IRType
    fr = bb.synthetic_frames(3)
    circ = build_swap_spec_circuit(fr[1], amp=0.23, n_slices=40, qubits=qubits)
    ir = str(circ.to_ir(ir_type=IRType.OPENQASM))
    print(f"circuit: verbatim={'verbatim' in ir} play={'play' in ir} rx={'rx(' in ir}  "
          f"(prep |11> + coupler flux + measure; OpenQASM serializes)")

    # (b) validate the fit on synthetic data: recover a known g_eff
    true_g = 9.0                                             # MHz (the swap rate to recover)
    chev, amp_res = synthetic_chevron(durations, amps, g_res_mhz=true_g, detuning_slope_mhz=400.0)
    res_row = int(np.argmin(chev.min(axis=1)))               # amplitude with the deepest swap
    g_fit = fit_swap_rate(durations, chev[res_row])
    print(f"\nfit validation (synthetic, true g_eff={true_g:.1f} MHz):")
    print(f"  resonant amplitude found: {amps[res_row]:.3f} (true {amp_res:.3f})")
    print(f"  g_eff recovered: {g_fit:.2f} MHz  (true {true_g:.1f})  err {abs(g_fit-true_g):.2f}")
    ok = abs(g_fit - true_g) < 1.5
    print(f"  {'PASS' if ok else 'CHECK'}: swap-rate fit recovers the coupling to <1.5 MHz")
    print("\nNEXT (paid): run this chevron on Cepheus -> g_eff(flux) + resonance -> coupler "
          "freq/J via from_calibration -> re-optimize coupler-only -> PREDICTIVE Level-B (the "
          "Level-A 0.42-sigma standard, now for the designed pulse).")


if __name__ == "__main__":
    main()
