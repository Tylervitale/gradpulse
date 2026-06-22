"""Robustness / miscalibration sweep for a saved gradpulse pulse.

Answers the first question any experimentalist asks of a tuned pulse: how tight
must the calibration be? It re-simulates the fixed (already-optimized) pulse --
no re-optimization -- under the two miscalibration axes a tune-up calibrates and
prints F_avg vs perturbation for each, plus the calibration window that keeps the
gate within 1e-3 of its nominal fidelity:

  * amplitude   -- AWG-gain / Rabi-calibration error (all amplitudes scaled)
  * frequency   -- static drive-frequency detuning (MHz), kept inside the 1/T
                   phase-wrap window so the curve stays monotonic

Usage:
    python examples/robustness_sweep.py                      # ./cz_pulse.json
    python examples/robustness_sweep.py --pulse path/to/pulse.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer


def _load(pulse_json: Path):
    meta = json.loads(pulse_json.read_text())
    env = np.load(pulse_json.parent / Path(meta["pulse_npy"]).name)
    prof = meta.get("profile", {})
    valid = {f.name for f in fields(ParametricCouplerProfile)}
    profile = ParametricCouplerProfile(**{k: v for k, v in prof.items() if k in valid})
    # Eval optimizer: bandwidth off + clamp activation so the saved smoothed
    # envelope is fed as the literal physical pulse (no double-smoothing) -- the
    # same convention the QuTiP validator uses.
    opt = ParametricCZOptimizer(
        profile, bandwidth_mhz=0.0, activation="clamp",
        n_channels=int(meta.get("n_channels", env.shape[1])),
        target_gate=str(meta.get("target_gate", "cz")),
        line_response=meta.get("line_response"),
        precision="double",
    )
    return opt, env, meta


def _window(x, fa, target):
    """Contiguous interval of x around 0 where F_avg stays >= target."""
    order = np.argsort(x)
    xs, fas = np.asarray(x)[order], np.asarray(fa)[order]
    z = int(np.argmin(np.abs(xs)))
    lo = hi = xs[z]
    i = z
    while i - 1 >= 0 and fas[i - 1] >= target:
        i -= 1; lo = xs[i]
    i = z
    while i + 1 < len(xs) and fas[i + 1] >= target:
        i += 1; hi = xs[i]
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulse", default="cz_pulse.json",
                    help="Pulse JSON written by examples/optimize_cz.py "
                         "(default: ./cz_pulse.json).")
    args = ap.parse_args()
    pulse = Path(args.pulse)
    if not pulse.exists():
        print(f"{pulse} not found. Generate one first: python examples/optimize_cz.py")
        return

    opt, env, meta = _load(pulse)
    nominal = opt.error_budget(env, dt=1.0)["F_avg"]
    target = nominal - 1e-3

    # Amplitude over +/-10%; drive-frequency on a fine grid (the conditional
    # phase accumulates fast over a long gate, so the raw window is sub-MHz).
    sweep = opt.robustness_sweep(
        env, dt=1.0,
        amp_fracs=np.linspace(-0.10, 0.10, 11),
        freq_mhz=np.linspace(-0.30, 0.30, 25),
    )

    print(f"  Pulse: {pulse.name}  (target gate {meta.get('target_gate', 'cz')})")
    print(f"  Nominal F_avg = {nominal:.6f};  calibration window keeps "
          f"F_avg >= {target:.6f}\n")

    labels = {"amplitude": "AMPLITUDE (fractional gain error) -- clean tolerance",
              "frequency": "FREQUENCY (MHz drive detuning) -- RAW, pre-virtual-Z"}
    notes = {"amplitude": "",
             "frequency": "       (conservative: virtual-Z recalibration relaxes this)"}
    for axis in ("amplitude", "frequency"):
        s = sweep[axis]
        print(f"  {labels[axis]}")
        if notes[axis]:
            print(notes[axis])
        # Subsample the (possibly fine) grid to ~11 printed rows.
        step = max(1, len(s["x"]) // 11)
        for xi, fa in list(zip(s["x"], s["F_avg"]))[::step]:
            bar = "#" * int(round(60 * max(0.0, (fa - 0.9) / 0.1)))
            print(f"    {xi:+7.3f}  F_avg={fa:.6f}  {bar}")
        lo, hi = _window(s["x"], s["F_avg"], target)
        print(f"    -> within 1e-3 of nominal for {lo:+.3f} .. {hi:+.3f} "
              f"({s['unit']})\n")


if __name__ == "__main__":
    main()
