"""Spectator (always-on ZZ) crosstalk for a saved gradpulse pulse.

The dominant multi-qubit error the single-pair model otherwise omits: a
neighbouring qubit coupled to a gate qubit by a static (always-on) ZZ shifts that
qubit's frequency, perturbing the gate. This re-simulates the fixed (already
optimized) pulse -- no re-optimization -- with an effective frozen-neighbour
detuning, and prints, versus the ZZ rate zeta:

  * idle        neighbour in |0>          (= the nominal gate)
  * excited     neighbour frozen in |1>   (RAW coherent penalty; a *known* state
                is ~fully virtual-Z removable, so this is a conservative bound)
  * unmeasured  neighbour state unknown   (channel averaged over its population, at
                the gate's nominal frame -- a dephasing channel; re-tuning the
                virtual-Z for the mean shift removes part, the spread is irreducible)

plus the largest zeta that keeps the added (marginal) unmeasured-neighbour infidelity
below 1e-3 -- a conservative ZZ tolerance for the device.

The reduction (off-resonant neighbour's ZZ == an effective detuning) is validated
against a full 3-transmon QuTiP simulation in tests/test_spectators.py. The
complementary RESONANT / frequency-collision regime -- where a near-resonant
spectator dynamically swaps population and cannot be frozen -- is then swept with
resonant_collision_fidelity (an explicitly evolving, exchange-coupled spectator).

Usage:
    python examples/spectator_crosstalk.py                      # ./cz_pulse.json
    python examples/spectator_crosstalk.py --pulse path/to/pulse.json
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
    # envelope is the literal physical pulse (no double-smoothing) -- the same
    # convention the QuTiP validator uses.
    opt = ParametricCZOptimizer(
        profile, bandwidth_mhz=0.0, activation="clamp",
        n_channels=int(meta.get("n_channels", env.shape[1])),
        target_gate=str(meta.get("target_gate", "cz")),
        line_response=meta.get("line_response"),
        precision="double",
    )
    return opt, env, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulse", default="cz_pulse.json",
                    help="Pulse JSON written by examples/optimize_cz.py "
                         "(default: ./cz_pulse.json).")
    ap.add_argument("--pop", type=float, default=0.5,
                    help="P(neighbour in |1>): 0.5 = maximally-uncertain (default).")
    args = ap.parse_args()
    pulse = Path(args.pulse)
    if not pulse.exists():
        print(f"{pulse} not found. Generate one first: python examples/optimize_cz.py")
        return

    opt, env, meta = _load(pulse)
    zetas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]               # MHz, typical .. severe

    print(f"  Pulse: {pulse.name}  (target gate {meta.get('target_gate', 'cz')})")
    print(f"  A neighbour on EACH gate qubit, P(|1>) = {args.pop}")
    print("  'added r' = average-gate infidelity the unmeasured neighbour ADDS "
          "(excludes the gate's own error).\n")
    print("    zeta(MHz)   idle      excited(raw)  unmeasured   added r")
    print("    " + "-" * 56)

    tol = 1e-3
    zz_window = None
    for z in zetas:
        r = opt.spectator_fidelity(env, dt=1.0, zeta_mhz=z, spectator_pop=args.pop)
        print(f"    {z:7.3f}   {r['f_proc_idle']:.6f}  {r['f_proc_excited']:.6f}    "
              f"{r['f_proc_spectator_avg']:.6f}   {r['delta_r_spectator']:.2e}")
        if r["delta_r_spectator"] <= tol:
            zz_window = z

    print()
    if zz_window is not None:
        print(f"  -> unmeasured-neighbour infidelity stays below {tol:.0e} up to "
              f"zeta ~ {zz_window} MHz.")
    else:
        print(f"  -> even the smallest zeta here exceeds {tol:.0e} infidelity; this "
              f"gate needs tighter ZZ suppression (a coupler null / echo).")
    print("  Note: a *known* neighbour state is ~fully removable by virtual-Z; the")
    print("  'unmeasured' column is the conservative nominal-frame cost (re-tuning")
    print("  for the mean shift removes ~half; the spread is the irreducible part).")

    # Resonant / frequency-collision regime: the spectator can no longer be frozen.
    # Sweep its detuning from a gate qubit toward 0 (an exact collision) and watch
    # the gate crater as population resonantly swaps into the now-evolving neighbour.
    print("\n  Resonant / frequency-collision regime (evolving exchange-coupled "
          "spectator, J = 8 MHz):")
    print("    detuning(MHz)   F_proc      spectator pop. swapped   added r")
    print("    " + "-" * 62)
    sweep = opt.resonant_collision_fidelity(
        env, dt=1.0, detuning_mhz=[2000.0, 400.0, 200.0, 100.0, 50.0, 0.0],
        j_mhz=8.0, couples_to=2)
    for det, f, leak, dr in zip(sweep["detuning_mhz"], sweep["f_proc"],
                                sweep["spectator_leakage"], sweep["delta_r_collision"]):
        print(f"    {det:9.0f}     {f:.6f}        {leak:.4f}            {dr:+.2e}")
    print("  -> far off-resonant recovers the bare gate; near resonance the exchange "
          "is resonant,")
    print("     population swaps into the spectator, and the gate fidelity collapses "
          "-- a frequency")
    print("     collision the frozen-spectator (static-ZZ) model above cannot capture.")


if __name__ == "__main__":
    main()
