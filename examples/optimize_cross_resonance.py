"""Optimize a cross-resonance ZX(pi/2) gate (the fixed-frequency architecture).

The second gate architecture: fixed-frequency transmons + always-on exchange,
entangled by a cross-resonance drive. ZX(pi/2) is locally equivalent to CNOT.

    python examples/optimize_cross_resonance.py

Produces ``zx_pulse.npy`` / ``zx_pulse.json``; cross-check against QuTiP with:

    python -m gradpulse.validate --pulse zx_pulse.json

It also runs a DRAG ablation. CR is the regime where DRAG *could* earn its keep:
the strong, off-resonant CR tone drives the control's |1>-|2> transition -- a
leakage channel the (near-quiet) parametric CZ doesn't have -- and the
derived-quadrature (Motzoi) correction targets exactly that. How much it actually
helps is regime-dependent and reported, not assumed: deep in the dispersive regime
the in-phase shaping already suppresses the leakage, so DRAG barely moves the
number (and can slightly hurt); it pays off most at strong drive / small
control-|1>-|2> detuning.
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from gradpulse.crossresonance import (CrossResonanceProfile,
                                       CrossResonanceZXOptimizer, DEVICE)

profile = CrossResonanceProfile()
DT_NS, N_SLICES = 1.0, 180   # shorter gate => less decoherence (F_proc ~ 0.997)

# 1. Optimize the ZX(pi/2) gate (DRAG on, with an active target-cancellation tone).
opt = CrossResonanceZXOptimizer(profile, bandwidth_mhz=60.0,
                                use_drag=True, use_target_cancel=True)
ITERS, SEEDS = 300, 2
res = opt.optimize(n_slices=N_SLICES, dt_ns=DT_NS, iterations=ITERS, n_seeds=SEEDS, lr=0.05)
f_proc, f_avg = res["best_fidelity"], res["best_fidelity_avg"]
print(f"optimized ZX(pi/2):  F_proc = {f_proc:.5f}   F_avg = {f_avg:.5f}")
print(f"  leakage = {res['best_leakage']:.2e}   virtual-Z frame = "
      f"{[round(v, 4) for v in res['virtual_z']]} rad")
print(f"  per-seed F_proc = {[round(f, 5) for f in res['all_fidelities']]}")

# 1b. Truncation convergence (n_levels). Default n_levels=4 is converged for this
#     strong-drive gate (see CrossResonanceProfile); confirm by re-scoring the SAME
#     pulse one level higher -- the shift should be ~1e-4, below the decoherence floor.
opt5 = CrossResonanceZXOptimizer(CrossResonanceProfile(n_levels=5), bandwidth_mhz=60.0,
                                 use_drag=True, use_target_cancel=True)
_xt = torch.tensor(res["best_raw_param"], device=DEVICE, dtype=opt5.rdtype).unsqueeze(0)
_vzt = torch.tensor(res["virtual_z"], device=DEVICE, dtype=opt5.rdtype)
_rho5 = opt5.simulate_choi_batch(_xt, dt=DT_NS)
f_proc_5 = float(opt5._process_fidelity(_rho5, _vzt)[0])
print(f"\ntruncation convergence (default n_levels={profile.n_levels}):")
print(f"  F_proc: {f_proc:.5f} (@{profile.n_levels})  ->  {f_proc_5:.5f} (@5)   "
      f"shift = {f_proc - f_proc_5:+.5f}   (converged: |shift| ~1e-4, below decoh. floor)")

# 2. DRAG ablation: identical settings, DRAG off (a fair on-vs-off comparison).
opt_nodrag = CrossResonanceZXOptimizer(profile, bandwidth_mhz=60.0,
                                       use_drag=False, use_target_cancel=True)
res_nodrag = opt_nodrag.optimize(n_slices=N_SLICES, dt_ns=DT_NS,
                                 iterations=ITERS, n_seeds=SEEDS, lr=0.05)
print("\nDRAG ablation (CR is the drive-dominated regime where DRAG can matter):")
print(f"  DRAG on : F_proc = {f_proc:.5f}   leakage = {res['best_leakage']:.2e}")
print(f"  DRAG off: F_proc = {res_nodrag['best_fidelity']:.5f}   "
      f"leakage = {res_nodrag['best_leakage']:.2e}")
dleak = res_nodrag["best_leakage"] - res["best_leakage"]
print(f"  -> leakage(off) - leakage(on) = {dleak:+.2e} "
      f"(DRAG helps most at strong drive / small control-|1>-|2> detuning;"
      f" reported, not assumed)")

# 3. Save for the independent QuTiP cross-check (re-derives DRAG + applies virtual-Z
#    from the saved metadata, so the DRAG-on pulse cross-checks apples-to-apples).
np.save("zx_pulse.npy", res["best_waveform"])
meta = {
    "architecture": "cross_resonance",
    "pulse_npy": "zx_pulse.npy",
    "pulse_dt_ns": DT_NS,
    "n_channels": int(res["best_waveform"].shape[1]),
    "bandwidth_mhz": opt.bandwidth_mhz,
    "use_drag": opt.use_drag,
    "use_target_cancel": opt.use_target_cancel,
    "echo": opt.echo,                       # echoed-CR sequence (validator mirrors it)
    "virtual_z": res["virtual_z"],
    "grape_f": float(f_proc),
    "profile": asdict(profile),
}
Path("zx_pulse.json").write_text(json.dumps(meta, indent=2))
print("\nSaved zx_pulse.npy + zx_pulse.json")
print("Cross-check against QuTiP with:")
print("    python -m gradpulse.validate --pulse zx_pulse.json")

# 4. Error budget: how much of the infidelity is removable control/leakage error
#    vs the T1/T_phi decoherence floor (same diagnostic as the parametric CZ).
eb = opt.error_budget(res["best_raw_param"], dt=DT_NS, vz=res["virtual_z"])
print("\nerror budget (ZX90):")
print(f"  total infidelity   r_total       = {eb['r_total']:.2e}")
print(f"  coherent/leakage   r_coherent    = {eb['r_coherent']:.2e}")
print(f"  decoherence floor  r_decoherence = {eb['r_decoherence']:.2e}")
print(f"  channel unitarity  u             = {eb['unitarity']:.5f}")

# 5. RWA validity as a HEADLINE number: does the RWA-optimized pulse survive the full
#    Hamiltonian, or did it exploit the approximation? Restores each drive's
#    counter-rotating partner (the Bloch-Siegert term at 2*omega_d) and re-scores the
#    SAME pulse -- a tiny delta means the pulse is honest, not RWA-overfit.
cr = opt.counter_rotating_fidelity(res["best_raw_param"], dt=DT_NS, vz=res["virtual_z"])
print("\nbeyond-RWA validity (same pulse, counter-rotating terms restored):")
print(f"  F_proc (RWA reference)  = {cr['f_proc_rwa']:.6f}")
print(f"  F_proc (full H)         = {cr['f_proc_counter_rot']:.6f}")
print(f"  beyond-RWA infidelity   = {cr['delta_r_counter_rot']:+.2e}   "
      f"({'survives the full H' if abs(cr['delta_r_counter_rot']) < 1e-3 else 'RWA-sensitive'})")
print("  to REMOVE even this residual, polish against the full Hamiltonian directly:")
print("      ref = opt.refine_beyond_rwa(res['best_raw_param'], vz=res['virtual_z'])")
print("      # ref['f_proc_after'] vs ref['f_proc_before']: the beyond-RWA-optimized")
print("      # fidelity (counter-rotating terms inside the gradient loop), no longer")
print("      # confined to the RWA.")
