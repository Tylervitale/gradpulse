"""Minimal end-to-end example: optimize a CZ gate for a parametric-coupler pair.

Install the package first (``pip install -e .`` from the repo root), then run:

    python examples/optimize_cz.py

Produces ``cz_pulse.npy`` (the bandwidth-smoothed [n_slices, n_channels] envelope
in [0, 1]) and ``cz_pulse.json`` (metadata + device profile the QuTiP validator
consumes). Then cross-check the result against an independent QuTiP simulation:

    python -m gradpulse.validate --pulse cz_pulse.json
"""
import json
from dataclasses import asdict

import numpy as np
import torch

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer
from gradpulse.parametric import DEVICE

# 1. Device profile. These are representative defaults; replace with your qubit
#    pair's calibrated values, e.g. via ParametricCouplerProfile.from_braket_calibration(
#    "device_properties.json", (4, 5), freq_ghz_q1=4.85, freq_ghz_q2=5.05) (frequency
#    and anharmonicity aren't in that schema, so pass them explicitly).
profile = ParametricCouplerProfile(
    freq_ghz_q1=4.85, freq_ghz_q2=5.05,
    anharm_ghz_q1=-0.20, anharm_ghz_q2=-0.20,
    t1_ns_q1=30_000, t2_ns_q1=25_000,
    t1_ns_q2=30_000, t2_ns_q2=25_000,
    g_max_mhz=12.0, omega_max_mhz=50.0,
)

# 2. Optimizer.
#    use_drag=False: this coupler-activated gate keeps single-qubit drives near-quiet,
#    so DRAG isn't the operative leakage lever, and the saved [0,1] envelope stays a
#    complete description for an apples-to-apples QuTiP cross-check. (With DRAG on,
#    opt.iq_waveform(raw) exports the full complex I/Q -- see examples/export_openpulse.py.)
#    Other options: step_order=2 for a 2nd-order Strang integrator (see dt_convergence
#    below); precision='double' to resolve fine-dt behavior below the complex64 floor;
#    n_channels>=4 + coupler_phase_mode='frequency' to make the drive detuning optimizable.
opt = ParametricCZOptimizer(
    profile,
    bandwidth_mhz=80.0,
    use_drag=False,
    n_channels=3,          # q1 drive, q2 drive, coupler envelope
    activation="sigmoid",  # smooth saturation -> no clipping artifacts
    step_order=1,          # 1 = default first-order; 2 = 2nd-order Strang step
)

# 3. Multi-seed GRAPE: Adam search from several perturbed warm-starts, L-BFGS polish.
DT_NS = 1.0
N_SLICES = 150            # 150 ns gate
result = opt.optimize_multi_seed(
    n_seeds=4,
    iterations=200,
    n_slices=N_SLICES,
    dt_ns=DT_NS,
    warm_start_mode="parametric_cz",
    use_process_fidelity=True,   # best_fidelity is reported as process fidelity
    lbfgs_polish=True,
)

waveform = result["best_waveform"]          # [n_slices, n_channels], in [0, 1]
print(f"best process fidelity : {result['best_fidelity']:.6f}")
print(f"per-seed fidelities   : {np.round(result['all_fidelities'], 5).tolist()}")
print(f"waveform shape        : {waveform.shape}  (n_slices, n_channels)")

# 4. Save in the format gradpulse.validate (QuTiP cross-check) expects. The device
#    profile is saved with the pulse so the validator reproduces the same physics.
np.save("cz_pulse.npy", waveform)
meta = {
    "pulse_npy": "cz_pulse.npy",
    "pulse_dt_ns": DT_NS,
    "n_channels": int(waveform.shape[1]),
    "bandwidth_mhz": opt.bandwidth_mhz,
    "smoother_type": opt.smoother_type,
    "target_gate": opt.target_gate,
    "line_response": opt.line_response,
    "grape_f": float(result["best_fidelity"]),
    "profile": asdict(profile),
}
with open("cz_pulse.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved cz_pulse.npy + cz_pulse.json")
print("Cross-check against QuTiP with:")
print("    python -m gradpulse.validate --pulse cz_pulse.json")

# 5. Confirm the reported fidelity is converged in the integration step dt.
#    dt_convergence holds the pulse fixed, refines dt, and Richardson-extrapolates the
#    dt->0 limit for both integrator orders. Feeding back best_raw_param (the exact
#    simulator input) reproduces the headline fidelity bit-for-bit at dt = 1 ns.
raw = torch.tensor(result["best_raw_param"], device=DEVICE, dtype=opt.rdtype)
conv = opt.dt_convergence(raw, dt=DT_NS, refinements=(1, 2, 4))
print("\ndt-convergence of F_proc (pulse fixed, integration step refined):")
for d, f1, f2 in zip(conv["dt"], conv["order1"], conv["order2"]):
    print(f"    dt = {d:6.4f} ns   order1 = {f1:.6f}   order2 = {f2:.6f}")
print(f"    order1 at dt = {DT_NS} ns reproduces best_fidelity exactly: "
      f"{abs(conv['order1'][0] - result['best_fidelity']):.1e} gap")
print(f"    dt->0 (Richardson):  order1 = {conv['order1_extrap']:.6f}   "
      f"order2 = {conv['order2_extrap']:.6f}")
print(f"    integrator-splitting error at dt = {DT_NS} ns: "
      f"{conv['splitting_err_at_dt']:.1e}")

# 5b. The complex64 floor (~1e-6) limits how fine a dt sweep is meaningful; rerun in
#     double precision (same pulse/physics) so the first-order error keeps halving
#     cleanly instead of flattening into round-off noise.
opt_d = ParametricCZOptimizer(
    profile, bandwidth_mhz=opt.bandwidth_mhz, use_drag=False,
    n_channels=int(waveform.shape[1]), activation="sigmoid", precision="double",
)
raw_d = torch.tensor(result["best_raw_param"], device=DEVICE, dtype=opt_d.rdtype)
conv_d = opt_d.dt_convergence(raw_d, dt=DT_NS, refinements=(1, 2, 4, 8))
print("\nsame sweep in double precision (precision='double'):")
for d, f1 in zip(conv_d["dt"], conv_d["order1"]):
    print(f"    dt = {d:6.4f} ns   order1 = {f1:.8f}")
