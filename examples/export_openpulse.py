"""Export an optimized gate to vendor-neutral OpenPulse 3.0 / OpenQASM 3, with the
DRAG quadrature baked into the complex I/Q so the exported pulse is COMPLETE.

    python examples/export_openpulse.py

``smoothed_waveform`` gives only the real in-phase envelope; with DRAG the simulator
also drives a derived quadrature that envelope omits. ``iq_waveform`` returns the full
complex drive (in-phase + quadrature) in physical rad/ns, and the OpenPulse exporter
preserves it. The emitted program is round-trip-verified against an INDEPENDENT
parser, so it is both valid OpenPulse 3.0 and lossless. (Qiskit removed ``qiskit.pulse``
in 2.0; this targets the live open standard instead.) Needs the ``[openpulse]`` extra.
"""
from gradpulse import CrossResonanceProfile, CrossResonanceZXOptimizer
from gradpulse import openpulse_export as ope

# Cross-resonance uses DRAG heavily, so it shows the I/Q export off best.
opt = CrossResonanceZXOptimizer(CrossResonanceProfile(), use_drag=True)
res = opt.optimize(n_slices=240, dt_ns=1.0, n_seeds=2, iterations=120)
print(f"optimized ZX(pi/2) F_proc = {res['best_fidelity']:.6f}")

import torch
from gradpulse.parametric import DEVICE
x = torch.tensor(res["best_raw_param"], device=DEVICE, dtype=opt.rdtype)

# The COMPLETE complex drive (in-phase + DRAG quadrature).
iq = opt.iq_waveform(x, dt=1.0)
print(f"I/Q channels: {iq['labels']}  (complex, units {iq['units']})")
import numpy as np
print(f"max |quadrature| on control drive: {np.max(np.abs(iq['iq'][:, 0].imag)):.4g} rad/ns "
      f"(this is what the real-envelope export would have dropped)")

# Emit + offline-verify the OpenPulse 3.0 program.
report = ope.openpulse_readiness_report(iq, dt_ns=1.0, gate_name="grad_zx", qubits=(0, 1))
with open("grad_zx_openpulse.qasm", "w") as f:
    f.write(report["program"])
print("\nSaved grad_zx_openpulse.qasm (OpenQASM 3 / OpenPulse 3.0).")
print("The CR gate also carries a virtual-Z frame (applied to the following 1q gates):",
      res["virtual_z"])
