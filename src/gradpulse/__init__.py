"""gradpulse: A differentiable, multi-solver-validated, open-system pulse optimizer for predictive superconducting-gate fidelities

Built and maintained by Pure State Labs Inc. (https://purestatelabs.com).

Give it a qubit pair and a target gate. It finds the control pulse by backpropagating
through an open-system (Lindblad) simulation, so T1, T2, and leakage are traded off
against control error while the pulse is shaped. Every fidelity it reports is cross-checked
by three independent solvers. Runs on a laptop (CPU or a consumer GPU).

Quickstart
----------
    import gradpulse as gp

    r = gp.optimize_cz()                              # representative device
    print(r["best_fidelity"])                         # process fidelity to CZ
    r["optimizer"].error_budget(r["best_raw_param"])  # control/leakage vs decoherence

For full control, build the pieces yourself (same shape for iSWAP, cross-resonance,
and the N-qubit optimizer):

    from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer
    opt = ParametricCZOptimizer(ParametricCouplerProfile())
    r = opt.optimize_multi_seed(n_seeds=4, iterations=200, n_slices=150)

Load a real device with ParametricCouplerProfile.from_calibration(...) or
.from_ibm_backend(...). Plot results with gradpulse.viz.

What's here
----------
Gate models, each with an independent QuTiP cross-check of its operators:
    ParametricCZOptimizer        tunable-coupler CZ / iSWAP (the headline gate)
    CrossResonanceZXOptimizer    fixed-frequency cross-resonance ZX(pi/2)
    MultiQubitOptimizer          N-qubit register, crosstalk inside the loop
Each pairs with a *Profile dataclass. One-call wrappers cover the common cases:
optimize_cz, optimize_iswap, tunable_coupler_cz, coupler_in_loop_cz.

Three independent solvers back every fidelity:
    the optimizer's own PyTorch/Trotter integrator,
    gradpulse.validate    a QuTiP rebuild (needs the [validate] extra), and
    gradpulse.liouville   a NumPy-only exact-generator solver, one per architecture
                          (liouville_f_proc / liouville_cr_f_proc /
                          liouville_nqubit_closed_f_proc) so the library-independent
                          leg covers all three, not just the parametric CZ.

Analysis lives on the optimizers: error_budget, robustness_sweep, quasi_static_fidelity,
spectator_fidelity, resonant_collision_fidelity, and more. Supporting modules: rb
(interleaved-RB estimator), literature (model-vs-device anchors), headtohead (in-loop
vs multiply-after), hardware (hardware-in-the-loop calibration), braket_bridge and
openpulse_export (vendor-neutral pulse export), mps (large-N evaluator).

See README.md for the full picture, API_REFERENCE.md for every signature, and examples/
for one runnable script per feature.
"""

from .parametric import (
    ParametricCouplerProfile,
    ParametricCZOptimizer,
    RepresentativeDefaultsWarning,
)
from .crossresonance import (
    CrossResonanceProfile,
    CrossResonanceZXOptimizer,
)
from .multiqubit import (
    MultiQubitProfile,
    MultiQubitOptimizer,
)
from .diagnostics import pauli_transfer_matrix, channel_unitarity
from .convenience import (optimize_cz, optimize_iswap, tunable_coupler_cz,
                          coupler_in_loop_cz)
from .basis import FourierBasis
# NumPy-only independent solvers; ship at top level without the [validate] extra.
# One per architecture, so the triple-solver discipline is library-independent for
# all three (not just the parametric CZ).
from .liouville import (liouville_f_proc, liouville_cr_f_proc,
                        liouville_nqubit_closed_f_proc)
from .mps import ChainTEBD  # NumPy-only MPS evaluator; see gradpulse.mps

__all__ = [
    "ParametricCouplerProfile", "ParametricCZOptimizer", "RepresentativeDefaultsWarning",
    "CrossResonanceProfile", "CrossResonanceZXOptimizer",
    "MultiQubitProfile", "MultiQubitOptimizer",
    "pauli_transfer_matrix", "channel_unitarity",
    "optimize_cz", "optimize_iswap", "tunable_coupler_cz", "coupler_in_loop_cz",
    "FourierBasis", "liouville_f_proc", "liouville_cr_f_proc",
    "liouville_nqubit_closed_f_proc", "ChainTEBD",
]
__version__ = "0.6.0"
__author__ = "Pure State Labs Inc."
__license__ = "MIT"
__copyright__ = "Copyright (c) 2026 Pure State Labs Inc."
