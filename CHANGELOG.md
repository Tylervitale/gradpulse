# Changelog

This file tracks notable changes to gradpulse. It loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). All dates are UTC.

We're still pre-1.0, so minor releases can include breaking API changes.

## [0.6.0] - 2026-06-17

First public release. gradpulse is a differentiable, multi-solver-validated, open-system pulse optimizer for predictive superconducting-gate fidelities.

You give gradpulse a qubit pair (a generic profile or a real device's calibration) and a target gate. It finds the microwave and flux pulse that runs that gate, and it does the optimization through a full open-system simulation: $T_1$ relaxation, $T_\phi$ dephasing, and leakage into higher transmon levels are all in the forward pass. So the optimizer trades gate speed against decoherence as it shapes the pulse, and the pulse it returns is optimized against that noise rather than against a perfect simulation.

It also won't hand you a fidelity it can't back up. Every number gradpulse reports is reproduced by three solvers that share no code, and the underlying model reproduces the measured gate error of real, published devices.

### Added

#### Optimizers and architectures
- Parametric-coupler CZ optimizer. Autodiff GRAPE on PyTorch with decoherence in
  the loop, exact leakage-aware process fidelity, a QuTiP cross-check, and the
  NumPy Liouvillian as a third solver.
- N-qubit support in `multiqubit.py`. GRAPE runs on any coupling graph and you
  can target a subset of qubits with identity on the rest, so crosstalk and
  frequency collisions are handled inside the optimization. Also checked against
  QuTiP.
- Cross-resonance gate, including the echoed sequence (`echo=True`) that turns
  the static ZZ into a single-qubit IZ you can remove, the way CR CNOTs get
  calibrated on real hardware.
- Explicit tunable-coupler model with a live coupler transmon in the loop, plus
  `coupler_in_loop_cz`, an opt-in explicit-coupler CZ. It reuses the
  `MultiQubitOptimizer` engine and reports coupler leakage and the SW parameter.
- `optimize_cz` and `optimize_iswap` one-call wrappers.
- Simultaneous multi-gate optimization and gradient checkpointing.

#### Solvers and validation
- A NumPy-only, exact-generator Liouvillian solver that doesn't depend on any
  external library, now covering all three architectures. That's parametric CZ,
  cross-resonance (`liouville_cr_f_proc`, which understands echo and virtual Z),
  and the closed-system N-qubit register (`liouville_nqubit_closed_f_proc`, with
  the spectator coupling folded into the drift). All of these are exported at the
  top level. CR matches the QuTiP referee to about 1e-7 (what's left is QuTiP's
  own Trotter split), and the N-qubit closed propagator matches the optimizer to
  machine precision, around 1e-15.
- Fock-truncation convergence study and the beyond-RWA correction
  (`counter_rotating_fidelity`), with support for an arbitrary 4x4 target gate.
- Analyses for spectator always-on ZZ, resonant collisions, and TLS defects.
  Each one is validated against a full multi-transmon QuTiP simulation.
- Divergence guards and convergence diagnostics in all three optimizers.
- `process_fidelity_sparse`, a sparse/Krylov path for cases past the dense wall.
- Analytic filter-function robustness that accounts for leakage (within about 1%
  of Monte Carlo).
- `tests/test_liouville.py` with cross-resonance (4 modes) and N-qubit (3 modes)
  cross-checks, behind a pure-NumPy import guard.

#### Calibration, control, and export
- Noisy closed-loop calibration: finite-shot, leakage-aware interleaved-RB
  estimates with shot noise in the QuTiP device backend.
- Band-limited spectral/CRAB optimization (`basis.py`, `optimize_spectral`).
- Full I/Q export plus OpenPulse 3.0 / OpenQASM 3, with an independent AST parser
  to check it (we needed this since `qiskit.pulse` was dropped in Qiskit 2.0).

#### Tooling and examples
- `gradpulse.viz` plotting helpers and a few tutorial notebooks.
- A head-to-head benchmark against `qutip-qtrl`, roughly at parity on wall-clock.
- `examples/leakage_in_the_loop.py`, which sweeps a cross-resonance gate's
  duration and uses the package's own `error_budget` to split each optimized
  gate's error into the decoherence floor the coherence formula captures and the
  coherent control plus leakage part it misses. The result is a clear crossover:
  at long durations the gate is coherence-limited and the formula tracks the real
  error, while at short durations it's leakage-limited and the formula
  under-predicts (by roughly 60x at 110 ns). It also caps the BLAS/OpenMP thread
  count at import so the stacked solvers don't oversubscribe the machine.

[0.6.0]: https://github.com/PureStateLabs/gradpulse/releases/tag/v0.6.0
