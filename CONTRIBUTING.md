# Contributing to gradpulse

Thanks for your interest in `gradpulse`. Contributions, bug reports, and
questions are all welcome.

## Reporting a bug or requesting a feature

Open an issue on the [GitHub issue tracker](https://github.com/PureStateLabs/gradpulse/issues).
For bugs, please include:

- what you ran (the command or a minimal code snippet),
- what you expected and what happened instead (full traceback if there is one),
- your environment: OS, Python version, and `torch.__version__` (and whether a
  GPU was used).

## Asking a question / getting support

If something is unclear rather than broken, open an issue with the
**question** label. There is no separate mailing list; the issue tracker is
the single place for support.

## Contributing code

1. Fork the repository and create a branch off `main`.
2. Install in editable mode with the validation extra:
   ```bash
   pip install -e ".[validate]"   # adds QuTiP for the cross-check
   pip install pytest
   ```
3. Make your change. Please keep the public API (`ParametricCouplerProfile`,
   `ParametricCZOptimizer`, the second architecture `CrossResonanceProfile`,
   `CrossResonanceZXOptimizer`, and the third `MultiQubitProfile`,
   `MultiQubitOptimizer`) backward-compatible unless the change is the point
   of the PR.
4. Run the test suite, which must pass before you open a PR:
   ```bash
   pytest tests/
   ```
   The smoke test runs in seconds on CPU. If your change touches the physics or
   the optimizer, also run the independent QuTiP cross-check and the structural
   checks:
   ```bash
   python examples/optimize_cz.py                       # optimize the 150 ns CZ, write cz_pulse.json
   python -m gradpulse.validate --pulse cz_pulse.json   # independent QuTiP cross-check
   python examples/validation_checks.py                 # gradient check, convergence, leakage, timing
   ```
   Deeper (slower) checks, worth running for physics changes: the cross-check
   across a grid of operating points and the simulated randomized-benchmarking
   bridge to a hardware-style estimator:
   ```bash
   python examples/validation_sweep.py                  # cross-check across durations/coupling/channels
   python examples/randomized_benchmarking.py           # leakage-aware interleaved-RB vs analytic F_avg
   ```
   Analysis / robustness tools (forward passes on the shipped pulse), and the
   composite paper figure (needs the `[viz]` extra, `pip install -e ".[viz]"`):
   ```bash
   python examples/robustness_sweep.py                  # amplitude/frequency calibration tolerance
   python examples/spectator_crosstalk.py               # always-on-ZZ neighbour penalty vs zeta
   python examples/optimize_robust.py --iterations 60   # robust ensemble optimization demo
   python examples/make_paper_figure.py                 # 6-panel summary (gate, dynamics, error budget, triple-solver, robustness, collision) -> paper/figure.png
   ```
   Second architecture (cross-resonance ZX), the measured-calibration demo, and
   the hardware-in-the-loop scaffold:
   ```bash
   python examples/optimize_cross_resonance.py          # fixed-frequency ZX(pi/2); DRAG ablation
   python -m gradpulse.validate --pulse zx_pulse.json   # QuTiP cross-check (auto-detects architecture)
   python examples/optimize_from_calibration.py         # optimize against a real device's measured T1/T2
   python examples/hardware_in_the_loop.py              # close the sim<->device gap from measured RB
   python examples/braket_export.py                     # Braket pulse export + cost (needs the [braket] extra)
   ```
   Third architecture (general N-qubit), the coupler-in-the-loop CZ, and the
   v0.5 additions (band-limited optimization, analytic filter-function robustness,
   complete-I/Q OpenPulse export, simultaneous gates, the qutip-qtrl benchmark):
   ```bash
   python examples/optimize_multiqubit.py               # N-qubit subset-target gate amid spectators
   python examples/coupler_in_loop_cz.py                # explicit-coupler CZ from a pair profile + leakage budget
   python examples/optimize_spectral.py                 # band-limited (Fourier/CRAB) optimization
   python examples/filter_function.py                   # analytic dephasing robustness (no Monte-Carlo)
   python examples/export_openpulse.py                  # OpenPulse 3.0 export (needs the [openpulse] extra)
   python examples/simultaneous_gates.py                # two gates at once on disjoint pairs
   python examples/gradient_checkpointing.py            # checkpoint_segments memory/compute trade
   python examples/benchmark_vs_qutip_qtrl.py           # head-to-head vs qutip-qtrl (needs the [benchmark] extra)
   ```
5. Open a pull request describing what changed and why. CI (`pytest` on Python
   3.10-3.13) runs automatically on the PR.

## Adding a control channel (the cross-check contract)

The independent QuTiP cross-check is only meaningful if it evolves the *same*
Hamiltonian the optimizer does. A new control-channel type therefore touches
**three** places, and they must stay in lockstep or the cross-check will silently
validate a *different* model and still "pass":

1. **Operator builder:** the time-independent operator the channel multiplies
   (e.g. `_build_coupler_ops` in `parametric.py`, the `Xop`/`Cop`/`Fop` dicts in
   `multiqubit.py`).
2. **Simulator hot path:** where that operator is added to the per-slice
   Hamiltonian with its amplitude/scale (the slice loop in the same module).
3. **QuTiP cross-check:** the rebuilt-from-scratch Hamiltonian in `validate.py`
   (`cross_check` / `multiqubit_cross_check`), which must add the identical term.

Two guards make this contract enforceable rather than a footgun:

- **Share one evolution core between the optimizer-side and standalone
  cross-checks** so steps 2-3 cannot drift apart. The cross-resonance path does
  this via `validate._qutip_cr_fproc`, used by both the file-based `cross_check`
  and the direct `cr_cross_check`; prefer that pattern for new channels.
- **Add a cross-check test whose pulse actually exercises the new channel** (a
  random non-zero amplitude on it), asserting agreement to ≲1e-6. A test that
  leaves the new channel at zero amplitude will pass against the wrong model;
  see `tests/test_echo_cr.py` and `tests/test_coupler_in_loop.py` for the shape.

If you cannot reach ≲1e-6 agreement with the new channel driven, the two
Hamiltonians differ. Fix that before anything else; a passing optimization on a
model the cross-check does not reproduce is not a validated result.

## Scope

`gradpulse` is intentionally compact: differentiable open-system (Lindblad)
GRAPE for a single entangling two-qubit gate on superconducting transmons.
Contributions that sharpen that core (additional gates, better validation,
clearer docs, performance) are very welcome. Contributions that broaden it into
a general-purpose control framework are better suited to a separate project.

## Code of conduct

Be respectful and constructive. Harassment or abuse of any kind is not
tolerated in issues, pull requests, or any other project space.
