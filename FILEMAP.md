# gradpulse file map

A complete inventory of every file in the repository and what it does, grouped by
area. Line counts (raw `wc -l`) are given for the source modules to show where the
weight sits.

For *why* (the physics and how the validation works) see the [README](README.md)
and the module docstrings; for the flat API index see [API_REFERENCE.md](API_REFERENCE.md);
for runnable usage see [`examples/`](examples/).

---

## Root: config, packaging, docs

| File | What it does |
|---|---|
| [README.md](README.md) | Project overview: the device-validation results, the triple-solver cross-check, the four architectures, the analysis/noise suite, hardware export, quickstart, install, and scope. |
| [RESULTS.md](RESULTS.md) | Hardware-validation log: Cepheus device facts, the 160-gate prediction-vs-measurement sweep and its σ analysis, the convergence and consistent-calibration triple-checks, the literature anchors, and the cost summary. |
| [API_REFERENCE.md](API_REFERENCE.md) | Flat, searchable API index: every public class/method/function, signatures, return dicts, a conventions table, and a customization cookbook. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributor guide; documents the **cross-check contract** (a new control channel must touch operator-builder + simulator + QuTiP validator in lockstep). |
| [CITATION.cff](CITATION.cff) | Citation metadata (author, version, license, repository URL, keywords). |
| [LICENSE](LICENSE) | MIT license text. |
| [pyproject.toml](pyproject.toml) | Build/packaging: name, version, deps (torch, numpy), optional extras (`validate`/`viz`/`sparse`/`braket`/`openpulse`/`benchmark`), pytest config, `src/` layout, console-script entry point. |
| [requirements.txt](requirements.txt) | Full dev-environment mirror of every pyproject.toml extra (core + qutip + qutip-qtrl + scipy + matplotlib + braket + openpulse + pytest) for a one-command, zero-skip test setup. |
| [.gitattributes](.gitattributes) | Repo-wide line-ending policy (`* text=auto eol=lf`, LF on every platform) + binary markings for `*.png`/`*.npy`/`*.pdf`/`*.gz` so they're never normalized. |
| [.gitignore](.gitignore) | Ignores Python caches/build artifacts/`*.log`/generated root-level pulses, paper build artifacts (`.pdf`/`.aux`/...), computed/regeneratable `examples/cepheus/` outputs, and `submissions/`; paper `.tex`/`.bib` and figures stay tracked. |
| [.github/workflows/tests.yml](.github/workflows/tests.yml) | CI: matrix pytest on Python 3.10-3.13 (CPU torch) + a separate `[validate]` job that runs the QuTiP cross-check tests. |

---

## `src/gradpulse/`: the importable package (~12.2K lines)

### Core engine & architectures

| File | Lines | What it does |
|---|---:|---|
| [parametric.py](src/gradpulse/parametric.py) | 2008 | **Architecture #1 (core):** parametric-coupler CZ. 9-D open-system (Lindblad) simulator with batched `matrix_exp`, DRAG, multi-seed Adam + L-BFGS GRAPE, exact Choi process fidelity, band-limited (CRAB) spectral optimization, and in-loop robust objectives (coherent-only, quasi-static, whole-band filter-function). |
| [analysis.py](src/gradpulse/analysis.py) | 1130 | `ParametricCZAnalysisMixin` mixed into the optimizer: error budget / channel unitarity, robustness sweep, quasi-static / colored / filter-function dephasing (scoring **and** the shared differentiable in-loop estimator), spectator-ZZ, resonant collision, lossy TLS defect, and dt-convergence. |
| [crossresonance.py](src/gradpulse/crossresonance.py) | 1415 | **Architecture #2:** fixed-frequency cross-resonance ZX(π/2). Batched multi-seed GRAPE (all seeds in one forward/backward), DRAG, echoed-CR, virtual-Z framing, beyond-RWA check + refinement, and the same spectator/collision analysis. |
| [multiqubit.py](src/gradpulse/multiqubit.py) | 1054 | **Architecture #3:** general N-qubit GRAPE on an arbitrary coupling graph. Subset-target gate + identity on spectators, simultaneous gates, explicit tunable-coupler, open/closed system, sparse-Krylov eval, gradient checkpointing. |

### Profiles, helpers, basis

| File | Lines | What it does |
|---|---:|---|
| [profiles.py](src/gradpulse/profiles.py) | 500 | `ParametricCouplerProfile` device dataclass + calibration loaders (`from_braket_calibration`, `from_calibration`, `from_ibm_backend`) + the representative-defaults warning. |
| [convenience.py](src/gradpulse/convenience.py) | 188 | One-call wrappers: `optimize_cz`, `optimize_iswap`, `tunable_coupler_cz`, `coupler_in_loop_cz`. |
| [basis.py](src/gradpulse/basis.py) | 95 | `FourierBasis`, the band-limited (CRAB) synthesis matrix used by `optimize_spectral`. |
| [diagnostics.py](src/gradpulse/diagnostics.py) | 58 | NumPy-only channel diagnostics: Pauli transfer matrix + channel unitarity (Wallman et al. 2015). |
| [_device.py](src/gradpulse/_device.py) | 32 | Single source of truth for the compute device (`GRADPULSE_DEVICE` env → else CUDA-if-available). |

### Independent validation & solvers

| File | Lines | What it does |
|---|---:|---|
| [validate.py](src/gradpulse/validate.py) | 1499 | **Independent QuTiP cross-check** for every feature: matched piecewise-constant + adaptive `mesolve`, both pair architectures, spectator / collision / TLS / coupler-elimination / multiqubit cross-checks. |
| [liouville.py](src/gradpulse/liouville.py) | 690 | **Third, QuTiP-free solver:** pure-NumPy Liouvillian superoperator with a full-generator matrix exponential (independently bounds the Trotter splitting error). One per architecture: `liouville_f_proc` (parametric CZ), `liouville_cr_f_proc` (cross-resonance), `liouville_nqubit_closed_f_proc` (N-qubit closed), so the library-independent leg spans all three. |
| [mps.py](src/gradpulse/mps.py) | 467 | `ChainTEBD`, an evaluation-only matrix-product-state evaluator (trajectory-unraveled) to score pulses past the dense ~4-qubit wall; reports a restricted-ensemble *witness*, not F_proc. |
| [rb.py](src/gradpulse/rb.py) | 360 | Simulated leakage-aware **interleaved randomized benchmarking** (2-qubit Clifford group, native-gate superoperators), bridging the analytic F_avg to a hardware-style estimator. |
| [benchmark.py](src/gradpulse/benchmark.py) | 247 | Head-to-head GRAPE benchmark of gradpulse's engine vs `qutip-qtrl` on an identical closed-system problem. |
| [literature.py](src/gradpulse/literature.py) | 501 | **Data-driven hardware-anchor validation:** torch-free load/judge for cited JSON device anchors (`examples/anchors/`). Builds the three coherence variants via `from_calibration` and judges the decoherence floor against the published number, with the `coherence_limited` flag selecting equality vs lower-bound. Derives `T2` in code from a published `T1`+`T_phi` (`effective_t2_ns`) so no hand-computed `T2` lives in the data. |
| [headtohead.py](src/gradpulse/headtohead.py) | 138 | **In-loop vs multiply-after demonstration:** `run_head_to_head` sweeps gate duration and runs both recipes (reusing `optimize_multi_seed(diss_scale=1/0)`) to measure where decoherence-in-the-loop beats the `F_coherent·e^{−t/T}` budget. Orchestrates the optimizer. |

### Hardware on-ramp & export

| File | Lines | What it does |
|---|---:|---|
| [hardware.py](src/gradpulse/hardware.py) | 483 | Hardware-in-the-loop scaffold: `HardwareBackend` protocol, `SimulatedBackend` / `QuTiPDeviceBackend` / `BraketPulseBackend`, coherence inference, and the `calibrate_to_hardware` loop. |
| [braket_bridge.py](src/gradpulse/braket_bridge.py) | 695 | Offline Amazon Braket export + the interleaved-RB hardware harness: envelope → `PulseSequence` / `ArbitraryWaveform` (round-trip verified), OpenPulse 3.0 serialization, cost/feasibility estimates, IRB circuit generation (reuses the `rb.py` Clifford group; ideal-return-to-\|00⟩ verified offline); `cz_durations_from_native_calibration` reads real per-pair CZ gate times out of a device's native-gate pulse calibration (the input that fixed the Cepheus study's flat-60 ns assumption). **Level B**: `build_bench_cz_pulse_sequence` binds a gradpulse-designed pulse to the device's CZ frame as a `pulse_gate` inside the verbatim box, `bench_cz_peak_from_native_calibration` anchors it to the device's calibrated flux peak, and `verify_levelb_offline` gates the serialization (fidelity needs the QPU; pulses don't run on the local simulator). |
| [openpulse_export.py](src/gradpulse/openpulse_export.py) | 304 | Vendor-neutral OpenQASM 3 / OpenPulse 3.0 `defcal` text export (complex I/Q preserved), re-parse-verified against an independent parser. |
| [viz.py](src/gradpulse/viz.py) | 158 | Matplotlib helpers: `plot_pulse`, `plot_convergence`, `plot_error_budget`, `plot_robustness`. |

### Package entry points

| File | Lines | What it does |
|---|---:|---|
| [__init__.py](src/gradpulse/__init__.py) | 91 | Public API surface + package docstring; re-exports the three architectures, convenience wrappers, `FourierBasis`, the three `liouville_*` solvers, `ChainTEBD`. |
| [__main__.py](src/gradpulse/__main__.py) | 113 | CLI: `python -m gradpulse` welcome banner + `--version` (reads the version without importing torch). |
| `gradpulse.egg-info/*` | n/a | Auto-generated editable-install metadata (PKG-INFO, SOURCES.txt, ...). Not source; regenerated by setuptools. |

---

## `tests/`: 41 test files + fixtures (~6.3K lines)

| File | What it verifies |
|---|---|
| [test_reproducibility.py](tests/test_reproducibility.py) | **Ship gate:** three independent solvers agree on the headline CZ F_proc (no hardcoded number); consensus-blessed drift checkpoint. |
| [test_smoke.py](tests/test_smoke.py) | Optimizer runs end-to-end and returns a valid, better-than-trivial pulse. |
| [test_integration.py](tests/test_integration.py) | Step-order 2 (Strang), dt-convergence, frequency-mode coupler, `mesolve` unbiasedness, operator-builder parity. |
| [test_calibration.py](tests/test_calibration.py) | Braket / IBM / normalized calibration loaders, unit conversion, representative-defaults warning. |
| [test_residuals.py](tests/test_residuals.py) | `best_raw_param` round-trip, double precision, coupling rolloff vs detuning. |
| [test_diagnostics.py](tests/test_diagnostics.py) | Error budget, unitarity, detuning primitive, robustness, quasi-static, finite-T, static ZZ. |
| [test_modes.py](tests/test_modes.py) | 4/6-channel modes, firbrick smoother, warm-starts, target gates, line response. |
| [test_levels.py](tests/test_levels.py) | Fock-truncation convergence (CZ @3, CR @4), custom target unitary, beyond-RWA check. |
| [test_crossresonance.py](tests/test_crossresonance.py) | CR ZX gate convergence, error budget, DRAG, QuTiP cross-check. |
| [test_echo_cr.py](tests/test_echo_cr.py) | Echoed-CR refocuses static ZZ; optimizer↔QuTiP parity with echo applied. |
| [test_multiqubit.py](tests/test_multiqubit.py) | N-qubit construction, subset targets, open/closed paths, state-transfer, QuTiP cross-check. |
| [test_tunable_coupler.py](tests/test_tunable_coupler.py) | Explicit tunable-coupler CZ wiring + machine-precision QuTiP cross-check. |
| [test_coupler_in_loop.py](tests/test_coupler_in_loop.py) | Coupler-in-loop diagnostics (leakage, SW small-parameter) + cross-check. |
| [test_coupler_elimination.py](tests/test_coupler_elimination.py) | Schrieffer-Wolff elimination residual scales as (g/Δ)². |
| [test_spectators.py](tests/test_spectators.py) | Always-on-ZZ spectator = effective detuning; validated vs 27-D / multi-transmon QuTiP. |
| [test_collision.py](tests/test_collision.py) | Resonant frequency-collision (evolving spectator) vs QuTiP, both architectures. |
| [test_tls.py](tests/test_tls.py) | Lossy two-level-defect diagnostic vs QuTiP. |
| [test_colored_noise.py](tests/test_colored_noise.py) | 1/f^α colored-noise MC: slow limit = quasi-static, motional narrowing, cross-qubit correlation. |
| [test_filter_function.py](tests/test_filter_function.py) | Analytic filter function = direct error generator and agrees with MC. |
| [test_spectral.py](tests/test_spectral.py) | Band-limited Fourier/CRAB optimization. |
| [test_simultaneous_gates.py](tests/test_simultaneous_gates.py) | Parallel CZ×CZ on disjoint pairs; combined target cross-checked. |
| [test_checkpointing.py](tests/test_checkpointing.py) | Gradient checkpointing: identical value + gradient, lower memory. |
| [test_sparse_eval.py](tests/test_sparse_eval.py) | Sparse/Krylov fidelity matches dense; runs at N=6. |
| [test_mps.py](tests/test_mps.py) | TEBD→exact convergence, untruncated MPS = statevector, trajectory witness vs dense. |
| [test_rb.py](tests/test_rb.py) | Clifford group order, IRB recovers depolarizing fidelity, leakage bias. |
| [test_hardware.py](tests/test_hardware.py) | Coherence inference + closed calibration loop (Simulated + QuTiP backends). |
| [test_noisy_calibration.py](tests/test_noisy_calibration.py) | Finite-shot IRB estimator unbiasedness + shot-noise scaling. |
| [test_iq_export.py](tests/test_iq_export.py) | Complete I/Q (DRAG baked in) + OpenPulse round-trip. |
| [test_braket_bridge.py](tests/test_braket_bridge.py) | Offline Braket export, cost arithmetic, credential-wall behavior, IRB circuit generation (ideal return-to-\|00⟩, native-CZ build), and Level-B pulse binding (bench-pulse build + serialize, flux-peak anchoring, verbatim+`play`, reference-circuits-unaffected, `verify_levelb_offline`). |
| [test_cz_durations.py](tests/test_cz_durations.py) | The native-cal CZ-duration parser: picks the flux play over decoy channels, trims pulse padding, the three duration modes (buffer/active/effective), dt scaling, hyphen keying, and robustness to empty/templated waveforms. Pure, no SDK. |
| [test_irb_resolution.py](tests/test_irb_resolution.py) | Interleaved-RB resolution study: the survival model reproduces the hardware canary (length-1 and depth-128), and a free-asymptote fit recovers the true gate error under asymmetric readout. |
| [test_benchmark.py](tests/test_benchmark.py) | qutip-qtrl benchmark engine reaches the optimum. |
| [test_optimizer_guards.py](tests/test_optimizer_guards.py) | Divergence guards catch injected NaN; convergence diagnostics reported. |
| [test_liouville.py](tests/test_liouville.py) | Third solver: `_expm` correctness, asserts it imports no scipy/qutip/torch, and that the NumPy-only leg reproduces QuTiP (parametric + cross-resonance) and the optimizer (N-qubit closed) across all three architectures. |
| [test_machine_precision.py](tests/test_machine_precision.py) | Independent solvers converge to ~10⁻¹³ as `dt→0` (Richardson); operating-point gap is provably first-order Trotter splitting, not model disagreement. |
| [test_cross_check_mutation.py](tests/test_cross_check_mutation.py) | **The cross-check has teeth (non-vacuous):** injects classic master-equation faults (T1/drive/dephasing errors via the public profile API, plus a dropped Lindblad anticommutator via the one justified monkeypatch) and proves each diverges from the optimizer by 1700-43000× the agreement gap. |
| [test_literature.py](tests/test_literature.py) | Data-driven hardware anchors: schema/provenance guards, coherence-variant derivation, `coherence_limited` equality-vs-lower-bound judgment, and the `analytic` floor mode for devices without published Hamiltonian parameters (Stehlik). |
| [test_headtohead.py](tests/test_headtohead.py) | Coherent-only (`diss_scale=0`) and dephasing-robust (`robust_dephasing_sigma_mhz`) objectives: guards, estimator-alignment with `quasi_static_fidelity`, and the head-to-head summary structure/gaps. |
| [test_filter_in_loop.py](tests/test_filter_in_loop.py) | Whole-band filter-function objective (`robust_filter_sigma_mhz`): machine-precision alignment of the in-loop estimator with `filter_function_fidelity`, differentiability, σ² scaling, standalone guards, and a slow end-to-end band-hardening check. |
| [test_import_hygiene.py](tests/test_import_hygiene.py) | Guards against bare `import validate`-style imports (must use `gradpulse....`). |
| [test_viz.py](tests/test_viz.py) | matplotlib plot-helper smoke tests. |
| [tests/fixtures/reference_cz_pulse.json](tests/fixtures/reference_cz_pulse.json) · [.npy](tests/fixtures/reference_cz_pulse.npy) | The blessed 150 ns CZ pulse + metadata, the consensus-pinned headline artifact every analysis test reuses. |
| [tests/fixtures/braket_device_calibration.json](tests/fixtures/braket_device_calibration.json) | Real 107-qubit Braket standardized device-properties fixture for the loader tests. |

---

## `examples/`: 48 scripts + data + notebooks (~6.2K lines)

The Cepheus hardware-validation study (Section below) accounts for nineteen of the scripts
(all under `examples/cepheus/`); the rest are one-feature demos.

| File | What it demonstrates |
|---|---|
| [examples/README.md](examples/README.md) | Index of all examples, grouped by task. |
| [optimize_cz.py](examples/optimize_cz.py) | **Start here:** minimal end-to-end CZ + dt-convergence. |
| [optimize_iswap.py](examples/optimize_iswap.py) | iSWAP (the coupler's native gate) by changing one argument. |
| [optimize_cross_resonance.py](examples/optimize_cross_resonance.py) | CR ZX(π/2), DRAG ablation, truncation convergence, beyond-RWA. |
| [optimize_multiqubit.py](examples/optimize_multiqubit.py) | Two-qubit gate inside a 3-qubit register (crosstalk in the loop). |
| [coupler_in_loop_cz.py](examples/coupler_in_loop_cz.py) | CZ with the coupler explicitly modeled + leakage budget. |
| [optimize_from_calibration.py](examples/optimize_from_calibration.py) | Optimize against measured Rigetti Cepheus parameters. |
| [optimize_robust.py](examples/optimize_robust.py) | Robust ensemble optimization vs a miscalibration ensemble. |
| [robustness_sweep.py](examples/robustness_sweep.py) | Calibration-tolerance sweep of a saved pulse. |
| [spectator_crosstalk.py](examples/spectator_crosstalk.py) | Always-on-ZZ + resonant-collision sweeps. |
| [filter_function.py](examples/filter_function.py) | Analytic dephasing robustness vs Monte-Carlo. |
| [randomized_benchmarking.py](examples/randomized_benchmarking.py) | Leakage-aware IRB vs analytic F_avg bridge. |
| [unitarity_rb.py](examples/unitarity_rb.py) | Purity (unitarity) RB to split coherent from incoherent error (u=1 coherent, u<1 incoherent); verified pure-depolarizing → u<1 and pure-coherent → u=1. Costs ~4.5-9× a plain IRB (9-basis purity measurement). |
| [optimize_spectral.py](examples/optimize_spectral.py) | Band-limited (Fourier/CRAB) optimization. |
| [simultaneous_gates.py](examples/simultaneous_gates.py) | Two gates at once on disjoint pairs. |
| [gradient_checkpointing.py](examples/gradient_checkpointing.py) | Memory-saving checkpointing demo. |
| [mps_large_register.py](examples/mps_large_register.py) | MPS witness at N=6 with chi-convergence. |
| [validation_checks.py](examples/validation_checks.py) | Gradient check, dt-convergence, leakage, timing (the paper's "Validation" numbers). |
| [validation_sweep.py](examples/validation_sweep.py) | QuTiP cross-check across a grid of operating points (asserts < 1e-3). |
| [validate_against_literature.py](examples/validate_against_literature.py) | Discover every cited JSON anchor in `examples/anchors/` and reproduce each device's published coherence budget (runs real GRAPE or analytic floor). Adding a device is dropping in a JSON file; see [`examples/anchors/README.md`](examples/anchors/README.md). |
| [stehlik_predict_vs_measured.py](examples/stehlik_predict_vs_measured.py) | Breadth analysis: analytic coherence floor vs measured error for all 11 Stehlik 2021 pairs, with the no-selection lower-bound story. |
| [decoherence_in_the_loop.py](examples/decoherence_in_the_loop.py) | In-loop vs optimise-coherent-then-multiply, swept over gate duration (the paper's head-to-head); prints the measured gap and writes `paper/decoherence_in_the_loop.png` with `[viz]`. |
| [leakage_in_the_loop.py](examples/leakage_in_the_loop.py) | Companion crossover: duration-swept cross-resonance gate, each optimized gate's error split by `error_budget` into the decoherence floor and the coherent control+leakage part: coherence-limited at long durations, leakage-limited at short (formula under-predicts ~60× at 110 ns). Triple-solver checked; writes `paper/leakage_in_the_loop.png` with `[viz]`. |
| [hardware_in_the_loop.py](examples/hardware_in_the_loop.py) | Closed sim↔device calibration loop. |
| [braket_export.py](examples/braket_export.py) | Offline Braket pulse export + cost. |
| [export_openpulse.py](examples/export_openpulse.py) | OpenPulse 3.0 export with DRAG baked into the I/Q. |
| [benchmark_vs_qutip_qtrl.py](examples/benchmark_vs_qutip_qtrl.py) | Head-to-head vs qutip-qtrl. |
| [benchmark_vs_c3.py](examples/benchmark_vs_c3.py) | Head-to-head vs C3 (c3-toolset), with portability notes. |
| [make_paper_figure.py](examples/make_paper_figure.py) | Generates the six-panel summary figure from the reference pulse. |
| [make_validation_figure.py](examples/make_validation_figure.py) | Generates the three-panel validation figure (dt-convergence, four-way solver agreement, cross-architecture cross-checks) from the reference pulse. |
| [examples/data/rigetti_cepheus_calibration.json](examples/data/rigetti_cepheus_calibration.json) | Bundled real Rigetti Cepheus-1-108Q calibration. |
| [examples/anchors/](examples/anchors/) | Cited literature device anchors: one JSON per device (`sung_2021_cz`, `marxer_2023_cz`, `stehlik_2021_cz`) plus a README documenting schema/provenance; consumed by `literature.py` and `validate_against_literature.py`. |
| [examples/stehlik_2021_table1.json](examples/stehlik_2021_table1.json) | The 11-pair breadth table behind `stehlik_predict_vs_measured.py`. |
| [examples/notebooks/01_intro_cz.ipynb](examples/notebooks/01_intro_cz.ipynb) | Notebook: CZ end-to-end with plots. |
| [examples/notebooks/02_real_device.ipynb](examples/notebooks/02_real_device.ipynb) | Notebook: same on the bundled real calibration. |
| [examples/notebooks/03_hardware_validation.ipynb](examples/notebooks/03_hardware_validation.ipynb) | Notebook: the flagship live-QPU result. Coherence floor vs measured CZ error across all 160 Cepheus pairs, the σ refinement, and the scatter (free; reads the committed sweep). |

### Cepheus hardware-validation study

The scripts behind [RESULTS.md](RESULTS.md), all under `examples/cepheus/`. They read the live
Rigetti Cepheus-1-108Q calibration (cached locally), predict each pair's CZ error, and compare
against the device's published measurement. Most run offline against the cached calibration;
hardware submission and Level-B benchmarking are through `run_irb_on_braket.py` and
`levelb_pulse_benchmark_offline.py` below.

| File | What it does |
|---|---|
| [cepheus_predict_vs_measured.py](examples/cepheus/cepheus_predict_vs_measured.py) | Free first pass: analytic coherence floor vs measured error across all 193 pairs, no optimization. |
| [cepheus_grape_sweep_all.py](examples/cepheus/cepheus_grape_sweep_all.py) | The load-bearing run: full GRAPE decoherence floor for all 160 active gates at each pair's real CZ duration, with per-duration `f_coh` caching and crash-safe resume. |
| [cepheus_grape_floor_pairs.py](examples/cepheus/cepheus_grape_floor_pairs.py) | The same floor on a handful of pairs, to separate the analytic-floor artifact from genuinely non-coherence-limited gates. |
| [cepheus_lowerbound_scatter.py](examples/cepheus/cepheus_lowerbound_scatter.py) | The **headline figure**: the unselected one-sided lower bound over all 160 active pairs (floor ≤ measured, within the RB error bar on ~150/160; median floor 0.66×). Nothing selected on the prediction. Reads `cepheus_grape_sweep_realdur.json` (no AWS) and writes `paper/cepheus_scatter.png`. |
| [cepheus_sigma_validation.py](examples/cepheus/cepheus_sigma_validation.py) | The σ **refinement** (not the headline, and not an independent test): σ = \|predicted − measured\| / RB standard-error on the saturation subset (floor within 0.8-1.25× of measured, a model-defined regime), gauging how tight the bound is where the gate is coherence-limited. Std-errors pinned to `cepheus_cz_std_errors.json` for offline reproducibility. |
| [cepheus_convergence_check.py](examples/cepheus/cepheus_convergence_check.py) | Triple-check that the per-pair spread is not under-optimization (16 seeds vs 2, more iterations). |
| [cepheus_consistent_recheck.py](examples/cepheus/cepheus_consistent_recheck.py) | Triple-check on a single fresh calibration; surfaces the calibration-self-consistency ceiling. |
| [cepheus_realdur_check.py](examples/cepheus/cepheus_realdur_check.py) | Exact before/after for the target pairs when the flat-60 ns assumption is replaced by real durations. |
| [cepheus_irb_resolution_study.py](examples/cepheus/cepheus_irb_resolution_study.py) | Monte-Carlo over the real 11520-Clifford group: which RB design and budget resolves a 0.4% gate under T2 and asymmetric readout. |
| [cepheus_peak_cal_study.py](examples/cepheus/cepheus_peak_cal_study.py) | Offline proof (real Clifford group, shot+sequence noise) that the Level-B peak cal must select by MAX SURVIVAL, not `min(fit r_cz)` which clamps on noise and wasted the first $33 sweep. Backs `select_best_peak`/`fit_resolved`. |
| [cepheus_coupler_only_ceiling.py](examples/cepheus/cepheus_coupler_only_ceiling.py) | The honest hardware-realizable Level-B ceiling: optimize coupler-flux-only (+virtual-Z, qubit-frequency channels disabled, as Cepheus's fixed-frequency qubits require). ~0.69 F_avg with representative coupler params -- the 0.97 demo used non-transferable qubit-frequency control. |
| [cepheus_coupler_param_sweep.py](examples/cepheus/cepheus_coupler_param_sweep.py) | Sweep the (unmeasured) coupler frequency to find where a coupler-only CZ becomes achievable in-model -> the target for on-device coupler characterization (Path 2). |
| [cepheus_faithful_model.py](examples/cepheus/cepheus_faithful_model.py) | Faithful (16,25) model: Cepheus-measured qubits + COMPLETE Rigetti-prototype coupler set (g1c/g2c/g12, anharms, coupler 2.644->3.622 GHz). F_avg 0.940, 0.4% leak -- strongest in-model GO-capability evidence. PROTOTYPE params, not Cepheus-confirmed; sensitivity (`cepheus_coupler_sensitivity.json`) shows coupling g is the high-impact unknown (g=60->0.65 vs 90->0.92). |
| [cepheus_closed_loop_cal.py](examples/cepheus/cepheus_closed_loop_cal.py) | Joint/staged closed-loop calibrator (flux scale + smooth shape modes + virtual-Z) -- backend-agnostic `joint_cal`/`staged_cal(measure_fn,...)`; sim surrogate for the free rehearsal, `BraketRBMeasure` for the on-device run. Rehearsal-validated: survival cost (not biased RB-fit) + JOINT 2-D virtual-Z grid (phases coupled) recovers 91% of the gap; naive Nelder-Mead/sequential-1D get stuck (Path 1, the route to a hardware GO). |
| [cepheus_speed_headroom.py](examples/cepheus/cepheus_speed_headroom.py) | Gate-duration sweep testing whether a FASTER CZ could beat native (lower coherence floor). Result: no speed headroom in-model -- F_avg worsens as the gate shortens (the coupling can't complete the swap faster), so beating native needs stronger coupling/coherence (hardware), not a cleverer pulse. |
| [cepheus_coupler_characterization.py](examples/cepheus/cepheus_coupler_characterization.py) | Swap-spectroscopy of the \|11>-\|02> avoided crossing (prep \|11> -> coupler flux(amp,dur) -> measure P\|11>): the measurement that yields the coupler params -> makes Level-B PREDICTIVE (Level-A 0.60σ median). Builds the on-device circuit (offline-verified) + validates the swap-rate fit on synthetic data (recovers g_eff to <0.3 MHz). |
| [run_irb_on_braket.py](examples/cepheus/run_irb_on_braket.py) | Interleaved RB of a CZ on a real Braket QPU, the `device.run()` step. Offline default verifies every circuit returns to \|00⟩, rehearses on the local simulator, and prints the exact bill; `--submit --device-arn` runs **Level A** (native-CZ) IRB with canary/cost/online guards. **Level B** (`--pulse --pulse-file`) benchmarks a gradpulse-designed pulse on the device's CZ frame; offline it checks serialization (the local sim can't run pulses). |
| [levelb_pulse_benchmark_offline.py](examples/cepheus/levelb_pulse_benchmark_offline.py) | The free half of "close the loop": optimize a gradpulse CZ (parametric, or `--tunable` for the Cepheus-matched baseband coupler), bind it to a Braket benchmarked-gate pulse, verify it serializes (OpenPulse/OpenQASM, verbatim+`play`, Clifford closes), and save the activation waveform for `run_irb_on_braket.py --pulse-file`. The gate fidelity is the half that needs the QPU. |
| [cepheus_rebuild_levelb_pulse.py](examples/cepheus/cepheus_rebuild_levelb_pulse.py) | Designs a high-fidelity tunable-coupler CZ for pair 16-25 with the **physical** objective (`fidelity="cz_data_virtualz"`): measured qubit freqs/T1/T2 + a representative dispersive coupler, warm-started from the device's own native CZ shape (`cepheus_cz_shape.npy`). Reaches F≈0.95 + the virtual-Z phases. Capability demo: representative coupler, open-loop (see the honest scope note in the file header). |
| `examples/cepheus/cepheus_*.{json,npy}` · `levelb_*` | Cached study inputs/outputs: the calibration snapshot, extracted per-pair CZ durations, the flat-60 ns and real-duration GRAPE sweeps, the pinned CZ standard-errors (`cepheus_cz_std_errors.json`), the Level-B activation waveforms, and the **raw paid-hardware run dumps** preserved from the device sessions: `cepheus_irb_1625_levelA.json` (the $57 Level-A IRB), `cepheus_scout_peak.json` (the $6.67 Level-B scout), `cepheus_stage1_peak.json` (the $33 stage-1 peak). Computed/regeneratable outputs are git-ignored (see `.gitignore`). |

---

## `paper/`: paper sources

| File | What it does |
|---|---|
| [paper/paper.md](paper/paper.md) | The JOSS software paper (Markdown); ships with the repo and is submitted via the pyOpenSci/JOSS route. |
| [paper/paper.bib](paper/paper.bib) | Bibliography for the JOSS paper (`paper.md`). |
| [paper/gradpulse_arxiv.tex](paper/gradpulse_arxiv.tex) | Extended arXiv preprint source (revtex4-2, two-column, inline bibliography); compiles with bundled Tectonic. Not yet submitted. |
| `paper/gradpulse_arxiv.pdf` | Compiled preprint PDF (git-ignored as a build artifact; rebuild from the `.tex`). |
| [paper/figure.png](paper/figure.png) | The generated six-panel summary figure (see `examples/make_paper_figure.py`); embedded in the README. |
| [paper/validation_figure.png](paper/validation_figure.png) | The generated three-panel validation figure (see `examples/make_validation_figure.py`). |
| [paper/cepheus_scatter.png](paper/cepheus_scatter.png) | The unselected prediction-vs-measured lower-bound scatter over all 160 Cepheus pairs (see `examples/cepheus/cepheus_lowerbound_scatter.py`); embedded in the paper's hardware-validation section. |
| [paper/leakage_in_the_loop.png](paper/leakage_in_the_loop.png) | The cross-resonance leakage-vs-decoherence crossover figure produced by `examples/leakage_in_the_loop.py` (generated output; not embedded in the paper or README). |

---

*gradpulse is built and maintained by [Pure State Labs Inc.](https://purestatelabs.com), MIT-licensed.*

### New Integration Modules

| File | What it does |
|---|---|
| [src/gradpulse/dlp.py](src/gradpulse/dlp.py) | Differentiable Logical Programming: soft logical constraints embedded directly into continuous GRAPE optimization. |
| [src/gradpulse/microscheduler.py](src/gradpulse/microscheduler.py) | Cycle-Aware Micro-Scheduling: precisely packs dependency-resolved analog pulses onto nanosecond boundaries. |
| [src/gradpulse/distortion.py](src/gradpulse/distortion.py) | Cable Distortion Hack: active pre-distortion (iterative deconvolution) to correct for cryogenic wiring transfer functions. |
| [src/gradpulse/mitigation.py](src/gradpulse/mitigation.py) | Zero-Noise Extrapolation: noise scaling and polynomial/exponential extrapolation routines for error mitigation. |
| [src/gradpulse/scheduling.py](src/gradpulse/scheduling.py) | Dependency Graphs: Directed Acyclic Graph structures for ensuring causal, commutation-aware execution of quantum operations. |
| [src/gradpulse/compression.py](src/gradpulse/compression.py) | Pulse-Level Compression: Run-Length Encoding and spline-based downsampling for AWG memory efficiency. |
| [src/gradpulse/rl.py](src/gradpulse/rl.py) | Reinforcement Learning: Gymnasium environments for discrete sequence discovery beyond local gradients. |
| [src/gradpulse/hardware.py](src/gradpulse/hardware.py) | Hardware Backends & Bayesian Optimization: Closed-loop interfaces, including Gaussian Process calibration loops. |
| [src/gradpulse/viz.py](src/gradpulse/viz.py) | Visualization Suite: dynamic Bloch trajectories, color-coded heatmaps, and pulse spectrograms. |
