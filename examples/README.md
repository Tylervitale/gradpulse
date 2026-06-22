# Examples

One runnable script per feature. Each is self-contained. Run it directly:

```bash
python examples/optimize_cz.py
```

New here? Start with **`optimize_cz.py`**, then branch out by what you need below.
Everything here runs on a laptop with no cloud account, except the two scripts
explicitly marked **needs AWS** in the Cepheus section.

**Prefer notebooks?** [`notebooks/01_intro_cz.ipynb`](notebooks/01_intro_cz.ipynb) walks through a CZ end-to-end with plots; [`notebooks/02_real_device.ipynb`](notebooks/02_real_device.ipynb) does it on the bundled real Rigetti Cepheus calibration; [`notebooks/03_hardware_validation.ipynb`](notebooks/03_hardware_validation.ipynb) reproduces the flagship live-QPU result: the coherence floor vs measured CZ error across all 160 Cepheus pairs (free; reads the committed sweep). (Needs `gradpulse[viz]` + Jupyter.)

## The core idea, measured

| Script | What it does |
|---|---|
| [`decoherence_in_the_loop.py`](decoherence_in_the_loop.py) | The package's central claim, demonstrated rather than asserted: on a decoherence-pressured device, optimizing with the noise **in the loop** beats `F ≈ F_coherent · exp(-t_g/T)` applied afterward. Reports the delivered-fidelity gap, split into a pulse-shaping and a duration-selection part. |
| [`leakage_in_the_loop.py`](leakage_in_the_loop.py) | The leakage companion: sweeps the duration of a **cross-resonance gate** and splits each optimized gate's error (via the package's own `error_budget`) into the decoherence floor the coherence formula captures and the coherent control+leakage part it cannot. Shows a clean crossover: slow gates are coherence-limited (the formula tracks the truth), fast gates become **leakage-limited** (coherent leakage dwarfs the decoherence floor, so the formula under-predicts the true error by tens of ×). That fast regime is where the open-system, leakage-aware loop earns its keep, vs the coherence-limited anchors, where a formula suffices. Every `F_proc` is cross-checked by the independent NumPy Liouvillian (+ QuTiP). |

## Optimize a gate

| Script | What it does |
|---|---|
| [`optimize_cz.py`](optimize_cz.py) | **Start here.** Minimal end-to-end CZ on a parametric-coupler pair. |
| [`optimize_iswap.py`](optimize_iswap.py) | iSWAP: the parametric coupler's native two-qubit gate. |
| [`optimize_cross_resonance.py`](optimize_cross_resonance.py) | Cross-resonance ZX(π/2), the fixed-frequency architecture (≈ CNOT). |
| [`optimize_multiqubit.py`](optimize_multiqubit.py) | A two-qubit gate inside a larger register, with crosstalk/collisions optimized *in the loop*. |
| [`coupler_in_loop_cz.py`](coupler_in_loop_cz.py) | CZ with the tunable coupler modelled **explicitly** in the loop (from a pair profile): reports the coupler-leakage budget the dispersively-eliminated pair model has identically zero of. |

## Use a real device

| Script | What it does |
|---|---|
| [`optimize_from_calibration.py`](optimize_from_calibration.py) | Optimize against **measured** device parameters instead of the representative defaults. Loads the real **Rigetti Cepheus-1-108Q** calibration in [`data/`](data/); point it at your own `device.properties.standardized` JSON to switch hardware. |

## Analyze & harden a pulse

| Script | What it does |
|---|---|
| [`optimize_robust.py`](optimize_robust.py) | Robust optimization *against* a miscalibration ensemble. |
| [`robustness_sweep.py`](robustness_sweep.py) | Map fidelity vs amplitude / drive-frequency miscalibration for a saved pulse. |
| [`spectator_crosstalk.py`](spectator_crosstalk.py) | Always-on-ZZ spectator crosstalk, plus the resonant-collision regime. |
| [`filter_function.py`](filter_function.py) | **Analytic** dephasing robustness (filter function): no Monte Carlo; validated against the MC sweeps. |
| [`randomized_benchmarking.py`](randomized_benchmarking.py) | Bridge the analytic process fidelity to a leakage-aware interleaved-RB estimator. |
| [`unitarity_rb.py`](unitarity_rb.py) | Unitarity (purity) RB: split the total gate error into a **coherent** part you could calibrate away and an **incoherent** (decoherence) part you cannot. |

## Optimization machinery

| Script | What it does |
|---|---|
| [`optimize_spectral.py`](optimize_spectral.py) | Band-limited (Fourier/CRAB) optimization: fewer parameters, band-limited *by construction*. |
| [`simultaneous_gates.py`](simultaneous_gates.py) | Optimize **two gates at once** on disjoint pairs under one shared crosstalk budget. |
| [`gradient_checkpointing.py`](gradient_checkpointing.py) | Cut autograd memory (`checkpoint_segments`) to reach larger open-system registers. |
| [`mps_large_register.py`](mps_large_register.py) | Score a pulse on a register too big for the dense simulator (N = 6, 8, ...) with the evaluation-only MPS evaluator (`ChainTEBD`): a restricted-ensemble fidelity **witness** for low-entanglement gates. |

## Benchmark against other tools

| Script | What it does |
|---|---|
| [`benchmark_vs_qutip_qtrl.py`](benchmark_vs_qutip_qtrl.py) | Head-to-head vs qutip-qtrl GRAPE on the identical (closed-system) problem: parity (needs the `[benchmark]` extra). |
| [`benchmark_vs_c3.py`](benchmark_vs_c3.py) | Head-to-head vs C3 (c3-toolset), the closest true peer: open-system, autodiff, hardware-backed. Documents what it took to run C3 1.4 on a current Python and the honest single-qubit comparison. |

## Independent validation (needs the `[validate]` extra → QuTiP)

| Script | What it does |
|---|---|
| [`validation_checks.py`](validation_checks.py) | Supplementary QuTiP cross-checks (the validation hierarchy). |
| [`validation_sweep.py`](validation_sweep.py) | Sweep the QuTiP cross-check across operating points. |

## Model vs. published device (no extra needed)

A pure T₁/T_φ model is a **lower bound** on a measured error-per-gate, so it equals a
device's measured fidelity only when that gate is coherence-limited.

| Script | What it does |
|---|---|
| [`validate_against_literature.py`](validate_against_literature.py) | Reproduce the measured error of every published gate in [`anchors/`](anchors/) (Sung 2021, Marxer 2023, Stehlik 2021) from its T₁/T₂, cross-checked against the analytic coherence-limit formula. **Adding a device is dropping a JSON in `anchors/`, no Python.** |
| [`stehlik_predict_vs_measured.py`](stehlik_predict_vs_measured.py) | Breadth version of the lower-bound test: all 11 tunable-coupler CZ pairs from Stehlik 2021 Table I: the floor sits at or below every measured number and saturates only where the gate is itself coherence-limited. |

## Export to hardware

| Script | What it does |
|---|---|
| [`braket_export.py`](braket_export.py) | Export an optimized pulse to Amazon Braket pulse-level hardware (needs the `[braket]` extra). |
| [`export_openpulse.py`](export_openpulse.py) | Export to vendor-neutral OpenPulse 3.0 / OpenQASM 3 with the DRAG quadrature baked into the I/Q (needs the `[openpulse]` extra). |
| [`hardware_in_the_loop.py`](hardware_in_the_loop.py) | Close the sim↔device gap by refining the model from measured RB. |

## Live-device validation: Rigetti Cepheus-1-108Q

The model was validated against a live 108-qubit QPU. These scripts reproduce that
study; all are **free** (they use only the device's published calibration) except
`run_irb_on_braket.py`, which submits circuits and **costs real QPU credits**. Full
write-up and numbers are in [`../RESULTS.md`](../RESULTS.md).

Scripts and data live in [`cepheus/`](cepheus/).

| Script | What it does |
|---|---|
| [`cepheus/cepheus_predict_vs_measured.py`](cepheus/cepheus_predict_vs_measured.py) | **Free.** The analytic coherence floor vs the measured CZ error across all coupled pairs; the cheap first look. |
| [`cepheus/cepheus_grape_sweep_all.py`](cepheus/cepheus_grape_sweep_all.py) | **Free.** The rigorous version: re-optimize the CZ with each pair's live T₁/T₂ *in the loop* at its real gate duration. Writes the scatter data the figures read. |
| [`cepheus/cepheus_lowerbound_scatter.py`](cepheus/cepheus_lowerbound_scatter.py) | **Free.** The full, unselected scatter (floor ≤ measured on ~150/160 pairs) → `paper/cepheus_scatter.png`. |
| [`cepheus/cepheus_sigma_validation.py`](cepheus/cepheus_sigma_validation.py) | **Free.** The statistically correct metric: is the prediction inside the *measurement's own* RB error bar? (median 0.60σ on the coherence-limited subset). |
| [`cepheus/run_irb_on_braket.py`](cepheus/run_irb_on_braket.py) | **Needs AWS · costs credits.** The one step the bridge stops at: `device.run()` on silicon. Level A benchmarks the device's native CZ (validates the model); Level B benchmarks a gradpulse-designed pulse (tests the optimizer on hardware). Has canary / cost-cap / online-check guardrails. |

The other `cepheus_*` / `levelb_*` files in [`cepheus/`](cepheus/) are the reproducible
intermediates and saved outputs (including paid-QPU results) behind those scripts and
RESULTS.md: coupler characterization and sweeps, peak/closed-loop calibration studies,
and the Level-B pulse builds. They are the campaign's lab notebook, not standalone feature
demos.

## Figures

| Script | What it does |
|---|---|
| [`make_paper_figure.py`](make_paper_figure.py) | The six-panel summary figure for the preprint (needs the `[viz]` extra → matplotlib). |
| [`make_validation_figure.py`](make_validation_figure.py) | The validation figure: dt-convergence of the Trotter splitting error, four-way solver agreement, and cross-architecture QuTiP agreement, all recomputed from the committed reference pulse (`[viz]`). |

---

Most scripts save their optimized pulse to `*_pulse.json` / `*_pulse.npy` at the repo
root (git-ignored). The analysis and validation scripts read those back, so run an
`optimize_*.py` first if a script expects a saved pulse.

*gradpulse is built and maintained by [Pure State Labs Inc.](https://purestatelabs.com), MIT-licensed.*
