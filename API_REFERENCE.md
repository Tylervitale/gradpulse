# gradpulse API reference

Generated from the in-source docstrings: a flat, searchable index of every public
class, method, and function: its signature, what it returns, and the one or two
facts you need to use it. For the *why* (the physics, how the validation works) read
the [README](README.md) and the module headers; for runnable end-to-end usage see
[`examples/`](examples/).

Scope reminder: every fidelity here is a **simulated** number, cross-checked
against an independent QuTiP integrator but **not** a measured hardware fidelity.
See [Limitations](README.md#limitations).

---

## Contents

- [Import surface](#import-surface)
- [Conventions](#conventions)
- [Customization cookbook](#customization-cookbook)
- [Device profiles](#device-profiles)
  - [`ParametricCouplerProfile`](#parametriccouplerprofile)
  - [`CrossResonanceProfile`](#crossresonanceprofile)
- [`ParametricCZOptimizer`](#parametricczoptimizer)
- [`CrossResonanceZXOptimizer`](#crossresonancezxoptimizer)
- [General N-qubit: `gradpulse.multiqubit`](#general-n-qubit-gradpulsemultiqubit)
- [Large-register evaluator: `gradpulse.mps`](#large-register-evaluator-gradpulsemps)
- [One-call convenience wrappers: `gradpulse.convenience`](#one-call-convenience-wrappers-gradpulseconvenience)
- [Band-limited basis: `gradpulse.basis`](#band-limited-basis-gradpulsebasis)
- [Independent validation: `gradpulse.validate`](#independent-validation-gradpulsevalidate)
- [Third solver: `gradpulse.liouville`](#third-solver-gradpulseliouville)
- [Randomized benchmarking: `gradpulse.rb`](#randomized-benchmarking-gradpulserb)
- [Literature anchors: `gradpulse.literature`](#literature-anchors-gradpulseliterature)
- [In-loop vs multiply-after: `gradpulse.headtohead`](#in-loop-vs-multiply-after-gradpulseheadtohead)
- [Hardware-in-the-loop: `gradpulse.hardware`](#hardware-in-the-loop-gradpulsehardware)
- [Amazon Braket on-ramp: `gradpulse.braket_bridge`](#amazon-braket-on-ramp-gradpulsebraket_bridge)
- [OpenPulse export: `gradpulse.openpulse_export`](#openpulse-export-gradpulseopenpulse_export)
- [qutip-qtrl benchmark: `gradpulse.benchmark`](#qutip-qtrl-benchmark-gradpulsebenchmark)
- [Channel diagnostics: `gradpulse.diagnostics`](#channel-diagnostics-gradpulsediagnostics)
- [Plotting: `gradpulse.viz`](#plotting-gradpulseviz-viz-extra)

---

## Import surface

```python
from gradpulse import (
    ParametricCouplerProfile, ParametricCZOptimizer,     # tunable-coupler CZ
    CrossResonanceProfile,    CrossResonanceZXOptimizer, # fixed-frequency ZX
    MultiQubitProfile,        MultiQubitOptimizer,       # general N-qubit register
    FourierBasis,                                        # band-limited (CRAB) synthesis basis
    liouville_f_proc,                                    # 3rd, QuTiP-free independent F_proc solver
    ChainTEBD,                                           # eval-only MPS evaluator for N>4 (witness)
    optimize_cz, optimize_iswap, tunable_coupler_cz,     # one-call convenience wrappers
    coupler_in_loop_cz,                                  # explicit-coupler CZ from a pair profile
)
import gradpulse.validate        as validate     # QuTiP cross-check (needs the [validate] extra)
import gradpulse.rb              as rb           # leakage-aware interleaved-RB estimator
import gradpulse.literature      as literature   # data-driven validation vs published gates
import gradpulse.hardware        as hardware     # hardware-in-the-loop scaffold
import gradpulse.braket_bridge   as braket_bridge   # Braket pulse export + cost (needs the [braket] extra)
import gradpulse.openpulse_export as openpulse      # vendor-neutral OpenPulse 3.0 export (needs the [openpulse] extra)
import gradpulse.benchmark       as benchmark   # head-to-head vs qutip-qtrl (needs the [benchmark] extra)
import gradpulse.diagnostics     as diagnostics  # PTM / channel unitarity
```

Everything in the top-level `from gradpulse import ...` line has no hard QuTiP
dependency; `gradpulse.validate` and the `QuTiPDeviceBackend` import QuTiP lazily.

**Command line.** `gradpulse` (or `python -m gradpulse`) prints a welcome banner and
quickstart; `gradpulse --version` prints the version and attribution. The Python API
above is the primary interface; `python -m gradpulse.validate --pulse <file>` runs the
independent QuTiP cross-check on a saved pulse.

---

## Conventions

| Convention | Value |
|---|---|
| **Units** | Frequencies in GHz, rates/detunings in MHz, time in ns. Internally converted to angular rad/ns (`2*pi*GHz`). |
| **Waveform layout** | `[n_slices, n_channels]` real array, one row per `dt_ns` slice. Channel order is per-optimizer (see each constructor). |
| **`best_raw_param` vs `best_waveform`** | `best_raw_param` is the optimizer's *parameter* (pre-activation/pre-smoothing); feed it back to `simulate_*`, `dt_convergence`, `counter_rotating_fidelity` to reproduce a result bit-for-bit. `best_waveform` is the smoothed physical envelope you'd hand to an AWG. |
| **Computational subspace** | The 4-D `{|00>, |01>, |10>, |11>}`, at Hilbert indices `[0, 1, n_levels, n_levels+1]`. Fidelities/leakage are reported on this subspace (d=4). |
| **`F_proc` vs `F_avg`** | Process fidelity and average gate fidelity, related by `F_avg = (d*F_proc + 1)/(d + 1)` with d=4. |
| **Precision** | `precision='single'` (complex64, default, fast) or `'double'` (complex128, ~10× slower, needed for fine-`dt` studies). |
| **Device** | Auto-selected: CUDA if available, else CPU (`gradpulse.parametric.DEVICE`). Pass tensors on `DEVICE`. |

---

## Customization cookbook

Common customizations and where each lives. Each is a constructor kwarg or a single
method call unless noted.

| To change... | Do this | Notes |
|---|---|---|
| **Device parameters** (T1/T2, freqs, anharm, coupling, drive ceiling) | `ParametricCouplerProfile(t1_ns_q1=..., freq_ghz_q1=..., g_max_mhz=..., ...)` | Plain dataclass kwargs; every parameter has an inline doc. |
| **Load a *measured*, fully device-specific profile** | `ParametricCouplerProfile.from_ibm_backend(backend, (q1, q2))` · `...from_calibration(norm_dict, (q1, q2))` · `...from_braket_calibration(path, (q1, q2))` | `from_ibm_backend` reads freq+anharm+T1/T2 from a Qiskit backend in one call; `from_calibration` takes a vendor-neutral normalized dict (any export maps onto it); Braket-standardized carries only T1/T2+CZ. Same loaders on `MultiQubitProfile` for N qubits. |
| **Target a different gate** | `ParametricCZOptimizer(prof, target_gate=...)`: `"cz"`, `"iswap"`, `"sqrt_iswap"`, **or any 4×4 `np.ndarray`** | Custom unitary is validated (unitarity-checked). iSWAP needs a resonant pair. |
| **Truncation / leakage convergence** | `CrossResonanceProfile(n_levels=...)` / `ParametricCouplerProfile(n_levels=...)` | Rebuilds every operator **and** the QuTiP cross-check. Defaults are the *converged* values: **4 for the strong-drive CR gate, 3 for the near-quiet CZ** (see [`CrossResonanceProfile`](#crossresonanceprofile)). Raise to re-confirm convergence in a new regime (shorter gate, stronger drive). Must be ≥ 3. |
| **Beyond-RWA (counter-rotating) error** | *measure:* `CrossResonanceZXOptimizer.counter_rotating_fidelity(raw_param, vz=...)` · *remove:* `...refine_beyond_rwa(raw_param, vz=...)` | CR only (its frame is unambiguous). `refine_beyond_rwa` polishes the pulse against the full time-dependent Hamiltonian so the optimizer is no longer confined to the RWA. |
| **Integration order / step** | `ParametricCZOptimizer(prof, step_order=2)` + `.dt_convergence(pulse)` | `step_order=2` is Strang O(dt²). |
| **A measured AWG / line response** | `ParametricCZOptimizer(prof, line_response=<array or {"type":"exponential","tau_ns":...}>)` | Pre-compensates the distortion the qubit sees. |
| **Add a penalty term to the loss** | edit the loss in `optimize_multi_seed` (Adam) / `_lbfgs_refine` (polish) | Copy the `leakage_penalty` / `bandwidth_penalty` pattern. A few lines. |
| **Control channels** | `n_channels ∈ {3,4,6}` + `coupler_phase_mode`, `smoother_type`, `use_drag` | *Configurable among presets*, not arbitrarily extensible; a brand-new channel type touches the operator builder, the simulate loop, and the QuTiP validator together. |
| **Robust (miscalibration-averaged) optimization** | `optimize_multi_seed(robust_amp_jitter=..., robust_freq_jitter_mhz=..., robust_g_jitter=..., robust_t12_jitter=...)` | Builds tolerance into the loss. |
| **Optimize against slow 1/f dephasing** (not just score it) | `optimize_multi_seed(robust_dephasing_sigma_mhz=σ)` | Puts quasi-static dephasing *inside* the gradient on the same Gauss-Hermite grid `quasi_static_fidelity` scores with. Standalone; `robust_dephasing_nodes**2` evals/step. |
| **Optimize against the whole 1/f band** (not just `F(0)`) | `optimize_multi_seed(robust_filter_sigma_mhz=σ)` | Adds the leakage-inclusive filter-function infidelity over `robust_filter_band_mhz` to the objective; hardens the mid-band the quasi-static limit misses, using the same estimator `filter_function_fidelity` scores with. |
| **Coherent-only objective** (the multiply-after baseline) | `optimize_multi_seed(diss_scale=0.0)` | Optimizes the unitary+leakage fidelity (decoherence off in the loop); the recipe `gradpulse.headtohead` runs against the in-loop objective. |
| **Finite temperature / static ZZ** | `ParametricCouplerProfile(n_thermal_q1=..., chi_zz_mhz=...)` | Both mirrored in the QuTiP cross-check. |
| **Noise spectrum (full band)** | `quasi_static_fidelity` (slow) · `colored_noise_fidelity(..., f_high_mhz=...)` (intermediate 1/f^α) · Lindblad T1/T_φ (white) | Three diagnostics span the spectrum; colored reduces to quasi-static at `f_high·T≪1`. |
| **Cross-qubit correlated noise** | `colored_noise_fidelity(..., correlation=ρ)` | `ρ∈[-1,1]`: 0 independent, +1 common-mode, −1 differential: spatially-correlated noise that independent per-qubit draws can't represent. |
| **Crosstalk (1 or N neighbours)** | `spectator_fidelity(zeta_mhz=...)` · `multi_spectator_fidelity(neighbours)` | Always-on ZZ as effective frozen-neighbour detuning; detunings sum. |
| **Frequency collision (resonant spectator)** | `resonant_collision_fidelity(detuning_mhz=..., j_mhz=...)` | Evolving exchange-coupled spectator; sweep `detuning_mhz` as a list for the curve. Reports gate collapse + population swapped into the spectator. |
| **Lossy TLS defect** (coherent quantum bath) | `tls_defect_fidelity(detuning_mhz=..., g_mhz=..., t1_tls_ns=...)` | An explicit exchange-coupled two-level defect *with its own T1* in the full open system, coherent-bath physics a classical-noise channel cannot represent. |
| **Band-limited (CRAB / Fourier) optimization** | `opt.optimize_spectral(f_max_mhz=...)` | Optimizes Fourier coefficients (`FourierBasis`) instead of per-slice values → band-limited *by construction*, ~6× fewer params, no smoother/anti-cheating penalty. Returns `out_of_band_fraction` (measured). |
| **Complete I/Q (DRAG baked into the waveform)** | `opt.iq_waveform(raw_param)` (both pair optimizers) | Full complex drive (in-phase + DRAG quadrature) in physical rad/ns. Feed a channel straight to `braket_bridge.to_braket_waveform` or `openpulse_export.to_openpulse_program`. |
| **Vendor-neutral pulse export** | `openpulse_export.to_openpulse_program(iq, dt_ns=...)` · `verify_openpulse_roundtrip(...)` | OpenQASM 3 / OpenPulse 3.0 `defcal` text, complex I/Q preserved, re-parse-verified offline. (`qiskit.pulse` was removed in Qiskit 2.0; this is the live standard.) |
| **Analytic dephasing robustness (no Monte-Carlo)** | `opt.filter_function(pulse)` · `opt.filter_function_fidelity(pulse, sigma_mhz=..., alpha=...)` | Per-frequency sensitivity F(f) (leakage-inclusive) + its integral against a 1/f^α PSD. Overlay a measured S(f). Validated ~1% vs the MC methods. |
| **Lower autograd memory (larger registers)** | `checkpoint_segments=S` (or `"auto"`) on `simulate_*`, `optimize_multi_seed`, `optimize` (multiqubit) | Recomputes the slice loop in backward → memory ~O(Nt/S) at ~2× forward compute. Same optimum (value+grad match plain to ~1e-6). `optimize_multi_seed(checkpoint_segments="auto")` picks `round(√Nt)` for long pulses (≥64 slices), `0` otherwise. |
| **Optimize several gates at once** | `MultiQubitOptimizer(target_gate=['cz','cz'], target_qubits=[(0,1),(2,3)])` | Simultaneous gates on disjoint groups under one shared crosstalk budget; combined target QuTiP-cross-checked. |
| **Score a register past the dense ~4-qubit wall** | `MultiQubitOptimizer.process_fidelity_sparse(...)` (closed) · `gradpulse.ChainTEBD.witness_open(...)` (open witness) | Sparse-Krylov `2**N`-state F_proc, or a trajectory-MPS witness in the weakly-entangling regime. The MPS number is a witness, **not** `F_proc`; see [`gradpulse.mps`](#large-register-evaluator-gradpulsemps). |
| **Benchmark vs an independent GRAPE** | `benchmark.run_benchmark(gate="cnot")` | gradpulse engine vs qutip-qtrl on the identical control problem (same H/target/grid + L-BFGS); reports fidelity, wall-clock, iters. |

---

## Device profiles

### `ParametricCouplerProfile`

`@dataclass`: tunable-transmon, parametric-coupler device parameters. Defaults are
representative published-typical values, **not** a measurement of any device. Constructing
a profile whose every device field is left at the default raises
`RepresentativeDefaultsWarning` (a top-level export), a "this isn't your device" guard
that stays silent the moment you set or load real parameters.

| Field | Default | Meaning |
|---|---|---|
| `qubit_pair` | `(4, 5)` | Labels for reporting / calibration lookup. |
| `n_levels` | `3` | Fock levels kept per transmon (Hilbert dim `n_levels**2`). 3 resolves the dominant \|2⟩ leakage + the \|11⟩-\|02⟩ CZ mechanism and is *converged* for the near-quiet CZ (a 4-level re-score moves F_proc by ~2e-4). Raise to re-check in a new regime. Must be ≥ 3. (The strong-drive CR gate defaults to 4; see [`CrossResonanceProfile`](#crossresonanceprofile).) |
| `t1_ns_q1`, `t1_ns_q2` | `30_000` | T1 energy relaxation (ns). |
| `t2_ns_q1`, `t2_ns_q2` | `25_000` | T2 dephasing (ns). |
| `n_thermal_q1`, `n_thermal_q2` | `0.0` | Bath photon occupation n_th (finite-T excitation jumps). Typical 0.01-0.05. |
| `freq_ghz_q1`, `freq_ghz_q2` | `4.85`, `5.05` | Qubit frequencies (GHz) at the flux operating point. |
| `anharm_ghz_q1`, `anharm_ghz_q2` | `-0.200` | Anharmonicity (GHz, negative). |
| `g_max_mhz` | `12.0` | Effective parametric coupling rate after Schrieffer-Wolff (MHz). |
| `omega_max_mhz` | `50.0` | Drive-amplitude saturation Rabi rate (MHz). |
| `chi_zz_mhz` | `0.0` | Static parasitic ZZ shift (MHz); 0 disables. |
| `native_cz_fidelity` | `0.988` | Reference CZ fidelity for comparison reporting only (not used in optimization). |
| `native_cz_duration_ns` | `150.0` | Reference CZ duration. |
| `notes` | `[]` | Free-form provenance strings. |

**`classmethod from_braket_calibration(path, qubit_pair, *, freq_ghz_q1=None, freq_ghz_q2=None, **overrides)`**
Load measured T1/T2 and native-CZ fidelity for a qubit pair from a Braket
`standardized_gate_model_qpu_device_properties` JSON. That schema carries **no**
qubit frequency or anharmonicity; pass `freq_ghz_q1/q2` (and anharmonicities via
`**overrides`) when you have them. Returns a populated `ParametricCouplerProfile`.

**`classmethod from_calibration(cal, qubit_pair, **overrides) -> (profile, notes)`**
Vendor-neutral one-call loader. `cal` is a *normalized* dict keyed by qubit index,
each carrying any of `freq_ghz`/`freq_hz`, `anharmonicity_mhz`/`_hz`, `t1_ns`/`t1_us`,
`t2_ns`/`t2_us` (SI or convenience units both accepted); an optional `two_qubit` block
supplies CZ fidelity. Any real device export maps onto this shape. Missing fields fall
back to the profile defaults and are listed in the returned `notes` (so nothing is
silently invented). Same classmethod on `MultiQubitProfile` for N qubits. The
module-level helper `normalize_qubit_node(node) -> {freq_ghz, anharm_ghz, t1_ns, t2_ns}`
(in `gradpulse.profiles`) maps one raw per-qubit entry (Hz/s **or** GHz/ns) onto those
canonical keys, so you can assemble the `cal` dict from any export.

**`classmethod from_ibm_backend(backend, qubit_pair, **overrides) -> (profile, notes)`**
Reads qubit frequency, anharmonicity, T1 and T2 directly from a Qiskit `BackendV1`/
`BackendV2` (or `Target`) in one call, no fields to supply by hand. Internally
normalizes the backend to the `from_calibration` dict, so the `notes` semantics match.
Same classmethod on `MultiQubitProfile`.

### `CrossResonanceProfile`

`@dataclass`: fixed-frequency transmon pair for the cross-resonance ZX gate. The
**control** qubit is driven at the **target's** frequency.

| Field | Default | Meaning |
|---|---|---|
| `qubit_pair` | `(0, 1)` | `(control, target)`. |
| `n_levels` | `4` | Fock levels per transmon. **Default 4, the *converged* truncation for this strong-drive gate** (3 levels overstate F_proc by hiding \|2⟩→\|3⟩ leakage; 4→5 is flat to ~1e-4). See the note below. Must be ≥ 3. |
| `freq_ghz_control`, `freq_ghz_target` | `5.00`, `4.85` | Frequencies (GHz). Only their difference (the CR detuning) matters. |
| `anharm_ghz_control`, `anharm_ghz_target` | `-0.33` | Anharmonicity (GHz). Fixed-frequency transmons run near −0.33. |
| `j_coupling_mhz` | `3.0` | Always-on transverse exchange J (MHz); dispersive regime \|Δ_c\| ≫ J. |
| `omega_max_mhz` | `60.0` | Drive saturation Rabi rate (MHz). |
| `chi_zz_mhz` | `0.0` | Optional static ZZ (MHz); 0 disables. |
| `t1_ns_control`, `t1_ns_target` | `150_000` | T1 (ns); fixed-frequency transmons are typically longer-lived. |
| `t2_ns_control`, `t2_ns_target` | `120_000` | T2 (ns). |
| `native_cnot_fidelity` | `0.990` | Reference CNOT fidelity (reporting only). |
| `native_cnot_duration_ns` | `300.0` | Reference CNOT duration. |
| `notes` | `[]` | Provenance strings. |

> **Truncation note (CR): why the default is 4.** Measured on this package: a CR
> pulse optimized in the 3-level model drops ~3% F_proc (0.99→0.95) when honestly
> re-scored at 4, because the 3-level optimizer is blind to \|2⟩→\|3⟩ leakage on the
> strongly-driven control. At `n_levels=4` the gate **is** converged; re-scoring a
> 4-level-optimal pulse at 5 moves F_proc by ~1e-4 (below the decoherence floor) and
> the achievable fidelity is flat across 4/5/6, so 4 is the default, and the shipped
> headline (F_proc≈0.9974, QuTiP-cross-checked at 4 levels to ~3e-5) is a converged
> number. Drop to 3 only for quick relative studies, never for a leakage or
> absolute-fidelity claim. The coupler-activated CZ is near-quiet and stays converged
> at 3 levels, so `ParametricCouplerProfile` keeps default 3.

---

## `ParametricCZOptimizer`

Autodiff GRAPE optimizer for a parametric-coupler CZ (and the iSWAP family via
`target_gate`). Output pulse format: `[n_slices, n_channels]` real envelope.

### Constructor

```python
ParametricCZOptimizer(
    profile=None,                 # ParametricCouplerProfile (defaults if None)
    bandwidth_mhz=80.0,           # control bandwidth limit (0 disables smoothing)
    use_drag=False, drag_order=2, # derived-quadrature DRAG (0=off, 1, 2)
    n_channels=3,                 # 3 | 4 | 6  (see below)
    smoother_type="gaussian",     # "gaussian" (6 dB/oct) | "firbrick" (sharp FIR)
    activation="clamp",           # "clamp" | "sigmoid" (smooth, bandwidth-faithful)
    step_order=1,                 # 1 = Trotter (matches validator) | 2 = Strang O(dt^2)
    coupler_phase_mode="phase",   # "phase" | "frequency" (needs n_channels>=4)
    delta_max_mhz=30.0,           # drive-detuning ceiling for coupler_phase_mode="frequency"
    coupler_g_linewidth_mhz=None, # optional Lorentzian coupling rolloff vs detuning
    line_response=None,           # AWG/line impulse response (array or {"type":"exponential","tau_ns":t})
    target_gate="cz",             # "cz" | "iswap" | "sqrt_iswap" | a 4x4 np.ndarray
    precision="single",           # "single" (complex64) | "double" (complex128)
)
```

- **`n_channels`.** `3`: `q1_drive, q2_drive, coupler_envelope`. `4`: adds a
  parametric coupler control channel (interpreted by `coupler_phase_mode`). `6`:
  adds per-qubit Stark/Z drives so the optimizer pre-compensates AC-Stark shifts.
- **`coupler_phase_mode`.** `"phase"`: channel 4 → absolute drive phase θ = π·u4.
  `"frequency"`: channel 4 → instantaneous drive **detuning** δ whose running
  integral is θ(t), making the drive frequency a real (offset-capable) control.
- **`target_gate`.** A string selects a named gate; **a 4×4 `np.ndarray` targets
  an arbitrary two-qubit unitary** (validated unitary, so a typo can't define a
  non-physical target). `opt.target_gate` reads `"custom"` in that case.

### `optimize_multi_seed(...) -> dict`

```python
optimize_multi_seed(
    label="parametric_grape", n_seeds=4, iterations=200, n_slices=150,
    warm_start_mode="parametric_cz", lr=0.01,
    lbfgs_polish=True, lbfgs_iters=50, dt_ns=1.0,
    use_process_fidelity=True,
    lr_schedule="cosine", lr_min_factor=0.05, warmup_frac=0.05,
    leakage_penalty=0.0, bandwidth_penalty=0.0, bandwidth_filter_mhz=250.0,
    robust_g_jitter=0.0, robust_t12_jitter=0.0,
    robust_amp_jitter=0.0, robust_freq_jitter_mhz=0.0,
    robust_dephasing_sigma_mhz=0.0, robust_dephasing_nodes=3,
    robust_filter_sigma_mhz=0.0, robust_filter_alpha=1.0,
    robust_filter_band_mhz=(1e-3, 5.0), robust_filter_n_freq=96,
    diss_scale=1.0,
    warm_start_pulse=None, grad_clip=1e3, checkpoint_segments=0, rng=None,
)
```
Multi-seed Adam GRAPE with optional L-BFGS polish and a miscalibration-averaged
("robust") loss. `checkpoint_segments=S>1` gradient-checkpoints the slice loop to cut
autograd memory (threaded through Adam **and** the L-BFGS polish); `grad_clip` bounds
the per-step gradient norm (divergence guard). **Returns** `{"best_fidelity",
"best_waveform", "best_raw_param", "all_fidelities", "best_seed_idx",
"lbfgs_polished", "history", "converged", "final_grad_norm", "recent_gain",
"n_nonfinite_steps"}`.

- **`diss_scale`.** Scalar on the in-loop Lindblad dissipator. `1.0` (default)
  optimizes the true open-system `F_proc`; **`0.0` optimizes the coherent
  (unitary+leakage) fidelity**, the "optimize-coherent, multiply-by-e^{−t/T}
  afterward" recipe, exposed so `gradpulse.headtohead` (below) can run it honestly
  against the in-loop objective.
- **`robust_dephasing_sigma_mhz`.** If `>0`, optimize the `F_proc` of the channel
  **averaged over quasi-static (Gaussian RMS σ) per-qubit dephasing**, on the same
  Gauss-Hermite grid `quasi_static_fidelity` scores with; i.e. put slow 1/f
  dephasing *inside* the gradient instead of only scoring it. Standalone (not
  combinable with the `robust_*_jitter` axes); `best_fidelity` is then the
  dephasing-averaged value. Costs `robust_dephasing_nodes**2` evolutions/step.
- **`robust_filter_sigma_mhz`.** If `>0`, add the first-order 1/fᵅ dephasing
  infidelity from the **leakage-inclusive filter function** to the objective:
  `loss = (1 − F_nominal) + σ_rad²·⟨F⟩_band`. Where `robust_dephasing_sigma_mhz`
  hardens only `F(0)` (the slow limit), this hardens the **whole band**
  `robust_filter_band_mhz` (default `(1e-3, 5.0)` MHz) with PSD exponent
  `robust_filter_alpha`, using the *same estimator* `filter_function_fidelity`
  scores with (asserted equal to machine precision). Standalone (not combinable with
  `robust_dephasing` or the jitter axes, which double-count the slow band). Costs one
  nominal sim plus `n_seeds` toggling-frame builds/step.

### `optimize_spectral(...) -> dict`

```python
optimize_spectral(
    n_harmonics=None, f_max_mhz=None,        # band-limit (default = self.bandwidth_mhz)
    n_slices=150, dt_ns=1.0, n_seeds=4, iterations=300, lr=0.05,
    warm_start_mode="parametric_cz", use_process_fidelity=True,
    lbfgs_polish=True, lbfgs_iters=50, leakage_penalty=0.0,
    amp_penalty=20.0,                        # steers away from the [0,1] clamp (keeps it band-limited)
    coeff_jitter=0.03, grad_clip=1e3, seed=42, verbose=True,
)
```
GRAPE in a **band-limited Fourier (CRAB-style) basis** instead of per-slice: each
control is a sum of sinusoids at harmonics of 1/T up to `f_max_mhz`, so the pulse is
band-limited *by construction*: no post-hoc smoother, no anti-cheating FFT penalty,
and ~`2·f_max·T` coefficients/channel vs `n_slices`. Seeds start as the least-squares
projection of the usual warm start onto the basis. **Returns** `best_fidelity`,
`best_waveform` `[n_slices, n_channels]` in `[0,1]`, `all_fidelities`, `history`,
`converged`, `lbfgs_polished` **plus** `best_coeffs` (the optimized parameters; there is
**no** `best_raw_param` in spectral mode), `basis` (the `FourierBasis`), `n_params` (vs
`n_params_piecewise`), `max_overshoot`, and **`out_of_band_fraction`** (the *measured*
residual energy above `f_max`; only the clamp can introduce any). The envelope
cross-checks against QuTiP with no smoother (`validate.qutip_f_proc` consumes it
directly).

### Simulation / scoring (forward passes)

| Method | Purpose |
|---|---|
| `simulate_gradient_batch(u_stack, dt=1.0, checkpoint_segments=0, ...)` | Differentiable batched evolution → final density matrices (the gradient hot path). `checkpoint_segments=S>1` gradient-checkpoints the slice loop (autograd memory ~O(Nt/S) at ~2× forward compute; identical result). |
| `simulate_choi_batch(u_stack, dt=1.0, diss_scale=1.0, detuning_offset=0.0, detuning_traj=None, checkpoint_segments=0)` | Differentiable Choi-state evolution (for process fidelity). `diss_scale=0` turns decoherence off; `detuning_offset` injects a static frequency offset (scalar or `(δ1, δ2)`); `detuning_traj` injects a per-slice frequency trajectory (the colored-noise path); `checkpoint_segments` as above. |
| `smoothed_waveform(x_raw, dt=1.0)` | The physical `[n_slices, n_channels]` envelope for a raw parameter (real, in-phase only). |
| `iq_waveform(x_raw, dt=1.0) -> dict` | The **complete** per-channel complex drive the simulator applied (in-phase **plus** the derived DRAG quadrature) in physical rad/ns, so an exported pulse is self-contained. Returns `{"iq" [.,n_ch] complex, "labels", "dt_ns", "units", "peak", "n_channels"}`. With `use_drag=False` the imaginary part is zero and it reduces to `smoothed_waveform` rescaled. Feed `iq["iq"][:,c]` to `braket_bridge` / `openpulse_export`. |
| `out_of_band_fraction(envelope, dt_ns=1.0, f_max_mhz=None) -> float` | The measured fraction of per-channel AC spectral energy above `f_max_mhz` (default `self.bandwidth_mhz`), the standalone check that any envelope really is band-limited (`0` = perfectly). The same quantity `optimize_spectral` returns for its own result. |

### Analysis (pure forward passes on an optimized pulse)

**`error_budget(u_stack, dt=1.0) -> dict`**: splits infidelity into a
control/leakage part (closed-system, `diss_scale=0`) and the decoherence floor,
plus the channel **unitarity** (Wallman et al. 2015) as an independent
coherent-vs-incoherent split. Returns `{"F_proc", "F_avg", "F_proc_closed",
"r_total", "r_control_leakage", "r_decoherence", "leakage", "unitarity",
"r_incoherent_unitarity", "coherent_excess"}`.

**`robustness_sweep(u_stack, dt=1.0, amp_fracs=None, freq_mhz=None) -> dict`**:
fidelity vs amplitude (AWG-gain) and static drive-detuning miscalibration. Returns
`{"amplitude": {...}, "frequency": {...}}`, each `{"x", "unit", "F_proc",
"F_avg"}`. The amplitude axis is the cleanly quotable tolerance; the frequency axis
is a conservative pre-virtual-Z worst case (keep `freq_mhz` within ~1/(2T)).

**`quasi_static_fidelity(u_stack, dt=1.0, sigma_mhz=0.3, n_nodes=5, include_decoherence=True) -> dict`**:
averages the gate over slow, 1/f-like frequency noise (the non-Markovian
dephasing the Lindblad T_φ misses) via deterministic Gauss-Hermite quadrature
(`n_nodes**2` channel evals, no RNG). The **slow** end of the noise spectrum.
Returns `{"F_proc", "F_avg", "F_proc_nominal", "sigma_mhz", "n_evals"}`.

**`colored_noise_fidelity(u_stack, dt=1.0, sigma_mhz=0.3, alpha=1.0, f_low_mhz=1e-3, f_high_mhz=5.0, n_traj=128, n_tones=40, seed=0, include_decoherence=True, correlation=0.0) -> dict`**:
the **intermediate** band between `quasi_static_fidelity` (slow) and the white
Lindblad T_φ (fast). Direct 1/f^α colored-noise Monte-Carlo: synthesizes per-qubit
frequency trajectories with PSD ∝ 1/f^α over `[f_low, f_high]` (RMS `sigma_mhz`),
evolves the channel under each (time-dependent detuning), and averages. Trajectories
run in parallel over the batch. Reduces to `quasi_static_fidelity` when `f_high·T≪1`
and motionally narrows as the band widens. `correlation=ρ∈[-1,1]` sets the cross-qubit
spatial correlation of the two trajectories: `0` independent draws (default), `+1`
common-mode (shared field), `−1` differential, capturing correlated noise that
independent per-qubit draws cannot. Returns `{"F_proc", "F_avg", "F_proc_nominal",
"sigma_mhz", "alpha", "n_traj", "f_low_mhz", "f_high_mhz", "correlation"}`.

**`filter_function(u_stack, dt=1.0, f_max_mhz=None, n_freq=300) -> dict`**:
the analytic, **no-Monte-Carlo** dephasing **filter function** F(f): the
leakage-inclusive first-order sensitivity to per-qubit frequency noise, built from the
toggling-frame number operators. `1 − F ≈ (1/2π)∫S(ω)F(ω)dω` for any noise PSD S. The
fast complement to the `quasi_static`/`colored_noise` Monte-Carlo. Returns
`{"freq_mhz", "omega", "F" (total), "F_per_qubit", "F0"}`; `F0 = F(0)` is the
quasi-static value, so `sigma²·F0` is the static-dephasing infidelity (matches
`quasi_static_fidelity`). Overlay your device's S(f) on `F`.

**`filter_function_fidelity(u_stack, dt=1.0, sigma_mhz=0.3, alpha=1.0, f_low_mhz=1e-3, f_high_mhz=5.0, n_freq=400, quasi_static=False) -> dict`**:
integrates `filter_function` against a 1/f^α PSD of RMS `sigma_mhz` over
`[f_low, f_high]` → an analytic process-fidelity estimate (first order in the noise;
agrees with `colored_noise_fidelity` for small σ). `quasi_static=True` collapses to the
exact slow limit `sigma_rad²·F(0)`. Returns `{"F_proc", "F_avg", "infidelity",
"sigma_mhz", "alpha", "f_low_mhz", "f_high_mhz", "F0"}`. (Parametric-CZ optimizer only.)

**`spectator_fidelity(u_stack, dt=1.0, zeta_mhz=0.1, spectator_pop=0.5) -> dict`**:
fidelity penalty from an always-on ZZ to an idle neighbour, modelled as an
effective frozen-neighbour detuning (validated to machine precision against a 27-D
3-transmon QuTiP sim). `zeta_mhz` scalar or `(ζ1, ζ2)`. Returns `{"f_proc_idle",
"f_proc_excited", "f_proc_spectator_avg", "f_avg_idle", "f_avg_spectator_avg",
"delta_r_spectator", "zz_phase_rad", ...}`. `delta_r_spectator` is the conservative
marginal infidelity an *unmeasured* neighbour adds at the nominal frame.

**`multi_spectator_fidelity(u_stack, neighbours, dt=1.0) -> dict`**:
the N-neighbour generalization of `spectator_fidelity`. `neighbours` is a list of
`(gate_qubit, zeta_mhz, pop)` with `gate_qubit ∈ {0, 1}`; detunings on a qubit sum,
and the channel is averaged over all neighbour-state combinations. Validated against
an explicit multi-transmon QuTiP sim (`validate.spectator_cross_check_multi`). Returns
`{"f_proc_idle", "f_proc_spectator_avg", "f_avg_idle", "f_avg_spectator_avg",
"delta_r_spectator", "n_neighbours", "n_configs"}`. (CR optimizer: same signature plus
`vz=`.)

**`resonant_collision_fidelity(u_stack, dt=1.0, detuning_mhz=0.0, j_mhz=8.0, couples_to=2, diss_scale=1.0) -> dict`**:
the **complement** of `spectator_fidelity`: a near-resonant spectator that
dynamically swaps population (a frequency collision), which no frozen detuning can
capture. Propagates an explicit *evolving* third transmon coupled by transverse
exchange `J(a_g†a_s + a_g a_s†)` in the full `n_levels³`-D open system. `detuning_mhz`
is the spectator's detuning **from the coupled gate qubit** (0 = exact collision); pass
a scalar for one point or a **list/array to sweep the whole collision curve in one
batched call**. `couples_to ∈ {1, 2}`. Returns `{"detuning_mhz", "f_proc", "f_avg",
"f_proc_isolated", "f_avg_isolated", "delta_r_collision", "spectator_leakage", ...}`
(scalars for a scalar detuning, lists for a sweep); `spectator_leakage` is the
population swapped into the spectator. J→0 reproduces the bare gate to machine
precision; cross-checked by `validate.collision_cross_check`. **CR optimizer:**
`resonant_collision_fidelity(x_raw, dt=1.0, vz=None, detuning_mhz=0.0, j_mhz=8.0,
couples_to="control", diss_scale=1.0)`: `couples_to ∈ {"control", "target"}`,
cross-checked by `validate.cr_collision_cross_check` (this is the dominant yield-limiter
for fixed-frequency CR lattices).

**`tls_defect_fidelity(u_stack, dt=1.0, detuning_mhz=0.0, g_mhz=2.0, t1_tls_ns=500.0, couples_to=1, diss_scale=1.0) -> dict`**:
gate fidelity vs. an explicit, **lossy two-level-system (TLS) defect**: the physics a
classical noise model (quasi-static / colored / white Lindblad) cannot capture: a coherent
quantum bath mode that vacuum-Rabi-swaps a real excitation with the qubit **and** carries
its own short T1, so near resonance it both swaps population out of the computational
subspace and drains it irreversibly. Evolves the gate pair plus one explicit defect in the
full `(n_levels²·2)`-D open system (18-D at `n_levels=3`): transverse exchange `g_mhz`, the
defect's frequency (via `detuning_mhz`), and its own T1 Lindblad jump. `g_mhz=0` reproduces
the bare gate to machine precision (guards the lift); cross-checked by
`validate.tls_defect_cross_check`. The lossy cousin of `resonant_collision_fidelity`;
`couples_to ∈ {1, 2}`.

**`dt_convergence(u_stack, dt=1.0, refinements=(1,2,4), metric="process") -> dict`**:
shrinks the integration step (holding the physical pulse fixed) and reports
fidelity at **both** step orders plus a Richardson dt→0 extrapolation. Pass
`best_raw_param`. Use `precision="double"` to see the clean O(dt)/O(dt²) trend
below the float32 noise floor.

---

## `CrossResonanceZXOptimizer`

Autodiff GRAPE for a cross-resonance ZX(π/2) gate (fixed-frequency pair). Controls:
`ch0` = control in-phase CR drive (the entangling drive), `ch1` = optional target
in-phase IX-cancellation tone. ZX(π/2) is matched up to a single-qubit-Z frame (two
angles optimized jointly and **reported**).

### Constructor

```python
CrossResonanceZXOptimizer(
    profile=None,             # CrossResonanceProfile (defaults if None)
    bandwidth_mhz=60.0,       # control bandwidth limit
    use_drag=True,            # derived-quadrature DRAG on the driven qubit(s)
    use_target_cancel=True,   # enable the ch1 active IX-cancellation tone (n_channels 2 vs 1)
    echo=False,               # echoed-CR sequence: control-qubit π at midpoint + 2nd-half drive flip
    precision="single",       # "single" | "double"
)
```

`echo=True` runs the **echoed** cross-resonance sequence: a control-qubit π pulse at
the gate midpoint with the CR drive sign-flipped on the second half. This refocuses
every term that anticommutes with X_control (the static ZZ and the IX/ZI error terms) into a removable single-qubit frame (the static ZZ phase a post-gate virtual-Z
cannot otherwise absorb), exactly as fixed-frequency CNOTs are echoed on hardware. The
QuTiP cross-check applies the identical echo through one shared evolution core, so the
echoed pulse stays independently validated (`validate.cr_cross_check`).

### `optimize(n_slices=300, dt_ns=1.0, iterations=400, n_seeds=3, lr=0.05, leak_weight=2.0, seed0=0, grad_clip=1e3, verbose=False, diss_scale=1.0) -> dict`

Multi-seed Adam GRAPE toward ZX(π/2); the loss is `(1 − F_proc) + leak_weight·leak`.
All `n_seeds` restarts optimize **together in one batched forward/backward**
(parameters `[n_seeds, n_slices, n_channels]`, per-seed virtual-Z frame
`[n_seeds, 2]`), the same way the parametric `optimize_multi_seed` does, so a run
is ~`n_seeds`× faster than a per-seed loop, with results equivalent up to optimizer
noise. **Returns** `{"best_fidelity", "best_fidelity_avg", "best_leakage",
"best_waveform", "best_raw_param", "virtual_z", "all_fidelities", "history",
"converged", "final_grad_norm", "n_nonfinite_steps", "n_slices", "dt_ns",
"echo"}`. `virtual_z` is `[φ_control, φ_target]`. `diss_scale=0.0` optimizes the **coherent**
(unitary+leakage) objective instead of the open-system one: the CR analogue of the
parametric `diss_scale`, for honest in-loop-vs-coherent comparisons.

### Analysis

**`error_budget(x_raw, dt=1.0, vz=None) -> dict`**: mirrors the parametric
`error_budget` (coherent/leakage vs decoherence floor + unitarity), in the
optimized virtual-Z frame. Returns `{"f_proc", "r_total", "r_coherent",
"r_decoherence", "unitarity", "r_incoherent_floor_from_unitarity",
"r_coherent_from_unitarity"}`.

**`counter_rotating_fidelity(x_raw, dt=1.0, vz=None, substeps=200, diss_scale=0.0) -> dict`**:
**the beyond-RWA validity check.** Re-simulates the *same* pulse with each
drive's counter-rotating partner (oscillating at 2·ω_d, a Bloch-Siegert-type shift)
restored, via fine `substeps` sub-stepping, and reports how far the process
fidelity moves. The RWA reference uses the same sub-stepped integrator *without* the
term, so the delta is pure counter-rotating physics, not a scheme artifact. Lives on
CR (not the parametric CZ) because CR has a single, unambiguous drive frame and is
also the strong-drive gate where RWA error matters most. `diss_scale=0` (default)
isolates the coherent error. Returns `{"f_proc_rwa", "f_proc_counter_rot",
"delta_r_counter_rot", "f_avg_rwa", "f_avg_counter_rot", "omega_d_ghz", "substeps",
"dt_fine_ns"}`.

**`refine_beyond_rwa(x_raw, dt=1.0, vz=None, substeps=20, iterations=20, lr=0.01) -> dict`**:
the **optimization** counterpart of `counter_rotating_fidelity` (which only
diagnoses). Starting from the RWA-optimized `x_raw`, it gradient-descends the process
fidelity *with the counter-rotating term inside the gradient loop* (fine sub-stepping),
so the optimizer **removes** the beyond-RWA residual instead of only measuring it: the
pulse is no longer confined to the RWA. A one-shot polish (heavier per step than the
RWA loop). Returns `{"best_raw_param", "f_proc_before", "f_proc_after",
"delta_removed", "vz", "substeps"}`.

**`iq_waveform(x_raw, dt=1.0) -> dict`**: the **complete** complex CR drive(s) (in-phase plus the derived DRAG quadrature) in physical rad/ns, hardware-export-ready
(same contract as the parametric `iq_waveform`). Feed a channel to `braket_bridge` /
`openpulse_export`.

**`spectator_fidelity(x_raw, dt=1.0, vz=None, zeta_mhz=0.1, ...) -> dict`**:
always-on-ZZ spectator penalty, CR analogue of the parametric method.

**`resonant_collision_fidelity(x_raw, dt=1.0, vz=None, detuning_mhz=0.0, j_mhz=8.0, couples_to="control", ...) -> dict`**:
frequency-collision diagnostic (evolving exchange-coupled spectator), CR analogue of
the parametric method; `couples_to ∈ {"control", "target"}`. The dominant yield-limiter
for fixed-frequency lattices. Cross-checked by `validate.cr_collision_cross_check`.

---

## General N-qubit: `gradpulse.multiqubit`

GRAPE on an arbitrary N-qubit register, so crosstalk/collisions are optimized
against (not just scored). Exact density-matrix simulation ⇒ cost is exponential in N
(open-system practical to ~4 qubits via the exact Choi path; the **memory-light
state-transfer** objective below pushes the open-system reach further; closed-system/
unitary to ~6-7). Cross-checked by `validate.multiqubit_cross_check` (independent QuTiP
rebuild, Δ ~ 4×10⁻¹⁴).

**`@dataclass MultiQubitProfile(n_qubits=3, freqs_ghz=..., anharm_mhz=..., t1_ns=..., t2_ns=..., couplings={(i,j): g_mhz}, n_levels=3, f_ref_ghz=None, notes=[])`**
N transmons on a coupling graph. Per-qubit lists must have length `n_qubits`;
`couplings` keys are edges `i<j` in MHz; `f_ref_ghz` defaults to the mean frequency
(set it to a driven qubit's frequency to drive that qubit on resonance).

**`class MultiQubitOptimizer(profile, target_gate="cz", target_qubits=(0,1), drive_qubits=None, tunable_edges=None, omega_max_mhz=60.0, g_max_mhz=20.0, bandwidth_mhz=80.0, use_drag=False, freq_control_qubits=None, delta_max_mhz=200.0, open_system=True, precision="single", verbose=True)`**
- `target_gate`: `"cz"/"cnot"/"iswap"/"sqrt_iswap"` or any `2**k × 2**k` unitary, realized on `target_qubits` (k indices, any subset incl. non-adjacent) with **identity on every other qubit**.
- **Simultaneous gates:** pass a **list** of gates and a matching list of *disjoint* qubit groups (`target_gate=['cz','cz'], target_qubits=[(0,1),(2,3)]`) to optimize parallel gates under one shared crosstalk budget. A single gate name with grouped qubits applies it to every group; a 1-qubit group may be a bare int. Groups must be disjoint (checked). The combined target is the tensor product, identity elsewhere.
- `drive_qubits` (default all): which qubits get a drive channel. `tunable_edges` (default the target-internal edges): coupling edges that are control channels; other edges stay always-on as crosstalk.
- `freq_control_qubits` (default none): qubits whose **frequency** is the control (a flux-tuned **tunable coupler** modelled as an explicit evolving transmon), with detuning ceiling `delta_max_mhz`: the faithful model where dispersive Schrieffer-Wolff elimination is not (e.g. a fast flux pulse).
- `open_system`: `True` = Lindblad Choi (decoherence + leakage aware); `False` = unitary (coherent + leakage, far cheaper, larger N).

| Method | Returns |
|---|---|
| `optimize(n_slices=200, dt_ns=1.0, iterations=250, n_seeds=2, lr=0.05, leak_weight=1.0, seed0=0, fidelity="choi", n_states=32, state_seed=0, grad_clip=1e3, checkpoint_segments=0, warm_start=None, edge_rest_slices=0, verbose=False)` | `{best_fidelity, F_avg, best_waveform [n_slices, n_channels], best_raw_param, leakage, history, n_qubits, target_qubits, fidelity_mode, converged, ...}`. Loss = (1−F_proc) + leak_weight·leakage over the 2ᴺ computational subspace. `fidelity="choi"` (default) = exact 4ᴺ Choi stack (open-system N≲4); `fidelity="state_transfer"` = memory-light estimate from `n_states` Haar-random inputs (O(n_states) propagated ops, MC variance ~1/n_states) that extends the open-system reach; `fidelity="cz_data_virtualz"` (open system, 2 data qubits + 1 ancilla coupler) optimizes the **physical** CZ: the gate on the data qubits with the coupler idle in \|0⟩, single-qubit virtual-Z free (as the device's native CZ applies via `shift_phase`); the strict default `CZ⊗I` instead penalizes coupler-excited inputs that never occur and caps a tunable-coupler CZ well below its true fidelity. With this mode the result also carries `virtual_z_phases (φ0, φ2)` to apply on the data qubits. `state_seed` fixes a deterministic objective. (Ignored when `open_system=False`.) `checkpoint_segments=S>1` gradient-checkpoints the slice loop (memory ~O(Nt/S), ~2× forward), the memory-side reach lever, complementary to `state_transfer`. `warm_start` seeds GRAPE from a known-good control envelope (a `[n_slices, n_channels]` array in the `[0,1]` smoothed-control convention, `0.5`=rest for a bipolar flux channel, or a list, one per seed) instead of random init; decisive for the hard 27-D tunable-coupler CZ, where seeding the device's flat-top CZ shape converges toward a clean gate while random init plateaus in leaky local optima. Seeds past the list fall back to random, so warm and random compete. `edge_rest_slices=r>0` forces every control to its rest value (`x=0.5`, `u=0`) over the first/last `r` slices via a raised-cosine ramp, so the gate is **composable**: it starts and ends with the coupler/drives idle and can be chained (as interleaved RB does) without leaving the coupler detuned. Default `0` is byte-for-byte the unconstrained path; use it for any pulse you will actually play in a sequence. |
| `state_transfer_fidelity(waveform, dt_ns=1.0, n_states=32, seed=0, diss_scale=1.0) -> dict` | Standalone memory-light scorer (same estimator the `state_transfer` objective uses). Returns `{F_avg, F_proc, n_states}`; converges to the exact `process_fidelity` as `n_states` grows. |
| `process_fidelity(waveform, dt_ns=1.0, diss_scale=1.0) -> float` | Exact F_proc of an already-smoothed pulse (no re-smoothing), matching the optimizer + QuTiP. |
| `process_fidelity_sparse(waveform, dt_ns=1.0) -> float` | Closed-system F_proc via **sparse Krylov** state propagation: propagates the `2**N` computational states with `scipy.sparse.linalg.expm_multiply` instead of the dense `4**N` Choi stack: memory `O(2**N·D)`, no dense propagator, so it scores larger N (closed-system; the step between the dense path and [`gradpulse.mps`](#large-register-evaluator-gradpulsemps)). |
| `cost_estimate() -> dict` | `{hilbert_dim, choi_ops, matmul_dim, work_per_step, warning}`, flagging large systems. |

---

## Large-register evaluator: `gradpulse.mps`

**Evaluation-only** matrix-product-state evaluator to score a pulse past the dense
~4-qubit wall **in the weakly-entangling regime**. NumPy-only; exported at top level as
`gradpulse.ChainTEBD`. The exponential does not vanish. It *moves*: an MPS compresses
*states*, not the `4**N` basis operators a 2-design averages, so it **cannot** produce the
exact `F_proc` cheaply. What it returns is an honest **restricted-ensemble
average-gate-fidelity witness** (mean input-output fidelity over a finite low-entanglement
input ensemble), reported as such; it does **not** map through `F_avg=(d·F_proc+1)/(d+1)`
and typically *over*-estimates (it under-samples the hard-to-preserve entangled inputs).
Its bias is measured against the dense value where both run; chi-convergence
(`max_discarded`) is the ship gate.

**`class ChainTEBD`**: second-order Trotter (TEBD) evolution built from the *same* local
model as `MultiQubitOptimizer`.

| Member | Purpose |
|---|---|
| `from_optimizer(opt) -> ChainTEBD` | Build directly from a configured `MultiQubitOptimizer` (shares its Hamiltonian + coupling graph). |
| `process_fidelity_tebd(waveform, dt_ns=1.0, substeps=1) -> float` | Exact closed-system `F_proc` over the full `2**N` subspace via TEBD, the **validation** path (agrees with `MultiQubitOptimizer.process_fidelity_sparse` as `substeps→∞`), not the cheap one. |
| `evolve_mps(local_kets, waveform, dt_ns=1.0, substeps=1, chi_max=64)` | Evolve a product input as a bond-`χ`-truncated MPS, `O(N·χ²·d)` per state. |
| `witness_open(ensemble, waveform, dt_ns=1.0, substeps=1, chi_max=64, n_traj=200, seed=0) -> dict` | The open-system **witness**: trajectory-unraveled MPS average over a restricted product-state ensemble, returned with its `max_discarded` truncation so chi-convergence is auditable. **NOT** `F_proc`. |
| `evolve_statevector` · `product_mps` · `mps_to_vector` · `evolve_trajectory` | Lower-level blocks: exact statevector TEBD, MPS construction/contraction, single-trajectory evolution. |

---

## One-call convenience wrappers: `gradpulse.convenience`

Collapse the "make a profile, make an optimizer, optimize" boilerplate into a single
call with good defaults; each returns the optimizer's result dict plus
`result["optimizer"]` for follow-up analysis. The full control surface stays on the
classes above.

| Function | What it does |
|---|---|
| `optimize_cz(profile=None, *, n_seeds=4, iterations=200, n_slices=150, dt_ns=1.0, bandwidth_mhz=80.0, precision="single", **kw) -> dict` | One-call CZ on a parametric-coupler pair (representative `ParametricCouplerProfile` if `profile=None`). Extra kwargs pass through to `optimize_multi_seed`. |
| `optimize_iswap(...) -> dict` | Same, targeting iSWAP. |
| `tunable_coupler_cz(freqs_ghz=(4.40,5.50,4.60), anharm_mhz=(-200,-150,-200), g_qubit_coupler_mhz=85.0, t1_ns=..., t2_ns=..., n_levels=3, delta_max_mhz=300.0, precision="double", verbose=True, **kw) -> MultiQubitOptimizer` | Builds the **explicit** qubit-coupler-qubit 3-element CZ optimizer (coupler evolved as a live flux-tuned transmon), on `MultiQubitOptimizer`. Call `.optimize(...)` on it. |
| `coupler_in_loop_cz(profile=None, *, coupler_freq_ghz=5.9, coupler_anharm_mhz=-250.0, gc_mhz=95.0, coupler_t1_ns=1.5e4, coupler_t2_ns=1.0e4, n_levels=3, n_seeds=2, iterations=150, n_slices=160, dt_ns=1.0, delta_max_mhz=300.0, precision="double", **kw) -> dict` | Opt-in **coupler-in-the-loop** CZ from a pair profile: reads the two qubits' params off `profile` (a `ParametricCouplerProfile`), re-introduces the Schrieffer-Wolff-eliminated coupler as a live transmon between them, and optimizes the flux-activated CZ on the QuTiP-cross-checked `MultiQubitOptimizer` engine. Returns the optimize dict plus `coupler_leakage` (the DOF the eliminated pair model has identically zero of), `sw_param=(gc/Δ)²`, `J_eff_mhz`, `coupler_freq_ghz`, and `optimizer` (feed to `validate.multiqubit_cross_check`). Heavy (27-dim open system); raise `iterations`/`n_seeds` for production. The rigorous residual of the elimination itself stays `validate.coupler_elimination_cross_check`. |

---

## Band-limited basis: `gradpulse.basis`

The synthesis basis behind `ParametricCZOptimizer.optimize_spectral`, a CRAB/Fourier
parameterization that is band-limited by construction (no post-hoc smoother).

**`class FourierBasis(n_slices, dt_ns=1.0, f_max_mhz=None, n_harmonics=None, dtype=torch.float32, device=None)`**
A fixed `[n_slices, n_basis]` real synthesis matrix `Phi`: a DC column plus a cos/sin
pair at each harmonic of `f0 = 1/T` (T = `n_slices·dt_ns`) up to `f_max_mhz` (default
the slice-Nyquist; pass your AWG bandwidth). `n_harmonics` overrides the harmonic count
directly (capped below Nyquist so the basis never aliases). `n_basis = 2·K + 1`.

| Member | Purpose |
|---|---|
| `synthesize(coeffs) -> tensor` | `coeffs [..., n_basis, n_channels] → control [..., n_slices, n_channels]` (every component ≤ `f_max`, so band-limited by construction). |
| `Phi`, `n_basis`, `n_harmonics`, `f_max_mhz`, `frequencies_mhz` | The basis matrix and its resolved spectral content. |
| `dc_coeff_for_level(level) -> float` | The DC coefficient synthesizing a constant `level`. |

---

## Independent validation: `gradpulse.validate`

Rebuilds the Hamiltonian + Lindblad operators in **QuTiP** (a different library) and
replays a saved pulse under a matched piecewise-constant scheme. Requires the
`[validate]` extra (QuTiP). Auto-rebuilds at any `n_levels` on the profile.

| Entry point | Purpose |
|---|---|
| **CLI** `python -m gradpulse.validate --pulse path/to/pulse.json [--mesolve]` | Auto-detects architecture from pulse metadata; prints `F_proc(QuTiP) − F_proc(gradpulse)` with a PASS gate at ±0.001. `--mesolve` adds the adaptive-solver cross-check below (parametric_cz pulses). |
| `qutip_f_proc(profile, waveform, target_gate="cz", dt_ns=1.0) -> float` | The independent process fidelity for a waveform under a profile (matched piecewise-constant scheme, same integration method as the optimizer, independent operator build). |
| `mesolve_zoh_fproc(profile, waveform, target_gate="cz", dt_ns=1.0, line_response=None) -> float` | Independent `F_proc` from QuTiP's **adaptive** master-equation solver (`mesolve`) on a zero-order-hold staircase of the pulse, run interval-by-interval. A *different numerical method* than the matched scheme, so it confirms the piecewise-constant integrator is **unbiased** (converges to the true continuous-time Lindblad solution), not merely self-consistent under `dt_convergence`. On the bundled CZ it agrees with the `dt→0`-refined value to `~10⁻⁷`. |
| `cross_check(pulse_json, profile_overrides=None) -> dict` | Programmatic form of the CLI check on a saved pulse file. |
| `cr_cross_check(optimizer, waveform, vz=None, echo=None, dt_ns=1.0) -> float` | In-process (no file) independent QuTiP `F_proc` for a cross-resonance pulse, **echo-aware**; shares the single `_qutip_cr_fproc` evolution core with the file-based check, so the echoed gate is validated by the same code path the optimizer is mirrored against. `echo=None` reads the flag off the optimizer. |
| `spectator_cross_check_3transmon(profile, waveform, ...) -> dict` | Validates the effective-spectator reduction against a full 3-transmon (27-D) QuTiP simulation. |
| `spectator_cross_check_multi(profile, waveform, target_gate, dt_ns, neighbours) -> float` | Multi-spectator: validates the additive-detuning reduction against an explicit ((n_levels²)·2^N)-D QuTiP sim with N frozen |1⟩ neighbours. |
| `collision_cross_check(profile, waveform, target_gate, dt_ns, detuning_mhz, j_mhz, couples_to=2) -> dict` | Validates `resonant_collision_fidelity` (parametric): full (n_levels³)-D QuTiP sim with an **evolving** exchange-coupled spectator. Returns `{"f_proc", "spectator_leakage"}`. |
| `cr_collision_cross_check(profile, waveform, vz, dt_ns, detuning_mhz, j_mhz, couples_to="control", *, use_drag=True) -> dict` | CR analogue of `collision_cross_check` (64-D at the default n_levels=4); DRAG re-derived, vz-framed ZX(π/2) target. |
| `cr_counter_rotating_cross_check(profile, waveform, vz, dt_ns, ...) -> dict` | Independent **adaptive-propagator** QuTiP check of `counter_rotating_fidelity`: rebuilds the time-dependent counter-rotating Hamiltonian and integrates it with a different library *and* scheme, returning the beyond-RWA `delta_r_counter_rot`. |
| `coupler_elimination_cross_check(freq_ghz, anharm_ghz, coupler_detuning_mhz, gc_mhz, ...) -> dict` | Validates the Schrieffer-Wolff coupler elimination: explicit 3-body vs 2-body effective exchange; returns `J_eff_mhz`, `max_coupler_pop`, `max_traj_diff` (all O((g/Δ)²)). |
| `tls_defect_cross_check(profile, waveform, target_gate, dt_ns, detuning_mhz, g_mhz, t1_tls_ns, couples_to=1) -> dict` | Validates `tls_defect_fidelity`: an explicit lossy two-level defect (transverse exchange + its own T1 jump) rebuilt in QuTiP in the full `(n_levels²·2)`-D open system. |
| `multiqubit_cross_check(optimizer, waveform, dt_ns=1.0) -> dict` | Independent QuTiP rebuild of a `MultiQubitOptimizer` (open system); evolves the saved pulse over the 2ᴺ subspace and returns `{"f_qutip", "f_torch", "delta"}` (Δ ~ 4×10⁻¹⁴). |

---

## Third solver: `gradpulse.liouville`

A **third, QuTiP-free** independent solver, so every headline number is reproduced by three
solvers that share no library, representation, or time-splitting (what makes the
"triple-solver validated" claim load-bearing). NumPy-only (a self-contained Padé `_expm`, no
QuTiP/PyTorch/SciPy), so it ships at top level and runs without the `[validate]` extra. It
builds the full Lindbladian **superoperator** (the Liouvillian) and propagates `vec(ρ)`
through the exact matrix exponential of the *whole* generator per slice, so a match
independently bounds the first-order **Trotter splitting error** that the matched QuTiP
cross-check structurally cannot probe. **One solver per architecture**, so the
library-independent leg covers all three pair/register models, not just the parametric CZ,
which would otherwise leave the cross-resonance and N-qubit gates checked only by two
QuTiP-based legs sharing a library.

**`liouville_f_proc(profile, waveform, target_gate="cz", dt_ns=1.0, line_response=None, detuning_offset=0.0) -> float`**
Drop-in analogue of `validate.qutip_f_proc`: same saved `[0,1]` envelope, same profile, the
same exact 16-operator Choi process fidelity, through a wholly separate solver.
`detuning_offset` adds a static qubit-frequency offset (rad/ns; scalar → common `δ·(N1+N2)`,
or `(δ1, δ2)` per qubit), so the robustness / quasi-static / spectator-ZZ paths can be
cross-checked too. Exported at top level as `gradpulse.liouville_f_proc`.

**`liouville_cr_f_proc(profile, waveform, vz=(0,0), echo=False, use_drag=True, dt_ns=1.0) -> float`**
The cross-resonance analogue: the same exact, leakage-aware `F_proc` that
`CrossResonanceZXOptimizer._process_fidelity` and `validate.cr_cross_check` compute, through
the NumPy-only Liouvillian. `waveform` is the smoothed signed drive (`result['best_waveform']`);
`vz` the saved virtual-Z frame; `echo` runs the echoed sequence (second-half sign-flip + ideal
control π at midpoint and end); `use_drag` re-derives the Motzoi quadratures. Reproduces the
QuTiP referee to `~1e-7` (the residual is the QuTiP leg's own Trotter split). Exported as
`gradpulse.liouville_cr_f_proc`.

**`liouville_nqubit_closed_f_proc(optimizer, waveform, dt_ns=1.0) -> float`**
The N-qubit analogue for the **closed-system** path: mirrors
`MultiQubitOptimizer._propagate_unitary` + `_process_fidelity_unitary` in pure NumPy, with an
independently reconstructed subset target and the spectator coupling carried in the drift (so
in-loop crosstalk is inside the check). `waveform` is the smoothed `[0,1]` control consumed
verbatim, as in `MultiQubitOptimizer.process_fidelity`. No Trotter split in the closed system,
so it meets the optimizer at machine precision (`~1e-15`) in double precision. The open-system
Choi path keeps its QuTiP cross-check (its Lindblad dissipator is the same form as the pair
models). Exported as `gradpulse.liouville_nqubit_closed_f_proc`.

---

## Randomized benchmarking: `gradpulse.rb`

numpy-only. Simulates the leakage-aware **interleaved-RB estimator** a hardware
experiment reports, and bridges it to the analytic process fidelity (the
estimand-vs-estimator distinction). Enumerates the full 11,520-element 2-qubit
Clifford group as native-gate words.

| Function | Purpose |
|---|---|
| `two_qubit_cliffords() -> CliffordGroup` | BFS enumeration of the 11,520 two-qubit Cliffords as native-gate words. |
| `gate_superoperator(opt, u_stack, dt=1.0) -> np.ndarray` | The noisy gate as an 81×81 superoperator from an optimizer + pulse. |
| `native_superops(cz_superop) -> dict` | Native-gate superoperators built around a given noisy CZ. |
| `superop_from_unitary(U9)`, `superop_from_basis_action(evolved_basis)` | Superoperator constructors. |
| `depolarizing_gate_superop(p, U4=None, ...)` | A depolarizing reference gate (for the validation test). |
| **`interleaved_rb(cz_superop, lengths=(1,2,4,8,16,24,32), n_sequences=40, seed=0, f_avg_analytic=None) -> dict`** | Runs reference + interleaved sequences; fits naive single-exponential **and** leakage-aware (Wood-Gambetta) decays. Returns the decays, `r_cz_naive`, `r_cz_leakage_aware`, `f_cz_irb`, `leakage_per_clifford_L1`, `seepage_per_clifford_L2`, and (if `f_avg_analytic` given) `r_analytic` + the bridge gap. |

---

## Literature anchors: `gradpulse.literature`

Torch-free. Validates the decoherence **model** against published hardware gates in a
**data-driven** way: each device is one cited JSON anchor in `examples/anchors/`, so
adding a device is dropping in a file; the ground truth lives in attributed data, not
in code. The judging logic is unit-tested without GRAPE (`tests/test_literature.py`);
the multi-minute optimization driver is `examples/validate_against_literature.py`.

| Function | Purpose |
|---|---|
| **`load_anchor(path) -> dict`** | Parse + schema-validate an anchor. **Rejects** one with no `provenance.citation`, no `validation.measured_f_avg` (must be an F_avg in (0,1)), or no boolean `validation.coherence_limited`; an un-attributable or unfalsifiable "validation" cannot be added. |
| `anchor_to_profiles(anchor) -> (none, t1, full)` | Build the three coherence variants. Only the measured T1/T2 is in the anchor; **T1-only** (`T2=2·T1`) and **no-decoherence** (`T1→∞`) are *derived* via `ParametricCouplerProfile.from_calibration`, so there is no second editable copy of the coherence numbers. |
| `gate_config(anchor) -> dict` | Optimizer/gate settings with package defaults filled in (kept separate so this module stays torch-free). |
| **`judge(anchor, f_coh, f_t1, f_full, analytic_epg=None) -> dict`** | The scientific decision, in F_avg/EPG terms. `coherence_limited: true` → PASS iff `decoherence_error/measured_error` ∈ `ratio_band` (default 0.5-1.5×). `false` → PASS iff the floor is a valid **lower bound** (`ratio ≤ 1`), and the residual is reported as `unexplained_error`. Independently flags `gate_closes` (a large coherent residual makes the comparison inconclusive). Pass `analytic_epg` to also record the GRAPE-independent floor alongside the GRAPE one. |
| `analytic_coherence_limit_epg(profile, gate_ns) -> float` | The **GRAPE-independent** coherence-limited average gate error, closed-form from T1/T2 and gate duration: `(2·t_g/5)·Σ_q(1/T1_q + 1/T_φ_q)` with `1/T_φ = 1/T2 − 1/(2·T1)` (floored at 0). Needs no optimization, so it's an independent sanity bound on the GRAPE decoherence floor (they should agree to within GRAPE scatter). |
| `judge_analytic(anchor, analytic_epg) -> dict` | `judge`'s sibling for devices that publish T1/T2 + gate time but **not** the per-pair frequencies/anharmonicities a GRAPE floor needs (e.g. Stehlik 2021). Applies the same `coherence_limited` band / lower-bound logic to `analytic_coherence_limit_epg` instead of the optimizer's floor. No `gate_closes` guard (no optimizer ran); records `gate_closes=None`. |
| `effective_t2_ns(t1_ns, tphi_exp_ns) -> float` | Derive the Markovian `T2` from `T1` and an exponential `T_phi` (`1/T2 = 1/(2 T1) + 1/T_phi`). An anchor qubit may publish `tphi_exp_ns` instead of `t2_ns` and the loader derives `T2` here, so a paper that reports `T_phi` (e.g. gate-effective coherence) needs no hand-computed `T2` in the file. A quoted Gaussian/non-Markovian `tphi_gauss_ns` is recorded but not folded in (a single-`T2` model cannot represent it). |
| `format_report(verdict) -> str`, `discover_anchors(dir) -> [Path]`, `f_avg(f_proc) -> float` | Render a verdict; list `*.json` anchors; F_proc→F_avg for d=4. |

---

## In-loop vs multiply-after: `gradpulse.headtohead`

Demonstrates (rather than asserts) the package's central claim: that optimizing with
decoherence *in the loop* beats the common "optimize the coherent gate, then multiply
by an `e^{−t_g/T}` budget" recipe. Reuses one primitive (`optimize_multi_seed` with
`diss_scale=1.0` for the true open-system objective vs `0.0` for the coherent objective) across a
sweep of gate durations on a device where leakage and decoherence genuinely compete.
Nothing is hard-coded; every operating point falls out of the sweep. Driver + figure:
[`examples/decoherence_in_the_loop.py`](examples/decoherence_in_the_loop.py).

The complementary regime, where *leakage* rather than decoherence dominates, is mapped
by [`examples/leakage_in_the_loop.py`](examples/leakage_in_the_loop.py): a cross-resonance
duration sweep that uses `error_budget` to split each optimized gate's error and show the
crossover from coherence-limited (the one-line coherence formula tracks the truth) to
leakage-limited (the formula under-predicts the true error by ~60× at 110 ns, a 6×10⁻²
gate certified near-perfect). Every fidelity is checked by the independent NumPy Liouvillian
and QuTiP.

**`run_head_to_head(profile, durations_ns, *, dt_ns=1.0, n_seeds=2, iterations=150, lbfgs_iters=40, lr=0.02, precision="double", seed=0, verbose=True, warm_start_chain=False) -> dict`**
For each duration it optimizes the in-loop objective (`A`) and the coherent objective
(`B`) from the same seed, then scores `B` two ways: `predicted` = its coherent
fidelity × the analytic coherence budget (`literature.analytic_coherence_limit_epg`),
and `delivered` = its true open-system fidelity. **Returns** `{"rows": [...per
duration...], "summary": {...}}`. The summary gives each method's chosen duration and
the delivered-fidelity gap, decomposed into a **pulse-shaping** part (in-loop's
gradient finds a better pulse at fixed duration) and a **duration-selection** part
(the analytic budget points the recipe at the wrong duration), plus the budget's
over-prediction at its own chosen duration. `warm_start_chain=True` seeds each
duration from the previous one's solution (resampled to the new slice count), which
converges much faster on a fine sweep; `A` chains from `A` and `B` from `B`, so the
A/B fairness is preserved.

**`resample_pulse(wf, n_new) -> np.ndarray`**: linearly resample a `[n_old, C]` pulse
envelope onto `n_new` slices (the warm-start primitive; reusable for any sweep that
re-optimizes across a changing slice count).

---

## Hardware-in-the-loop: `gradpulse.hardware`

Implement one method for your device and the loop refines the model's effective
coherence from the measured gate fidelity, closing the sim↔device gap.

**`class HardwareBackend`**: the hook. Implement
`measure_gate(waveform, dt_ns, meta=None) -> GateMeasurement`.

**`@dataclass GateMeasurement`**: `f_avg`, `source`, and metadata returned by a backend.

| Backend | What the "device" is |
|---|---|
| `SimulatedBackend(profile)` | gradpulse's own simulator with (typically worse) coherence. Fast, no extra deps. |
| `QuTiPDeviceBackend(profile, shots=None, n_irb_sequences=30)` | The **independent** QuTiP integrator, closing a genuine model-vs-truth gap with no QPU. `source == "qutip_independent"`. Set `shots=N` and the "measurement" becomes a **finite-shot interleaved-RB fit** (binomial sampling + least-squares decay fit) instead of the exact number, so the loop is exercised against realistic measurement noise (`source == "qutip_independent+shotnoise"`). |
| `BraketPulseBackend(device=None, shots=1000, device_name="Rigetti-Cepheus-1-108Q", clifford_compiler=None)` | The concrete real-silicon seam (alias: `BraketBackendTemplate`). `build_gate_sequence(waveform)` → real `braket.pulse.PulseSequence`; `estimate_cost(...)` → `$`; `measure_gate(...)` builds everything and raises a precise error at `device.run()` unless given a live `AwsDevice` + `clifford_compiler`. |

**`calibrate_to_hardware(initial_profile, backend, rounds=3, opt_kwargs=None, dt_ns=1.0, n_channels=3, bandwidth_mhz=80.0, precision="double", n_measure=1, verbose=False) -> dict`**
Each round: optimize on the current model, hand the physical pulse to the backend,
read back measured F_avg, infer the coherence scaling that reconciles them, fold it
in, repeat. Returns `{"refined_profile", "history"}` where each `history` entry has
`{"round", "f_model_avg", "f_hardware_avg", "gap", "coherence_scale", "t1_ns_q1",
"t2_ns_q1", "source"}`. With a shot-noisy backend, **`n_measure>1` averages that many
independent measurements per round**, the lever that beats the statistical error down
(~1/√n_measure), and the entry then also carries `f_avg_sem`.

`simulate_noisy_irb(f_avg_true, shots=1000, n_sequences=30, ...) -> float` is the
underlying estimator: it draws a binomially sampled RB survival curve about the exact
`f_avg_true` and fits it back, so the returned number carries genuine sampling noise
(unbiased to leading order; spread ~ 1/√(shots·n_sequences)).

Helpers: `predicted_process_fidelity(profile, waveform, ...)`,
`predicted_f_avg(profile, waveform, ...)`, `apply_coherence_scale(profile, scale)`,
`infer_coherence_scale(profile, waveform, measured_f_avg, ...)`.

---

## Amazon Braket on-ramp: `gradpulse.braket_bridge`

Optional `[braket]` extra (`amazon-braket-sdk`). Everything here is **offline-verifiable**:
no AWS account, credentials, or cost. The one step it does *not* do is `device.run()`,
the only thing that closes simulation ≠ hardware. Beyond pulse export it also generates the
**interleaved-RB circuits** (verbatim-boxed, offline-verified to return to the ground state)
and the **Level-B** binding that plays a gradpulse pulse as a benchmarked `pulse_gate`.

| Function | Purpose |
|---|---|
| `to_braket_waveform(envelope, *, waveform_id="grad_env", normalize=True) -> (ArbitraryWaveform, peak)` | Export one channel (real or complex I/Q) to a Braket waveform; returns the physical `peak` that normalization divided out. |
| `verify_waveform_roundtrip(envelope, *, normalize=True) -> float` | Max abs error between exported and original samples (~0 = faithful). |
| `synthetic_frames(n_channels, *, base_freq_hz=5e9, dt_s=1e-9) -> list[Frame]` | Placeholder frames for offline construction (use `device.frames` on hardware). |
| `build_gate_pulse_sequence(waveform, frames, *, normalize=True) -> PulseSequence` | Bind a `[n_slices, n_channels]` envelope to frames; `seq.to_ir()` is valid OpenPulse 3.0. |
| `estimate_experiment_cost(n_circuits, n_shots, device=...) -> CostEstimate` | Per-task + per-shot cost from `BRAKET_QPU_PRICING` (dated 2026-06). |
| `irb_circuit_count(lengths, n_seeds) -> int` | `2·len(lengths)·n_seeds` (reference + interleaved). |
| `largest_irb_under_budget(budget_usd=50, device=..., n_shots=500, lengths=(1,2,4,8,16)) -> dict` | Largest IRB design fitting the budget, with rough fidelity resolution. |
| `native_rb_sequences(lengths, n_seeds, *, seed=0, interleaved=False) -> list[dict]` | The interleaved-RB **circuits** as native-gate words (`h`/`s`/`cz`; a tagged `CZ_BENCH` after each Clifford when `interleaved`), built before submission. |
| `to_braket_rb_circuit(gates, qubits=(0,1), *, bench_cz_pulse=None, verbatim=True, buffer_bench_cz=False) -> Circuit` | A native-gate word → `braket.circuits.Circuit` inside a **verbatim box** (Rigetti preserves the sequence exactly, mandatory for RB). `buffer_bench_cz=True` wraps each benchmarked CZ in barriers so back-to-back-CZ context cannot inflate the interleaved estimator; `bench_cz_pulse` binds a gradpulse pulse as a `pulse_gate` (Level B). |
| `ideal_survival_probability(gates) -> float` · `survival_from_counts(counts) -> float` | The offline correctness gate (an ideal sequence returns to ground ≈ 1.0, proving the circuits are right *before* spending) and the P(ground) survival observable from a measurement-counts dict. |
| `cz_durations_from_native_calibration(cal, *, mode="active", threshold=0.01, dt_ns=1.0) -> dict` | Real per-pair CZ gate **durations** (ns) parsed from a Braket native-gate-calibration dict, the input that replaced the study's flat-60 ns assumption. `mode ∈ {buffer, active, effective}`. |
| `build_bench_cz_pulse_sequence(flux_waveform, flux_frame, *, peak_amplitude=1.0, drive_frames=None, virtual_z=(0,0), normalize=True) -> PulseSequence` | **Level B:** bind a gradpulse gate-activation waveform to the device's CZ frame as one `pulse_gate`. |
| `bench_cz_peak_from_native_calibration(cal, site) -> float` | **Level B:** peak \|flux amplitude\| of the device's native CZ on a pair; anchors a gradpulse shape to the device's own calibrated flux full-scale. |
| `verify_levelb_offline(bench_cz_pulse, *, qubits=(0,1), n_cliffords=4, seed=0) -> dict` | **Level B:** everything checkable without a QPU. The bench pulse serializes to OpenPulse, and an interleaved-RB circuit carrying it serializes to OpenQASM 3 with the verbatim pragma + an embedded `play`. |
| `hardware_readiness_report(waveform, *, device=..., budget_usd=50, n_shots=500, verbose=True) -> dict` | Runs the whole offline chain; states what's validated and the one step left. |

---

## OpenPulse export: `gradpulse.openpulse_export`

Optional `[openpulse]` extra (the `openpulse` parser ships with `amazon-braket-sdk`).
Vendor-neutral OpenQASM 3 / **OpenPulse 3.0** export, the live open standard, since
`qiskit.pulse` was **removed in Qiskit 2.0**. Pure text; nothing here needs a device,
account, or cost. Accepts the saved real envelope, a complex I/Q array, **or** the dict
from `optimizer.iq_waveform` (complex I/Q with DRAG baked in).

| Function | Purpose |
|---|---|
| `to_openpulse_program(waveform, dt_ns=1.0, *, gate_name="grad_gate", qubits=(0,1), frame_freqs_hz=None, labels=None, normalize=True) -> str` | Emit a `defcalgrammar "openpulse"` + `defcal <gate> $q0, $q1 {...}` program. `normalize` scales each channel into the unit disk (a frame plays a fraction of its calibrated full scale); the physical peak is written into a comment. |
| `verify_openpulse_roundtrip(waveform, dt_ns=1.0, *, normalize=True, **kw) -> float` | Emit → parse back with the **independent** `openpulse` parser → max abs sample error. `~0` means valid **and** lossless; `inf` if a channel failed to round-trip. The offline guarantee (analogue of the Braket bridge's `verify_waveform_roundtrip`). |
| `parse_openpulse_waveforms(program) -> dict` | `{waveform_name: complex ndarray}` reconstructed from the AST (also handy to check a third party's program). |
| `to_qiskit_schedule(waveform, *, gate_name="grad_gate", normalize=True)` | Native `qiskit.pulse.ScheduleBlock`, **only** if `qiskit.pulse` is importable (qiskit < 2.0); otherwise raises a clear error pointing at `to_openpulse_program`. |
| `openpulse_readiness_report(waveform, dt_ns=1.0, *, gate_name=..., qubits=(0,1), verbose=True, **kw) -> dict` | Runs the full offline chain (emit + independent re-parse) and states the one step left (running the `defcal` on a device). |

---

## qutip-qtrl benchmark: `gradpulse.benchmark`

Optional `[benchmark]` extra (`qutip-qtrl`). Turns the statement-of-need claim into
evidence: runs gradpulse's own autodiff `matrix_exp` GRAPE (Adam + L-BFGS) and
qutip-qtrl's analytic-gradient GRAPE on the **identical** control problem (same drift,
controls, target, time grid, and L-BFGS optimizer class), so the only variable is the
engine. qutip-qtrl is optional; the gradpulse side runs regardless (with a note if
absent).

| Function | Purpose |
|---|---|
| `standard_two_qubit_problem(gate="cnot") -> (H0, Hc, U_target, n_ts, evo_time)` | The benchmark task: exchange drift + local X/Y controls, target `"cnot"/"iswap"/"cz"`, the canonical closed-system 2-qubit synthesis problem qutip-qtrl itself uses. |
| `grape_autodiff(H0, Hc, U_target, n_ts, evo_time, *, lbfgs_iters=100, seed=0, device="cpu") -> dict` | The gradpulse engine on a generic `(H0, Hc, U_target)`. Returns `{method, fidelity, wall_s, iters}`. |
| `grape_qutip_qtrl(H0, Hc, U_target, n_ts, evo_time, *, max_iter=400) -> dict` | Same problem via `qutip-qtrl`'s `optimize_pulse_unitary` (raises `ImportError` if not installed). |
| `run_benchmark(gate="cnot", *, max_iter=200, seed=0, device="cpu", verbose=True) -> dict` | Runs both and returns/prints the comparison: both reach the optimum to ~machine precision at the same order-of-magnitude wall-clock. |

---

## Channel diagnostics: `gradpulse.diagnostics`

| Function | Purpose |
|---|---|
| `pauli_transfer_matrix(comp_superop) -> np.ndarray` | The 16×16 Pauli transfer matrix from a computational-subspace superoperator (proper basis change). |
| `channel_unitarity(comp_superop) -> float` | The channel **unitarity** (Wallman et al. 2015), the coherent-vs-incoherent diagnostic used by `error_budget`. |

---

## Plotting: `gradpulse.viz` (`[viz]` extra)

Quick matplotlib views of any result/analysis dict. Each takes an optional `ax`/`axes` and returns the matplotlib `Axes` (or `Figure`), so you can restyle, compose, or `savefig`. Imported lazily, so the core keeps no hard matplotlib dependency. Demonstrated end-to-end in [`examples/notebooks/01_intro_cz.ipynb`](examples/notebooks/01_intro_cz.ipynb).

| Function | Plots |
|---|---|
| `plot_pulse(result, dt_ns=None, ax=None, channel_labels=None) -> Axes` | Each control channel's envelope vs time (from `result["best_waveform"]`). |
| `plot_convergence(result, ax=None, infidelity=True) -> Axes` | Optimization curve from `result["history"]`, `1-F` on a log axis by default. |
| `plot_error_budget(budget, ax=None) -> Axes` | Bar chart of the `error_budget()` decomposition (control/leakage vs the decoherence floor). |
| `plot_robustness(sweep, axes=None) -> Figure` | Fidelity vs miscalibration, one subplot per axis of a `robustness_sweep()`. |

```python
import gradpulse as gp
from gradpulse import viz
r = gp.optimize_cz()
viz.plot_pulse(r); viz.plot_convergence(r)
viz.plot_error_budget(r["optimizer"].error_budget(r["best_raw_param"]))
viz.plot_robustness(r["optimizer"].robustness_sweep(r["best_raw_param"]))
```

---

*gradpulse is built and maintained by [Pure State Labs Inc.](https://purestatelabs.com), MIT-licensed.*


## Differentiable Logical Programming (`dlp.py`)

Embeds declarative soft logic constraints inside GRAPE's differentiable loss function.

- `Proposition`: Base class for declarative propositions.
- `Rule(premise: Proposition, penalty_weight: float)`: An IF-THEN constraint evaluated during optimization.
- `SoftLogic`: Continuous approximations for logic gates (`soft_and`, `soft_or`, `implies`).
- `SoftRelational`: Continuous approximations for relational operators (`greater_than`, `less_than`, `equals`).

## Advanced Scheduling (`scheduling.py` & `microscheduler.py`)

Dependency graphs and tight nanosecond packing for sequence execution.

- `DependencyGraph`: Models causal execution dependencies as a Directed Acyclic Graph.
- `OperationNode`: Represents a scheduled pulse or gate.
- `check_commutation(op1, op2)`: Safely checks if disjoint gates can commute.
- `MicroScheduler(dt_ns, channel_constraints)`: Maps a DependencyGraph onto continuous nanosecond boundaries, respecting ring-down delays.

## Calibration and Bayesian Optimization (`hardware.py`)

Derivative-free optimization and active learning for closed-loop hardware execution.

- `BayesianCalibrationLoop(backend, bounds)`: A surrogate-model optimizer (using Expected Improvement or Probability of Improvement).
- `HardwareBackend`: Base interface for submitting pulses to physical QPUs.
- `calibrate_to_hardware(...)`: Main closed-loop calibration routine.

## Distortion and Error Mitigation (`distortion.py` & `mitigation.py`)

Active corrections for physical control constraints.

- `Predistorter(kernel)`: Inverts cryogenic wiring transfer functions using iterative Tikhonov regularization.
- `ZNE(scales, method, scaling_type)`: Manager for Zero-Noise Extrapolation experiments.
- `stretch_pulse`, `fold_pulse`: Native noise-scaling routines for mitigation.

## Reinforcement Learning (`rl.py`)

Gymnasium environments to discover non-intuitive discrete parameter regimes.

- `CrossResonanceEnv`: Exposes the GRAPE optimizer as an RL environment to find initial seed macros.
- `train_ppo(...)`: Harness to train a Soft Actor-Critic / PPO agent using stable-baselines3.

## Waveform Compression (`compression.py`)

Reduces payload sizes for AWG memory limits.

- `compress_rle`: Constant-amplitude run-length encoding.
- `compress_delta`: Time-domain delta encoding.
- `compress_spline`: Spline-based downsampling.
- `verify_compression`: Strict check ensuring decompression remains within machine bounds.

## Visualization (`viz.py`)

Rich, color-coded representations of quantum dynamics.

- `plot_state_heatmap`: Time evolution of N-level system populations with continuous colormaps.
- `plot_bloch_trajectory`: Gradient-colored Bloch sphere trajectories representing leakage.
- `plot_spectrogram`: Time-frequency diagnostic views.
