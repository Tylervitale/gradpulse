# gradpulse: hardware validation log

A record of the numbers from validating `gradpulse` against the Rigetti Cepheus-1-108Q QPU on
Amazon Braket, plus the simulation cross-checks. Hardware numbers cost real money; simulation
numbers reproduce from the scripts named alongside them. Dates are UTC.

---

## 1. The device

Rigetti Cepheus-1-108Q, public ARN `arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q`.
Online, CZ-native (gates `rx, rz, cz, barrier`, which matches the model), 193 coupled pairs,
pulse-level access. Pricing was checked against the AWS Braket pricing page rather than trusting
the repo's own constant: $0.30 per task plus $0.000425 per shot, no minimum shots.

Calibration for the pair we benchmarked, qubits 16 and 25 (snapshot 2026-06-20 00:32 UTC):

| | q16 | q25 |
|---|---|---|
| T1 | 41.1 µs | 38.9 µs |
| T2 | 16.2 µs | 14.3 µs |
| 1-qubit RB fidelity | 0.99920 | 0.99906 |
| readout fidelity (symmetric) | 0.973 | 0.929 |
| CZ fidelity (interleaved RB) | 0.99594 ± 5.3e-4 (error 0.406%) | |

We ran three real tasks on qubits 16-25, 100 shots each, about $1.03 total. Two findings, both
understood and neither a device fault:

- A length-1 circuit gave survival 0.87, then 0.74 on the byte-identical circuit two hours later.
  The errors piled into the `10` outcome (6 → 18), which is asymmetric readout on one qubit; the
  symmetric readout number (0.973) hides it. RB's decay rate is SPAM-robust, so this lowers the
  signal amplitude but does not bias the extracted gate error.
- A depth-128 circuit (384 CZ, ~44 µs long) floored at survival 0.24, the fully-mixed value 1/d.
  The circuit runs about 3× the qubits' ~14 µs T2, so it decoheres completely. That is the
  expected coherence limit, and it confirms the gate is coherence-limited.

---

## 2. Prediction vs. measurement, all 160 gates

The core result. For each of Cepheus's 160 active pairs, `gradpulse` reads the real CZ duration
from the device's native-gate pulse calibration and recomputes the open-system decoherence floor
at that pair's published T1/T2, then compares to the published CZ error.

**The unselected claim (headline).** The prediction is one-sided (the floor must not exceed a
coherence-limited gate's measured error), so the honest test is the full scatter of all 160 pairs,
nothing selected on the prediction (`examples/cepheus/cepheus_lowerbound_scatter.py`; plotted in the paper).
The floor sits at or below the measured error, within the measurement's own ±12-42% RB uncertainty,
on **150/160** pairs, with a median floor of **0.66×** measured, the lower-bound behavior expected
when gates carry control/crosstalk error above their coherence limit. Only 4 pairs exceed measured
by >2×, one the impossible-T2 calibration entry flagged below.

**Saturation σ (refinement, not the headline).** How tightly does the floor *saturate* the
measurement where it does? A published CZ error has its own RB standard error (±12-42% on this
chip), so a relative "percent off" is meaningless; the honest metric is the distance in error bars,
σ = |predicted − measured| / standard-error (`examples/cepheus/cepheus_sigma_validation.py`). The standard
errors are pinned to a dated snapshot (`examples/cepheus/cepheus_cz_std_errors.json`, pulled 2026-06-21) so
this is reproducible offline and does not drift with the day's recalibration. On the 41 saturation
pairs (floor within 0.8-1.25× of measured *and* carrying a valid published RB error bar):

| agreement | gates |
|---|---|
| within 1σ | 32 / 41 |
| within 2σ | 41 / 41 |
| within 3σ | 41 / 41 |
| median | 0.60σ |

The prediction is statistically indistinguishable from the measurement on every gate in this
subset (all within 2σ). But the subset is *defined by* the saturation (floor ≈ measured), so this
measures how tight the bound is in the coherence-limited regime; it is not an independent test of
the prediction. The falsifiable claim is the one-sided bound over all 160, above. Per pair, the
misleading "% off" numbers turn into fractions of a σ:

| pair | t_g | measured ± err | gradpulse | σ |
|---|---|---|---|---|
| 2-3 | 50 ns | 0.88% ± 0.87% | 0.88% | 0.0 |
| 73-82 | 62 ns | 0.67% ± 0.17% | 0.67% | 0.0 |
| 78-87 | 66 ns | 1.37% ± 0.21% | 1.40% | 0.1 |
| 40-41 | 50 ns | 2.37% ± 0.36% | 2.48% | 0.3 |

78-87's "1.9% off" is a 0.026% absolute gap against a number known to ±0.21%. That is 0.1σ.

**Reading the real gate time.** This used to assume a flat 60 ns. The true per-pair durations
(30-94 ns, median 55) live in the native-gate calibration, free to read; the parser is
`braket_bridge.cz_durations_from_native_calibration`. Using them is the correct calculation, and
it does not tighten the fit; it makes `gradpulse` a cleaner lower bound (overall median ratio
0.69× → 0.66×). It also exposed that some flat-60 "≈1.0×" matches were a gate-time coincidence:
34-35 has a 38 ns gate, so 60 ns inflated its floor to fake a match (1.03× → 0.77× corrected).
A second bug surfaced here too: CZ node order differs between the standardized and native cals,
so 14 pairs were silently falling back to 60 ns until the lookup was made order-independent.

**The 116 that aren't coherence-limited.** 102 sit well below their floor (genuine control,
crosstalk, or TLS error that a T1/Tφ model correctly does not predict), and 14 over-predict (the
calibration-consistency effect below). For the lower-bound pairs, `gradpulse` reports the floor
and never a fudged match.

**The scatter is not the optimizer.** Re-optimizing the floor for 40-41 and 78-87 at 16 seeds,
versus the sweep's 2, gave no improvement; 3× the iterations moved it under 0.03%
(`examples/cepheus/cepheus_convergence_check.py`). So (2 seeds, 400 iters) is converged and the
spread is real, not under-optimization.

**The real ceiling is calibration self-consistency.** The headline σ paired snapshot floors with
live error bars. Re-validating six pairs on a single fresh calibration (T1/T2, measured error,
and error bar all from one pull) gives median 1.28σ with one 5.69σ outlier, 40-41
(`examples/cepheus/cepheus_consistent_recheck.py`). The cause is in the data, not the model: on that pull
q41's T2 reads 0.7 µs (it was 1.4 µs in the snapshot), so the floor is 4.25%, yet the measured CZ
is 1.31%, *below* its own coherence floor, which is impossible for simultaneous measurements. The
device measures T1/T2 and the CZ benchmark at different times and drifts ~2× over hours (a free
re-pull moved the same pairs −45% to +165%). Because `gradpulse` never lets a gate beat its
coherence limit, it flags inconsistent calibration entries instead of absorbing them.

**Why a lower bound and not 1×, like the literature anchors.** The cal exposes only idling-point
T1/T2, not the gate-effective values Sung and Marxer used. Gate time is now correct, so the
residual is purely idle-vs-gate-effective coherence, which Braket does not publish. A missing
input, not a model error, and we don't paper over it with a fudge factor.

---

## 3. Literature anchors

Three independently characterized devices from three groups (MIT, IQM, IBM), shipped as cited
JSON in `examples/anchors/`.

| device | gate | measured EPG | gradpulse | ratio | floor | authors |
|---|---|---|---|---|---|---|
| Sung 2021 (PRX 11, 021058) | 60 ns tunable-coupler CZ | 2.4e-3 | 2.4e-3 | 0.99× | GRAPE | "close to T1 limit" |
| Marxer 2023 (PRX Q 4, 010314) | 33 ns CZ | 1.9e-3 | 1.93e-3 | 1.01× | GRAPE | "mostly coherence limited" |
| Stehlik 2021 (PRL 127, 080505) | 130 ns CZ (pair 11) | 4.9e-3 | 5.1e-3 | 1.05× | analytic | error rises w/ t_g "due to decoherence" |

Marxer's own coherence limit is 1.7e-3; `gradpulse`'s analytic estimate is 1.75e-3. Sung/Marxer
use the papers' gate-effective T1/T2 (GRAPE floor). Stehlik publishes T1/T2/gate-time but not
per-pair freq/anharm, so it is judged on the closed-form analytic floor (`floor_method:
"analytic"`). Pair 11 is selected by **physics** (shortest coherence + longest gate = the only
decoherence-dominated pair), not by its ratio.

**Stehlik breadth (the no-selection result).** Across all 11 published Stehlik pairs gradpulse's
coherence floor is a one-sided lower bound: floor ≤ measured on **11/11** (median **0.37×**),
saturating (1.05×) only on pair 11. The fast, well-coherent gates sit at 0.2-0.4×: gradpulse
correctly attributes most of their error to coherent/control sources, matching the authors' own
statement that short-gate error is loss of adiabaticity, not decoherence. This reproduces the
paper's error-budget structure on a peer-reviewed device, complementing the live 160-pair Cepheus
study (`examples/stehlik_predict_vs_measured.py`).

---

## 4. Triple-solver cross-check (simulation)

The headline CZ process fidelity is 0.99033, agreed by the optimizer, an independent NumPy
Liouvillian, and QuTiP. The Liouvillian gives 0.99032921 against the blessed 0.99032899, a gap of
2.2e-7 that is the first-order Trotter splitting error. Driving dt → 0 with Richardson
extrapolation brings the independent solvers to ~1e-13, which shows the operating-point gap is
discretization rather than a disagreement between models.

---

## 5. RB resolution study (simulation)

`examples/cepheus/cepheus_irb_resolution_study.py` runs Monte Carlo over the real 11520-element 2-qubit
Clifford group. It reproduces the canary: length-1 survival 0.73 modeled vs 0.74 measured, depth
128 survival 0.21 vs 0.24, with a readout-induced asymptote of 0.197 rather than 1/d = 0.25.

For the true r_CZ ≈ 0.406%, the best design is depth-128 with a free asymptote (0.40% ± 0.10%);
the long sequences pin the asymptote that asymmetric readout otherwise biases. A fixed-1/d fit is
tighter but biased +0.10%. The default run (7 sequences × 500 shots, ~$57) recovers 0.39% ± 0.13%.

---

## 6. Decoherence in the loop (simulation)

- Optimizing coherently and multiplying by e^(−t_g/T) over-predicts the delivered CZ fidelity by
  ~3e-3; the in-loop objective reports what it delivers. The pulse-shaping edge itself is small
  (~3e-4) because the CZ is decoherence-limited, and widens for leaky gates.
- The dephasing-robust objective buys +2.27e-2 quasi-static fidelity for −1.43e-3 nominal.
- The filter-function variant matches but does not beat the quasi-static target in the 1/f regime.
  An honest negative, reported as one.
- Leakage in the loop (`examples/leakage_in_the_loop.py`) maps the "widens for leaky gates" point
  directly: sweeping a cross-resonance gate's duration and splitting each optimized gate's error
  with `error_budget` gives a clean crossover. Long gates stay coherence-limited (the floor tracks
  the true error to within run-to-run scatter); fast gates turn leakage-limited, where coherent
  leakage dwarfs the decoherence floor and the coherence formula under-predicts the true error by
  ~60x (110 ns: 6.1e-2 true vs a 1.0e-3 floor). Triple-solver clean (NumPy Liouville vs QuTiP,
  d=5e-8).

---

## 7. Level B: running a gradpulse-designed pulse on silicon (first light; GO not yet reached)

Sections 1-3 compare a gradpulse *prediction* to a device *measurement*. Level B is the harder
experiment: bind a gradpulse-**designed** pulse to the device's CZ frame and benchmark it. A
gradpulse pulse has **run on the device**; it does **not yet match native**, and the analysis
below explains exactly why and what would close the gap.

**Pulse-control first light (done).** A gradpulse pulse plays on Cepheus: `CZ_BENCH` pulse_gate
inside the RB verbatim box, accepted and executed (pair 16-25). A length-1 native-shape canary
measured **survival 0.46** vs native's ~0.76, so open-loop transfer is currently **well below
native**, as expected. Wiring (`build_bench_cz_pulse_sequence`, anchored to the device CZ flux
peak) and serialization are offline-validated (`verify_levelb_offline`, `tests/test_braket_bridge.py`).

**An honest correction.** An earlier in-model design hit F_proc = 0.986, but that used the model's
**qubit-frequency-control** channels, which Cepheus (fixed-frequency transmons) does **not have**
during a CZ (its native gate is coupler flux + `shift_phase` only). Dropping them and re-optimizing
virtual-Z costs **+0.156 F_avg** (`examples/cepheus/cepheus_coupler_only_ceiling.py`): the hardware-realizable
coupler-only ceiling is ~0.69 with *representative* coupler params. A coupler-frequency sweep
(`cepheus_coupler_param_sweep.py`) shows coupler-only reaches ~0.89 at a lower coupler frequency, so
the cap is the *unmeasured* coupler params, not the control set; native itself (coupler-only) is
~0.994, proving the device supports it.

**The 0.69 was a wrong-frequency artifact: faithful params reach 0.94.** The ~0.69 above used a
representative coupler *above* the qubits (6.80 GHz). A Rigetti 2026 adiabatic-CZ paper (PROTOTYPE,
**not** confirmed Cepheus-1-108Q) places the tunable coupler *below* the qubits: idle 2.644 GHz bare,
tuning up ~978 MHz. With that complete, self-consistent set (g₁c=96.2, g₂c=83.9, g₁₂=3.96 MHz, qubit
anharm −227/−221, coupler −178), the faithful model (`examples/cepheus/cepheus_faithful_model.py`, Cepheus
q16/q25 + prototype coupler) reaches **F_avg 0.940, 0.4% leakage**, a clean GO-capable gate. So 0.69
was the wrong coupler frequency, not a real ceiling. A sensitivity sweep
(`cepheus_coupler_sensitivity.json`) shows **coupling g is the high-sensitivity unknown** (g=60→0.65
vs 90→0.92), so the prototype values are a strong *plausibility* floor but the Cepheus-exact g still
needs `RUN_SWEEPS`/Rigetti. (Verified: those Hamiltonian params are structurally absent from
`device.gate_calibrations`; the coupler is element 140 on a baseband flux frame, no drive frame.)

**The route to matching native, validated in simulation.** On-device closed-loop calibration:
warm-start from the native shape, then tune flux **scale** + the two **virtual-Z** phases against
measured RB. Two non-obvious fixes were decisive (`examples/cepheus/cepheus_closed_loop_cal.py`): select on
**max interleaved survival**, not the RB fit (the fit is biased +0.40 for near-depolarized gates, so
argmax-of-fit picks the *worst* gate); and sweep a **joint 2-D virtual-Z grid**, not sequential 1-D
(the two phases are coupled). With both, the staged cal recovers **91%** of the open-loop→ceiling gap
under realistic shot noise vs 5% for naive Nelder-Mead/sequential. The tool implements it
(`run_irb_on_braket.py` Stage 1 peak + Stage 2 joint-2-D virtual-Z, max-survival). **Not yet run on
hardware.** Simulation validates the *method*, not the on-device number.

**gradpulse can match native, not beat it.** Native is coherence-limited (§1-2). The only pulse
lever to beat it is a faster gate, and a duration sweep (`cepheus_speed_headroom.py`) shows **no
speed headroom**: shorter gates are worse (the coupling can't complete the swap faster), so beating
native needs better coherence or stronger coupling: hardware, not a cleverer pulse. This reconfirms
§2's thesis from the speed angle.

**Making Level B predictive (the fixable caveat).** §2 showed gradpulse predicts measured CZ error to
0.42σ given accurate coherence params; the only reason the Level-B number isn't predictable is the
guessed coupler. `examples/cepheus/cepheus_coupler_characterization.py` builds the measurement that fixes
that: swap-spectroscopy of the |11⟩-|02⟩ avoided crossing (prep |11⟩ → coupler flux(amp,dur) →
P|11⟩, binary readout suffices). The on-device circuit serializes; the swap-rate fit recovers the
coupling to <0.3 MHz on synthetic data. Running it (~$221) yields the coupler params → re-optimize
coupler-only → predictive Level-B at the §2 standard.

---

## 8. Cost

| item | cost |
|---|---|
| one canary task (100 shots) | $0.34 |
| spent so far on hardware (all canaries) | ~$8 |
| **Level-A validation run** (our own native-CZ IRB, 112 circuits × 500 shots) | **~$57** |
| coupler swap-spectroscopy chevron (192 circuits × 2000 shots) | ~$221 |
| full Level-B staged cal (Stage 1 peak + Stage 2 joint-2-D virtual-Z) | ~$280-310 |

The 160-gate sweep, the cal rehearsals, the speed sweep, and the spectroscopy fit-validation are all
**free** (simulation). Only task submission costs money. **Level-B GO (match native) requires the
coupler measurement + cal, ~$500 total, well beyond a $100 budget.** Level A (gradpulse predicting
our own measured native CZ) is the bulletproof result and fits in ~$57.

**Bottom line.** From T1/T2 alone, `gradpulse`'s decoherence floor is a one-sided lower bound on
the measured CZ error: across all 160 Cepheus pairs it sits at or below measured (within the
measurement's RB uncertainty) on 150, median 0.66×. Where a gate is coherence-limited the floor
saturates the measurement: on Cepheus, a median 0.42σ across the 44 saturation pairs (that subset
is defined by the saturation, so it gauges tightness, not independent accuracy), and on two
published devices 0.99× and 1.01× at gate-effective inputs. The accuracy is bounded by the device's
measurement and calibration, not by the model.

---

## 9. Pair 81-90, crosstalk, and what Braket exposes (2026-06-21)

### What the Braket typed API exposes, and what it does not

Read calibration via the typed accessors, not `json.loads(device.properties.json())` dict-digging:
`dev.properties.standardized` (`oneQubitProperties` / `twoQubitProperties`),
`dev.properties.provider.specs` (QCS ISA: `architecture / benchmarks / instructions / name`),
`dev.topology_graph` (networkx).

- **Exposed per qubit:** T1, T2, readout, isolated 1Q RB, **simultaneous 1Q RB**.
- **Exposed per edge:** CZ interleaved RB only (there is **no** simultaneous-2Q RB).
- **Not exposed anywhere:** anharmonicity, coupler frequency, qubit-coupler g. The Level-B
  Hamiltonian gap is therefore **structural, not an API miss**, confirmed by enumerating the
  typed objects. Qubit frequencies live only in the pulse charge frames (~4.65 / 4.81 GHz on
  16/25; the coupler frame is baseband 0).

### Pair 81-90: a better gate, but control-limited

Today's calibration (2026-06-21):

| | Q81 | Q90 |
|---|---|---|
| T1 | 54.5 µs | 53.9 µs |
| T2 | 25.7 µs | 43.2 µs |
| readout | 0.953 | 0.978 |
| CZ fidelity (interleaved RB) | 0.99500 (error 0.500%) | |

This is a top-3%-fidelity pair with ~2× the coherence of 16-25. gradpulse's coherence floor here is
**0.193% (analytic) / 0.212% (GRAPE)**, both ~0.4× the measured 0.500%. So 81-90 is
**control-limited**: its coherence is so good that ~60% of its (small) error is control/calibration,
which the coherence model correctly does **not** predict (an honest lower bound). The corollary: a
clean predict-then-measure *tight* match needs a *coherence-limited* pair (floor ≈ measured), not
81-90. A native-CZ canary on 81-90 (length-1 interleaved, 100 shots) gave survival **0.890**, 2-bit
keys, clean; pipeline validated on the new pair, device healthy.

### Crosstalk: measured vs gradpulse

The **simultaneous 1Q RB** is free, already-measured crosstalk: Q81 isolated 0.99941 vs
simultaneous 0.99886 = **5.5×10⁻⁴** crosstalk hit.

A tunable coupler is *designed* to null static ZZ at its idle sweet spot, and Rigetti's paper
reports the architecture reaches **ZZ ≈ 0 at idle**; the public 27.3 kHz figure is from a
*deliberately shifted* idling point used for benchmarking, not the operating point. So at the real
operating point the static-ZZ contribution is **~0**, and gradpulse's static-ZZ spectator model
(machine-precision validated against full 27-D QuTiP) correctly predicts **~0** static-ZZ
crosstalk. That is the model being *right*: it captures that a well-tuned coupler has ~no static ZZ.

The measured 5.5×10⁻⁴ is therefore **almost entirely dynamic** crosstalk (driven-spectator +
microwave/addressing crosstalk during simultaneous operation), a mechanism *no* static model
(gradpulse included) covers. gradpulse remains an honest lower bound on total crosstalk: the static
part it models is ~0 at the operating point, and the residual is dynamic and out of scope. (Feeding
the model the 27.3 kHz *shifted-point* value gives ~1.6×10⁻⁴, but that ZZ is not the operating
value; at ZZ≈0 the static contribution is ~0, not a "30% match.") Crosstalk is also **dormant in
isolated CZ RB** (spectators sit in |0⟩), so it does not explain the isolated 81-90 gap. That gap
is miscalibration.

### Run options to prove gradpulse on our own data

The strong, clean proof is **predict-then-measure the native CZ on a coherence-limited pair**
(tight match, our own measurement, pre-registered). Costs at AWS-verified pricing
($0.30/task + $0.000425/shot):

| run | circuits × shots | cost |
|---|---|---|
| isolated CZ IRB, one pair (validation-grade) | 112 × 500 | $57.40 |
| + canary | 1 × 100 | $0.34 |
| simultaneous-1Q-RB (already measured) | n/a | $0.00 |

A **leaner config** (fewer seeds/shots/lengths) brings this down substantially while still
resolving a ~0.5% gate; the cost-vs-resolution tradeoff is characterized in
`examples/cepheus/cepheus_irb_resolution_study.py`, and the resolving config must be confirmed by a
pre-flight resolution check before spending (the "don't under-resolve" rule). The
simultaneous-CZ-RB add-on is **deliberately dropped**: with ZZ≈0 at the operating point gradpulse
correctly predicts ~0 *static* crosstalk, so a simultaneous run would measure *dynamic* crosstalk
that gradpulse does not model. That is an out-of-scope mechanism, not a gradpulse validation.

---

## 10. The Level-A run on 16-25, and why interleaved RB read 2.5× high (2026-06-21)

We ran the validation: pair **16-25**, native-CZ interleaved RB, 112 circuits × 500 shots, **$57**.
Pre-registered before submission: gradpulse **0.51%** (analytic) / 0.61% (GRAPE); device spec
**0.49%**; expected band 0.49-0.54%. The run completed cleanly (both gating canaries passed; the
length-128 / 2410-gate depth canary confirmed the verbatim box was preserved, survival 0.36 not
collapsed). **Measured naive interleaved r_CZ = 1.25% (free asymptote) / 1.01% (fixed-1/d)**,
2.0-2.5× the prediction *and* the device's own published spec. The fits are tight (σ≈0.16%), so it
resolved a genuinely different value, not noise.

**This is NOT the [§-failure mode of an earlier $33 scout]** (broken circuit → flat/garbage decay).
Here the reference arm is a *clean, spec-matching RB measurement*: the circuits, the
H/S→RX/RZ/CZ decomposition, and the device are all correct. The proof and the root cause:

**The reference arm matches spec → the device CZ is ~0.5%, exactly as gradpulse predicted.**
Using the *real* per-Clifford gate counts from the 11520-element group (**1.88 CZ + 4.03 RX**, RZ
virtual = zero-error on Rigetti):

| quantity | value |
|---|---|
| reference RB error/Clifford, measured | **1.248%** (per-seed robust: 1.25% ± 0.22%, n=7) |
| predicted from spec (1.88·0.49% + 4.03·0.07%) | **1.208%** → ratio **1.04×** |
| ⇒ implied **embedded** CZ error | **~0.51%** (≤0.66% even if 1Q error = 0) |
| gradpulse first-principles prediction | 0.51% · device spec 0.49%, **all three agree** |

So the **same `cz q[16] q[25]` instruction** reads ~0.51% when embedded in random Cliffords but
1.25% when measured by interleaving. The 2.5× lives entirely in the **interleaved estimator**, not
the gate.

**Mechanism: eliminated three Markovian suspects in gradpulse's own leakage-aware
`interleaved_rb`, then found the real one in the circuit structure:**

| hypothesis | test | verdict |
|---|---|---|
| coherent error (cond-phase / single-Z) | 81×81 IRB, same true infidelity | recovered 0.98-1.00× → **no** |
| leakage to \|2⟩ | sweep leak at fixed 0.49% in-subspace | inflates the **reference** arm *more* (2.27× at 1% leak) than interleaved; hardware reference is clean → **no** |
| finite seeds (7, disjoint ref/int draws) | pure-depol 0.49%, n=7, 60 trials | r_CZ = 0.492% ± 0.054%, never ≥1.0% → **no** |

No per-gate (Markovian) error model reproduces "clean reference + 2× interleaved", and *that
elimination is the evidence*: the effect is **non-Markovian / context-dependent**, the one class a
per-gate superoperator simulation cannot represent by construction. The smoking gun is in the
circuits themselves: interleaving a CZ after every Clifford creates **9× more back-to-back CZ
adjacencies** (11.8% of CZs vs 1.3% in the reference). Back-to-back flux pulses with no
single-qubit gate / settling between them error more than well-separated CZs (residual flux
distortion, coupler not returned to idle), a context effect interleaved RB conflates into "the
gate error." The inserted CZ therefore *measures* ~1%+ even though the isolated gate is ~0.5%.

**What this means.** gradpulse's isolated-CZ prediction (0.51%) is **validated** by the reference
arm, the spec, and the back-calculated embedded CZ, three independent anchors at ~0.5%. The naive
interleaved number is a documented interleaved-RB fragility (context-dependence), not a gate fault,
device drift, or code bug. The $57 bought (a) a real, our-own-data hardware confirmation of the CZ
error and (b) a genuine methods finding: **naive interleaved RB via Braket binary readout
overestimates this CZ ~2.5×, and gradpulse's Markovian cross-checks correctly localize the cause to
circuit context.** *Honest caveat:* the 9× adjacency jump is measured and the excess error of
back-to-back flux pulses is established hardware physics, but the per-pair excess was not directly
measured here; a clean isolated-CZ interleaved number on this device would need context control
(buffer/pad the inserted CZ), i.e. another run.

**The fix (offline-verified, hardware-unverified).** `to_braket_rb_circuit(..., buffer_bench_cz=True)`
(exposed as `run_irb_on_braket.py --buffer-bench-cz`) wraps each benchmarked CZ in `barrier`s
(native on Cepheus, identity, scheduling-level) so it is never abutted to a Clifford's own CZ.
Offline-verified: benchmarked-CZ adjacency drops **11.4% → 0.0%** in the serialized circuits, the
default is byte-identical (all prior runs/tests unchanged), and the noiseless pipeline still returns
to |00⟩ (regression: `test_buffer_bench_cz_isolates_benchmarked_gate_and_is_optin`). Whether it
recovers the isolated ~0.5% **on silicon** is unverified: the effect is non-Markovian (simulation
cannot confirm it) and any residual may be flux-predistortion-limited (the verbatim box bypasses
predistortion). It is the right next experiment, not a closed loop. The load-bearing validation does
not depend on it: the reference arm already confirms the gate at ~0.5%.
