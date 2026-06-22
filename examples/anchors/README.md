# Literature anchors

Each JSON file here is one **published** two-qubit gate that gradpulse's decoherence
model is validated against. `examples/validate_against_literature.py` discovers every
`*.json` in this directory, builds the device from it, runs three independent GRAPE
optimizations (no decoherence / T1-only / full T1+T_phi), and checks the decoherence
floor against the published number. The loading and judging logic is the torch-free
[`gradpulse.literature`](../../src/gradpulse/literature.py) module, unit-tested in
[`tests/test_literature.py`](../../tests/test_literature.py).

**Adding a device is dropping a JSON file here: no Python.** The point is that the
ground truth lives in cited data, not in code where a typo silently becomes a
"validation".

### Shipped anchors

| File | Device | Measured | gradpulse floor / measured |
|---|---|---|---|
| `sung_2021_cz.json` | Sung et al. 2021, *Phys. Rev. X* **11**, 021058, tunable-coupler CZ, 60 ns | 99.76% (2.4×10⁻³) | 0.99× (GRAPE) |
| `marxer_2023_cz.json` | Marxer et al. 2023, *PRX Quantum* **4**, 010314, long-distance-coupler CZ, 33 ns | 99.81% (1.9×10⁻³) | 1.01× (GRAPE) |
| `stehlik_2021_cz.json` | Stehlik et al. 2021, *Phys. Rev. Lett.* **127**, 080505, IBM tunable-coupler CZ, pair 11, 130 ns | 99.51% (4.9×10⁻³) | 1.05× (analytic) |

Three independent groups (MIT, IQM, IBM). The first two authors explicitly call their gate
coherence-limited (`coherence_limited: true`). The
Marxer anchor's `provenance.notes` documents the one subtlety, and the harness makes it
auditable rather than a buried choice: its gate parks the qubits and pulses the *coupler*,
so the coherence governing the gate is the **gate-effective** value the authors compute, not
the static idling-point measurement. The data file carries the paper's quoted
`t1_ns` + `tphi_exp_ns` and the loader derives `T2` in code (`effective_t2_ns`), so no
hand-computed `T2` sits in the file. `tests/test_literature.py` asserts that the *static*
idling times over-predict the error by ~1.7× while the effective times land at the measured
number, and quantifies the omitted Gaussian-dephasing component (<1% of the limit, which a
single-`T2` Markovian model cannot represent). Read it before adding a similar tunable-coupler
device: `tphi_exp_ns` (with optional `tphi_gauss_ns` for the record) is the supported way to
give a qubit's dephasing when a paper reports `T_phi` rather than `T2`.

The **Stehlik** anchor uses `"floor_method": "analytic"`: that paper publishes T1/T2 and gate
time for 11 pairs but **not** the per-pair frequencies/anharmonicities a GRAPE floor would
need, so the honest, fully source-derived metric is the closed-form coherence limit
`(2 t_g/5) Σ_q(1/T1+1/T_phi)`, which depends only on published quantities (gives 1.05×). The
`freq_ghz`/`anharm_ghz` in that file are representative placeholders so the profile builds; they
do not enter the analytic floor. Pair 11 is the one selected by **physics** (shortest coherence
*and* longest gate = the only decoherence-dominated pair), not by its agreement: across all 11
pairs gradpulse's floor is a strict lower bound (median 0.37×), saturating only here. That
full breadth result (the no-selection lower-bound story, complementing the live 160-pair
Cepheus study on a peer-reviewed device) is [`examples/stehlik_predict_vs_measured.py`](../stehlik_predict_vs_measured.py)
over [`examples/stehlik_2021_table1.json`](../stehlik_2021_table1.json).

## Schema

```jsonc
{
  "name": "Sung 2021 CZ (60 ns)",            // required
  "architecture": "parametric",              // only "parametric" supported today
  "provenance": {                            // required; an anchor with no citation is not data
    "citation": "Y. Sung et al., Phys. Rev. X 11, 021058 (2021)",   // required
    "doi": "10.1103/PhysRevX.11.021058",
    "url": "https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.021058",
    "notes": "Where each number comes from; quote the paper."
  },
  "qubits": {                                // required; from_calibration schema
    "0": {"freq_ghz": 4.16, "anharm_ghz": -0.220, "t1_ns": 60000.0, "t2_ns": 103000.0},
    "1": {"freq_ghz": 4.00, "anharm_ghz": -0.210, "t1_ns": 30000.0, "t2_ns": 16000.0}
    // instead of "t2_ns", a qubit may give "tphi_exp_ns" (exponential T_phi); the loader
    // derives T2 = 1/(1/(2 T1) + 1/T_phi) in code. "tphi_gauss_ns" is recorded but unused
    // (a single-T2 Markovian model cannot represent Gaussian/non-Markovian dephasing).
  },
  "qubit_pair": [0, 1],                      // required; which two qubits the gate acts on
  "device": {"g_max_mhz": 45.0, "omega_max_mhz": 50.0},   // extra ParametricCouplerProfile overrides
  "gate": {                                  // optimizer/gate settings (all have defaults)
    "kind": "cz", "n_slices": 60, "dt_ns": 1.0, "bandwidth_mhz": 200.0,
    "use_drag": true, "drag_order": 2, "n_channels": 4,
    "precision": "double", "n_seeds": 4, "iterations": 500
  },
  "validation": {                            // required
    "measured_f_avg": 0.9976,                // required; the published RB number, as F_avg
    "t1_limit_f_avg": 0.9985,                // optional; the paper's quoted T1 limit
    "coherence_limited": true,               // required; selects equality vs lower-bound (see below)
    "ratio_band": [0.5, 1.5],                // optional; PASS band for coherence_limited:true
    "floor_method": "grape"                  // optional; "grape" (default) or "analytic" -- use
                                             //   "analytic" when the paper gives T1/T2/t_g but not
                                             //   per-pair freq/anharm (judged on the closed-form floor)
  }
}
```

Units are flexible: `from_calibration` also accepts `t1_s`/`t2_s` (seconds) and
`freq_hz`/`anharm_hz`; any omitted per-qubit field keeps a representative default and is
recorded in the profile `notes`.

## The one rule that matters: `coherence_limited`

A pure T1/T_phi Markovian floor is a **lower bound** on a measured error-per-gate: it
omits residual coherent control error, ZZ/classical crosstalk, leakage, and
non-Markovian noise, all of which a hardware RB number contains.

- **`coherence_limited: true`**, the authors state the gate is at/near its coherence
  limit, so the floor should *equal* the measured number. The harness checks
  `decoherence_error / measured_error` is within `ratio_band` (default `0.5-1.5×`).
  *Only set this if the paper says so.* Sung lands at ~0.99×.

- **`coherence_limited: false`**, the gate is **not** coherence-limited (measured error
  exceeds its own T1/T2 budget). The harness then asks only that the floor be a valid
  lower bound (`ratio ≤ 1`) and reports the residual `measured − floor` as the
  un-modelled (coherent/crosstalk/leakage/non-Markovian) error. This is how a device the
  model legitimately under-predicts is handled **honestly** (reported as a lower bound)
  instead of being dropped for "not matching". A `false` anchor whose floor *exceeds*
  measured (`ratio > 1`) FAILS, because predicting more decoherence error than the
  device's total error is a real model failure.

Do **not** tune any number to make an anchor pass. If a `true` anchor lands far from 1×,
the honest fix is to recheck the published parameters or reclassify it as `false`, not
to widen the band.
