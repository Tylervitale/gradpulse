"""gradpulse.literature - data-driven validation against published hardware gates.

gradpulse reports a *simulator* fidelity. The QuTiP and Liouvillian cross-checks
prove that number is computed correctly, but they cannot prove the *model* matches a
real device -- two independent solvers of the same model agree even if the model is
wrong. The missing check is to feed gradpulse the published parameters of a real,
characterized two-qubit gate and show its decoherence-limited floor reproduces the
device's published coherence budget.

This module makes that check **data-driven** instead of hardcoded. A published gate is
a single JSON *anchor* carrying both the device physics *and* the validation metadata
-- measured fidelity, the optional T1 limit, the gate timing, the provenance
(citation + DOI), and one load-bearing flag, ``coherence_limited``. Adding a device is
then dropping a JSON file into ``examples/anchors/``; no Python is written, and the
ground truth lives in curated data with its citation attached, not in code where a
typo silently becomes a "validation".

The honesty that motivates the whole exercise is encoded in the schema, not just the
prose:

* A pure T1/T_phi Markovian floor is a **lower bound** on a measured error-per-gate.
  It omits residual coherent control error, classical and ZZ crosstalk, leakage, and
  non-Markovian noise -- all of which a hardware RB number contains.
* So the floor equals the measured number **only when the gate is itself
  coherence-limited**. ``coherence_limited: true`` asserts equality (floor ~ measured);
  ``coherence_limited: false`` asserts only the inequality the physics guarantees
  (floor <= measured) and reports the residual as un-modelled error. A device whose
  measured error *exceeds* its own coherence budget is handled honestly -- the model
  correctly under-predicts it -- rather than dropped for "not matching".

Two pure (torch-free, fast) entry points do the work, so the judgment is unit-testable
without running GRAPE:

* :func:`load_anchor` - parse + schema-validate a JSON anchor (rejects one with no
  citation, no measured fidelity, or no ``coherence_limited`` flag).
* :func:`anchor_to_profiles` - build the three coherence variants (none / T1-only /
  full) by reusing :meth:`ParametricCouplerProfile.from_calibration`.
* :func:`judge` - given the three optimized fidelities, decide pass/fail under the
  anchor's ``coherence_limited`` claim and return a structured verdict.
* :func:`format_report` - render a verdict as text.

The slow part -- running real GRAPE to get those three fidelities -- lives in
``examples/validate_against_literature.py``, which imports this module, so the data and
the judgment ship in the package while the multi-minute optimization stays an example.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

from .profiles import ParametricCouplerProfile


# Reported metrics use F_avg throughout (d=4) since that is how papers quote measured
# fidelity/EPG; mixing an F_proc deficit against an F_avg EPG under-counts by 4/5.
def f_avg(f_proc: float) -> float:
    return (4.0 * f_proc + 1.0) / 5.0


def effective_t2_ns(t1_ns: float, tphi_exp_ns: float) -> float:
    """The Markovian ``T2`` from a relaxation time and an *exponential* pure-dephasing time.

    Textbook relation ``1/T2 = 1/(2 T1) + 1/T_phi``. Some papers (and, for a tunable
    coupler, the gate-*effective* coherence analysis) report ``T1`` and an exponential
    ``T_phi`` separately rather than ``T2``. An anchor may then publish the paper's actual
    ``t1_ns`` and ``tphi_exp_ns`` and let the harness derive ``T2`` here -- so no
    hand-computed ``T2`` sits in the data file, only quantities quoted from the source.

    This represents only the *exponential* (Markovian) part of dephasing. A Gaussian /
    non-Markovian component (a quoted ``T_phi,Gauss``) is deliberately NOT folded in: a
    single-``T2`` Lindblad model cannot represent it, and pretending otherwise via an
    effective rate would hide a real model limitation. Its (small) contribution to a
    device's coherence limit should be quantified separately and, if it matters, treated
    with the non-Markovian tools (``quasi_static_fidelity`` / filter functions).
    """
    return 1.0 / (1.0 / (2.0 * float(t1_ns)) + 1.0 / float(tphi_exp_ns))


def analytic_coherence_limit_epg(profile, gate_ns: float) -> float:
    """Closed-form coherence-limited average gate error (F_avg terms) for the pair.

    A GRAPE-independent prediction straight from the device T1/T2 and the gate duration::

        1 - F_avg  ~=  (2 t_g / 5) * sum_q ( 1/T1_q + 1/T_phi_q ),
        1/T_phi = 1/T2 - 1/(2 T1)   (floored at 0; T2 > 2 T1 would give negative dephasing).

    The ``2 t_g / 5`` prefactor is the two-qubit (d=4) coherence limit. It is validated
    empirically: for the Sung anchor it returns 2.33e-3, on top of the GRAPE decoherence
    floor and the measured 2.4e-3 (``tests/test_literature.py``). Because it needs no
    optimization, it is an independent sanity bound on the GRAPE floor -- if the two
    disagree by more than GRAPE scatter, something is wrong with one of them.
    """
    def _per_qubit(t1, t2):
        inv_tphi = max(0.0, 1.0 / t2 - 1.0 / (2.0 * t1))
        return 1.0 / t1 + inv_tphi

    s = (_per_qubit(profile.t1_ns_q1, profile.t2_ns_q1)
         + _per_qubit(profile.t1_ns_q2, profile.t2_ns_q2))
    return (2.0 * gate_ns / 5.0) * s


# A coherence-limited floor should *equal* measured; band around 1.0x absorbs the fact
# that neither number is exact (the Sung anchor lands at ~0.99x). Override via
# validation.ratio_band.
_DEFAULT_RATIO_BAND = (0.5, 1.5)
# coherence_limited=false: the floor must not EXCEED measured (a lower bound); slack
# absorbs optimizer/numerical noise. ratio > 1+slack is unphysical (predicts more
# decoherence error than the device's total error).
_LOWER_BOUND_SLACK = 0.10
# The comparison is only meaningful if the gate closes without decoherence (so the gap
# IS decoherence, not optimizer failure); flag inconclusive otherwise.
_GATE_CLOSES_FRACTION = 0.25


# ----------------------------------------------------------------------------
# Load + schema validation
# ----------------------------------------------------------------------------
def load_anchor(path) -> dict:
    """Parse and schema-validate a literature anchor JSON.

    The required fields are exactly the ones whose absence would make a "validation"
    meaningless or unattributable, so the loader -- not a reviewer's vigilance -- is
    what guarantees every anchor is (a) attributable to a publication and (b) carries
    the measured number and the coherence claim it is being judged against::

        {
          "name": "Sung 2021 CZ (60 ns)",
          "architecture": "parametric",                 # only "parametric" is supported
          "provenance": {"citation": "...", "doi": "...", "url": "...", "notes": "..."},
          "qubits": {"0": {"freq_ghz":.., "anharm_ghz":.., "t1_ns":.., "t2_ns":..},
                     "1": {...}},                        # from_calibration schema; a qubit
          #   may give "tphi_exp_ns" instead of "t2_ns" (T2 derived: 1/T2=1/(2 T1)+1/T_phi),
          #   plus an optional "tphi_gauss_ns" recorded but unused (non-Markovian).
          "qubit_pair": [0, 1],
          "device":   {"g_max_mhz": 45.0, "omega_max_mhz": 50.0},   # profile overrides
          "gate":     {"kind":"cz","n_slices":60,"dt_ns":1.0,"bandwidth_mhz":200.0,
                       "use_drag":true,"drag_order":2,"n_channels":4},
          "validation": {"measured_f_avg":0.9976, "t1_limit_f_avg":0.9985,
                         "coherence_limited":true, "ratio_band":[0.5,1.5]}
        }

    Returns the parsed dict (with ``_path`` added for diagnostics). Raises ``ValueError``
    with a specific message on any missing/contradictory required field.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        anchor = json.load(fh)
    if not isinstance(anchor, dict):
        raise ValueError(f"{path.name}: anchor must be a JSON object.")

    def _require(container, key, where):
        if key not in container or container[key] in (None, ""):
            raise ValueError(f"{path.name}: missing required field '{where}{key}'.")
        return container[key]

    _require(anchor, "name", "")
    arch = anchor.get("architecture", "parametric")
    if arch != "parametric":
        # The judging logic and from_calibration both assume the parametric-coupler
        # profile. Fail loudly rather than silently mis-build a different architecture.
        raise ValueError(
            f"{path.name}: architecture '{arch}' is not supported by the literature "
            "harness yet (only 'parametric'). Build its profile explicitly instead."
        )

    prov = _require(anchor, "provenance", "")
    if not isinstance(prov, dict):
        raise ValueError(f"{path.name}: 'provenance' must be an object.")
    _require(prov, "citation", "provenance.")  # an anchor with no citation is not data.

    qubits = _require(anchor, "qubits", "")
    if not isinstance(qubits, dict) or len(qubits) < 2:
        raise ValueError(f"{path.name}: 'qubits' must map >=2 qubit indices to params.")
    pair = _require(anchor, "qubit_pair", "")
    if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
        raise ValueError(f"{path.name}: 'qubit_pair' must be [q1, q2].")

    val = _require(anchor, "validation", "")
    if not isinstance(val, dict):
        raise ValueError(f"{path.name}: 'validation' must be an object.")
    measured = _require(val, "measured_f_avg", "validation.")
    if not (0.0 < float(measured) < 1.0):
        raise ValueError(
            f"{path.name}: validation.measured_f_avg={measured} must be a fidelity in "
            "(0, 1) (an F_avg, not an error)."
        )
    if "coherence_limited" not in val or not isinstance(val["coherence_limited"], bool):
        raise ValueError(
            f"{path.name}: validation.coherence_limited (true/false) is required -- it "
            "selects whether the floor is compared for equality or as a lower bound."
        )
    band = val.get("ratio_band")
    if band is not None and not (isinstance(band, (list, tuple)) and len(band) == 2
                                 and band[0] <= band[1]):
        raise ValueError(f"{path.name}: validation.ratio_band must be [low, high].")

    fm = val.get("floor_method", "grape")
    if fm not in ("grape", "analytic"):
        raise ValueError(
            f"{path.name}: validation.floor_method must be 'grape' (the GRAPE decoherence "
            "floor) or 'analytic' (the closed-form coherence limit, for a device that "
            "publishes T1/T2 and gate time but not the per-pair frequencies/anharmonicities "
            "a GRAPE floor would need)."
        )

    # A qubit may publish tphi_exp_ns instead of t2_ns; derive T2 here so the data file
    # holds only source-quoted quantities. Explicit t2_ns always wins.
    for qi, node in qubits.items():
        if not isinstance(node, dict):
            continue
        if node.get("t2_ns") is None and node.get("tphi_exp_ns") is not None:
            t1 = node.get("t1_ns")
            if t1 is None:
                raise ValueError(
                    f"{path.name}: qubit {qi} gives tphi_exp_ns but no t1_ns to derive "
                    "T2 from (1/T2 = 1/(2 T1) + 1/T_phi)."
                )
            node["t2_ns"] = effective_t2_ns(t1, node["tphi_exp_ns"])

    anchor["_path"] = str(path)
    return anchor


# ----------------------------------------------------------------------------
# Build the three coherence variants from the anchor
# ----------------------------------------------------------------------------
# 1e8 ns = 100 ms is ~1000x a typical transmon T1, so relaxation is negligible over a
# ~60 ns gate; T2 = 2*T1 zeroes pure dephasing.
_INF_T1_NS = 1.0e8


def anchor_to_profiles(anchor: dict):
    """Return ``(prof_none, prof_t1, prof_full)`` for the three coherence assumptions.

    Only the *measured* T1/T2 (the physics) lives in the anchor; the no-decoherence and
    T1-only variants are **derived** from it, so there is no redundant, separately-
    editable copy of the coherence numbers to drift:

    * ``prof_full`` - the device's measured T1 and T2.
    * ``prof_t1``   - measured T1, ``T2 = 2*T1`` (pure dephasing removed, relaxation kept).
    * ``prof_none`` - ``T1 -> inf``, ``T2 = 2*T1`` (both removed; the gate-closes check).

    Device knobs that ``from_calibration`` does not parse (``g_max_mhz``,
    ``omega_max_mhz``, ``n_levels``, ...) are passed through from ``anchor['device']``
    as profile overrides.
    """
    pair = (int(anchor["qubit_pair"][0]), int(anchor["qubit_pair"][1]))
    overrides = dict(anchor.get("device", {}))

    prof_full = ParametricCouplerProfile.from_calibration(anchor, pair, **overrides)
    t1a, t1b = prof_full.t1_ns_q1, prof_full.t1_ns_q2

    # T2 == 2*T1 exactly is physical (1/T_phi == 0), so neither the unphysical-T2 nor
    # the representative-defaults warning fires on replace().
    prof_t1 = replace(prof_full, t2_ns_q1=2.0 * t1a, t2_ns_q2=2.0 * t1b)
    prof_none = replace(prof_full,
                        t1_ns_q1=_INF_T1_NS, t1_ns_q2=_INF_T1_NS,
                        t2_ns_q1=2.0 * _INF_T1_NS, t2_ns_q2=2.0 * _INF_T1_NS)
    return prof_none, prof_t1, prof_full


def gate_config(anchor: dict) -> dict:
    """Optimizer/gate settings for this anchor, with the package defaults filled in.

    Kept separate from the profile so ``literature`` stays torch-free: the example
    driver reads this dict and constructs the (torch-backed) optimizer.
    """
    g = dict(anchor.get("gate", {}))
    g.setdefault("kind", "cz")
    g.setdefault("n_slices", 60)
    g.setdefault("dt_ns", 1.0)
    g.setdefault("bandwidth_mhz", 200.0)
    g.setdefault("use_drag", True)
    g.setdefault("drag_order", 2)
    g.setdefault("n_channels", 4)
    # double precision resolves the fine ~1e-3 decoherence error well below the
    # single-precision ~1e-6 floor; cost is negligible at this size.
    g.setdefault("precision", "double")
    g.setdefault("n_seeds", 4)
    g.setdefault("iterations", 500)
    return g


# ----------------------------------------------------------------------------
# Judge: does the decoherence floor reproduce the published budget?
# ----------------------------------------------------------------------------
def judge(anchor: dict, f_coh: float, f_t1: float, f_full: float,
          analytic_epg: Optional[float] = None) -> dict:
    """Decide whether the decoherence floor reproduces the anchor's published budget.

    Pure and torch-free: it takes the three *process* fidelities from independent
    optimizations (no decoherence / T1-only / full T1+T_phi) and returns a structured
    verdict. All quantities are reported in F_avg / EPG terms (see :func:`f_avg`).

    The comparison is selected by ``anchor['validation']['coherence_limited']``:

    * **true**  - the floor should *equal* measured. PASS iff the ratio
      ``decoherence_error / measured_error`` lands in ``ratio_band`` (default
      ``(0.5, 1.5)``; the Sung anchor gives ~0.99x).
    * **false** - the floor is only a *lower bound*. PASS iff ``ratio <= 1 + slack``;
      the residual ``measured_error - decoherence_error`` is reported as the
      un-modelled (coherent / crosstalk / leakage / non-Markovian) error. ``ratio > 1``
      would mean predicting more decoherence error than the device's total error -- a
      real model failure -- so it FAILS.

    Independently, ``gate_closes`` records whether the no-decoherence residual is small
    versus the measured error; if it is not, the verdict is marked inconclusive because
    a gap could be optimizer error masquerading as decoherence.

    ``analytic_epg`` (optional, from :func:`analytic_coherence_limit_epg`) is recorded
    for reporting as a GRAPE-independent sanity bound on the decoherence error; it is not
    a pass criterion (the GRAPE floor is the one being judged against measured).
    """
    val = anchor["validation"]
    measured = float(val["measured_f_avg"])
    coherence_limited = bool(val["coherence_limited"])

    a_coh, a_t1, a_full = f_avg(f_coh), f_avg(f_t1), f_avg(f_full)
    coherent_residual = 1.0 - a_coh          # optimizer quality (gate-closes check)
    dec_err = a_coh - a_full                 # gradpulse decoherence error, F_avg terms
    t1_err = a_coh - a_t1                    # the T1-only contribution
    meas_epg = 1.0 - measured                # measured error per gate
    ratio = dec_err / meas_epg if meas_epg > 0 else float("inf")

    gate_closes = coherent_residual <= _GATE_CLOSES_FRACTION * meas_epg

    if coherence_limited:
        band = tuple(val.get("ratio_band", _DEFAULT_RATIO_BAND))
        in_band = band[0] <= ratio <= band[1]
        passed = bool(in_band and gate_closes)
        unexplained = None
        claim = "equality"
    else:
        passed = bool(ratio <= 1.0 + _LOWER_BOUND_SLACK and gate_closes)
        # What the T1/T_phi model does NOT account for (only meaningful as a lower
        # bound, i.e. when measured exceeds the floor).
        unexplained = max(0.0, meas_epg - dec_err)
        band = None
        claim = "lower_bound"

    return {
        "name": anchor.get("name", "?"),
        "coherence_limited": coherence_limited,
        "claim": claim,
        "f_avg_no_decoh": a_coh,
        "f_avg_t1_only": a_t1,
        "f_avg_full": a_full,
        "coherent_residual": coherent_residual,
        "decoherence_error": dec_err,
        "t1_error": t1_err,
        "measured_f_avg": measured,
        "measured_epg": meas_epg,
        "t1_limit_f_avg": val.get("t1_limit_f_avg"),
        "ratio": ratio,
        "ratio_band": band,
        "unexplained_error": unexplained,
        "analytic_epg": analytic_epg,
        "gate_closes": gate_closes,
        "passed": passed,
    }


def judge_analytic(anchor: dict, analytic_epg: float) -> dict:
    """Judge using the GRAPE-INDEPENDENT analytic coherence floor as the floor itself.

    For a device that publishes T1/T2 and gate duration but NOT the per-pair
    frequencies/anharmonicities a GRAPE floor would need (e.g. Stehlik 2021, whose Table I
    gives coherence times and gate times across 11 pairs but no Hamiltonian parameters),
    the closed-form ``(2 t_g/5) sum_q(1/T1+1/T_phi)`` IS the honest, fully source-derived
    floor -- it depends only on published quantities. This applies the same
    ``coherence_limited`` band / lower-bound logic as :func:`judge`, but on ``analytic_epg``
    rather than the optimizer's decoherence error.

    Unlike :func:`judge` there is no ``gate_closes`` guard: with no optimization there is
    no coherent residual to measure. The analytic floor assumes the gate closes coherently
    (true for these real, published gates at their published durations); the verdict records
    ``gate_closes=None`` rather than implying an optimizer ran.
    """
    val = anchor["validation"]
    measured = float(val["measured_f_avg"])
    coherence_limited = bool(val["coherence_limited"])
    meas_epg = 1.0 - measured
    ratio = analytic_epg / meas_epg if meas_epg > 0 else float("inf")

    if coherence_limited:
        band = tuple(val.get("ratio_band", _DEFAULT_RATIO_BAND))
        passed = bool(band[0] <= ratio <= band[1])
        unexplained = None
        claim = "equality (analytic floor)"
    else:
        passed = bool(ratio <= 1.0 + _LOWER_BOUND_SLACK)
        unexplained = max(0.0, meas_epg - analytic_epg)
        band = None
        claim = "lower_bound (analytic floor)"

    return {
        "name": anchor.get("name", "?"),
        "coherence_limited": coherence_limited,
        "claim": claim,
        "floor_method": "analytic",
        "decoherence_error": analytic_epg,
        "measured_f_avg": measured,
        "measured_epg": meas_epg,
        "t1_limit_f_avg": val.get("t1_limit_f_avg"),
        "ratio": ratio,
        "ratio_band": band,
        "unexplained_error": unexplained,
        "analytic_epg": analytic_epg,
        "gate_closes": None,   # not checked in analytic mode (no optimization runs)
        "passed": passed,
    }


def _format_analytic(v: dict) -> str:
    """Render a :func:`judge_analytic` verdict (no GRAPE lines)."""
    lines = [f"\n=== {v['name']} ===",
             "  floor method                    : analytic coherence limit, GRAPE-"
             "independent",
             "                                    (2 t_g/5) sum_q(1/T1+1/T_phi) -- paper "
             "gives T1/T2/t_g, not per-pair freq/anharm"]
    lines.append(f"  decoherence-limited floor (EPG) : {v['decoherence_error']:.2e}")
    lines.append(f"  published measured              : F_avg={v['measured_f_avg']:.5f}"
                 f"   error/EPG {v['measured_epg']:.2e}")
    if v["coherence_limited"]:
        band = v["ratio_band"]
        lines.append(f"  --> floor / measured error      : {v['ratio']:.2f}x"
                     f"   [coherence-limited: expect ~1x, band {band[0]:g}-{band[1]:g}]")
    else:
        lines.append(f"  --> floor / measured error      : {v['ratio']:.2f}x"
                     f"   [lower bound: expect <1x; un-modelled "
                     f"{v['unexplained_error']:.2e} of {v['measured_epg']:.2e}]")
    lines.append(f"  VERDICT: {'PASS' if v['passed'] else 'FAIL'}")
    return "\n".join(lines)


def format_report(verdict: dict) -> str:
    """Render a :func:`judge` (or :func:`judge_analytic`) verdict as a text block."""
    v = verdict
    if v.get("floor_method") == "analytic":
        return _format_analytic(v)
    lines = [f"\n=== {v['name']} ==="]
    lines.append(
        f"  coherent floor (no decoherence) : F_avg={v['f_avg_no_decoh']:.5f}"
        f"   (optimizer residual {v['coherent_residual']:.1e};"
        f" gate {'closes' if v['gate_closes'] else 'DOES NOT close -- inconclusive'})"
    )
    t1lim = (f"  (paper T1 limit {1.0 - v['t1_limit_f_avg']:.2e})"
             if v["t1_limit_f_avg"] else "")
    lines.append(
        f"  T1-only                         : F_avg={v['f_avg_t1_only']:.5f}"
        f"   T1 error {v['t1_error']:.2e}{t1lim}"
    )
    lines.append(
        f"  full T1+T_phi (Markovian)       : F_avg={v['f_avg_full']:.5f}"
        f"   decoherence error {v['decoherence_error']:.2e}"
    )
    lines.append(
        f"  decoherence-limited floor       : F_avg={1.0 - v['decoherence_error']:.5f}"
    )
    lines.append(
        f"  published measured              : F_avg={v['measured_f_avg']:.5f}"
        f"   error/EPG {v['measured_epg']:.2e}"
    )
    if v.get("analytic_epg") is not None:
        lines.append(
            f"  analytic coherence limit        : EPG {v['analytic_epg']:.2e}"
            f"   (GRAPE-independent; (2 t_g/5) sum_q 1/T1+1/T_phi -- sanity bound on the"
            f" {v['decoherence_error']:.2e} GRAPE floor)"
        )
    if v["coherence_limited"]:
        band = v["ratio_band"]
        lines.append(
            f"  --> floor / measured error      : {v['ratio']:.2f}x"
            f"   [coherence-limited: expect ~1x, band {band[0]:g}-{band[1]:g}]"
        )
        lines.append(
            "      The authors report this gate is coherence-limited, so the T1/T_phi "
            "floor IS the measured number."
        )
    else:
        lines.append(
            f"  --> floor / measured error      : {v['ratio']:.2f}x"
            f"   [NOT coherence-limited: floor is a strict lower bound, expect <1x]"
        )
        lines.append(
            f"      un-modelled error (coherent/crosstalk/leakage/non-Markovian): "
            f"{v['unexplained_error']:.2e} of the {v['measured_epg']:.2e} measured EPG "
            "lies beyond the T1/T_phi floor."
        )
    lines.append(f"  VERDICT: {'PASS' if v['passed'] else 'FAIL'}")
    return "\n".join(lines)


def discover_anchors(directory) -> list:
    """Return sorted paths of ``*.json`` anchors in ``directory`` (non-recursive)."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.json") if p.is_file())
