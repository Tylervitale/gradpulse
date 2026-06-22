"""Data-driven literature validation: ``gradpulse.literature``.

These tests exercise the load / build / judge machinery WITHOUT running GRAPE (the
multi-minute optimization lives in ``examples/validate_against_literature.py``), so the
scientific decision logic -- the schema guards and the coherence_limited equality-vs-
lower-bound judgment -- is fast and unit-tested:

  - a real anchor (the migrated Sung CZ) loads and reproduces the exact device numbers
    that the prior hardcoded validate_sung_cz() used (regression guard on the migration),
  - the loader REJECTS an anchor with no citation, no measured fidelity, or no
    coherence_limited flag (so an un-attributable "validation" cannot be added),
  - the three coherence variants (none / T1-only / full) are derived correctly,
  - judge() PASSES a coherence-limited floor that matches measured, FAILS one out of
    band or one where the gate did not close, and handles a NOT-coherence-limited anchor
    as a lower bound (PASS when the floor under-predicts, FAIL when it over-predicts).

Run:  pytest tests/test_literature.py
"""
import json
from pathlib import Path

import pytest

from gradpulse import ParametricCouplerProfile
from gradpulse.literature import (
    analytic_coherence_limit_epg,
    anchor_to_profiles,
    discover_anchors,
    effective_t2_ns,
    f_avg,
    format_report,
    gate_config,
    judge,
    judge_analytic,
    load_anchor,
)

ANCHOR_DIR = Path(__file__).parent.parent / "examples" / "anchors"
SUNG = ANCHOR_DIR / "sung_2021_cz.json"
MARXER = ANCHOR_DIR / "marxer_2023_cz.json"
STEHLIK = ANCHOR_DIR / "stehlik_2021_cz.json"
STEHLIK_TABLE = Path(__file__).parent.parent / "examples" / "stehlik_2021_table1.json"


# ---------------------------------------------------------------------------
# Load + schema guards
# ---------------------------------------------------------------------------
def test_sung_anchor_loads():
    a = load_anchor(SUNG)
    assert a["name"] == "Sung 2021 CZ (60 ns)"
    assert a["architecture"] == "parametric"
    assert "021058" in a["provenance"]["citation"]
    assert a["validation"]["coherence_limited"] is True
    assert a["validation"]["measured_f_avg"] == pytest.approx(0.9976)


def test_discover_finds_sung():
    found = discover_anchors(ANCHOR_DIR)
    assert SUNG.resolve() in {p.resolve() for p in found}


def _valid_anchor():
    return {
        "name": "T", "architecture": "parametric",
        "provenance": {"citation": "Some Author, Some Journal (2024)"},
        "qubits": {
            "0": {"freq_ghz": 4.1, "anharm_ghz": -0.2, "t1_ns": 50000.0, "t2_ns": 40000.0},
            "1": {"freq_ghz": 4.0, "anharm_ghz": -0.2, "t1_ns": 50000.0, "t2_ns": 40000.0},
        },
        "qubit_pair": [0, 1],
        "validation": {"measured_f_avg": 0.997, "coherence_limited": True},
    }


def _write(tmp_path, data):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_valid_minimal_anchor_loads(tmp_path):
    load_anchor(_write(tmp_path, _valid_anchor()))  # must not raise


def test_rejects_missing_citation(tmp_path):
    a = _valid_anchor()
    a["provenance"].pop("citation")
    with pytest.raises(ValueError, match="citation"):
        load_anchor(_write(tmp_path, a))


def test_rejects_missing_provenance(tmp_path):
    a = _valid_anchor()
    a.pop("provenance")
    with pytest.raises(ValueError, match="provenance"):
        load_anchor(_write(tmp_path, a))


def test_rejects_missing_coherence_flag(tmp_path):
    a = _valid_anchor()
    a["validation"].pop("coherence_limited")
    with pytest.raises(ValueError, match="coherence_limited"):
        load_anchor(_write(tmp_path, a))


def test_rejects_non_bool_coherence_flag(tmp_path):
    a = _valid_anchor()
    a["validation"]["coherence_limited"] = "yes"
    with pytest.raises(ValueError, match="coherence_limited"):
        load_anchor(_write(tmp_path, a))


def test_rejects_measured_out_of_range(tmp_path):
    a = _valid_anchor()
    a["validation"]["measured_f_avg"] = 1.5  # not a fidelity
    with pytest.raises(ValueError, match="measured_f_avg"):
        load_anchor(_write(tmp_path, a))


def test_rejects_unsupported_architecture(tmp_path):
    a = _valid_anchor()
    a["architecture"] = "cross_resonance"
    with pytest.raises(ValueError, match="architecture"):
        load_anchor(_write(tmp_path, a))


# ---------------------------------------------------------------------------
# Build: the three coherence variants. Doubles as a regression guard that the JSON
# reproduces the exact numbers the prior hardcoded validate_sung_cz() used.
# ---------------------------------------------------------------------------
def test_anchor_reproduces_legacy_sung_physics():
    a = load_anchor(SUNG)
    _, _, full = anchor_to_profiles(a)
    assert full.qubit_pair == (0, 1)
    # device frequencies / anharmonicities (legacy `base` dict)
    assert full.freq_ghz_q1 == pytest.approx(4.16)
    assert full.freq_ghz_q2 == pytest.approx(4.00)
    assert full.anharm_ghz_q1 == pytest.approx(-0.220)
    assert full.anharm_ghz_q2 == pytest.approx(-0.210)
    assert full.g_max_mhz == pytest.approx(45.0)
    assert full.omega_max_mhz == pytest.approx(50.0)
    # measured T1/T2 (legacy prof_full)
    assert full.t1_ns_q1 == pytest.approx(60000.0)
    assert full.t1_ns_q2 == pytest.approx(30000.0)
    assert full.t2_ns_q1 == pytest.approx(103000.0)
    assert full.t2_ns_q2 == pytest.approx(16000.0)


def test_coherence_variants_are_derived_correctly():
    a = load_anchor(SUNG)
    none, t1, full = anchor_to_profiles(a)
    # T1-only: relaxation kept, pure dephasing removed (T2 == 2*T1), legacy prof_t1.
    assert t1.t1_ns_q1 == pytest.approx(60000.0)
    assert t1.t1_ns_q2 == pytest.approx(30000.0)
    assert t1.t2_ns_q1 == pytest.approx(120000.0)
    assert t1.t2_ns_q2 == pytest.approx(60000.0)
    # No decoherence: T1 -> ~inf, T2 == 2*T1 (legacy prof_none used 1e8 / 2e8).
    assert none.t1_ns_q1 >= 1e8
    assert none.t2_ns_q1 == pytest.approx(2.0 * none.t1_ns_q1)
    # The physics (frequencies) is identical across the three variants.
    assert none.freq_ghz_q1 == t1.freq_ghz_q1 == full.freq_ghz_q1


def test_analytic_coherence_limit_matches_known_sung_floor():
    # GRAPE-independent: straight from the measured T1/T2 + 60 ns gate. Pins the
    # (2 t_g/5) prefactor, and must land on the known GRAPE floor / measured ~2.4e-3.
    a = load_anchor(SUNG)
    _, _, full = anchor_to_profiles(a)
    tg = gate_config(a)["n_slices"] * gate_config(a)["dt_ns"]
    epg = analytic_coherence_limit_epg(full, tg)
    assert epg == pytest.approx(2.333e-3, abs=2e-5)        # the pinned analytic value
    assert epg == pytest.approx(1.0 - a["validation"]["measured_f_avg"], abs=1.5e-4)


def test_effective_t2_from_t1_and_tphi_is_textbook():
    # 1/T2 = 1/(2 T1) + 1/T_phi. Pure relaxation (T_phi -> inf) gives T2 = 2 T1.
    assert effective_t2_ns(20_000.0, 1e12) == pytest.approx(40_000.0, rel=1e-6)
    # Marxer Q1 effective: T1,eff=14us, Tphi,1,eff=67us  (quoted from the paper).
    assert effective_t2_ns(14_000.0, 67_000.0) == pytest.approx(19_747.0, abs=5.0)
    # Marxer Q2 effective: T1,eff=43us, Tphi,1,eff=43us.
    assert effective_t2_ns(43_000.0, 43_000.0) == pytest.approx(28_667.0, abs=5.0)


def test_marxer_anchor_derives_t2_in_code_from_published_values():
    # The anchor publishes the paper's quoted T1,eff and Tphi,1,eff; the harness derives
    # T2 (no hand-computed T2 in the data file). Confirm the loader did the derivation.
    a = load_anchor(MARXER)
    assert a["qubits"]["0"]["t2_ns"] == pytest.approx(effective_t2_ns(14_000.0, 67_000.0))
    assert a["qubits"]["1"]["t2_ns"] == pytest.approx(effective_t2_ns(43_000.0, 43_000.0))
    _, _, full = anchor_to_profiles(a)
    assert full.t2_ns_q1 == pytest.approx(19_747.0, abs=5.0)
    assert full.t2_ns_q2 == pytest.approx(28_667.0, abs=5.0)


def test_marxer_reproduces_published_coherence_limit():
    # Fed the paper's gate-effective coherence times, gradpulse's analytic floor reproduces
    # the authors' computed coherence limit eps_limit=1.7e-3 and lands at the measured 1.9e-3.
    a = load_anchor(MARXER)
    assert a["validation"]["coherence_limited"] is True
    assert a["validation"]["measured_f_avg"] == pytest.approx(0.9981)
    _, _, full = anchor_to_profiles(a)
    tg = gate_config(a)["n_slices"] * gate_config(a)["dt_ns"]   # 33 ns
    epg = analytic_coherence_limit_epg(full, tg)
    assert epg == pytest.approx(1.75e-3, abs=5e-5)             # ~ paper eps_limit 1.7e-3
    assert 0.5 <= epg / (1.0 - 0.9981) <= 1.5                  # coherence-limited band (~0.92x)


def test_marxer_static_idling_times_overpredict_effective_do_not():
    # TRANSPARENCY: the effective-vs-static choice is audited, not buried. Feeding gradpulse
    # the STATIC Table III idling-point T1/T2 (Q1 13/14 us, Q2 42/8 us -- measured with the
    # coupler un-pulsed) OVER-predicts the error, because that static T2 is not the gate's
    # effective T2. The gate-effective times (used by the anchor) land at the measured number.
    meas_epg = 1.0 - 0.9981
    tg = 33.0
    static = ParametricCouplerProfile(
        freq_ghz_q1=4.102, freq_ghz_q2=3.892, anharm_ghz_q1=-0.215, anharm_ghz_q2=-0.217,
        t1_ns_q1=13_000.0, t1_ns_q2=42_000.0, t2_ns_q1=14_000.0, t2_ns_q2=8_000.0)
    epg_static = analytic_coherence_limit_epg(static, tg)
    assert epg_static / meas_epg > 1.4                         # static visibly over-predicts (~1.7x)

    _, _, effective = anchor_to_profiles(load_anchor(MARXER))
    epg_eff = analytic_coherence_limit_epg(effective, tg)
    assert epg_eff / meas_epg < 1.1                            # effective lands at measured (~0.9x)


def test_marxer_omitted_gaussian_dephasing_is_below_one_percent():
    # gradpulse's single-T2 Lindblad floor omits the authors' Gaussian dephasing component
    # (Tphi,2,eff: Q1=17us, Q2=6us). Quantify exactly what is omitted: the (tg/Tphi2)^2 terms,
    # which must be < 1% of the coherence limit (so the Markovian floor is a faithful bound).
    tg = 33.0
    eps_limit = 1.7e-3
    gaussian = sum(0.4 * (tg / tphi2) ** 2 for tphi2 in (17_000.0, 6_000.0))
    assert gaussian == pytest.approx(1.36e-5, abs=2e-6)       # the exact omitted contribution
    assert gaussian < 0.01 * eps_limit                        # < 1% of the limit


def test_all_three_anchors_discovered():
    found = {p.name for p in discover_anchors(ANCHOR_DIR)}
    assert {"sung_2021_cz.json", "marxer_2023_cz.json",
            "stehlik_2021_cz.json"} <= found


# ---------------------------------------------------------------------------
# Stehlik 2021 (IBM) -- analytic-floor anchor + the 11-pair lower-bound breadth
# ---------------------------------------------------------------------------
def test_floor_method_validation(tmp_path):
    a = _valid_anchor()
    a["validation"]["floor_method"] = "analytic"
    load_anchor(_write(tmp_path, a))                 # 'analytic' is accepted
    a["validation"]["floor_method"] = "bogus"
    with pytest.raises(ValueError, match="floor_method"):
        load_anchor(_write(tmp_path, a))


def test_stehlik_anchor_loads_as_analytic_coherence_limited():
    a = load_anchor(STEHLIK)
    assert a["validation"]["coherence_limited"] is True
    assert a["validation"]["floor_method"] == "analytic"
    assert "080505" in a["provenance"]["citation"]
    # pair 11: measured EPG 0.0049 -> F_avg 0.9951
    assert a["validation"]["measured_f_avg"] == pytest.approx(0.9951)


def test_stehlik_pair11_analytic_floor_in_coherence_band():
    """The honest, fully paper-sourced metric: analytic floor / measured EPG ~ 1.05x,
    inside the coherence-limited band; judged WITHOUT GRAPE (no Hamiltonian params)."""
    a = load_anchor(STEHLIK)
    g = gate_config(a)
    _, _, full = anchor_to_profiles(a)
    epg = analytic_coherence_limit_epg(full, g["n_slices"] * g["dt_ns"])
    v = judge_analytic(a, epg)
    assert v["passed"] is True
    assert v["ratio"] == pytest.approx(1.05, abs=0.05)
    assert v["gate_closes"] is None                  # analytic mode runs no optimizer
    assert "analytic" in format_report(v)


# pair 9 has T2 > 2*T1 in Table I (self-inconsistent); gradpulse legitimately warns and
# clamps its dephasing -- the same handling as drifted calibration entries. Expected here.
@pytest.mark.filterwarnings("ignore:ParametricCouplerProfile q")
def test_stehlik_table_floor_is_lower_bound_on_all_11_pairs():
    """The no-selection breadth result: gradpulse's coherence floor never exceeds the
    measured error across all 11 published Stehlik pairs, and saturates only on pair 11
    (the one pair with both short coherence and a long gate)."""
    data = json.loads(STEHLIK_TABLE.read_text(encoding="utf-8"))
    ratios = {}
    for p in data["pairs"]:
        prof = ParametricCouplerProfile(
            t1_ns_q1=p["t1_us"] * 1e3, t2_ns_q1=p["t2_us"] * 1e3,
            t1_ns_q2=p["t1_us"] * 1e3, t2_ns_q2=p["t2_us"] * 1e3)
        ratios[p["pair"]] = analytic_coherence_limit_epg(prof, p["gate_ns"]) / p["epg"]
    assert len(ratios) == 11
    # lower bound holds everywhere (10% slack for the one saturating pair)
    assert all(r <= 1.10 for r in ratios.values())
    # exactly pair 11 saturates -> coherence-limited
    assert [pp for pp, r in ratios.items() if 0.5 <= r <= 1.5] == [11]
    assert ratios[11] == pytest.approx(1.05, abs=0.05)


def test_gate_config_defaults_and_overrides():
    base = gate_config({})
    assert base["kind"] == "cz" and base["n_slices"] == 60 and base["dt_ns"] == 1.0
    assert base["precision"] == "double"
    over = gate_config({"gate": {"n_slices": 120, "iterations": 800}})
    assert over["n_slices"] == 120 and over["iterations"] == 800
    assert over["bandwidth_mhz"] == 200.0  # untouched default still present


# ---------------------------------------------------------------------------
# Judge: the scientific decision. Inputs are PROCESS fidelities; f_avg() converts.
# ---------------------------------------------------------------------------
def _coh_anchor(measured, band=None):
    v = {"measured_f_avg": measured, "coherence_limited": True}
    if band is not None:
        v["ratio_band"] = band
    return {"name": "coh", "validation": v}


def _noncoh_anchor(measured):
    return {"name": "noncoh",
            "validation": {"measured_f_avg": measured, "coherence_limited": False}}


def _f_proc_for_f_avg(target_a):
    """Invert f_avg: the process fidelity whose F_avg is target_a."""
    return (5.0 * target_a - 1.0) / 4.0


def test_judge_coherence_limited_passes_when_floor_matches():
    # measured EPG = 0.0024; build f_full so the decoherence error == measured EPG (1x).
    a = _coh_anchor(0.9976)
    f_coh = 1.0 - 1e-5                       # gate closes (tiny coherent residual)
    a_coh = f_avg(f_coh)
    f_full = _f_proc_for_f_avg(a_coh - 0.0024)
    f_t1 = _f_proc_for_f_avg(a_coh - 0.0012)
    v = judge(a, f_coh, f_t1, f_full)
    assert v["gate_closes"] is True
    assert v["ratio"] == pytest.approx(1.0, abs=0.05)
    assert v["passed"] is True
    assert v["claim"] == "equality"


def test_judge_coherence_limited_fails_out_of_band():
    a = _coh_anchor(0.9976)                  # meas EPG 0.0024
    f_coh = 1.0 - 1e-5
    f_full = 0.95                            # decoherence error ~16x the measured EPG
    v = judge(a, f_coh, f_coh, f_full)
    assert v["ratio"] > 1.5
    assert v["passed"] is False


def test_judge_flags_gate_that_does_not_close():
    # Floor/measured ratio is ~1x, but the optimizer left a large coherent residual,
    # so the "gap is decoherence" premise fails -> inconclusive (passed False).
    a = _coh_anchor(0.9976)
    f_coh = 0.99                             # a_coh ~ 0.992; residual 0.008 >> 0.25*0.0024
    a_coh = f_avg(f_coh)
    f_full = _f_proc_for_f_avg(a_coh - 0.0024)   # ratio ~ 1x
    v = judge(a, f_coh, f_coh, f_full)
    assert v["gate_closes"] is False
    assert v["passed"] is False


def test_judge_lower_bound_passes_when_model_underpredicts():
    # NOT coherence-limited: measured EPG 0.005, model floor only 0.003 -> valid lower
    # bound, and 0.002 is reported as un-modelled error.
    a = _noncoh_anchor(0.995)
    f_coh = 1.0 - 1e-5
    a_coh = f_avg(f_coh)
    f_full = _f_proc_for_f_avg(a_coh - 0.003)
    v = judge(a, f_coh, f_coh, f_full)
    assert v["claim"] == "lower_bound"
    assert v["ratio"] < 1.0
    assert v["passed"] is True
    assert v["unexplained_error"] == pytest.approx(0.002, abs=2e-4)


def test_judge_lower_bound_fails_when_model_overpredicts():
    # A NOT-coherence-limited floor that EXCEEDS the measured total error is unphysical.
    a = _noncoh_anchor(0.995)                # meas EPG 0.005
    f_coh = 1.0 - 1e-5
    a_coh = f_avg(f_coh)
    f_full = _f_proc_for_f_avg(a_coh - 0.008)    # decoherence error 0.008 > 0.005
    v = judge(a, f_coh, f_coh, f_full)
    assert v["ratio"] > 1.0
    assert v["passed"] is False


def test_format_report_smoke_both_branches():
    # Equality branch.
    a = _coh_anchor(0.9976)
    f_coh = 1.0 - 1e-5
    f_full = _f_proc_for_f_avg(f_avg(f_coh) - 0.0024)
    txt = format_report(judge(a, f_coh, f_coh, f_full))
    assert "coherence-limited" in txt and ("PASS" in txt or "FAIL" in txt)
    # Lower-bound branch.
    a2 = _noncoh_anchor(0.995)
    f_full2 = _f_proc_for_f_avg(f_avg(f_coh) - 0.003)
    txt2 = format_report(judge(a2, f_coh, f_coh, f_full2))
    assert "lower bound" in txt2 and "un-modelled" in txt2
