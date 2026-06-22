"""Calibration loader: ``ParametricCouplerProfile.from_braket_calibration``.

Loads a real Braket *standardized* device-properties JSON (a 107-qubit fixture
under ``tests/fixtures/``) and checks that:

  - measured T1/T2 land in the profile with the right units (seconds -> ns),
  - the pair's CZ fidelity is picked up regardless of key order,
  - frequency/anharmonicity stay at representative defaults (that schema carries
    neither) and that fact is recorded in ``notes``,
  - ``**overrides`` apply and unknown fields are rejected,
  - the error paths (missing qubit, missing CZ, wrong schema) raise clearly.

The values asserted below are the real measured numbers for pair (4, 5) in the
fixture, so this doubles as a regression guard on the parsing/units.

Run:  pytest tests/        OR        python tests/test_calibration.py
"""
import json
import warnings
from pathlib import Path

import pytest

from gradpulse import ParametricCouplerProfile, RepresentativeDefaultsWarning

FIXTURE = Path(__file__).parent / "fixtures" / "braket_device_calibration.json"


def test_loads_measured_t1_t2_for_pair():
    prof = ParametricCouplerProfile.from_braket_calibration(FIXTURE, (4, 5))
    assert prof.qubit_pair == (4, 5)
    # Measured values for q4/q5 (seconds in the file -> ns in the profile).
    assert prof.t1_ns_q1 == pytest.approx(19_664, rel=1e-3)   # q4 T1 = 19.66 us
    assert prof.t1_ns_q2 == pytest.approx(39_771, rel=1e-3)   # q5 T1 = 39.77 us
    assert prof.t2_ns_q1 == pytest.approx(12_735, rel=1e-3)   # q4 T2 = 12.74 us
    assert prof.t2_ns_q2 == pytest.approx(11_814, rel=1e-3)   # q5 T2 = 11.81 us


def test_loads_measured_cz_fidelity():
    prof = ParametricCouplerProfile.from_braket_calibration(FIXTURE, (4, 5))
    # Real interleaved-RB CZ for the pair, replaces the 0.988 placeholder.
    assert prof.native_cz_fidelity == pytest.approx(0.98597, abs=1e-4)
    assert prof.native_cz_fidelity != ParametricCouplerProfile().native_cz_fidelity


def test_units_convert_from_raw_json():
    # Re-derive straight from the file: value_seconds * 1e9 == profile ns.
    raw = json.loads(FIXTURE.read_text())
    t1_q4_s = raw["oneQubitProperties"]["4"]["T1"]["value"]
    prof = ParametricCouplerProfile.from_braket_calibration(FIXTURE, (4, 5))
    assert prof.t1_ns_q1 == pytest.approx(t1_q4_s * 1e9, rel=1e-12)


def test_pair_order_is_symmetric():
    # twoQubitProperties is keyed "4-5"; requesting (5, 4) must still find CZ and
    # swap the per-qubit fields into the q1/q2 slots accordingly.
    a = ParametricCouplerProfile.from_braket_calibration(FIXTURE, (4, 5))
    b = ParametricCouplerProfile.from_braket_calibration(FIXTURE, (5, 4))
    assert b.qubit_pair == (5, 4)
    assert b.native_cz_fidelity == pytest.approx(a.native_cz_fidelity)
    assert b.t1_ns_q1 == pytest.approx(a.t1_ns_q2)   # q5 now in the q1 slot
    assert b.t1_ns_q2 == pytest.approx(a.t1_ns_q1)


def test_freq_anharm_keep_defaults_and_are_noted():
    default = ParametricCouplerProfile()
    prof = ParametricCouplerProfile.from_braket_calibration(FIXTURE, (4, 5))
    # The standardized schema has no frequency/anharmonicity: defaults, not loads.
    assert prof.freq_ghz_q1 == default.freq_ghz_q1
    assert prof.freq_ghz_q2 == default.freq_ghz_q2
    assert prof.anharm_ghz_q1 == default.anharm_ghz_q1
    assert prof.anharm_ghz_q2 == default.anharm_ghz_q2
    assert any("representative defaults" in n for n in prof.notes)


def test_overrides_apply_and_suppress_default_note():
    prof = ParametricCouplerProfile.from_braket_calibration(
        FIXTURE, (4, 5), freq_ghz_q1=4.4808, freq_ghz_q2=4.5797,
    )
    assert prof.freq_ghz_q1 == pytest.approx(4.4808)
    assert prof.freq_ghz_q2 == pytest.approx(4.5797)
    # T1/T2/CZ are still the loaded measurements.
    assert prof.t1_ns_q1 == pytest.approx(19_664, rel=1e-3)
    # Supplying freq explicitly suppresses the "defaults" note.
    assert not any("representative defaults" in n for n in prof.notes)


def test_require_cz_false_keeps_default_for_uncoupled_pair():
    # (0, 50) are both present but not a coupled pair, so there is no CZ entry.
    prof = ParametricCouplerProfile.from_braket_calibration(
        FIXTURE, (0, 50), require_cz=False,
    )
    assert prof.native_cz_fidelity == ParametricCouplerProfile().native_cz_fidelity
    assert any("none measured" in n for n in prof.notes)


def test_missing_cz_raises_by_default():
    with pytest.raises(ValueError, match="no measured CZ fidelity"):
        ParametricCouplerProfile.from_braket_calibration(FIXTURE, (0, 50))


def test_missing_qubit_raises():
    with pytest.raises(ValueError, match="not present"):
        ParametricCouplerProfile.from_braket_calibration(FIXTURE, (4, 9999))


def test_unknown_override_raises():
    with pytest.raises(TypeError, match="unknown ParametricCouplerProfile field"):
        ParametricCouplerProfile.from_braket_calibration(
            FIXTURE, (4, 5), not_a_real_field=1.0,
        )


def test_bad_schema_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "a device file"}))
    with pytest.raises(ValueError, match="standardized"):
        ParametricCouplerProfile.from_braket_calibration(bad, (4, 5))


def test_loaded_profile_drives_the_optimizer():
    # End-to-end: a calibration-loaded profile is a drop-in for the optimizer.
    from gradpulse import ParametricCZOptimizer
    prof = ParametricCouplerProfile.from_braket_calibration(FIXTURE, (4, 5))
    opt = ParametricCZOptimizer(prof, n_channels=3, activation="sigmoid")
    res = opt.optimize_multi_seed(
        n_seeds=1, iterations=3, n_slices=40, dt_ns=1.0,
        warm_start_mode="parametric_cz", use_process_fidelity=True,
        lbfgs_polish=False,
    )
    assert 0.0 <= float(res["best_fidelity"]) <= 1.0


# ---------------------------------------------------------------------------
# Universal normalized-dict loader + vendor adapters (close the "frequency and
# anharmonicity must be supplied separately" gap)
# ---------------------------------------------------------------------------
_SI_CAL = {
    "qubits": {
        0: {"frequency": 4.85e9, "anharmonicity": -0.31e9, "T1": 42e-6, "T2": 35e-6},
        1: {"frequency": 5.05e9, "anharmonicity": -0.29e9, "T1": 40e-6, "T2": 30e-6},
    },
    "two_qubit": {(0, 1): {"cz_fidelity": 0.991}},
}


def test_from_calibration_si_units_loads_everything():
    p = ParametricCouplerProfile.from_calibration(_SI_CAL, (0, 1))
    assert abs(p.freq_ghz_q1 - 4.85) < 1e-9 and abs(p.freq_ghz_q2 - 5.05) < 1e-9
    assert abs(p.anharm_ghz_q1 + 0.31) < 1e-9 and abs(p.anharm_ghz_q2 + 0.29) < 1e-9
    assert abs(p.t1_ns_q1 - 42_000) < 1e-6 and abs(p.t2_ns_q2 - 30_000) < 1e-6
    assert abs(p.native_cz_fidelity - 0.991) < 1e-12


def test_from_calibration_ghz_ns_units_and_default_fallback():
    cal = {"qubits": {3: {"freq_ghz": 4.7, "t1_ns": 50_000, "t2_ns": 41_000},
                      7: {"freq_ghz": 4.9, "t1_ns": 48_000, "t2_ns": 39_000}}}
    p = ParametricCouplerProfile.from_calibration(cal, (3, 7))
    assert abs(p.freq_ghz_q1 - 4.7) < 1e-9 and abs(p.t1_ns_q2 - 48_000) < 1e-6
    # anharmonicity absent -> default kept and noted
    assert abs(p.anharm_ghz_q1 + 0.2) < 1e-9
    assert any("kept representative defaults" in n for n in p.notes)


def test_from_calibration_overrides_win():
    p = ParametricCouplerProfile.from_calibration(_SI_CAL, (0, 1), freq_ghz_q1=4.4)
    assert abs(p.freq_ghz_q1 - 4.4) < 1e-12


def test_from_calibration_require_cz_raises_when_absent():
    cal = {"qubits": _SI_CAL["qubits"]}  # no two_qubit block
    with pytest.raises(ValueError, match="CZ fidelity"):
        ParametricCouplerProfile.from_calibration(cal, (0, 1), require_cz=True)


def test_from_calibration_missing_qubit_raises():
    with pytest.raises(ValueError, match="absent"):
        ParametricCouplerProfile.from_calibration(_SI_CAL, (0, 99))


class _PropsV1:
    def t1(self, q): return {0: 42e-6, 1: 40e-6}[q]
    def t2(self, q): return {0: 35e-6, 1: 30e-6}[q]
    def frequency(self, q): return {0: 4.85e9, 1: 5.05e9}[q]
    def qubit_property(self, q, name): return ({0: -0.31e9, 1: -0.29e9}[q], None)


class _BackendV1:
    def properties(self): return _PropsV1()


class _QP:
    def __init__(self, t1, t2, freq): self.t1, self.t2, self.frequency = t1, t2, freq


class _BackendV2:
    def qubit_properties(self, q):
        return {0: _QP(42e-6, 35e-6, 4.85e9), 1: _QP(40e-6, 30e-6, 5.05e9)}[q]


def test_from_ibm_backend_v1_extracts_freq_anharm_t1t2():
    p = ParametricCouplerProfile.from_ibm_backend(_BackendV1(), (0, 1))
    assert abs(p.freq_ghz_q1 - 4.85) < 1e-9 and abs(p.anharm_ghz_q1 + 0.31) < 1e-9
    assert abs(p.t1_ns_q1 - 42_000) < 1e-6 and abs(p.t2_ns_q2 - 30_000) < 1e-6


def test_from_ibm_backend_v2_extracts_what_it_exposes():
    # BackendV2 QubitProperties carry no anharmonicity -> default kept.
    p = ParametricCouplerProfile.from_ibm_backend(_BackendV2(), (0, 1))
    assert abs(p.freq_ghz_q2 - 5.05) < 1e-9 and abs(p.t1_ns_q1 - 42_000) < 1e-6
    assert abs(p.anharm_ghz_q1 + 0.2) < 1e-9


def test_multiqubit_from_calibration_three_qubits():
    from gradpulse import MultiQubitProfile
    cal = {"qubits": {**_SI_CAL["qubits"],
                      2: {"frequency": 5.25e9, "anharmonicity": -0.30e9,
                          "T1": 38e-6, "T2": 28e-6}}}
    prof = MultiQubitProfile.from_calibration(
        cal, qubits=[0, 1, 2], couplings={(0, 1): 12.0, (1, 2): 6.0})
    assert prof.n_qubits == 3
    assert prof.freqs_ghz[2] == 5.25 and prof.anharm_mhz[0] == -310.0
    assert prof.t1_ns == [42_000, 40_000, 38_000]
    assert prof.couplings == {(0, 1): 12.0, (1, 2): 6.0}


def test_multiqubit_from_ibm_backend_v1():
    from gradpulse import MultiQubitProfile
    prof = MultiQubitProfile.from_calibration(
        {"qubits": {0: {"frequency": 4.85e9, "anharmonicity": -0.31e9,
                        "T1": 42e-6, "T2": 35e-6},
                    1: {"frequency": 5.05e9, "anharmonicity": -0.29e9,
                        "T1": 40e-6, "T2": 30e-6}}},
        qubits=[0, 1], couplings={(0, 1): 12.0})
    assert abs(prof.anharm_mhz[0] + 310.0) < 1e-6
    assert abs(prof.anharm_mhz[1] + 290.0) < 1e-6


# ---------------------------------------------------------------------------
# Representative-defaults guardrail (#6): the caveat lives in code, not only docs.
# ---------------------------------------------------------------------------

def test_default_profile_warns_it_is_representative():
    """A pure-default profile warns that its numbers are representative, not measured."""
    with pytest.warns(RepresentativeDefaultsWarning):
        ParametricCouplerProfile()


def test_braket_calibrated_profile_does_not_warn_representative():
    """A real calibration moves T1/T2 off their defaults, so the guardrail stays silent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RepresentativeDefaultsWarning)
        ParametricCouplerProfile.from_braket_calibration(FIXTURE, (4, 5))  # must not raise


def test_hand_set_device_value_silences_representative_warning():
    """Any hand-set device field signals the user has taken ownership of the params."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RepresentativeDefaultsWarning)
        ParametricCouplerProfile(t1_ns_q1=21_000.0)  # one field off default -> silent


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
