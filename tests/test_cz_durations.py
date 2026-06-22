"""Pure offline tests for the native-calibration CZ-duration parser
(gradpulse.braket_bridge.cz_durations_from_native_calibration).

No AWS, no SDK, no network: a synthetic native-gate-calibration dict with a known
zero-padded triangular flux pulse, so the three duration definitions land on exact
integers. This is the fix that replaced the hardcoded 60 ns gate time in the Cepheus
coherence-floor validation with the device's own per-pair number.
"""
import numpy as np

from gradpulse import braket_bridge as bb


# Triangular flux pulse with padding: 3 leading + 2 trailing zeros around a 7-sample
# ramp (peak 1.0). buffer=12 samples, active(>1% peak)=7, effective(area/peak)=4.
_TRI = [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0]
_DECOY = [1.0] * 100  # a long pulse on a non-flux frame the parser must NOT pick


def _cal():
    amps = [[x, 0.0] for x in _TRI]
    return {
        "gates": {
            # realistic CZ: a charge-frame play (decoy) BEFORE the real flux play,
            # plus a shift_phase -- the parser must skip both and find the flux play.
            "16_25": {"cz": [{"name": "cz", "qubits": ["16", "25"], "calibrations": [
                {"name": "barrier", "arguments": []},
                {"name": "play", "arguments": [
                    {"name": "frame", "value": "Transmon_16_charge_tx"},
                    {"name": "waveform", "value": "wf_decoy"}]},
                {"name": "play", "arguments": [
                    {"name": "frame", "value": "Transmon_108_flux_tx_cz"},
                    {"name": "waveform", "value": "wf_tri"}]},
                {"name": "shift_phase", "arguments": [
                    {"name": "frame", "value": "Transmon_16_charge_tx"},
                    {"name": "phase", "value": 1.23}]},
            ]}]},
            # CZ whose only play is on a charge frame (no flux) -> omitted.
            "1_2": {"cz": [{"name": "cz", "qubits": ["1", "2"], "calibrations": [
                {"name": "play", "arguments": [
                    {"name": "frame", "value": "Transmon_1_charge_tx"},
                    {"name": "waveform", "value": "wf_tri"}]}]}]},
            # not a CZ site -> ignored.
            "5": {"rx": [{"name": "rx", "calibrations": []}]},
        },
        "waveforms": {
            "wf_tri": {"waveformId": "wf_tri", "amplitudes": amps},
            "wf_decoy": {"waveformId": "wf_decoy", "amplitudes": [[x, 0.0] for x in _DECOY]},
        },
    }


def test_active_duration_picks_flux_play_and_trims_padding():
    d = bb.cz_durations_from_native_calibration(_cal())          # mode='active', dt=1ns
    assert set(d) == {"16-25"}                 # 1-2 (charge-only) and 5 (no CZ) omitted
    assert d["16-25"] == 7.0                   # NOT 100 (decoy) and NOT 12 (buffer)


def test_three_modes_are_distinct_and_correct():
    cal = _cal()
    assert bb.cz_durations_from_native_calibration(cal, mode="buffer")["16-25"] == 12.0
    assert bb.cz_durations_from_native_calibration(cal, mode="active")["16-25"] == 7.0
    assert bb.cz_durations_from_native_calibration(cal, mode="effective")["16-25"] == 4.0


def test_dt_ns_scales_the_result():
    d = bb.cz_durations_from_native_calibration(_cal(), dt_ns=2.0)
    assert d["16-25"] == 14.0                  # 7 samples * 2 ns


def test_hyphen_keying_is_symmetric_and_string():
    d = bb.cz_durations_from_native_calibration(_cal())
    assert "16-25" in d and "16_25" not in d   # underscores -> hyphens


def test_robust_to_empty_or_templated_waveforms():
    assert bb.cz_durations_from_native_calibration({}) == {}
    assert bb.cz_durations_from_native_calibration({"gates": {}, "waveforms": {}}) == {}
    # a parametric waveform with no explicit samples is skipped, not crashed on
    templated = {
        "gates": {"7_8": {"cz": [{"name": "cz", "calibrations": [
            {"name": "play", "arguments": [
                {"name": "frame", "value": "q108_flux_cz"},
                {"name": "waveform", "value": "param"}]}]}]}},
        "waveforms": {"param": {"waveformId": "param", "name": "constant",
                                "arguments": [{"name": "length", "value": 6e-8}]}},
    }
    assert bb.cz_durations_from_native_calibration(templated) == {}
