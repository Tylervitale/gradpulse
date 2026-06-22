"""Offline tests for the Amazon Braket hardware on-ramp (gradpulse.braket_bridge).

Everything here runs WITHOUT AWS credentials, a device, or any cost -- it exercises
exactly the part of the hardware path that is verifiable in software: faithful
waveform export, OpenPulse program construction, cost arithmetic, and the honest
credential-wall behaviour of the backend. The one step these CANNOT cover --
device.run() on real silicon -- is the irreducible sim != hardware boundary.
"""
import numpy as np
import pytest

pytest.importorskip("braket.pulse", reason="needs amazon-braket-sdk")

from gradpulse import braket_bridge as bb  # noqa: E402
from gradpulse import hardware as hw  # noqa: E402


def _demo_waveform():
    t = np.linspace(0.0, 1.0, 120)
    drive = np.sin(np.pi * t) + 0.1j * np.cos(np.pi * t)   # complex I/Q (DRAG-like)
    coupler = np.sin(np.pi * t) ** 2
    return np.stack([drive, coupler], axis=1)


def test_waveform_roundtrip_is_machine_exact():
    wf = _demo_waveform()
    for c in range(wf.shape[1]):
        err = bb.verify_waveform_roundtrip(wf[:, c])
        assert err < 1e-12, f"channel {c} export not faithful: {err}"


def test_normalize_returns_physical_peak_and_unit_disk():
    env = 3.7 * np.sin(np.linspace(0, np.pi, 64))
    wf, peak = bb.to_braket_waveform(env, normalize=True)
    amps = np.asarray(wf.amplitudes)
    assert abs(peak - env.max()) < 1e-12
    assert np.max(np.abs(amps)) <= 1.0 + 1e-12       # inside the unit disk
    # un-normalized round-trips to the raw samples
    assert bb.verify_waveform_roundtrip(env, normalize=False) < 1e-12


def test_pulse_sequence_builds_and_serializes_openpulse():
    wf = _demo_waveform()
    frames = bb.synthetic_frames(wf.shape[1])
    seq = bb.build_gate_pulse_sequence(wf, frames)
    ir = seq.to_ir()
    assert "OPENQASM 3.0" in ir
    assert ir.count("play(") == wf.shape[1]          # one play per channel


def test_too_few_frames_raises():
    wf = _demo_waveform()
    with pytest.raises(ValueError):
        bb.build_gate_pulse_sequence(wf, bb.synthetic_frames(1))


def test_cost_arithmetic_matches_pricing_table():
    per_task, per_shot = bb.BRAKET_QPU_PRICING["Rigetti-Cepheus-1-108Q"]
    ce = bb.estimate_experiment_cost(100, 500, "Rigetti-Cepheus-1-108Q")
    assert ce.task_fee_usd == pytest.approx(per_task * 100)
    assert ce.shot_fee_usd == pytest.approx(per_shot * 100 * 500)
    assert ce.total_usd == pytest.approx(ce.task_fee_usd + ce.shot_fee_usd)


def test_largest_irb_under_budget_is_feasible_and_respects_cap():
    plan = bb.largest_irb_under_budget(50.0, "Rigetti-Cepheus-1-108Q", n_shots=500)
    assert plan["feasible"]
    assert plan["est_cost_usd"] <= 50.0
    # one more seed must exceed the budget (it is the *largest* fitting design)
    over = bb.estimate_experiment_cost(
        bb.irb_circuit_count(plan["lengths"], plan["n_seeds"] + 1), 500).total_usd
    assert over > 50.0


def test_unknown_device_raises():
    with pytest.raises(KeyError):
        bb.estimate_experiment_cost(10, 100, "NoSuchQPU")


def test_readiness_report_offline_chain():
    wf = _demo_waveform()
    rep = bb.hardware_readiness_report(wf, budget_usd=50.0, n_shots=500, verbose=False)
    assert rep["waveform_export_faithful"]
    assert rep["pulse_sequence_builds"]
    assert rep["irb_plan"]["feasible"]
    # the report must be explicit that the physical run is the one unmet step
    assert "device.run()" in rep["NOT_validated_needs_silicon"]


def test_backend_builds_offline_then_refuses_to_submit():
    """The backend constructs everything up to submission and stops honestly."""
    wf = _demo_waveform()
    be = hw.BraketPulseBackend(shots=500)             # no live device, no compiler
    seq = be.build_gate_sequence(wf)
    assert "OPENQASM 3.0" in seq.to_ir()
    assert be.estimate_cost(n_seeds=9).total_usd <= 50.0
    with pytest.raises(RuntimeError, match="cannot SUBMIT"):
        be.measure_gate(wf)


def test_template_name_is_backcompat_alias():
    assert hw.BraketBackendTemplate is hw.BraketPulseBackend


# ---- interleaved-RB circuit generation (the device.run input) -------------
def test_rb_sequences_ideal_return_to_zero_reference_and_interleaved():
    """The correctness gate: every generated sequence (reference AND interleaved)
    returns to |00> under ideal gates -- the recovery inverts the ideal product.
    This is what proves the circuits are right before any QPU spend."""
    for interleaved in (False, True):
        seqs = bb.native_rb_sequences([1, 2, 4, 8], n_seeds=3, seed=7,
                                      interleaved=interleaved)
        assert len(seqs) == 4 * 3
        for s in seqs:
            assert abs(bb.ideal_survival_probability(s["gates"]) - 1.0) < 1e-9


def test_interleaved_sequences_carry_the_benchmarked_cz():
    """Interleaved sequences insert exactly one benchmarked CZ per Clifford (m of
    them at length m); reference sequences carry none."""
    ref = bb.native_rb_sequences([5], n_seeds=1, seed=0, interleaved=False)[0]
    intl = bb.native_rb_sequences([5], n_seeds=1, seed=0, interleaved=True)[0]
    assert sum(g[0] == "CZ_BENCH" for g in ref["gates"]) == 0
    assert sum(g[0] == "CZ_BENCH" for g in intl["gates"]) == 5


def test_survival_from_counts():
    assert bb.survival_from_counts({"00": 750, "01": 100, "10": 100, "11": 50}) == 0.75
    assert bb.survival_from_counts({}) == 0.0


def test_to_braket_rb_circuit_wraps_in_verbatim_box():
    """Regression guard for the silent RB-killer: the hardware circuit MUST be a
    verbatim box. Without it Rigetti's compiler cancels each random Clifford against
    its recovery (net identity) and the decay goes flat -- you'd measure ~1.0 at every
    length and learn nothing, while the job still 'succeeds'."""
    seq = bb.native_rb_sequences([3], n_seeds=1, seed=1, interleaved=True)[0]
    circ = bb.to_braket_rb_circuit(seq["gates"], qubits=(10, 11))
    assert circ.qubit_count == 2
    qasm = circ.to_ir(ir_type="OPENQASM").source.lower()
    assert "verbatim" in qasm or "pragma" in qasm


def test_to_braket_rb_circuit_emits_only_cepheus_native_gates():
    """Regression guard for the other silent killer: only Cepheus-native gates
    (RX, RZ, CZ) may appear -- H/S are rejected inside a verbatim box. The Clifford
    words are transpiled to native upstream; the offline survival gate proves the
    transpilation is exact, so this guards that nothing non-native slips onto the wire."""
    seq = bb.native_rb_sequences([4], n_seeds=1, seed=2, interleaved=True)[0]
    body = bb.to_braket_rb_circuit(seq["gates"], qubits=(0, 1), verbatim=False)
    opnames = {type(instr.operator).__name__ for instr in body.instructions}
    assert opnames <= {"Rx", "Rz", "CZ"}, opnames
    assert len(body.instructions) == len(seq["gates"])


def test_local_simulator_noiseless_pipeline_returns_to_zero():
    """Strongest offline gate: the exact generate->circuit->run->parse pipeline on the
    noiseless LocalSimulator must yield survival ~1.0 at every length. If this passes,
    the QPU submission is the identical code path with a device ARN swapped in."""
    pytest.importorskip("braket")
    from braket.devices import LocalSimulator
    sim = LocalSimulator()
    for s in bb.native_rb_sequences([1, 2, 4, 8], n_seeds=1, seed=3):
        circ = bb.to_braket_rb_circuit(s["gates"], qubits=(0, 1), verbatim=False)
        r = sim.run(circ, shots=300).result()
        assert bb.survival_from_counts(r.measurement_counts) > 0.999


def _cz_back_to_back(qasm):
    """Count `cz` instructions immediately followed by another `cz` (no barrier
    between) in serialized OpenQASM -- the back-to-back-flux adjacency."""
    instrs = [l.strip() for l in qasm.splitlines()
              if l.strip().startswith(("cz ", "barrier ", "rx(", "rz("))]
    czi = [i for i, l in enumerate(instrs) if l.startswith("cz ")]
    return sum(1 for i in czi if i + 1 < len(instrs) and instrs[i + 1].startswith("cz "))


def test_buffer_bench_cz_isolates_benchmarked_gate_and_is_optin():
    """`buffer_bench_cz` wraps each benchmarked CZ in barriers so it is never abutted
    to a Clifford's own CZ. A real Level-A run (RESULTS.md S10) showed interleaving puts
    the benchmarked CZ back-to-back with a native CZ ~9x more than the reference arm,
    and back-to-back flux pulses error more -- inflating the naive interleaved-RB number
    ~2.5x above the isolated-gate error. The fix must be opt-in (default byte-identical),
    identity (still returns to |00>), and must remove the benchmarked-CZ adjacency."""
    seq = bb.native_rb_sequences([16], n_seeds=2, seed=5, interleaved=True)
    n_off = n_on = 0
    for s in seq:
        g = s["gates"]
        off = bb.to_braket_rb_circuit(g, qubits=(16, 25)).to_ir(ir_type="OPENQASM").source
        off_explicit = bb.to_braket_rb_circuit(
            g, qubits=(16, 25), buffer_bench_cz=False).to_ir(ir_type="OPENQASM").source
        on = bb.to_braket_rb_circuit(
            g, qubits=(16, 25), buffer_bench_cz=True).to_ir(ir_type="OPENQASM").source
        assert off == off_explicit                 # default is byte-identical
        assert "barrier" not in off and "barrier" in on
        n_off += _cz_back_to_back(off)
        n_on += _cz_back_to_back(on)
    assert n_off > 0                               # the context bias exists unbuffered
    assert n_on == 0                               # buffering removes every adjacency

    # the barrier is identity: the noiseless pipeline still returns to |00>
    pytest.importorskip("braket")
    from braket.devices import LocalSimulator
    sim = LocalSimulator()
    for s in bb.native_rb_sequences([1, 4, 8], n_seeds=1, seed=2, interleaved=True):
        circ = bb.to_braket_rb_circuit(
            s["gates"], qubits=(0, 1), verbatim=False, buffer_bench_cz=True)
        r = sim.run(circ, shots=300).result()
        assert bb.survival_from_counts(r.measurement_counts) > 0.999


# ---- Level B: benchmarking a gradpulse-DESIGNED pulse ---------------------
def _bench_pulse(peak=0.3, drives=True):
    flux_frame, d0, d1 = bb.synthetic_frames(3)
    flux = np.sin(np.pi * np.linspace(0, 1, 48)) ** 2
    return bb.build_bench_cz_pulse_sequence(
        flux, flux_frame, peak_amplitude=peak,
        drive_frames=((d0, d1) if drives else None), virtual_z=(0.2, -0.1))


def test_bench_pulse_builds_and_serializes_one_play():
    ir = str(_bench_pulse().to_ir())
    assert "OPENQASM 3.0" in ir
    assert ir.count("play(") == 1


def test_bench_peak_anchors_to_device_flux_scale():
    """The normalized gradpulse shape is scaled to the device's calibrated flux peak,
    so the played waveform's peak IS that anchor -- the open-loop transfer scale."""
    import re
    flux_frame = bb.synthetic_frames(1)[0]
    flux = 5.0 * np.sin(np.pi * np.linspace(0, 1, 24)) ** 2     # arbitrary input scale
    ir = str(bb.build_bench_cz_pulse_sequence(flux, flux_frame, peak_amplitude=0.37).to_ir())
    vals = [float(x) for x in
            re.search(r"grad_bench_cz\s*=\s*\{([^}]*)\}", ir).group(1).split(",")]
    assert max(abs(v) for v in vals) == pytest.approx(0.37)


def test_bench_pulse_warns_on_nonzero_rest_envelope():
    """Feeding a [0,1] envelope (rest=0.5) instead of a physical rest-0 flux is the
    classic Level-B mistake -- it plays a DC offset, a different gate. The binder must
    warn so it never silently transfers garbage; the physical u=2x-1 must not warn."""
    flux_frame = bb.synthetic_frames(1)[0]
    x = 0.5 + 0.3 * np.sin(np.pi * np.linspace(0, 1, 40))   # [0,1] envelope, rest~0.5
    with pytest.warns(RuntimeWarning, match="envelope"):
        bb.build_bench_cz_pulse_sequence(x, flux_frame, peak_amplitude=0.22)
    u = 2.0 * x - 1.0                                        # physical flux, rest~0
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # any RuntimeWarning -> failure
        bb.build_bench_cz_pulse_sequence(u, flux_frame, peak_amplitude=0.22)


def test_level_b_binds_pulse_inside_verbatim_not_native_cz():
    """A gradpulse bench pulse must be BOUND (a play inside the verbatim box), not
    silently replaced by the native CZ -- otherwise you'd think you measured your pulse
    when you measured the device's. The Cliffords stay native (pragma present)."""
    seq = bb.native_rb_sequences([3], n_seeds=1, seed=0, interleaved=True)[0]
    qasm = bb.to_braket_rb_circuit(
        seq["gates"], qubits=(16, 25),
        bench_cz_pulse=_bench_pulse()).to_ir(ir_type="OPENQASM").source
    assert "pragma braket verbatim" in qasm      # random Cliffords still exact
    assert "play(" in qasm                        # the gradpulse pulse is in the program


def test_level_b_reference_circuits_have_no_bench_pulse():
    """The bench pulse binds only to interleaved CZ_BENCH markers; a reference sequence
    (no markers) carries no play even when a bench pulse is passed -- which is exactly
    why the interleaved/reference ratio isolates the benchmarked gate."""
    ref = bb.native_rb_sequences([4], n_seeds=1, seed=0, interleaved=False)[0]
    qasm = bb.to_braket_rb_circuit(
        ref["gates"], qubits=(16, 25),
        bench_cz_pulse=_bench_pulse()).to_ir(ir_type="OPENQASM").source
    assert "play(" not in qasm


def test_bench_cz_peak_from_native_calibration_reads_peak():
    """The anchor peak is the device's OWN native-CZ flux peak, read from the native
    gate-calibration doc, with a symmetric node key (a-b or a_b)."""
    amps = [[0.0, 0.0], [0.21, 0.0], [0.42, 0.0], [0.21, 0.0], [0.0, 0.0]]
    cal = {"gates": {"16_25": {"cz": [{"name": "cz", "calibrations": [
            {"name": "play", "arguments": [
                {"name": "frame", "value": "q16_q25_cz_frame"},
                {"name": "waveform", "value": "wf_cz"}]}]}]}},
           "waveforms": {"wf_cz": {"amplitudes": amps}}}
    for key in ("16-25", "16_25"):
        assert bb.bench_cz_peak_from_native_calibration(cal, key) == pytest.approx(0.42)
    with pytest.raises((KeyError, ValueError)):
        bb.bench_cz_peak_from_native_calibration(cal, "99-100")


def test_verify_levelb_offline_passes_and_flags_silicon():
    """The offline gate passes for a well-formed bench pulse AND states plainly that the
    FIDELITY still needs the device (the local simulator runs gates, not pulses)."""
    rep = bb.verify_levelb_offline(_bench_pulse(), qubits=(16, 25), n_cliffords=4)
    assert rep["offline_ok"]
    assert rep["verbatim_pragma_present"] and rep["play_present"]
    assert rep["ideal_clifford_closes"]
    assert "device.run()" in rep["NOT_validated_needs_silicon"]


def test_combined_ab_run_count_and_cost():
    """The COMBINED Level-A+B run (examples/run_irb_on_braket.py --combined) shares ONE
    reference between the native-CZ and gradpulse-pulse interleaved sets: 56 ref + 56
    native + 56 pulse = 168 circuits, NOT 224 (= the reference paid for twice). This locks
    the headline cost ($86.10 batch + 3 canaries $1.03 = $87.13 on Cepheus) against drift."""
    lengths = [1, 2, 4, 8, 16, 32, 64, 128]
    n_seeds = 7
    ref = bb.native_rb_sequences(lengths, n_seeds, seed=0, interleaved=False)
    intl = bb.native_rb_sequences(lengths, n_seeds, seed=0, interleaved=True)
    assert len(ref) == 56 and len(intl) == 56
    n_combined = len(ref) + 2 * len(intl)              # ref + native-int + pulse-int
    n_separate = 2 * (len(ref) + len(intl))            # A=(ref+int) and B=(ref+int)
    assert n_combined == 168
    assert n_separate - n_combined == len(ref) == 56   # exactly one reference set saved

    batch = bb.estimate_experiment_cost(n_combined, 500)
    canaries = bb.estimate_experiment_cost(3, 100)     # sanity + native-depth + pulse-depth
    assert batch.total_usd == pytest.approx(86.10, abs=1e-6)
    assert canaries.total_usd == pytest.approx(1.0275, abs=1e-6)
    assert batch.total_usd + canaries.total_usd == pytest.approx(87.1275, abs=1e-6)


def test_combined_same_sequence_binds_native_or_pulse():
    """The combined run is a CONTROLLED comparison: the SAME interleaved sequence becomes
    the native CZ (bench=None) or the gradpulse pulse (bench set). Verify the interleaved
    sequence carries the bindable CZ_BENCH marker (so it can be either) while the shared
    reference does not, and that the binding swaps cz <-> play in the emitted program."""
    intl = bb.native_rb_sequences([4], n_seeds=1, seed=0, interleaved=True)[0]
    ref = bb.native_rb_sequences([4], n_seeds=1, seed=0, interleaved=False)[0]
    assert any(g[0] == "CZ_BENCH" for g in intl["gates"])        # bindable to native OR pulse
    assert not any(g[0] == "CZ_BENCH" for g in ref["gates"])     # reference has no gate-under-test

    native = bb.to_braket_rb_circuit(
        intl["gates"], qubits=(16, 25), bench_cz_pulse=None).to_ir(ir_type="OPENQASM").source
    pulse = bb.to_braket_rb_circuit(
        intl["gates"], qubits=(16, 25), bench_cz_pulse=_bench_pulse()).to_ir(ir_type="OPENQASM").source
    assert "play(" in pulse and "play(" not in native             # same sequence, swapped gate
