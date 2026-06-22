"""gradpulse.braket_bridge -- export an optimized pulse to Amazon Braket and build
the interleaved-RB circuits that benchmark it on hardware.

Two benchmark levels (see examples/run_irb_on_braket.py):
  Level A -- benchmark the device's NATIVE CZ; validates gradpulse's coherence model.
  Level B -- benchmark a gradpulse-DESIGNED pulse, played on the device's CZ frame as
    a pulse gate inside the verbatim box (build_bench_cz_pulse_sequence +
    to_braket_rb_circuit(bench_cz_pulse=...)); validates the optimizer on silicon.

Everything except device.run() is offline-verifiable. Note that Braket's local
simulator executes gates, not pulse programs, so a Level-B *fidelity* needs the QPU;
what is checkable offline is that the pulse and circuit serialize (OpenPulse 3.0 /
OpenQASM 3) and the waveform round-trips (verify_levelb_offline).

Requires the optional ``braket`` SDK (``pip install amazon-braket-sdk``); imported
lazily so the gradpulse core has no hard dependency on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def _require_braket():
    try:
        from braket.pulse import PulseSequence, ArbitraryWaveform, Frame, Port
        return PulseSequence, ArbitraryWaveform, Frame, Port
    except ImportError as e:  # pragma: no cover - exercised only without the SDK
        raise ImportError(
            "gradpulse.braket_bridge needs the Amazon Braket SDK. Install it with "
            "`pip install amazon-braket-sdk` (or the gradpulse `[braket]` extra)."
        ) from e


# Pricing -- current published Amazon Braket QPU rates (per-task + per-shot).
# Verified 2026-06; CONFIRM before relying on a dollar figure, as providers
# re-price. None per-shot = not published at build time.
BRAKET_QPU_PRICING = {
    # device key            : (per_task_usd, per_shot_usd)
    "Rigetti-Cepheus-1-108Q": (0.30, 0.000425),
    "Rigetti-Ankaa-3":        (0.30, 0.00090),
    "OQC":                    (0.30, None),
}
_PRICING_AS_OF = "2026-06"


# ---------------------------------------------------------------------------
# 1. Waveform export (fully offline, round-trip verified)
# ---------------------------------------------------------------------------
def to_braket_waveform(envelope: np.ndarray, *, waveform_id: str = "grad_env",
                       normalize: bool = True):
    """Export one channel's envelope to a ``braket.pulse.ArbitraryWaveform``.

    envelope : real or complex 1-D samples (complex = I/Q, e.g. a DRAG drive
               u + i*v). The optimizer's per-channel smoothed control.
    normalize : scale by 1/max|amplitude| so samples land in the unit disk. A
               Braket frame plays a waveform as a fraction of its *calibrated*
               full-scale amplitude, so only the SHAPE is portable; the absolute
               Rabi scale is set by the device's frame calibration, not by us.
               The peak physical amplitude is returned so you can match it to the
               device's calibrated drive strength.

    Returns (waveform, peak) where peak = max|envelope| (the physical scale that
    `normalize` divided out; 1.0 if normalize=False).
    """
    _, ArbitraryWaveform, _, _ = _require_braket()
    env = np.asarray(envelope).ravel()
    peak = float(np.max(np.abs(env))) if env.size else 0.0
    samples = (env / peak) if (normalize and peak > 0) else env
    wf = ArbitraryWaveform(samples.tolist(), id=waveform_id)
    return wf, peak


def verify_waveform_roundtrip(envelope: np.ndarray, *, normalize: bool = True) -> float:
    """Export an envelope and read it back; return max|exported - original|.

    The export is faithful iff this is ~0 (machine precision). This is the
    offline guarantee that nothing is lost crossing into the Braket waveform
    object -- the one part of the hardware path testable without a device.
    """
    env = np.asarray(envelope).ravel()
    wf, peak = to_braket_waveform(env, normalize=normalize)
    got = np.asarray(wf.amplitudes)
    ref = (env / peak) if (normalize and peak > 0) else env
    if got.size != ref.size:
        return float("inf")
    return float(np.max(np.abs(got - ref))) if got.size else 0.0


# ---------------------------------------------------------------------------
# 2. Gate PulseSequence construction (fully offline; serializes to OpenPulse 3.0)
# ---------------------------------------------------------------------------
def synthetic_frames(n_channels: int, *, base_freq_hz: float = 5.0e9,
                     dt_s: float = 1.0e-9):
    """Build placeholder ``Frame``/``Port`` objects so a PulseSequence can be
    constructed and serialized OFFLINE (for export tests / inspection).

    On real hardware you pass the device's actual frames
    (``device.frames`` / ``device.properties.pulse``) instead -- the frame and
    port identifiers are device-specific and are the user's plug-in point.
    Channel order matches the optimizer: q1 drive, q2 drive, coupler[, phase][, Stark].
    """
    _, _, Frame, Port = _require_braket()
    names = ["q1_drive", "q2_drive", "coupler", "coupler_phase", "stark"]
    frames = []
    for c in range(n_channels):
        nm = names[c] if c < len(names) else f"ch{c}"
        port = Port(port_id=f"{nm}_port", dt=dt_s)
        frames.append(Frame(frame_id=f"{nm}_frame", port=port,
                            frequency=base_freq_hz, phase=0.0))
    return frames


def build_gate_pulse_sequence(waveform: np.ndarray, frames: Sequence,
                              *, normalize: bool = True):
    """Bind a saved [n_slices, n_channels] envelope to ``frames`` as one
    ``braket.pulse.PulseSequence`` (the gate), one ArbitraryWaveform per channel
    played simultaneously, then a barrier. Returns the PulseSequence.

    Verify offline with ``seq.to_ir()`` (valid OpenPulse 3.0). This does NOT submit
    anything; ``frames`` may be ``synthetic_frames(...)`` for inspection or the
    device's real frames for execution.
    """
    PulseSequence, _, _, _ = _require_braket()
    wf = np.asarray(waveform)
    if wf.ndim == 1:
        wf = wf[:, None]
    n_ch = wf.shape[1]
    if len(frames) < n_ch:
        raise ValueError(f"need >= {n_ch} frames for {n_ch} channels, got {len(frames)}")
    seq = PulseSequence()
    for c in range(n_ch):
        w, _ = to_braket_waveform(wf[:, c], waveform_id=f"grad_ch{c}", normalize=normalize)
        seq = seq.play(frames[c], w)
    seq = seq.barrier(list(frames[:n_ch]))
    return seq


def build_bench_cz_pulse_sequence(flux_waveform: np.ndarray, flux_frame, *,
                                  peak_amplitude: float = 1.0,
                                  drive_frames: Optional[Sequence] = None,
                                  virtual_z: Sequence[float] = (0.0, 0.0),
                                  normalize: bool = True,
                                  waveform_id: str = "grad_bench_cz"):
    """Bind a gradpulse gate-activation waveform to a device CZ frame as ONE
    benchmarked-gate ``PulseSequence`` -- the Level-B pulse the interleaved RB measures.

    flux_waveform : 1-D samples of the gate-ACTIVATION channel, as a PHYSICAL activation
        that rests at 0 (no drive -> no flux). The samples are normalized by their peak
        |amplitude| and scaled to ``peak_amplitude``, so a value of 0 must mean "coupler
        at rest". For a baseband tunable-coupler device (Cepheus, Sycamore) this is the
        coupler flux of a ``tunable_coupler_cz`` result -- but that model's
        ``best_waveform[:, coupler_idx]`` is a [0,1] ENVELOPE where 0.5 is rest, so pass
        the physical flux ``u = 2*best_waveform[:, coupler_idx] - 1`` (NOT the raw [0,1]
        channel, which would play rest at a large DC offset -- a different, wrong pulse).
        For a parametrically-activated coupler, it is the parametric drive channel (already
        rest-0). Match the gradpulse architecture to the device's activation mechanism or
        the transfer is meaningless.
    flux_frame : the device's CZ/flux ``Frame`` (from ``device.frames``); the same
        frame the native CZ plays on, so the gradpulse shape drives the coupler the
        device has already characterized.
    peak_amplitude : physical full-scale the (unit-peak, if ``normalize``) shape is
        multiplied by. Set it to the device's OWN native-CZ flux peak
        (``bench_cz_peak_from_native_calibration``) to anchor the gradpulse pulse to a
        flux scale known to produce ~a CZ. This anchoring is open-loop: the map from a
        gradpulse model amplitude to a device DAC amplitude is not the identity, so the
        transferred pulse is a STARTING point for on-device calibration, not a
        calibrated gate. Expect it to under-perform the device's closed-loop-tuned
        native CZ until refined (that gap is what the HITL calibration hooks close).
    drive_frames : optional ``(frame_q0, frame_q1)`` for the single-qubit virtual-Z
        corrections that close the CZ, applied as ``shift_phase``. ``None`` -> flux only.
    virtual_z : ``(phi0, phi1)`` rad on ``drive_frames``. The device's virtual-Z is
        separately calibrated; supply a model estimate or refine with a short on-device
        phase sweep. Default ``(0, 0)`` plays the bare entangling flux.

    Returns a ``braket.pulse.PulseSequence`` (offline-serializable via ``seq.to_ir()``;
    does NOT submit). Braket's local simulator cannot execute pulse programs, so unlike
    the Level-A gate path this yields no off-device survival number -- its fidelity
    needs the QPU.
    """
    PulseSequence, ArbitraryWaveform, _, _ = _require_braket()
    env = np.asarray(flux_waveform).ravel()
    pk = float(np.max(np.abs(env))) if env.size else 0.0
    # An activation must rest at ~0 at its endpoints. If both ends sit far from 0 (e.g. a
    # [0,1] tunable-coupler envelope where rest=0.5 passed by mistake), the played pulse
    # holds a large DC flux offset -- a different, wrong gate. Warn loudly rather than
    # silently transfer garbage onto the QPU.
    if pk > 0 and env.size >= 2:
        rest = max(abs(float(env[0])), abs(float(env[-1]))) / pk
        if rest > 0.25:
            import warnings
            warnings.warn(
                f"build_bench_cz_pulse_sequence: waveform endpoints sit at {rest:.0%} of "
                "peak, not ~0 -- this looks like a [0,1] envelope, not a physical rest-0 "
                "activation. For a tunable_coupler_cz result pass u = 2*best_waveform[:,c]-1, "
                "not the raw channel. Proceeding, but the played pulse will carry a DC offset.",
                RuntimeWarning, stacklevel=2)
    shape = (env / pk) if (normalize and pk > 0) else env
    samples = (shape * float(peak_amplitude))
    wf = ArbitraryWaveform(np.asarray(samples).tolist(), id=waveform_id)

    seq = PulseSequence()
    frames = [flux_frame]
    if drive_frames is not None:
        f0, f1 = drive_frames
        phi0, phi1 = virtual_z
        if phi0:
            seq = seq.shift_phase(f0, float(phi0))
        if phi1:
            seq = seq.shift_phase(f1, float(phi1))
        frames = [flux_frame, f0, f1]
    seq = seq.play(flux_frame, wf)
    seq = seq.barrier(frames)
    return seq


# ---------------------------------------------------------------------------
# 3. Cost / feasibility (honest sub-$50 question)
# ---------------------------------------------------------------------------
@dataclass
class CostEstimate:
    device: str
    n_circuits: int
    n_shots: int
    task_fee_usd: float
    shot_fee_usd: float
    total_usd: float
    pricing_as_of: str
    note: str = ""


def estimate_experiment_cost(n_circuits: int, n_shots: int,
                             device: str = "Rigetti-Cepheus-1-108Q") -> CostEstimate:
    """Braket QPU cost = per-task fee * n_circuits + per-shot fee * n_circuits * n_shots.

    Each distinct circuit submission is a task. (Batching changes orchestration, not
    the per-task/per-shot fees.)
    """
    if device not in BRAKET_QPU_PRICING:
        raise KeyError(f"unknown device '{device}'; known: {list(BRAKET_QPU_PRICING)}")
    per_task, per_shot = BRAKET_QPU_PRICING[device]
    note = ""
    if per_shot is None:
        per_shot = 0.0
        note = f"per-shot price for '{device}' not in table -- shot fee omitted; confirm on the pricing page."
    task_fee = per_task * n_circuits
    shot_fee = per_shot * n_circuits * n_shots
    return CostEstimate(device=device, n_circuits=int(n_circuits), n_shots=int(n_shots),
                        task_fee_usd=task_fee, shot_fee_usd=shot_fee,
                        total_usd=task_fee + shot_fee, pricing_as_of=_PRICING_AS_OF,
                        note=note)


def irb_circuit_count(lengths: Sequence[int], n_seeds: int) -> int:
    """Interleaved RB = a reference and an interleaved sequence at each (length, seed)."""
    return 2 * len(list(lengths)) * int(n_seeds)


def largest_irb_under_budget(budget_usd: float = 50.0,
                             device: str = "Rigetti-Cepheus-1-108Q",
                             n_shots: int = 500,
                             lengths: Sequence[int] = (1, 2, 4, 8, 16)) -> dict:
    """Largest interleaved-RB design (most seeds) whose cost fits ``budget_usd``.

    Answers the user's question directly: what real measurement fits under $X?
    Reports the design, cost, and a rough fidelity resolution (~1/sqrt(seeds*shots)
    per length -- a guide, not a rigorous error bar). Returns ``n_seeds=0`` if even
    a single seed exceeds the budget (lower shots or pick a cheaper device).
    """
    lengths = list(lengths)
    best = None
    for n_seeds in range(1, 201):
        n_circ = irb_circuit_count(lengths, n_seeds)
        cost = estimate_experiment_cost(n_circ, n_shots, device).total_usd
        if cost > budget_usd:
            break
        best = (n_seeds, n_circ, cost)
    if best is None:
        return {"feasible": False, "device": device, "budget_usd": budget_usd,
                "n_shots": n_shots, "lengths": lengths,
                "reason": "even 1 seed exceeds budget; reduce shots or change device."}
    n_seeds, n_circ, cost = best
    # crude per-point resolution; IRB fidelity error bar is roughly this scale.
    resolution = 1.0 / np.sqrt(max(1, n_seeds) * max(1, n_shots))
    return {"feasible": True, "device": device, "budget_usd": budget_usd,
            "n_shots": n_shots, "lengths": lengths, "n_seeds": n_seeds,
            "n_circuits": n_circ, "est_cost_usd": cost,
            "approx_fidelity_resolution": float(resolution),
            "note": "coarse but real measured fidelity; tighten with more seeds/shots "
                    "(and budget). Resolution is a rough guide, not a fitted error bar."}


# ---------------------------------------------------------------------------
# 4. Interleaved-RB circuit generation (pure; the device.run() input)
# ---------------------------------------------------------------------------
# Reuses the 11520-element 2-qubit Clifford group from gradpulse.rb, TRANSPILED to the
# Rigetti-Cepheus native set -- RX(+/-pi/2), RZ(theta), CZ -- since that is the only
# gate set a verbatim box accepts (see to_braket_rb_circuit); H/S are not native and
# would be rejected or silently recompiled otherwise. The recovery Clifford is
# appended, so an ideal native run returns to |00>. Pure (no braket, no device) and
# unit-tested; only submit_irb (examples/run_irb_on_braket.py) touches silicon.
_HALF_PI = float(np.pi / 2.0)
_CZ4 = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)


def _rx2(theta):
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _rz2(theta):
    return np.array([[np.exp(-1j * theta / 2.0), 0.0],
                     [0.0, np.exp(1j * theta / 2.0)]], dtype=complex)


# Abstract generator -> native-gate word. H = RZ(pi/2) RX(pi/2) RZ(pi/2), S = RZ(pi/2),
# up to a global phase that cancels over the recovery-closed sequence.
_WORD_TO_NATIVE = {
    "H1": [("rz", (0,), _HALF_PI), ("rx", (0,), _HALF_PI), ("rz", (0,), _HALF_PI)],
    "H2": [("rz", (1,), _HALF_PI), ("rx", (1,), _HALF_PI), ("rz", (1,), _HALF_PI)],
    "S1": [("rz", (0,), _HALF_PI)],
    "S2": [("rz", (1,), _HALF_PI)],
    "CZ": [("cz", (0, 1))],
}
# The interleaved (benchmarked) gate is tagged distinctly so the circuit builder can
# bind it to the native CZ (Level A) or a custom gradpulse pulse (Level B), while the
# CZs *inside* Cliffords always stay native.
_BENCH_CZ = ("CZ_BENCH", (0, 1))


def _word_to_gates(word):
    out = []
    for g in word:
        out.extend(_WORD_TO_NATIVE[g])
    return out


def _gate_unitary_4x4(name, qubits, param=None):
    """Ideal 4x4 computational-subspace unitary for a native gate (for the offline
    return-to-|00> check). rx/rz act on one qubit; cz / CZ_BENCH are the ideal CZ."""
    if name in ("cz", "CZ_BENCH"):
        return _CZ4
    if name == "rx":
        one = _rx2(param)
    elif name == "rz":
        one = _rz2(param)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown native gate {name!r}")
    return np.kron(one, np.eye(2, dtype=complex)) if qubits[0] == 0 \
        else np.kron(np.eye(2, dtype=complex), one)


def native_rb_sequences(lengths: Sequence[int], n_seeds: int, *, seed: int = 0,
                        interleaved: bool = False):
    """Native-gate interleaved-RB sequences (the hardware circuits, pre-submission).

    Each entry is ``{"length", "seed_idx", "interleaved", "gates"}`` where ``gates``
    is a list of ``(gate_name, qubit_indices)`` over ``h/s/cz`` plus, when
    ``interleaved``, a tagged ``CZ_BENCH`` after every Clifford (the benchmarked
    gate). The recovery Clifford that inverts the ideal product is appended, so an
    ideal run returns to |00>. Mirrors gradpulse.rb's protocol but emits gate words
    instead of superoperators. Pure -- no braket, no device.
    """
    from .rb import two_qubit_cliffords
    group = two_qubit_cliffords()
    n_cliff = len(group)
    rng = np.random.default_rng(seed if not interleaved else seed + 1)
    out = []
    for m in lengths:
        for s in range(int(n_seeds)):
            gates, ideal = [], np.eye(4, dtype=complex)
            for _ in range(int(m)):
                idx = int(rng.integers(n_cliff))
                gates.extend(_word_to_gates(group.words[idx]))
                ideal = group.unitaries[idx] @ ideal
                if interleaved:
                    gates.append(_BENCH_CZ)
                    ideal = _CZ4 @ ideal
            rec_idx = group.index_of(ideal.conj().T)         # recovery = inverse
            gates.extend(_word_to_gates(group.words[rec_idx]))
            out.append({"length": int(m), "seed_idx": int(s),
                        "interleaved": bool(interleaved), "gates": gates})
    return out


def ideal_survival_probability(gates) -> float:
    """Offline correctness check: apply the ideal gates to |00> and return
    |<00|psi>|^2. Any sequence from native_rb_sequences must give ~1.0 (the recovery
    inverts the ideal product) -- this is what guarantees the circuits are right
    BEFORE spending a cent on the device."""
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1.0
    for g in gates:
        name, qubits = g[0], g[1]
        param = g[2] if len(g) > 2 else None
        psi = _gate_unitary_4x4(name, qubits, param) @ psi
    return float(np.abs(psi[0]) ** 2)


def survival_from_counts(counts) -> float:
    """P(|00>) from a measurement-counts dict ({'00': n00, '01': n01, ...}); the RB
    survival observable. Accepts braket's ``measurement_counts`` (a Counter)."""
    total = sum(int(v) for v in counts.values())
    if total == 0:
        return 0.0
    n00 = sum(int(v) for k, v in counts.items() if str(k)[:2] == "00")
    return n00 / total


def to_braket_rb_circuit(gates, qubits: Sequence[int] = (0, 1), *,
                         bench_cz_pulse=None, verbatim: bool = True,
                         buffer_bench_cz: bool = False):
    """Build a ``braket.circuits.Circuit`` from a native-gate sequence.

    The gates (RX(+/-pi/2), RZ, CZ) are placed inside a VERBATIM BOX so Rigetti's
    compiler preserves the sequence exactly. This is mandatory for RB: without it the
    compiler cancels each random Clifford against its recovery (net identity) and the
    decay curve goes flat -- you would measure ~1.0 at every length and learn nothing.
    The verbatim box also requires the native gate set, which is why H/S are transpiled
    to RX/RZ upstream (see _WORD_TO_NATIVE). Pass ``verbatim=False`` only for a
    noiseless local-simulator sanity run where recompilation is harmless.

    ``qubits`` are the two physical device qubits. ``CZ_BENCH`` markers bind to the
    native ``cz`` when ``bench_cz_pulse is None`` (Level A -- benchmarks the device
    CZ), or, when ``bench_cz_pulse`` is a ``PulseSequence`` (from
    ``build_bench_cz_pulse_sequence``), to a ``pulse_gate`` playing that gradpulse
    waveform (Level B -- benchmarks YOUR pulse). The pulse gate lives INSIDE the
    verbatim box, so the random Cliffords still run as exact native gates while only
    the interleaved gate is the gradpulse pulse -- which is what isolates its error in
    interleaved RB. The Level-B pulse needs the device's real CZ frame + calibrated
    flux scale; see examples/run_irb_on_braket.py for the device-side wiring.

    ``buffer_bench_cz`` (default False = byte-identical to the original) wraps each
    benchmarked CZ in ``barrier``s on the two qubits. This addresses a CONTEXT bias
    observed on a real Level-A run (RESULTS.md S10): interleaving a CZ after every
    Clifford put the benchmarked CZ back-to-back with a Clifford's own CZ ~9x more
    often than in the reference arm (11.8% of CZs vs 1.3%), and back-to-back flux
    pulses error more than well-separated ones, inflating the naive interleaved-RB
    number ~2.5x above the isolated-gate error. A barrier is a native, identity (so the
    ideal return-to-|00> is unchanged), scheduling-level boundary that stops the
    compiler abutting the benchmarked CZ's flux pulse with its neighbours. NOTE: this
    is offline-verified to remove the instruction-level adjacency only; whether it
    recovers the isolated ~0.5% on silicon is UNVERIFIED (the effect is non-Markovian,
    so simulation cannot confirm it, and a residual may be flux-predistortion-limited,
    which the verbatim box bypasses). It needs a hardware run to confirm.
    """
    try:
        from braket.circuits import Circuit
    except ImportError as e:  # pragma: no cover - exercised only without the SDK
        raise ImportError("to_braket_rb_circuit needs amazon-braket-sdk "
                          "(`pip install amazon-braket-sdk`).") from e
    q0, q1 = int(qubits[0]), int(qubits[1])
    phys = (q0, q1)
    body = Circuit()
    for g in gates:
        name, qs = g[0], g[1]
        param = g[2] if len(g) > 2 else None
        if name == "rx":
            body.rx(phys[qs[0]], param)
        elif name == "rz":
            body.rz(phys[qs[0]], param)
        elif name == "cz":
            body.cz(q0, q1)
        elif name == "CZ_BENCH":
            if buffer_bench_cz:
                body.barrier([q0, q1])           # isolate the benchmarked gate from
            if bench_cz_pulse is None:           # adjacent native-CZ flux pulses
                body.cz(q0, q1)                              # Level A: native CZ
            else:                                            # Level B: gradpulse pulse
                body.pulse_gate([q0, q1], bench_cz_pulse)
            if buffer_bench_cz:
                body.barrier([q0, q1])
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown native gate {name!r}")
    return Circuit().add_verbatim_box(body) if verbatim else body


def verify_levelb_offline(bench_cz_pulse, *, qubits: Sequence[int] = (0, 1),
                          n_cliffords: int = 4, seed: int = 0) -> dict:
    """Everything about a Level-B (gradpulse-pulse) IRB circuit checkable WITHOUT a QPU.

    Confirms: the bench pulse serializes to OpenPulse; an interleaved-RB circuit
    carrying it as a pulse gate serializes to OpenQASM 3 with the verbatim pragma AND
    an embedded ``play`` (so the random Cliffords are preserved and the gradpulse pulse
    is really in the program); and the ideal Clifford structure still closes to |00>.

    What it does NOT check: the gate's fidelity. Braket's local simulator runs gates,
    not pulse programs, so a Level-B number requires ``device.run()``. ``offline_ok``
    gates everything that CAN be verified before spending; the rest needs silicon.
    """
    seqs = native_rb_sequences([int(n_cliffords)], 1, seed=seed, interleaved=True)
    gates = seqs[0]["gates"]
    ideal_ok = abs(ideal_survival_probability(gates) - 1.0) < 1e-9
    openpulse = str(bench_cz_pulse.to_ir())
    circ = to_braket_rb_circuit(gates, qubits=qubits, bench_cz_pulse=bench_cz_pulse)
    qasm = circ.to_ir(ir_type="OPENQASM").source
    pragma = "pragma braket verbatim" in qasm
    has_play = "play(" in qasm
    return {
        "bench_pulse_openpulse_chars": len(openpulse),
        "circuit_openqasm_chars": len(qasm),
        "verbatim_pragma_present": bool(pragma),
        "play_present": bool(has_play),
        "ideal_clifford_closes": bool(ideal_ok),
        "offline_ok": bool(ideal_ok and pragma and has_play),
        "NOT_validated_needs_silicon": "pulse-program execution + the gate fidelity -- "
            "the local simulator runs gates, not pulses; a Level-B number needs "
            "device.run() (open-loop transfer, so expect it below the device's "
            "closed-loop-calibrated native CZ until on-device calibration).",
    }


# ---------------------------------------------------------------------------
# 5. Native gate-calibration introspection -- the REAL per-pair CZ duration
# ---------------------------------------------------------------------------
# The standardized device-properties doc carries T1/T2 and CZ fidelity but NOT the
# gate time, which lives in the separate native-gate-calibration doc (device.
# properties.pulse.nativeGateCalibrationsRef): each CZ is a `play` of a flux waveform
# on the coupler frame, and that waveform's length IS the gate duration. Pure
# JSON/array work: no SDK, no network, testable offline.
def _cz_flux_amplitude(cal: dict, waveform_id):
    """|amplitude| samples of a calibration waveform, or None if not sampled.

    Braket stores a sampled waveform as ``amplitudes = [[re, im], ...]``; a
    templated/parametric waveform has no explicit samples (returns None).
    """
    w = (cal.get("waveforms") or {}).get(waveform_id)
    if not isinstance(w, dict):
        return None
    amps = w.get("amplitudes")
    if amps is None:
        return None
    a = np.asarray(amps, dtype=float)
    if a.ndim == 2 and a.shape[1] >= 2:
        return np.abs(a[:, 0] + 1j * a[:, 1])
    return np.abs(a.ravel())


def _cz_amp_for_site(cal: dict, entry: dict):
    """|amplitude| samples of a site's CZ flux ``play``, or None. The CZ is a play on
    a frame whose id contains 'cz' or 'flux'; this walks the entry's calibrations to
    find it. Shared by the duration reader and the bench-anchoring peak reader."""
    cz_key = next((g for g in entry if str(g).lower() == "cz"), None)
    if cz_key is None:
        return None
    for item in entry[cz_key]:
        if str(item.get("name", "")).lower() != "cz":
            continue
        for c in item.get("calibrations", []):
            if c.get("name") != "play":
                continue
            args = {a.get("name"): a.get("value") for a in c.get("arguments", [])}
            frame = str(args.get("frame", "")).lower()
            if "cz" in frame or "flux" in frame:
                amp = _cz_flux_amplitude(cal, args.get("waveform"))
                if amp is not None:
                    return amp
    return None


def _duration_from_amplitude(amp: np.ndarray, mode: str, threshold: float) -> float:
    """Gate time in samples for one of three honest definitions of "duration"."""
    peak = float(np.max(amp))
    if peak <= 0.0:
        return 0.0
    if mode == "buffer":            # full scheduled slot (longest)
        return float(amp.size)
    if mode == "effective":         # area/peak = equivalent flat-top width (shortest)
        return float(amp.sum() / peak)
    if mode == "active":            # flux meaningfully on; the coherence-relevant time
        nz = np.where(amp > threshold * peak)[0]
        return float(nz[-1] - nz[0] + 1) if nz.size else 0.0
    raise ValueError(f"mode must be 'active'|'buffer'|'effective', got {mode!r}")


def cz_durations_from_native_calibration(cal: dict, *, mode: str = "active",
                                         threshold: float = 0.01,
                                         dt_ns: float = 1.0) -> dict:
    """Real per-pair CZ gate durations (ns) from a Braket native-gate-calibration dict.

    ``cal`` is the parsed JSON behind ``device.properties.pulse.nativeGateCalibrationsRef``
    (download it once: it is a presigned URL). For every coupled pair this finds the
    CZ's ``play`` on the coupler/flux frame and measures its waveform length.

    ``mode`` picks among three legitimate -- and genuinely different -- definitions of
    "gate time", because the flux pulse is a zero-padded raised cosine:

      - ``"buffer"``    : full sample count = the circuit time-slot the CZ occupies.
      - ``"active"``    : first-to-last sample above ``threshold * peak`` = the time the
                          flux is meaningfully on. This is the coherence-relevant duration
                          and the DEFAULT (empirically it is the median ~= the old 60 ns
                          assumption, and it is the time the qubits are actually detuned).
      - ``"effective"`` : area/peak = the equivalent flat-top width (shortest).

    The spread between these is ~2x (e.g. 80 / 60 / 38 ns medians on Cepheus-1-108Q),
    so the choice is itself a ~+-30% lever on any coherence-floor number -- report which
    you used. ``dt_ns`` is the sample period (1 ns on Rigetti flux channels).

    Returns ``{"a-b": duration_ns}`` keyed by hyphen-joined node ids (CZ symmetric).
    Pairs whose CZ has no sampled flux waveform are omitted.
    """
    gates = cal.get("gates", cal)
    out = {}
    for site, entry in gates.items():
        if not isinstance(entry, dict):
            continue
        amp = _cz_amp_for_site(cal, entry)
        if amp is None or amp.size == 0:
            continue
        dur = _duration_from_amplitude(amp, mode, threshold) * float(dt_ns)
        if dur > 0.0:
            out[str(site).replace("_", "-")] = dur
    return out


def bench_cz_peak_from_native_calibration(cal: dict, site) -> float:
    """Peak ``|flux amplitude|`` of the device's native CZ on ``site`` (a node-pair
    key like ``"16-25"`` or ``"16_25"``). Pass it as
    ``build_bench_cz_pulse_sequence(peak_amplitude=...)`` so a gradpulse shape is
    anchored to the device's OWN calibrated CZ flux full-scale -- the open-loop
    starting amplitude for benchmarking a gradpulse pulse against the native gate.
    Raises if the site has no sampled CZ flux waveform."""
    gates = cal.get("gates", cal)
    s = str(site)
    entry = (gates.get(s) or gates.get(s.replace("-", "_"))
             or gates.get(s.replace("_", "-")))
    if not isinstance(entry, dict):
        raise KeyError(f"no native-calibration entry for site {site!r}")
    amp = _cz_amp_for_site(cal, entry)
    if amp is None or amp.size == 0:
        raise ValueError(f"site {site!r} has no sampled CZ flux waveform")
    return float(np.max(amp))


# ---------------------------------------------------------------------------
# Readiness report -- ties the offline-verifiable chain together
# ---------------------------------------------------------------------------
def hardware_readiness_report(waveform: np.ndarray, *, n_channels: Optional[int] = None,
                              device: str = "Rigetti-Cepheus-1-108Q",
                              budget_usd: float = 50.0, n_shots: int = 500,
                              verbose: bool = True) -> dict:
    """Run the full OFFLINE-verifiable hardware path on a saved pulse and report.

    Checks (no credentials, no cost): waveform round-trips faithfully, the gate
    PulseSequence builds + serializes to OpenPulse, and a real interleaved-RB
    experiment fits under ``budget_usd``. States plainly the one step left --
    ``device.run()`` -- and the device-mismatch caveat. Returns a dict; prints a
    summary if ``verbose``.
    """
    wf = np.asarray(waveform)
    if wf.ndim == 1:
        wf = wf[:, None]
    n_ch = n_channels or wf.shape[1]

    roundtrip = max(verify_waveform_roundtrip(wf[:, c]) for c in range(n_ch))
    frames = synthetic_frames(n_ch)
    seq = build_gate_pulse_sequence(wf, frames)
    openpulse = seq.to_ir()
    plan = largest_irb_under_budget(budget_usd, device, n_shots)

    report = {
        "waveform_roundtrip_maxerr": roundtrip,
        "waveform_export_faithful": roundtrip < 1e-9,
        "openpulse_chars": len(openpulse),
        "pulse_sequence_builds": True,
        "irb_plan": plan,
        "validated_offline": ["waveform export (round-trip)",
                              "PulseSequence build + OpenPulse serialization",
                              "IRB protocol + analysis (gradpulse.rb.interleaved_rb)",
                              "cost estimate"],
        "NOT_validated_needs_silicon": "device.run() -- real measurement; needs AWS "
                                       "credentials + the estimated cost. This is the "
                                       "only step that closes sim != hardware.",
        "scientific_caveat": "pulse targets representative model params, not a specific "
                             "device qubit; re-optimize against the device's real "
                             "calibration before trusting an absolute hardware number.",
    }
    if verbose:
        print(f"[braket readiness] waveform round-trip max err = {roundtrip:.2e} "
              f"({'FAITHFUL' if report['waveform_export_faithful'] else 'LOSSY!'})")
        print(f"[braket readiness] gate PulseSequence builds + serializes "
              f"({report['openpulse_chars']} chars OpenPulse 3.0)")
        if plan.get("feasible"):
            print(f"[braket readiness] under ${budget_usd:.0f} on {device}: "
                  f"IRB with {plan['n_seeds']} seeds x {len(plan['lengths'])} lengths "
                  f"= {plan['n_circuits']} circuits @ {n_shots} shots "
                  f"~ ${plan['est_cost_usd']:.2f} "
                  f"(fidelity resolution ~{plan['approx_fidelity_resolution']:.1e})")
        else:
            print(f"[braket readiness] no IRB fits under ${budget_usd:.0f}: {plan['reason']}")
        print("[braket readiness] ONE step left -> device.run() with your AWS "
              "credentials. Everything above is built and offline-validated.")
    return report
