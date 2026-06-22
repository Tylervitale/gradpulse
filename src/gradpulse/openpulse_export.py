"""gradpulse.openpulse_export -- vendor-neutral OpenPulse 3.0 / OpenQASM 3 export.

The Braket bridge (``gradpulse.braket_bridge``) covers the AWS pulse-level path. This
module covers the *open standard*: it emits an optimized gate as an OpenQASM 3 program
with a ``defcalgrammar "openpulse"`` calibration block -- the same pulse grammar used
by IBM-style fixed-frequency / cross-resonance hardware. It depends on NOTHING but
numpy: the program is built as text.

WHY NOT ``qiskit.pulse``? Qiskit REMOVED the ``qiskit.pulse`` module in Qiskit 2.0
(2025). Targeting it would mean exporting to a deleted API. OpenPulse 3.0 is the live,
vendor-neutral successor (an extension of OpenQASM 3), so that is what we emit. If you
run an older Qiskit (<2.0) with ``qiskit.pulse`` still present, ``to_qiskit_schedule``
gives you a native ``ScheduleBlock``; otherwise use the text program below, which any
OpenPulse-aware stack ingests.

WHAT IS OFFLINE-VERIFIABLE (no hardware, no account, no cost):
  * The emitted program is round-trip faithful: parse it back with an INDEPENDENT
    OpenPulse parser (the ``openpulse`` package), pull the waveform samples out of the
    AST, and confirm they equal the exported samples to machine precision
    (``verify_openpulse_roundtrip``). This both proves the text is syntactically valid
    OpenPulse 3.0 AND that nothing was lost crossing into it.
  * Complex I/Q is preserved, so a DRAG drive exported from ``optimizer.iq_waveform``
    (in-phase + derived quadrature) lands complete -- the receiver re-derives nothing.

THE ONE STEP THIS DOES NOT CLOSE -- by construction:
  * Running the program on a device. As with the Braket bridge, only the user's real
    backend + credentials + frame calibration can do that, and the same scientific
    caveat applies: a pulse optimized for representative model parameters characterises
    the toolchain, not a device-matched gate, until re-optimized against the device's
    own calibration.

The ``openpulse`` parser (used for the round-trip check) ships with
``amazon-braket-sdk``; it is imported lazily so the gradpulse core needs neither it nor
this module.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Input coercion -- accept the legacy real envelope, a raw complex array, or the
# dict returned by ``optimizer.iq_waveform`` (complex I/Q with DRAG baked in).
# ---------------------------------------------------------------------------
def _coerce(waveform):
    """-> (samples [n_slices, n_ch] complex, labels list, dt_hint or None)."""
    if isinstance(waveform, dict) and "iq" in waveform:
        arr = np.asarray(waveform["iq"])
        labels = list(waveform.get("labels") or [])
        dt_hint = waveform.get("dt_ns")
    else:
        arr = np.asarray(waveform)
        labels = []
        dt_hint = None
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"waveform must be 1-D or 2-D, got shape {arr.shape}")
    arr = arr.astype(complex)
    if not labels or len(labels) != arr.shape[1]:
        labels = [f"ch{c}" for c in range(arr.shape[1])]
    return arr, labels, dt_hint


def _fmt_sample(z: complex, sig: int = 12) -> str:
    """OpenPulse literal for a complex sample: real, or ``a + b im`` / ``a - b im``."""
    re, im = float(z.real), float(z.imag)
    if abs(im) <= 1e-15:
        return f"{re:.{sig}g}"
    sign = "+" if im >= 0 else "-"
    return f"{re:.{sig}g} {sign} {abs(im):.{sig}g}im"


# ---------------------------------------------------------------------------
# 1. Program emission (pure text; valid OpenQASM 3 + OpenPulse grammar)
# ---------------------------------------------------------------------------
def to_openpulse_program(waveform, dt_ns: float = 1.0, *,
                         gate_name: str = "grad_gate",
                         qubits: Sequence[int] = (0, 1),
                         frame_freqs_hz: Optional[Sequence[float]] = None,
                         labels: Optional[Sequence[str]] = None,
                         normalize: bool = True) -> str:
    """Emit a gate as an OpenQASM 3 / OpenPulse 3.0 program string.

    waveform : the saved real [n_slices, n_ch] envelope, a complex [n_slices, n_ch]
        I/Q array, or the dict from ``optimizer.iq_waveform`` (complex, DRAG baked in).
    dt_ns : sample period (ns); written into a ``// dt`` header comment.
    gate_name, qubits : the ``defcal <gate_name> $q0, $q1 { ... }`` signature.
    frame_freqs_hz : per-channel drive frequency for each ``newframe`` (placeholder
        device frames, like braket_bridge.synthetic_frames; the device supplies the
        real ones at run time). Defaults to 5 GHz on every channel.
    normalize : scale each channel by 1/peak|.| so samples land in the unit disk --
        a frame plays a waveform as a fraction of its CALIBRATED full-scale amplitude,
        so only the shape is portable. The per-channel physical peak (in the input's
        units, e.g. rad/ns) is written into a comment so you can match it to the
        device's calibrated drive strength.

    Returns the program text. Verify it offline with ``verify_openpulse_roundtrip``.
    """
    arr, lbls, _ = _coerce(waveform)
    if labels is not None:
        if len(labels) != arr.shape[1]:
            raise ValueError(f"labels length {len(labels)} != n_channels {arr.shape[1]}")
        lbls = list(labels)
    n_slices, n_ch = arr.shape
    freqs = list(frame_freqs_hz) if frame_freqs_hz is not None else [5.0e9] * n_ch
    if len(freqs) < n_ch:
        raise ValueError(f"need >= {n_ch} frame_freqs_hz, got {len(freqs)}")
    qubits = list(qubits)

    peaks = np.max(np.abs(arr), axis=0)
    samples = arr.copy()
    if normalize:
        for c in range(n_ch):
            if peaks[c] > 0:
                samples[:, c] = arr[:, c] / peaks[c]

    L = []
    L.append("OPENQASM 3.0;")
    L.append('defcalgrammar "openpulse";')
    L.append(f"// gradpulse export: {n_slices} samples x {n_ch} channels, dt = {dt_ns} ns")
    L.append(f"// channels: {', '.join(lbls)}")
    L.append(f"// per-channel physical peak |amplitude| (input units): "
             f"{', '.join(f'{lbls[c]}={peaks[c]:.6g}' for c in range(n_ch))}")
    if normalize:
        L.append("// samples below are normalized to unit peak; multiply by the peak "
                 "above for physical amplitude.")
    L.append("cal {")
    for c in range(n_ch):
        L.append(f"    port {lbls[c]}_port;")
    for c in range(n_ch):
        L.append(f"    frame {lbls[c]}_frame = newframe({lbls[c]}_port, "
                 f"{freqs[c]:.6g}, 0.0);")
    for c in range(n_ch):
        body = ", ".join(_fmt_sample(z) for z in samples[:, c])
        L.append(f"    waveform {lbls[c]}_wf = {{{body}}};")
    L.append("}")
    q_args = ", ".join(f"${q}" for q in qubits)
    L.append(f"defcal {gate_name} {q_args} {{")
    for c in range(n_ch):
        L.append(f"    play({lbls[c]}_frame, {lbls[c]}_wf);")
    L.append(f"    barrier {q_args};")
    L.append("}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# 2. Round-trip verification (independent parser -> AST -> samples)
# ---------------------------------------------------------------------------
def _require_openpulse_parser():
    try:
        from openpulse import parse
        return parse
    except ImportError as e:  # pragma: no cover - exercised only without the parser
        raise ImportError(
            "verify_openpulse_roundtrip needs an OpenPulse parser. Install one with "
            "`pip install openpulse` (it also ships with amazon-braket-sdk)."
        ) from e


def _eval_literal(node) -> complex:
    """Evaluate an OpenQASM-AST numeric/imaginary literal expression to a complex."""
    t = type(node).__name__
    if t in ("FloatLiteral", "IntegerLiteral"):
        return complex(float(node.value), 0.0)
    if t == "ImaginaryLiteral":
        return complex(0.0, float(node.value))
    if t == "UnaryExpression":
        v = _eval_literal(node.expression)
        return -v if node.op.name == "-" else v
    if t == "BinaryExpression":
        a, b = _eval_literal(node.lhs), _eval_literal(node.rhs)
        return a + b if node.op.name == "+" else a - b
    raise ValueError(f"unexpected waveform-literal node {t}")


def parse_openpulse_waveforms(program: str) -> dict:
    """Parse a program with the INDEPENDENT openpulse parser and return
    ``{waveform_name: complex ndarray}`` reconstructed from the AST.

    Used by ``verify_openpulse_roundtrip``; also handy on its own to confirm a
    third party's program ingests as you expect.
    """
    parse = _require_openpulse_parser()
    tree = parse(program)
    out = {}
    for stmt in tree.statements:
        if type(stmt).__name__ != "CalibrationStatement":
            continue
        for s in stmt.body:
            init = getattr(s, "init_expression", None)
            if init is None or type(init).__name__ != "ArrayLiteral":
                continue
            name = s.identifier.name
            out[name] = np.array([_eval_literal(v) for v in init.values], dtype=complex)
    return out


def verify_openpulse_roundtrip(waveform, dt_ns: float = 1.0, *,
                               normalize: bool = True, **kwargs) -> float:
    """Emit a program, parse it back independently, and return the max abs error
    between exported and re-parsed samples.

    ~0 (machine precision) means the export is BOTH valid OpenPulse 3.0 (it parsed)
    AND lossless (samples survived). inf if a channel failed to round-trip. This is
    the offline guarantee for the OpenPulse path -- the analogue of the Braket
    bridge's ``verify_waveform_roundtrip``.
    """
    arr, lbls, _ = _coerce(waveform)
    if kwargs.get("labels") is not None:
        lbls = list(kwargs["labels"])
    n_ch = arr.shape[1]
    peaks = np.max(np.abs(arr), axis=0)
    ref = arr.copy()
    if normalize:
        for c in range(n_ch):
            if peaks[c] > 0:
                ref[:, c] = arr[:, c] / peaks[c]
    program = to_openpulse_program(waveform, dt_ns, normalize=normalize, **kwargs)
    got = parse_openpulse_waveforms(program)
    worst = 0.0
    for c in range(n_ch):
        name = f"{lbls[c]}_wf"
        if name not in got or got[name].size != ref.shape[0]:
            return float("inf")
        worst = max(worst, float(np.max(np.abs(got[name] - ref[:, c]))))
    return worst


# ---------------------------------------------------------------------------
# 3. Optional native qiskit.pulse path (only on qiskit < 2.0)
# ---------------------------------------------------------------------------
def to_qiskit_schedule(waveform, *, gate_name: str = "grad_gate", normalize: bool = True):
    """Build a native ``qiskit.pulse.ScheduleBlock`` -- ONLY if ``qiskit.pulse`` is
    importable (i.e. qiskit < 2.0, before the module was removed).

    Raises a clear ImportError on qiskit >= 2.0 pointing at ``to_openpulse_program``,
    which is the supported path on current qiskit.
    """
    try:
        from qiskit import pulse  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "qiskit.pulse was removed in Qiskit 2.0, so a native ScheduleBlock cannot "
            "be built on your install. Use to_openpulse_program(...) for the "
            "vendor-neutral OpenPulse 3.0 text program instead (works on any qiskit)."
        ) from e
    from qiskit import pulse
    arr, lbls, _ = _coerce(waveform)
    n_ch = arr.shape[1]
    peaks = np.max(np.abs(arr), axis=0)
    with pulse.build(name=gate_name) as sched:  # pragma: no cover - needs qiskit<2.0
        for c in range(n_ch):
            samples = arr[:, c] / peaks[c] if (normalize and peaks[c] > 0) else arr[:, c]
            ch = pulse.DriveChannel(c)
            pulse.play(pulse.Waveform(np.asarray(samples, dtype=complex)), ch)
    return sched


# ---------------------------------------------------------------------------
# 4. Readiness report (ties the offline-verifiable chain together)
# ---------------------------------------------------------------------------
def openpulse_readiness_report(waveform, dt_ns: float = 1.0, *,
                               gate_name: str = "grad_gate",
                               qubits: Sequence[int] = (0, 1),
                               verbose: bool = True, **kwargs) -> dict:
    """Run the full OFFLINE-verifiable OpenPulse path on a pulse and report.

    Emits the program, round-trips it through an independent parser, and states the
    one step left (running on a device). Returns a dict; prints a summary if verbose.
    """
    program = to_openpulse_program(waveform, dt_ns, gate_name=gate_name,
                                   qubits=qubits, **kwargs)
    err = verify_openpulse_roundtrip(waveform, dt_ns, gate_name=gate_name,
                                     qubits=qubits, **kwargs)
    arr, lbls, _ = _coerce(waveform)
    has_iq = bool(np.max(np.abs(arr.imag)) > 1e-12)
    report = {
        "openpulse_chars": len(program),
        "n_channels": arr.shape[1],
        "carries_iq_quadrature": has_iq,
        "roundtrip_maxerr": err,
        "roundtrip_faithful": err < 1e-9,
        "program": program,
        "validated_offline": ["OpenPulse 3.0 emission",
                              "independent re-parse (openpulse package)",
                              "lossless sample round-trip"],
        "NOT_validated_needs_hardware": "running the defcal on a device -- needs your "
                                        "backend, credentials, and frame calibration.",
        "scientific_caveat": "pulse targets representative model params; re-optimize "
                             "against the device's real calibration before trusting an "
                             "absolute hardware number.",
    }
    if verbose:
        print(f"[openpulse readiness] emitted {report['openpulse_chars']} chars of "
              f"OpenPulse 3.0 ({arr.shape[1]} channels"
              f"{', complex I/Q with DRAG' if has_iq else ', real envelope'})")
        print(f"[openpulse readiness] independent re-parse round-trip max err = "
              f"{err:.2e} ({'FAITHFUL + VALID' if report['roundtrip_faithful'] else 'PROBLEM'})")
        print("[openpulse readiness] ONE step left -> run the defcal on your backend "
              "with its real frame calibration. Everything above is offline-validated.")
    return report
