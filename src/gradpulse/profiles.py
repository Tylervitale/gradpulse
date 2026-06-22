"""gradpulse.profiles - device parameter profiles.

``ParametricCouplerProfile`` holds the physical parameters of a tunable-transmon
parametric-coupler pair (frequencies, anharmonicities, T1/T2, coupling/drive
rates, plus optional realism knobs). It is imported and re-exported by
``gradpulse.parametric`` (so ``from gradpulse import ParametricCouplerProfile``
keeps working) and is the single argument to ``ParametricCZOptimizer``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List


# ---- Device profile --------------------------------------------------------

class RepresentativeDefaultsWarning(UserWarning):
    """Warn that a profile is built from representative published-typical device
    parameters, not a real calibration -- so its fidelity is a design-tool number,
    not a hardware prediction. Emitted when *every* device-identity field (T1/T2,
    frequencies, anharmonicity) is still at its default; any ``from_*`` loader or
    hand-set device value silences it. Filter with::

        import warnings, gradpulse
        warnings.filterwarnings("ignore", category=gradpulse.RepresentativeDefaultsWarning)
    """


@dataclass
class ParametricCouplerProfile:
    """Tunable-transmon parametric-coupler profile.

    Defaults are representative published-typical transmon values, not
    measurements of any specific device. Override via constructor kwargs once
    you have calibration data for your assigned qubits (e.g. from Braket's
    `device.properties`).
    """
    qubit_pair: tuple = (4, 5)

    # Fock levels per transmon (Hilbert space = n_levels**2). Default 3 (qutrit) is
    # the minimal truncation resolving |2> leakage + the |11>-|02> CZ mechanism;
    # raise to 4+ to check truncation convergence. Must be >= 3.
    n_levels: int = 3
    # T1 (energy relaxation) in nanoseconds
    t1_ns_q1: float = 30_000.0
    t1_ns_q2: float = 30_000.0
    # T2 (dephasing) in nanoseconds
    t2_ns_q1: float = 25_000.0
    t2_ns_q2: float = 25_000.0
    # Thermal photon occupation n_th of each qubit's bath. 0.0 = ground-state bath
    # (default); > 0 adds a thermal-excitation jump and enhances relaxation to
    # (1+n_th)/T1. Typical superconducting values are ~0.01-0.05.
    n_thermal_q1: float = 0.0
    n_thermal_q2: float = 0.0
    # Qubit frequencies in GHz at flux operating point
    freq_ghz_q1: float = 4.85
    freq_ghz_q2: float = 5.05
    # Anharmonicity in GHz (negative)
    anharm_ghz_q1: float = -0.200
    anharm_ghz_q2: float = -0.200
    # Effective parametric coupling rate, MHz (after Schrieffer-Wolff)
    g_max_mhz: float = 12.0
    # Drive amplitude saturation Rabi rate, MHz
    omega_max_mhz: float = 50.0
    # Static parasitic ZZ from spectator qubits dressing |11>, MHz; typically
    # 0.01-0.1 MHz on current hardware. 0.0 disables crosstalk.
    chi_zz_mhz: float = 0.0
    # Native CZ reference (comparison reporting only, not used in opt);
    # from_braket_calibration overwrites with the measured interleaved-RB fidelity.
    native_cz_fidelity: float = 0.988
    native_cz_duration_ns: float = 150.0

    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Pure dephasing requires T2 <= 2*T1; warn rather than silently floor the
        # rate to ~0 (usually mixed T1/T2 units in a calibration file).
        import warnings
        for q, (t1, t2) in enumerate(((self.t1_ns_q1, self.t2_ns_q1),
                                      (self.t1_ns_q2, self.t2_ns_q2)), start=1):
            if t2 > 2.0 * t1:
                warnings.warn(
                    f"ParametricCouplerProfile q{q}: T2={t2:g} ns > 2*T1={2.0 * t1:g} ns "
                    f"is unphysical for pure dephasing (1/T_phi < 0); the dephasing rate "
                    f"will be floored to ~0. Check your T1/T2 calibration units.",
                    stacklevel=2)

        # If every device-identity field is still at default, no calibration was
        # supplied -- surface the caveat here, not just in the docstring. Any from_*
        # loader (or hand-set device value) moves a field off default and silences this.
        _IDENTITY = ("freq_ghz_q1", "freq_ghz_q2", "anharm_ghz_q1", "anharm_ghz_q2",
                     "t1_ns_q1", "t1_ns_q2", "t2_ns_q1", "t2_ns_q2")
        _default = {f.name: f.default for f in fields(self)}
        if all(getattr(self, n) == _default[n] for n in _IDENTITY):
            warnings.warn(
                "ParametricCouplerProfile is using representative published-typical "
                "device parameters (T1/T2, frequencies, anharmonicity), NOT a measurement "
                "of your device -- the resulting fidelity is a design-tool number, not a "
                "hardware prediction. Load a real calibration via from_braket_calibration "
                "/ from_ibm_backend / from_calibration before quoting a fidelity for a "
                "specific qubit. Silence: warnings.filterwarnings('ignore', "
                "category=gradpulse.RepresentativeDefaultsWarning).",
                RepresentativeDefaultsWarning, stacklevel=2)

    @classmethod
    def from_braket_calibration(
        cls,
        path: str,
        qubit_pair: tuple,
        *,
        require_cz: bool = True,
        **overrides,
    ) -> "ParametricCouplerProfile":
        """Build a profile from a Braket standardized device-properties JSON.

        Populates the fields a calibration file actually measures:

          - ``t1_ns_q1/q2`` and ``t2_ns_q1/q2`` from ``oneQubitProperties[q]``
            (the schema stores T1/T2 in seconds; converted to nanoseconds)
          - ``native_cz_fidelity`` from the pair's ``twoQubitGateFidelity`` CZ
            entry (interleaved RB)

        The *standardized* Braket schema does **not** carry qubit frequency or
        anharmonicity, so ``freq_ghz_q1/q2`` and ``anharm_ghz_q1/q2`` keep their
        representative defaults unless you pass them (or any other field) as a
        keyword via ``**overrides``. Provenance is appended to ``notes``.

        Parameters
        ----------
        path :
            Path to a Braket ``standardized_gate_model_qpu_device_properties``
            JSON (e.g. a device's ``properties.standardized`` saved to disk).
        qubit_pair : tuple[int, int]
            ``(q1, q2)``; ``q1`` maps to the ``*_q1`` fields, ``q2`` to ``*_q2``.
            Either ordering finds the pair's CZ fidelity.
        require_cz : bool, default True
            Raise if the pair has no measured CZ fidelity. If False, keep the
            default ``native_cz_fidelity`` and note that it was not measured.
        **overrides :
            Any ``ParametricCouplerProfile`` field to set after loading
            (e.g. ``freq_ghz_q1=4.48`` once known from another source).

        Raises
        ------
        ValueError
            If the file is not a standardized device-properties document, a
            requested qubit is absent, or (with ``require_cz``) the pair has no
            CZ fidelity.
        TypeError
            If ``**overrides`` names a field that does not exist on the profile.
        """
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)

        one_q = doc.get("oneQubitProperties")
        two_q = doc.get("twoQubitProperties", {})
        if not isinstance(one_q, dict):
            raise ValueError(
                f"{Path(path).name!r} is not a Braket standardized "
                "device-properties file (no 'oneQubitProperties')."
            )

        q1, q2 = int(qubit_pair[0]), int(qubit_pair[1])

        def _coh_ns(qi: int, which: str) -> float:
            node = one_q.get(str(qi))
            if node is None:
                raise ValueError(
                    f"qubit {qi} is not present in the calibration file "
                    f"(it lists {len(one_q)} qubits)."
                )
            entry = node.get(which) or {}
            if "value" not in entry:
                raise ValueError(f"qubit {qi} has no {which} entry.")
            unit = str(entry.get("unit", "S")).upper()
            scale = {"S": 1e9, "NS": 1.0}.get(unit)
            if scale is None:
                raise ValueError(f"unexpected {which} unit {unit!r} for qubit {qi}.")
            return float(entry["value"]) * scale

        loaded = dict(
            qubit_pair=(q1, q2),
            t1_ns_q1=_coh_ns(q1, "T1"), t1_ns_q2=_coh_ns(q2, "T1"),
            t2_ns_q1=_coh_ns(q1, "T2"), t2_ns_q2=_coh_ns(q2, "T2"),
        )

        # CZ fidelity: match the pair regardless of key order or separator.
        cz_fid = None
        want = {str(q1), str(q2)}
        for key, val in two_q.items():
            if set(key.replace("_", "-").split("-")) == want:
                for gate in val.get("twoQubitGateFidelity", []):
                    if str(gate.get("gateName", "")).upper() == "CZ":
                        cz_fid = float(gate["fidelity"])
                        break
                break
        if cz_fid is None and require_cz:
            raise ValueError(
                f"no measured CZ fidelity for pair {(q1, q2)} in the "
                "calibration file (pass require_cz=False to keep the default)."
            )
        if cz_fid is not None:
            loaded["native_cz_fidelity"] = cz_fid

        valid = {f.name for f in fields(cls)}
        unknown = set(overrides) - valid
        if unknown:
            raise TypeError(
                f"unknown ParametricCouplerProfile field(s): {sorted(unknown)}"
            )
        loaded.update(overrides)

        profile = cls(**loaded)

        cz_note = (f"native CZ {cz_fid:.4f}" if cz_fid is not None
                   else "native CZ = default (none measured)")
        profile.notes.append(
            f"Loaded measured T1/T2 and {cz_note} for qubits {(q1, q2)} from "
            f"Braket calibration {Path(path).name!r}."
        )
        if "freq_ghz_q1" not in overrides and "freq_ghz_q2" not in overrides:
            profile.notes.append(
                f"freq_ghz ({profile.freq_ghz_q1}/{profile.freq_ghz_q2}) and "
                f"anharm_ghz ({profile.anharm_ghz_q1}/{profile.anharm_ghz_q2}) "
                "are representative defaults -- the standardized Braket schema "
                "carries no frequency or anharmonicity."
            )
        return profile

    @classmethod
    def from_calibration(
        cls,
        data: dict,
        qubit_pair: tuple,
        *,
        require_cz: bool = False,
        **overrides,
    ) -> "ParametricCouplerProfile":
        """Build a profile from a vendor-neutral *normalized* calibration dict.

        This is the universal loader: unlike :meth:`from_braket_calibration` (whose
        schema carries no frequency or anharmonicity), this accepts a simple,
        documented structure that *does*, so a fully device-specific profile -- the
        two qubits' frequencies, anharmonicities, and T1/T2 -- loads in one call. Any
        device export becomes usable by mapping it onto this structure (the vendor
        adapters such as :meth:`from_ibm_backend` do exactly that and delegate here)::

            {
              "qubits": {
                 <index>: {"freq_ghz": .., "anharm_ghz": .., "t1_ns": .., "t2_ns": ..},
                 ...
              },
              "two_qubit": {                      # optional
                 (q1, q2): {"cz_fidelity": ..},   # key may be a tuple or "q1-q2"
              },
            }

        Every per-qubit field is individually optional: any omitted field keeps its
        representative default and is recorded in ``notes``. T1/T2 may instead be
        given as ``t1_s``/``t2_s`` (seconds) and frequency/anharmonicity as
        ``freq_hz``/``anharm_hz``; units are converted automatically.

        Parameters
        ----------
        data : dict
            The normalized calibration structure above.
        qubit_pair : tuple[int, int]
            ``(q1, q2)`` -- which entries of ``data["qubits"]`` map to the ``*_q1``
            and ``*_q2`` fields.
        require_cz : bool, default False
            Raise if no CZ fidelity is found for the pair.
        **overrides :
            Any profile field to set last (wins over the calibration).
        """
        qubits = data.get("qubits")
        if not isinstance(qubits, dict):
            raise ValueError(
                "calibration data must have a 'qubits' dict mapping qubit index "
                "-> {freq_ghz, anharm_ghz, t1_ns, t2_ns}."
            )
        q1, q2 = int(qubit_pair[0]), int(qubit_pair[1])

        def _node(qi):
            node = qubits.get(qi, qubits.get(str(qi)))
            if node is None:
                raise ValueError(
                    f"qubit {qi} is absent from the calibration "
                    f"(it lists qubits {sorted(map(str, qubits))})."
                )
            return node

        # field name -> (calibration keys to try, unit scale to the profile unit)
        _MAP = {
            "freq_ghz": (("freq_ghz", "frequency_ghz"), 1.0),
            "freq_ghz_from_hz": (("freq_hz", "frequency_hz", "frequency"), 1e-9),
            "anharm_ghz": (("anharm_ghz", "anharmonicity_ghz"), 1.0),
            "anharm_ghz_from_hz": (("anharm_hz", "anharmonicity_hz", "anharmonicity"), 1e-9),
            "t1_ns": (("t1_ns", "T1_ns"), 1.0),
            "t1_ns_from_s": (("t1_s", "T1", "t1"), 1e9),
            "t2_ns": (("t2_ns", "T2_ns"), 1.0),
            "t2_ns_from_s": (("t2_s", "T2", "t2"), 1e9),
        }

        def _pull(node, *specs):
            for spec in specs:
                keys, scale = _MAP[spec]
                for k in keys:
                    if k in node and node[k] is not None:
                        return float(node[k]) * scale
            return None

        loaded = dict(qubit_pair=(q1, q2))
        missing = []
        for tag, qi in (("q1", q1), ("q2", q2)):
            node = _node(qi)
            vals = {
                f"freq_ghz_{tag}": _pull(node, "freq_ghz", "freq_ghz_from_hz"),
                f"anharm_ghz_{tag}": _pull(node, "anharm_ghz", "anharm_ghz_from_hz"),
                f"t1_ns_{tag}": _pull(node, "t1_ns", "t1_ns_from_s"),
                f"t2_ns_{tag}": _pull(node, "t2_ns", "t2_ns_from_s"),
            }
            for field_name, v in vals.items():
                if v is not None:
                    loaded[field_name] = v
                else:
                    missing.append(field_name)

        cz_fid = _find_cz_fidelity(data.get("two_qubit", {}), q1, q2)
        if cz_fid is None and require_cz:
            raise ValueError(
                f"no CZ fidelity for pair {(q1, q2)} in the calibration "
                "(pass require_cz=False to keep the default)."
            )
        if cz_fid is not None:
            loaded["native_cz_fidelity"] = cz_fid

        valid = {f.name for f in fields(cls)}
        unknown = set(overrides) - valid
        if unknown:
            raise TypeError(
                f"unknown ParametricCouplerProfile field(s): {sorted(unknown)}"
            )
        loaded.update(overrides)
        profile = cls(**loaded)

        loaded_fields = sorted(set(loaded) - {"qubit_pair"} - set(overrides))
        profile.notes.append(
            f"Loaded {loaded_fields} for qubits {(q1, q2)} from a normalized "
            "calibration dict."
        )
        still_default = sorted(set(missing) - set(overrides))
        if still_default:
            profile.notes.append(
                f"{still_default} not in the calibration -- kept representative "
                "defaults."
            )
        return profile

    @classmethod
    def from_ibm_backend(
        cls,
        backend,
        qubit_pair: tuple,
        **overrides,
    ) -> "ParametricCouplerProfile":
        """Build a profile from a Qiskit backend (``BackendV1`` or ``BackendV2``).

        IBM backends report qubit frequency, anharmonicity, and T1/T2 together, so
        this returns a fully device-specific profile in one call (unlike the
        standardized-Braket schema, which lacks frequency/anharmonicity). Whatever a
        given backend does not expose (older devices may omit anharmonicity) keeps
        its representative default, recorded in ``notes``. Internally it normalizes
        the backend into the :meth:`from_calibration` structure and delegates.
        """
        cal = _ibm_backend_to_calibration(backend, qubit_pair)
        return cls.from_calibration(cal, qubit_pair, **overrides)


# ----------------------------------------------------------------------------
# Calibration helpers (vendor-neutral normalization)
# ----------------------------------------------------------------------------
# canonical field -> list of (accepted keys, scale to the canonical unit)
_QUBIT_FIELD_SPECS = {
    "freq_ghz": [(("freq_ghz", "frequency_ghz"), 1.0),
                 (("freq_hz", "frequency_hz", "frequency"), 1e-9)],
    "anharm_ghz": [(("anharm_ghz", "anharmonicity_ghz"), 1.0),
                   (("anharm_hz", "anharmonicity_hz", "anharmonicity"), 1e-9)],
    "t1_ns": [(("t1_ns", "T1_ns"), 1.0), (("t1_s", "T1", "t1"), 1e9)],
    "t2_ns": [(("t2_ns", "T2_ns"), 1.0), (("t2_s", "T2", "t2"), 1e9)],
}


def normalize_qubit_node(node: dict) -> dict:
    """Map one raw per-qubit calibration entry to canonical units
    ``{freq_ghz, anharm_ghz, t1_ns, t2_ns}``, accepting Hz/s or GHz/ns inputs.
    Missing fields are returned as ``None``."""
    out = {}
    for canon, options in _QUBIT_FIELD_SPECS.items():
        out[canon] = None
        for keys, scale in options:
            hit = next((node[k] for k in keys if node.get(k) is not None), None)
            if hit is not None:
                out[canon] = float(hit) * scale
                break
    return out


def _find_cz_fidelity(two_qubit: dict, q1: int, q2: int):
    """Return the pair's CZ fidelity from a normalized ``two_qubit`` mapping,
    matching the pair regardless of key order or separator. ``None`` if absent."""
    if not isinstance(two_qubit, dict):
        return None
    want = {str(q1), str(q2)}
    for key, val in two_qubit.items():
        if isinstance(key, (tuple, list)):
            have = {str(int(key[0])), str(int(key[1]))}
        else:
            have = set(str(key).replace("_", "-").split("-"))
        if have == want and isinstance(val, dict):
            for k in ("cz_fidelity", "cz", "fidelity"):
                if k in val and val[k] is not None:
                    return float(val[k])
    return None


def _ibm_backend_to_calibration(backend, qubits) -> dict:
    """Normalize a Qiskit backend into the :meth:`from_calibration` dict.

    Handles both ``BackendV1`` (a ``BackendProperties`` via ``backend.properties()``)
    and ``BackendV2`` (per-qubit ``QubitProperties`` via ``backend.qubit_properties``),
    and also accepts a ``BackendProperties`` object passed directly. Extracts
    frequency (Hz), T1/T2 (s), and -- when the backend exposes it -- anharmonicity
    (Hz). Fields absent on a given backend are simply omitted, so the downstream
    loader keeps their defaults.
    """
    qubits = [int(q) for q in qubits]

    # Resolve a BackendProperties object if the backend is V1 (or one was passed in).
    props = None
    get_props = getattr(backend, "properties", None)
    if callable(get_props):
        try:
            props = get_props()
        except Exception:
            props = None
    if props is None and hasattr(backend, "qubit_property"):
        props = backend  # a BackendProperties was passed directly

    def _qubit_node_v1(q):
        node = {}
        for name, key in (("T1", "t1_s"), ("T2", "t2_s")):
            fn = getattr(props, name.lower(), None)
            try:
                if callable(fn):
                    node[key] = float(fn(q))
            except Exception:
                pass
        try:
            node["frequency_hz"] = float(props.frequency(q))
        except Exception:
            pass
        try:
            val = props.qubit_property(q, "anharmonicity")
            node["anharmonicity_hz"] = float(val[0] if isinstance(val, (tuple, list)) else val)
        except Exception:
            pass
        return node

    def _qubit_node_v2(q):
        node = {}
        qp = backend.qubit_properties(q)
        for attr, key in (("t1", "t1_s"), ("t2", "t2_s"), ("frequency", "frequency_hz")):
            v = getattr(qp, attr, None)
            if v is not None:
                node[key] = float(v)
        anh = getattr(qp, "anharmonicity", None)
        if anh is not None:
            node["anharmonicity_hz"] = float(anh)
        return node

    nodes = {}
    if props is not None:
        for q in qubits:
            nodes[q] = _qubit_node_v1(q)
    elif hasattr(backend, "qubit_properties"):
        for q in qubits:
            nodes[q] = _qubit_node_v2(q)
    else:
        raise ValueError(
            "object does not look like a Qiskit backend: no .properties() "
            "(BackendV1) or .qubit_properties() (BackendV2)."
        )
    if all(not n for n in nodes.values()):
        raise ValueError(
            "could not extract any qubit parameters from the backend "
            f"for qubits {qubits}."
        )
    return {"qubits": nodes}
