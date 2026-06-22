"""gradpulse.multiqubit -- general N-qubit GRAPE: optimize a gate *in the presence
of* its neighbours, so crosstalk and frequency collisions are in the optimization
loop, not just evaluated after.

The two shipped architectures (parametric CZ, cross-resonance ZX) optimize a single
qubit *pair*; a spectator is something you score against (`spectator_fidelity`,
`resonant_collision_fidelity`). This module lifts GRAPE to an arbitrary register:

  * `MultiQubitProfile` -- N transmons (per-qubit frequency, anharmonicity, T1/T2)
    on an arbitrary coupling graph (`couplings={(i,j): g_mhz}`), uniform `n_levels`.
  * `MultiQubitOptimizer` -- autodiff GRAPE for a target gate on ANY subset of the
    register (identity on the rest), with per-qubit drives and per-edge tunable
    couplings. Because every qubit evolves and every coupling acts, optimizing a CZ
    on (0,1) with a coupled qubit 2 present *is* optimizing against that crosstalk /
    collision -- the optimizer is rewarded for leaving the spectator unchanged.

Frame & model (one common rotating frame at `f_ref`, the same convention the
parametric/CR modules use):

    H(t) = sum_q [ Delta_q n_q + (alpha_q/2) n_q(n_q-1) ]            (drift)
         + sum_{(i,j) in fixed edges} g_ij (a_i^dag a_j + a_i a_j^dag)
         + sum_{q in drive_qubits}  Omega_q(t) (a_q + a_q^dag) [+ DRAG quad.]
         + sum_{(i,j) in tunable edges} g_ij(t) (a_i^dag a_j + a_i a_j^dag)

with Delta_q = 2*pi*(f_q - f_ref) -- so when two qubits' detunings collide
(Delta_i ~ Delta_j) their exchange goes resonant, exactly the physics
`resonant_collision_fidelity` diagnoses. Markovian T1/T_phi Lindblad dissipators are
lifted per qubit (same rates as the pair models).

THE WALL (honest): this is *exact* density-matrix simulation, so cost is
exponential in N -- the Hilbert space is `n_levels**N` and the open-system process
fidelity evolves `4**N` Choi operators. No code escapes this; it is the cost of an
exact quantum simulation. Practical envelope on a workstation/GPU:

    open_system=True  (exact, leakage- AND decoherence-aware): N <= 4 (5 on a big GPU)
    open_system=False (unitary; coherent + leakage, no decoherence): N <= 6-7

The constructor prints the dimension and the per-step cost so you see what you are
asking for. `MultiQubitOptimizer(...).cost_estimate()` returns it without building.

Validated the same way as everything else: `validate.multiqubit_cross_check` rebuilds
the model in QuTiP (a different library) and reproduces F_proc to machine precision
(~1e-14 on the 3-qubit CZ-with-spectator, in double precision).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Optional, Sequence

import numpy as np
import torch

try:
    from .parametric import DEVICE
    from .profiles import normalize_qubit_node, _ibm_backend_to_calibration
except ImportError:  # pragma: no cover - direct-script execution
    from parametric import DEVICE
    from profiles import normalize_qubit_node, _ibm_backend_to_calibration

_2PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
@dataclass
class MultiQubitProfile:
    """N-transmon device on an arbitrary coupling graph.

    n_qubits        : register size N.
    freqs_ghz       : per-qubit transition frequency (len N). Differences drive
                      the detunings and hence the collision physics; absolute
                      values matter only through f_ref.
    anharm_mhz      : per-qubit anharmonicity (len N, negative for transmons).
    t1_ns, t2_ns    : per-qubit coherence times (len N).
    couplings       : {(i, j): g_mhz} static exchange on edges (i < j). These are
                      always-on; tunable couplings are selected on the optimizer.
    n_levels        : per-transmon Fock cutoff (uniform). 3 is the validated
                      default for dispersive gates (see the pair models).
    f_ref_ghz       : rotating-frame reference (default = mean of freqs_ghz). Set
                      it to a driven qubit's frequency to drive that qubit on
                      resonance.
    """
    n_qubits: int = 3
    freqs_ghz: Sequence[float] = (5.00, 5.10, 5.20)
    anharm_mhz: Sequence[float] = (-300.0, -300.0, -300.0)
    t1_ns: Sequence[float] = (40_000.0, 40_000.0, 40_000.0)
    t2_ns: Sequence[float] = (30_000.0, 30_000.0, 30_000.0)
    couplings: dict = field(default_factory=lambda: {(0, 1): 12.0, (1, 2): 12.0})
    n_levels: int = 3
    f_ref_ghz: Optional[float] = None
    notes: list = field(default_factory=list)

    def __post_init__(self):
        n = self.n_qubits
        for name in ("freqs_ghz", "anharm_mhz", "t1_ns", "t2_ns"):
            v = list(getattr(self, name))
            if len(v) != n:
                raise ValueError(f"{name} must have length n_qubits={n}, got {len(v)}")
            setattr(self, name, v)
        if self.n_levels < 2:
            raise ValueError("n_levels must be >= 2 (computational subspace is {0,1}).")
        for (i, j) in self.couplings:
            if not (0 <= i < n and 0 <= j < n and i != j):
                raise ValueError(f"coupling edge {(i, j)} out of range for N={n}")
        # Pure dephasing requires T2 <= 2*T1; warn rather than silently floor the
        # rate to ~0 (usually mixed T1/T2 units in a calibration file).
        import warnings
        for q, (t1, t2) in enumerate(zip(self.t1_ns, self.t2_ns)):
            if t2 > 2.0 * t1:
                warnings.warn(
                    f"MultiQubitProfile q{q}: T2={t2:g} ns > 2*T1={2.0 * t1:g} ns is "
                    f"unphysical for pure dephasing (1/T_phi < 0); the dephasing rate "
                    f"will be floored to ~0. Check your T1/T2 calibration units.",
                    stacklevel=2)
        if self.f_ref_ghz is None:
            self.f_ref_ghz = float(np.mean(self.freqs_ghz))

    @classmethod
    def from_calibration(cls, data: dict, qubits, *, couplings=None, **overrides):
        """Build an N-qubit profile from a vendor-neutral normalized calibration
        dict (the same structure as ``ParametricCouplerProfile.from_calibration``).

        ``qubits`` is the ordered list of device qubit indices to include (length
        N); their measured frequency, anharmonicity, and T1/T2 populate the per-qubit
        lists, so a real device loads in one call. Per-qubit fields absent from the
        calibration fall back to representative defaults (recorded in ``notes``).
        ``couplings`` (exchange ``g`` per edge, which calibration files do not carry)
        must be supplied here or via ``**overrides``.
        """
        qcal = data.get("qubits")
        if not isinstance(qcal, dict):
            raise ValueError("calibration data must have a 'qubits' dict.")
        qubits = [int(q) for q in qubits]
        n = len(qubits)
        d_default = cls(n_qubits=n, freqs_ghz=[5.0] * n, anharm_mhz=[-300.0] * n,
                        t1_ns=[40_000.0] * n, t2_ns=[30_000.0] * n,
                        couplings={}, f_ref_ghz=0.0)
        freqs, anh, t1, t2 = [], [], [], []
        missing = []
        for slot, q in enumerate(qubits):
            node = qcal.get(q, qcal.get(str(q)))
            if node is None:
                raise ValueError(f"qubit {q} absent from calibration "
                                 f"(has {sorted(map(str, qcal))}).")
            v = normalize_qubit_node(node)
            freqs.append(v["freq_ghz"] if v["freq_ghz"] is not None else d_default.freqs_ghz[slot])
            anh.append(v["anharm_ghz"] * 1000.0 if v["anharm_ghz"] is not None else d_default.anharm_mhz[slot])
            t1.append(v["t1_ns"] if v["t1_ns"] is not None else d_default.t1_ns[slot])
            t2.append(v["t2_ns"] if v["t2_ns"] is not None else d_default.t2_ns[slot])
            for fld, val in (("freq", v["freq_ghz"]), ("anharm", v["anharm_ghz"]),
                             ("t1", v["t1_ns"]), ("t2", v["t2_ns"])):
                if val is None:
                    missing.append(f"q{q}.{fld}")
        kw = dict(n_qubits=n, freqs_ghz=freqs, anharm_mhz=anh, t1_ns=t1, t2_ns=t2)
        if couplings is not None:
            kw["couplings"] = couplings
        valid = {f.name for f in __import__("dataclasses").fields(cls)}
        unknown = set(overrides) - valid
        if unknown:
            raise TypeError(f"unknown MultiQubitProfile field(s): {sorted(unknown)}")
        kw.update(overrides)
        prof = cls(**kw)
        prof.notes.append(f"Loaded device qubits {qubits} from a normalized "
                          "calibration dict.")
        if missing:
            prof.notes.append(f"{missing} not in calibration -- kept defaults.")
        return prof

    @classmethod
    def from_ibm_backend(cls, backend, qubits, *, couplings=None, **overrides):
        """Build an N-qubit profile from a Qiskit backend (``BackendV1``/``V2``):
        frequency, anharmonicity, and T1/T2 for the chosen ``qubits`` in one call.
        ``couplings`` (exchange ``g``) is not in calibration -- supply it here."""
        cal = _ibm_backend_to_calibration(backend, qubits)
        return cls.from_calibration(cal, qubits, couplings=couplings, **overrides)


# ---------------------------------------------------------------------------
# Operator helpers
# ---------------------------------------------------------------------------
def _ladder(d: int, dtype, device):
    a = torch.zeros((d, d), dtype=dtype, device=device)
    for k in range(1, d):
        a[k - 1, k] = math.sqrt(k)
    return a


def _kron_list(mats):
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out


def _embed(op_single, q: int, n: int, ident):
    """Place a single-qubit operator on slot q in the N-fold tensor product."""
    mats = [ident] * n
    mats[q] = op_single
    return _kron_list(mats)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
class MultiQubitOptimizer:
    """Autodiff GRAPE for a gate on a subset of an N-qubit register.

    target_gate    : a (2**k x 2**k) unitary (numpy/torch/list) OR one of
                     {"cz", "iswap", "cnot", "sqrt_iswap"} -- the gate to realize
                     on `target_qubits` (k of them). Identity is the target on every
                     other qubit, so the optimizer must leave spectators alone.
    target_qubits  : tuple of k qubit indices the gate acts on (order matters and
                     matches the gate's tensor-factor order).
    drive_qubits   : which qubits get a microwave drive channel (default: all).
    tunable_edges  : coupling edges whose strength is a control channel
                     (default: the edges among target_qubits that exist in the
                     profile graph). Edges NOT listed here stay at their static
                     profile value (always-on crosstalk the optimizer must fight).
    freq_control_qubits : elements whose detuning is a control channel (+/-
                     delta_max_mhz), i.e. a flux line tuning a transmon/coupler
                     frequency. This is how a TUNABLE-COUPLER CZ is driven: model
                     q0-coupler-q1 with always-on q-coupler edges, put the coupler
                     here, target CZ on (q0, q1) -- the coupler-mediated |11>-|02>
                     interaction the flux pulse activates is then captured exactly,
                     coupler explicitly evolved (no Schrieffer-Wolff elimination).
                     See ``tunable_coupler_cz`` in gradpulse.convenience.
    open_system    : True = exact Lindblad Choi (decoherence + leakage aware).
                     False = unitary propagation (coherent + leakage; far cheaper,
                     reaches larger N). See the module docstring's cost envelope.
    """

    _NAMED_GATES = {
        # two-qubit
        "cz": np.diag([1, 1, 1, -1]).astype(complex),
        "cnot": np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                          [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
        "iswap": np.array([[1, 0, 0, 0], [0, 0, 1j, 0],
                           [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex),
        "sqrt_iswap": np.array([[1, 0, 0, 0],
                                [0, 1 / math.sqrt(2), 1j / math.sqrt(2), 0],
                                [0, 1j / math.sqrt(2), 1 / math.sqrt(2), 0],
                                [0, 0, 0, 1]], dtype=complex),
        # single-qubit (useful as one element of a simultaneous-gate spec, e.g. a
        # CZ on (0,1) WHILE an X on q2 -- the optimizer must realise both at once).
        "i": np.eye(2, dtype=complex),
        "x": np.array([[0, 1], [1, 0]], dtype=complex),
        "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "z": np.array([[1, 0], [0, -1]], dtype=complex),
        "h": np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),
        "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    }

    def __init__(self, profile: MultiQubitProfile,
                 target_gate="cz", target_qubits: Sequence[int] = (0, 1),
                 drive_qubits: Optional[Sequence[int]] = None,
                 tunable_edges: Optional[Sequence[tuple]] = None,
                 omega_max_mhz: float = 60.0, g_max_mhz: float = 20.0,
                 bandwidth_mhz: float = 80.0, use_drag: bool = False,
                 freq_control_qubits: Optional[Sequence[int]] = None,
                 delta_max_mhz: float = 200.0,
                 open_system: bool = True, precision: str = "single",
                 verbose: bool = True):
        self.profile = profile
        self.N = profile.n_qubits
        self.d = profile.n_levels
        self.D = self.d ** self.N
        # self.target_specs is the normalized [(U, qubits), ...] (see
        # _normalize_target_specs); per-group membership feeds the default
        # tunable-edge choice below. self.target_qubits is the union.
        self.target_specs = self._normalize_target_specs(target_gate, target_qubits)
        self._group_of = {}
        for gi, (_, qs) in enumerate(self.target_specs):
            for q in qs:
                self._group_of[q] = gi
        self.target_qubits = tuple(sorted(self._group_of))
        self.open_system = bool(open_system)
        self.use_drag = bool(use_drag)
        self.bandwidth_mhz = float(bandwidth_mhz)
        self.rdtype = torch.float64 if precision == "double" else torch.float32
        self.cdtype = torch.complex128 if precision == "double" else torch.complex64
        self.OMEGA_MAX = _2PI * omega_max_mhz / 1000.0
        self.G_MAX = _2PI * g_max_mhz / 1000.0
        # See freq_control_qubits in the class docstring.
        self.DELTA_MAX = _2PI * delta_max_mhz / 1000.0
        self.freq_control_qubits = ([] if freq_control_qubits is None
                                    else [int(q) for q in freq_control_qubits])

        self.drive_qubits = (list(range(self.N)) if drive_qubits is None
                             else [int(q) for q in drive_qubits])
        if tunable_edges is None:
            # Default = couplings WITHIN a gate group; cross-group couplings stay
            # fixed/always-on -- for simultaneous gates that IS the crosstalk to fight.
            go = self._group_of
            tunable_edges = [e for e in profile.couplings
                             if e[0] in go and e[1] in go and go[e[0]] == go[e[1]]]
        self.tunable_edges = [tuple(e) for e in tunable_edges]
        self.fixed_edges = [e for e in profile.couplings
                            if tuple(e) not in set(self.tunable_edges)]

        self._build_operators()
        self._build_target()
        # channel layout: [drive qubits, tunable couplings, freq-control elements].
        # DRAG (when on) is derived from the in-phase drive, so it consumes no channel.
        self.n_drive_ch = len(self.drive_qubits)
        self.n_channels = (self.n_drive_ch + len(self.tunable_edges)
                           + len(self.freq_control_qubits))

        if verbose:
            c = self.cost_estimate()
            print(f"[multiqubit] N={self.N} qubits, n_levels={self.d} -> Hilbert "
                  f"dim {self.D}; {'open (Lindblad Choi, %d ops)' % (self._dcomp ** 2) if self.open_system else 'closed (unitary)'}; "
                  f"{self.n_channels} control channels; ~{c['matmul_dim']} matmul dim/step. "
                  f"{c['warning']}")

    # ---- operators -------------------------------------------------------
    def _build_operators(self):
        d, N = self.d, self.N
        cdt = self.cdtype
        a1 = _ladder(d, cdt, DEVICE)
        ad1 = a1.conj().t().contiguous()
        n1 = (ad1 @ a1)
        ident = torch.eye(d, dtype=cdt, device=DEVICE)
        I = torch.eye(self.D, dtype=cdt, device=DEVICE)
        self._I = I

        a = [_embed(a1, q, N, ident).contiguous() for q in range(N)]
        ad = [op.conj().t().contiguous() for op in a]
        nop = [(ad[q] @ a[q]).contiguous() for q in range(N)]
        self._a, self._ad, self._n = a, ad, nop

        f_ref = self.profile.f_ref_ghz
        H = torch.zeros((self.D, self.D), dtype=cdt, device=DEVICE)
        for q in range(N):
            delta = _2PI * (self.profile.freqs_ghz[q] - f_ref)        # rad/ns
            alpha = _2PI * self.profile.anharm_mhz[q] / 1000.0
            H = H + delta * nop[q] + 0.5 * alpha * (ad[q] @ ad[q] @ a[q] @ a[q])
        for (i, j) in self.fixed_edges:
            g = _2PI * self.profile.couplings[(i, j)] / 1000.0
            H = H + g * (ad[i] @ a[j] + a[i] @ ad[j])
        self._H_DRIFT = H.contiguous()

        # control operators
        self._X = {q: (a[q] + ad[q]).contiguous() for q in self.drive_qubits}
        self._Y = {q: (1j * (ad[q] - a[q])).contiguous() for q in self.drive_qubits}
        self._C = {e: (ad[e[0]] @ a[e[1]] + a[e[0]] @ ad[e[1]]).contiguous()
                   for e in self.tunable_edges}
        # frequency-control operator is the number operator (a detuning shift)
        self._F = {q: nop[q].contiguous() for q in self.freq_control_qubits}

        # dissipators (per qubit), same rate convention as the pair models
        self._L, loss = [], torch.zeros((self.D, self.D), dtype=cdt, device=DEVICE)
        for q in range(N):
            t1, t2 = self.profile.t1_ns[q], self.profile.t2_ns[q]
            rate_t1 = math.sqrt(1.0 / max(t1, 1.0))
            inv_tphi = 1.0 / max(t2, 1.0) - 1.0 / max(2.0 * t1, 1.0)
            rate_phi = math.sqrt(2.0 / max(1.0 / max(inv_tphi, 1e-9), 1e-9))
            for L in (rate_t1 * a[q], rate_phi * nop[q]):
                L = L.contiguous()
                self._L.append(L)
                loss = loss + 0.5 * (L.conj().t() @ L)
        self._L_LOSS_SUM = loss.contiguous()

        # computational-subspace indices: the 2**N states with each qubit in {0,1}
        idx = []
        for bits in product((0, 1), repeat=N):
            k = 0
            for b in bits:
                k = k * d + b
            idx.append(k)
        self._comp_idx = torch.tensor(idx, dtype=torch.long, device=DEVICE)
        self._dcomp = 2 ** N

    def _resolve_gate(self, g, k: int) -> np.ndarray:
        """A named/array gate -> a validated (2**k x 2**k) unitary ndarray."""
        if isinstance(g, str):
            key = g.lower()
            if key not in self._NAMED_GATES:
                raise ValueError(f"unknown gate {g!r}; known: {sorted(self._NAMED_GATES)} "
                                 "or pass an explicit unitary matrix")
            G = self._NAMED_GATES[key]
        else:
            G = np.asarray(g, dtype=complex)
        if G.shape != (2 ** k, 2 ** k):
            raise ValueError(f"gate must be {2**k}x{2**k} for {k} qubit(s), "
                             f"got {G.shape}")
        if not np.allclose(G.conj().T @ G, np.eye(2 ** k), atol=1e-8):
            raise ValueError("target_gate is not unitary.")
        return G

    @staticmethod
    def _is_single_gate_spec(g) -> bool:
        """True if g is ONE gate (a name or a 2-D matrix), False if a list of gates."""
        if isinstance(g, str):
            return True
        try:
            return np.ndim(np.asarray(g, dtype=complex)) == 2
        except (TypeError, ValueError):
            return False

    def _normalize_target_specs(self, target_gate, target_qubits):
        """Normalize the (gate, qubits) input into [(U_ndarray, qubits_tuple), ...].

        Single gate:        target_gate='cz',         target_qubits=(0, 1)
        Simultaneous gates: target_gate=['cz', 'cz'], target_qubits=[(0,1), (2,3)]
                            (one gate per disjoint qubit group). A single gate name
                            with grouped qubits applies that gate to every group; a
                            single-qubit group may be written as a bare int (2) or a
                            1-tuple ((2,)).
        """
        tq = list(target_qubits)
        multi = len(tq) > 0 and any(isinstance(q, (tuple, list)) for q in tq)
        if multi:
            groups = [tuple(int(x) for x in (q if isinstance(q, (tuple, list)) else (q,)))
                      for q in tq]
            if self._is_single_gate_spec(target_gate):
                gates = [target_gate] * len(groups)     # one gate replicated per group
            else:
                gates = list(target_gate)
            if len(gates) != len(groups):
                raise ValueError(
                    "for simultaneous gates, target_gate must be a list matching "
                    f"target_qubits ({len(groups)} groups), got {len(gates)} gate(s)")
            specs = [(self._resolve_gate(g, len(qs)), qs)
                     for g, qs in zip(gates, groups)]
        else:
            specs = [(self._resolve_gate(target_gate, len(tq)),
                      tuple(int(q) for q in tq))]
        seen = set()
        for _, qs in specs:
            for q in qs:
                if not (0 <= q < self.N):
                    raise ValueError(f"target qubit {q} out of range [0, {self.N})")
                if q in seen:
                    raise ValueError(
                        f"target qubit {q} is in two gate groups; simultaneous gates "
                        "must act on DISJOINT qubits")
                seen.add(q)
        return specs

    def _build_target(self):
        """Full 2**N target unitary: each spec's gate on its (disjoint) qubit group,
        identity on every other qubit. For one spec this is the single-gate case; for
        several it is the tensor product of the gates -- the simultaneous-gate target
        the optimizer must realise at once while leaving spectators undisturbed."""
        N, dcomp = self.N, self._dcomp
        U = np.zeros((dcomp, dcomp), dtype=complex)
        comp_states = list(product((0, 1), repeat=N))
        index = {s: r for r, s in enumerate(comp_states)}
        for r, s in enumerate(comp_states):
            # Apply each gate to its group; combine output amplitudes multiplicatively
            # (disjoint groups => the joint amplitude factorizes over the groups).
            partials = [(1.0 + 0j, list(s))]
            for G, qubits in self.target_specs:
                k = len(qubits)
                in_col = 0
                for q in qubits:
                    in_col = in_col * 2 + s[q]
                nxt = []
                for amp0, sout0 in partials:
                    for out_row in range(2 ** k):
                        amp = G[out_row, in_col]
                        if amp == 0:
                            continue
                        ob = [(out_row >> (k - 1 - p)) & 1 for p in range(k)]
                        sout = list(sout0)
                        for p, q in enumerate(qubits):
                            sout[q] = ob[p]
                        nxt.append((amp0 * amp, sout))
                partials = nxt
            for amp, sout in partials:
                U[index[tuple(sout)], r] += amp
        self.u_target = torch.tensor(U, dtype=self.cdtype, device=DEVICE)

    # ---- cost ------------------------------------------------------------
    def cost_estimate(self) -> dict:
        # Open-system evolves the full register's computational Choi basis
        # ((2**N)**2 operators), so identity-on-spectators is inside the objective.
        n_ops = (self._dcomp ** 2) if self.open_system else 1
        # Per-step work ~ (# evolved operators) * D^3 (the matrix triple products).
        # This, not D alone, is what makes a run slow; warn above ~5e7/step.
        work = n_ops * self.D ** 3
        big = work > 5e7
        warn = ("LARGE: expect slow steps / high memory -- consider open_system=False "
                "or fewer qubits/levels." if big else "tractable on a workstation/GPU.")
        return {"hilbert_dim": self.D, "choi_ops": n_ops, "matmul_dim": self.D,
                "work_per_step": work, "warning": warn}

    # ---- control smoothing ----------------------------------------------
    def _smoother(self, n_slices, dt_ns):
        sigma = max(1e-6, 1000.0 / (2.0 * math.pi * self.bandwidth_mhz * dt_ns))
        half = max(1, int(3 * sigma))
        x = torch.arange(-half, half + 1, dtype=self.rdtype, device=DEVICE)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        return (k / k.sum()).view(1, 1, -1)

    def _smooth(self, u, kernel):
        # u: [B, n_slices, C] in [0,1]; reflect-pad and convolve per channel.
        B, Nt, C = u.shape
        pad = kernel.shape[-1] // 2
        x = u.permute(0, 2, 1).reshape(B * C, 1, Nt)
        x = torch.nn.functional.pad(x, (pad, pad), mode="reflect")
        y = torch.nn.functional.conv1d(x, kernel.to(x.dtype))
        return y.reshape(B, C, Nt).permute(0, 2, 1)

    def _edge_rest_window(self, n_slices, r):
        """``[1, n_slices, 1]`` raised-cosine window: 0 at the very edges ramping to 1
        over ``r`` slices each side. Masking controls as ``0.5 + (xs-0.5)*w`` forces them
        to rest (``x=0.5``, ``u=0``) at the boundaries -- a composable gate. ``r<=0`` ->
        ``None`` (no masking; legacy path unchanged)."""
        if not r or r <= 0:
            return None
        r = int(min(r, n_slices // 2))
        w = torch.ones(n_slices, dtype=self.rdtype, device=DEVICE)
        ramp = 0.5 * (1.0 - torch.cos(
            math.pi * torch.arange(1, r + 1, dtype=self.rdtype, device=DEVICE) / (r + 1)))
        w[:r] = ramp
        w[-r:] = ramp.flip(0)
        return w.view(1, n_slices, 1)

    # ---- simulation ------------------------------------------------------
    def _hamiltonian_slice(self, u, i, Nt, dt):
        """Batched control Hamiltonian [B, D, D] at slice ``i`` from signed controls
        ``u`` ([-1,1]). The single per-slice H builder, used by ``_hamiltonian_seq``
        (default path) and rebuilt inside checkpointed segments (so the gradient-
        checkpointed propagation never materializes the whole [Nt, B, D, D] stack)."""
        B = u.shape[0]
        H = self._H_DRIFT.view(1, self.D, self.D).expand(B, -1, -1).clone()
        ch = 0
        for q in self.drive_qubits:
            amp = (u[:, i, ch] * self.OMEGA_MAX).view(B, 1, 1)
            H = H + amp * self._X[q]
            ch += 1
            if self.use_drag:
                # 1st-order DRAG quadrature v = -du/dt / alpha, derived from the
                # in-phase drive channel just read (ch-1); consumes no channel.
                alpha = _2PI * self.profile.anharm_mhz[q] / 1000.0
                if 0 < i < Nt - 1:
                    du = (u[:, i + 1, ch - 1] - u[:, i - 1, ch - 1]) * self.OMEGA_MAX / (2 * dt)
                else:
                    du = torch.zeros(B, dtype=u.dtype, device=DEVICE)
                H = H + (-du / alpha).view(B, 1, 1) * self._Y[q]
        for e in self.tunable_edges:
            g = (u[:, i, ch] * self.G_MAX).view(B, 1, 1)
            H = H + g * self._C[e]
            ch += 1
        for q in self.freq_control_qubits:
            det = (u[:, i, ch] * self.DELTA_MAX).view(B, 1, 1)
            H = H + det * self._F[q]
            ch += 1
        return H

    def _hamiltonian_seq(self, x_smooth, dt):
        """List over slices of the batched control Hamiltonian [B, D, D]."""
        B, Nt, C = x_smooth.shape
        u = 2.0 * x_smooth - 1.0                  # [-1, 1] signed
        return [self._hamiltonian_slice(u, i, Nt, dt) for i in range(Nt)]

    def _propagate_unitary(self, x_smooth, dt):
        Hs = self._hamiltonian_seq(x_smooth, dt)
        B = x_smooth.shape[0]
        U = self._I.view(1, self.D, self.D).expand(B, -1, -1).clone()
        for H in Hs:
            U = torch.linalg.matrix_exp(-1j * H * dt) @ U
        return U                                   # [B, D, D]

    def _choi_rho0(self, B):
        ci = self._comp_idx
        dc = self._dcomp
        rho0 = torch.zeros((B, dc * dc, self.D, self.D), dtype=self.cdtype, device=DEVICE)
        for i in range(dc):
            for j in range(dc):
                rho0[:, i * dc + j, ci[i], ci[j]] = 1.0
        return rho0

    def _propagate_choi(self, x_smooth, dt, diss_scale=1.0, rho0=None,
                        checkpoint_segments=0):
        """Open-system propagation of a density-matrix stack. With ``rho0=None`` it
        evolves the full Choi basis (exact process fidelity); pass ``rho0`` (shape
        ``[B, M, D, D]``) to evolve any other set of states (e.g. the state-transfer
        estimator's input states).

        ``checkpoint_segments=S>1`` gradient-checkpoints the slice loop into S
        contiguous segments: only the rho at each boundary is kept for backprop and
        the segment interiors (Hamiltonians included) are recomputed in backward.
        Autograd memory drops from O(Nt) to ~O(Nt/S + S) at ~2x forward compute --
        the memory wall, not compute, is what caps the 4**N open-system Choi stack,
        so this is what extends the reachable register size for evaluation/optimization.
        """
        B = x_smooth.shape[0]
        Nt = x_smooth.shape[1]
        rho = self._choi_rho0(B) if rho0 is None else rho0
        Ls = self._L
        loss = self._L_LOSS_SUM

        def _advance(rho, H):
            U = torch.linalg.matrix_exp(-1j * H * dt).unsqueeze(1)
            Ud = U.conj().transpose(-2, -1)
            rho = U @ rho @ Ud
            jump = sum(L @ rho @ L.conj().t() for L in Ls)
            anti = loss @ rho + rho @ loss
            return rho + dt * diss_scale * (jump - anti)

        n_ckpt = int(checkpoint_segments or 0)
        if n_ckpt > 1 and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            u = 2.0 * x_smooth - 1.0

            def _seg(rho_in, lo, hi):
                r = rho_in
                for i in range(int(lo), int(hi)):
                    r = _advance(r, self._hamiltonian_slice(u, i, Nt, dt))
                return r

            bounds = [round(k * Nt / n_ckpt) for k in range(n_ckpt + 1)]
            for s in range(n_ckpt):
                lo, hi = bounds[s], bounds[s + 1]
                if hi <= lo:
                    continue
                rho = checkpoint(_seg, rho, lo, hi, use_reentrant=False)
        else:
            for H in self._hamiltonian_seq(x_smooth, dt):
                rho = _advance(rho, H)
        return rho                                 # [B, M, D, D]

    # ---- fidelity --------------------------------------------------------
    def _process_fidelity_choi(self, rho_choi):
        ci = self._comp_idx
        dc = self._dcomp
        B = rho_choi.shape[0]
        proj = rho_choi[:, :, ci, :][:, :, :, ci]              # [B, dc^2, dc, dc]
        C = proj.reshape(B, dc, dc, dc, dc)                    # [B, i, j, a, c]
        U = self.u_target
        F = torch.einsum('ai,zijac,cj->z', U.conj(), C, U).real / (dc * dc)
        return F.clamp(0.0, 1.0)

    def _process_fidelity_unitary(self, Ufull):
        ci = self._comp_idx
        dc = self._dcomp
        Uc = Ufull[:, ci, :][:, :, ci]                         # [B, dc, dc]
        M = torch.einsum('ac,zac->z', self.u_target.conj(), Uc)  # Tr(U_t^dag Uc)
        return (M.real ** 2 + M.imag ** 2) / (dc * dc)

    def _leakage_choi(self, rho_choi):
        ci = self._comp_idx
        dc = self._dcomp
        diag_ops = [m * dc + m for m in range(dc)]             # |i><i|
        rho = rho_choi[:, diag_ops]
        diag = rho.diagonal(dim1=-2, dim2=-1).real
        comp_pop = diag[..., ci].sum(dim=-1)
        return (1.0 - comp_pop).mean(dim=1).clamp(0.0, 1.0)

    # ---- state-transfer fidelity (memory-light alternative to the 4**N Choi) --
    def _haar_comp_states(self, n_states, seed):
        """``n_states`` Haar-random pure states of the 2**N computational subspace,
        embedded in the full D-dim space, plus their images under the target gate.
        Returns (states[K, D], targets[K, D]); both are constants of the pulse."""
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        dc = self._dcomp
        psi = (torch.randn(n_states, dc, generator=g)
               + 1j * torch.randn(n_states, dc, generator=g)).to(self.cdtype).to(DEVICE)
        psi = psi / psi.norm(dim=1, keepdim=True)
        tgt = psi @ self.u_target.t()                          # (U_target psi)
        states = torch.zeros(n_states, self.D, dtype=self.cdtype, device=DEVICE)
        targets = torch.zeros(n_states, self.D, dtype=self.cdtype, device=DEVICE)
        states[:, self._comp_idx] = psi
        targets[:, self._comp_idx] = tgt
        return states, targets

    def _favg_from_states(self, rho_states, targets):
        """Average state fidelity ``mean_k <target_k| rho_k |target_k>`` over the
        propagated input states -- an (unbiased, for Haar inputs) estimate of the
        average gate fidelity F_avg. ``rho_states`` is [B, K, D, D]; targets [K, D]."""
        f = torch.einsum('ki,zkij,kj->zk', targets.conj(), rho_states, targets).real
        return f.mean(dim=1).clamp(0.0, 1.0)                   # [B]

    def _leakage_from_states(self, rho_states):
        """Mean leaked population (1 - computational-subspace population) over the
        propagated state stack [B, K, D, D]."""
        diag = rho_states.diagonal(dim1=-2, dim2=-1).real      # [B, K, D]
        comp_pop = diag[..., self._comp_idx].sum(dim=-1)       # [B, K]
        return (1.0 - comp_pop).mean(dim=1).clamp(0.0, 1.0)

    def state_transfer_fidelity(self, waveform, dt_ns: float = 1.0, n_states: int = 32,
                                seed: int = 0, diss_scale: float = 1.0) -> dict:
        """Memory-light estimate of the gate fidelity that avoids the exact 4**N Choi
        stack: propagate ``n_states`` Haar-random computational-subspace input states
        (open system) and average their state fidelity to the target. This trades
        exactness for memory -- O(n_states) propagated operators instead of 4**N --
        which is what lets the open-system optimizer reach larger registers; the
        estimate has Monte-Carlo variance ~1/n_states and converges to the exact
        F_proc as ``n_states`` grows. Returns
        {F_avg, F_proc, n_states} (F_proc inferred via F_avg=(d F_proc+1)/(d+1))."""
        x = torch.as_tensor(waveform, dtype=self.rdtype, device=DEVICE)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.clamp(0.0, 1.0)
        states, targets = self._haar_comp_states(n_states, seed)
        rho0 = torch.einsum('ki,kj->kij', states, states.conj()).unsqueeze(0)
        with torch.no_grad():
            rho = self._propagate_choi(x, dt_ns, diss_scale, rho0=rho0)
            favg = float(self._favg_from_states(rho, targets)[0])
        d = float(self._dcomp)
        return {"F_avg": favg, "F_proc": ((d + 1.0) * favg - 1.0) / d,
                "n_states": int(n_states)}

    def process_fidelity(self, waveform, dt_ns: float = 1.0,
                         diss_scale: float = 1.0) -> float:
        """F_proc of an already-smoothed physical pulse to the subset target.

        ``waveform`` is the saved ``best_waveform`` (the smoothed [0,1] envelope
        the simulator actually evolved) -- it is propagated verbatim, NOT
        re-smoothed (that would attenuate features twice). This is the same
        no-resmooth convention the QuTiP cross-check and the pair models' eval
        path use, so this number matches the optimizer's reported fidelity and the
        independent QuTiP value. Exact entanglement fidelity over the 2**N
        computational subspace; open system if the optimizer is open, unitary
        otherwise.
        """
        x = torch.as_tensor(waveform, dtype=self.rdtype, device=DEVICE)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.clamp(0.0, 1.0)
        with torch.no_grad():
            if self.open_system:
                rho = self._propagate_choi(x, dt_ns, diss_scale)
                return float(self._process_fidelity_choi(rho)[0])
            U = self._propagate_unitary(x, dt_ns)
            return float(self._process_fidelity_unitary(U)[0])

    def process_fidelity_sparse(self, waveform, dt_ns: float = 1.0) -> float:
        """Closed-system process fidelity via **sparse Krylov** state propagation --
        an evaluation path for systems past the dense ~4-qubit wall.

        Instead of building the ``D x D`` propagator (``_propagate_unitary``) or the
        ``4**N`` Choi stack (``_propagate_choi``), this propagates the ``2**N``
        computational-basis **state vectors** through each time slice with
        ``scipy.sparse.linalg.expm_multiply`` -- a Krylov matrix-vector product on
        the sparse, per-slice Hamiltonian. Memory is ``O(2**N * D)`` with no dense
        ``matrix_exp``, so it reaches larger ``N`` than ``process_fidelity``.

        Scope (honest): coherent + leakage aware, **no decoherence** (that needs
        density matrices -- the real memory wall this cannot escape) and **no
        gradients**, so gradient-based ``optimize()`` stays dense-bound; this is for
        *scoring* a pulse on a big register, not optimizing one. Uses the same
        metric as ``process_fidelity(open_system=False)`` and agrees with it to
        integrator precision on small systems. ``waveform`` is consumed verbatim
        (already-smoothed), matching ``process_fidelity``.
        """
        try:
            import scipy.sparse as sp
            from scipy.sparse.linalg import expm_multiply
        except ImportError as e:  # pragma: no cover
            raise ImportError("process_fidelity_sparse needs scipy "
                              "(pip install scipy).") from e
        wf = np.asarray(waveform, dtype=float)
        if wf.ndim == 1:
            wf = wf[:, None]
        u_s = 2.0 * wf - 1.0

        def _sp(op):
            return sp.csc_matrix(op.detach().cpu().numpy())

        Hd = _sp(self._H_DRIFT)
        Xop = {q: _sp(self._X[q]) for q in self.drive_qubits}
        Yop = {q: _sp(self._Y[q]) for q in self.drive_qubits}
        Cop = {e: _sp(self._C[e]) for e in self.tunable_edges}
        Fop = {q: _sp(self._F[q]) for q in self.freq_control_qubits}
        ci = self._comp_idx.cpu().numpy()
        dc = self._dcomp
        Nt = wf.shape[0]
        # computational-basis state vectors as columns of a D x dc matrix
        psi = np.zeros((self.D, dc), dtype=complex)
        psi[ci, np.arange(dc)] = 1.0
        om, gm, dm = self.OMEGA_MAX, self.G_MAX, getattr(self, "DELTA_MAX", 0.0)
        for i in range(Nt):
            H = Hd.copy()
            ch = 0
            for q in self.drive_qubits:
                H = H + (u_s[i, ch] * om) * Xop[q]
                if self.use_drag:   # derived-quadrature DRAG, mirrors _hamiltonian_seq
                    alpha = _2PI * self.profile.anharm_mhz[q] / 1000.0
                    du = ((u_s[i + 1, ch] - u_s[i - 1, ch]) * om / (2 * dt_ns)
                          if 0 < i < Nt - 1 else 0.0)
                    H = H + (-du / alpha) * Yop[q]
                ch += 1
            for e in self.tunable_edges:
                H = H + (u_s[i, ch] * gm) * Cop[e]; ch += 1
            for q in self.freq_control_qubits:
                H = H + (u_s[i, ch] * dm) * Fop[q]; ch += 1
            psi = expm_multiply(-1j * H * dt_ns, psi)
        Uc = psi[ci, :]                                # dc x dc comp-subspace evolution
        Ut = self.u_target.detach().cpu().numpy()
        M = np.trace(Ut.conj().T @ Uc)
        return float((M.real ** 2 + M.imag ** 2) / (dc * dc))

    # ---- optimization ----------------------------------------------------
    # ---- seed initialization --------------------------------------------
    @staticmethod
    def _resample_env(w, n_slices):
        """Linear-interp resample a warm-start envelope [L, C] -> [n_slices, C]."""
        L = w.shape[0]
        if L == n_slices:
            return w
        pos = torch.linspace(0, L - 1, n_slices, dtype=w.dtype)
        i0 = pos.floor().long().clamp(0, L - 1)
        i1 = (i0 + 1).clamp(0, L - 1)
        frac = (pos - i0.to(w.dtype)).unsqueeze(1)
        return w[i0] * (1 - frac) + w[i1] * frac

    def _init_raw(self, n_slices, warm_start, s, g):
        """Seed pre-sigmoid logits for GRAPE seed ``s``. Default (warm_start None or
        exhausted): small random ``0.1*randn`` -- byte-identical to the legacy path.
        If a warm-start envelope is given for this seed, invert the sigmoid so
        ``sigmoid(raw) == envelope`` (the Gaussian smoother leaves an already-smooth
        envelope ~unchanged), plus a tiny jitter so repeated warm seeds still diverge.
        A single envelope seeds only seed 0; a list seeds 0..len-1, rest random."""
        ws = None
        if warm_start is not None:
            if isinstance(warm_start, (list, tuple)):
                ws = warm_start[s] if s < len(warm_start) else None
            else:
                ws = warm_start if s == 0 else None
        if ws is None:
            return (0.1 * torch.randn(1, n_slices, self.n_channels, generator=g)
                    ).to(self.rdtype).to(DEVICE).requires_grad_(True)
        w = torch.as_tensor(ws, dtype=self.rdtype)
        if w.ndim == 1:
            w = w.unsqueeze(1).expand(-1, self.n_channels).contiguous()
        if w.shape[1] != self.n_channels:
            raise ValueError(f"warm_start has {w.shape[1]} channels; model has "
                             f"{self.n_channels}.")
        w = self._resample_env(w, n_slices).clamp(1e-4, 1.0 - 1e-4)
        raw0 = torch.log(w / (1.0 - w)).view(1, n_slices, self.n_channels)
        jit = 0.02 * torch.randn(1, n_slices, self.n_channels, generator=g).to(self.rdtype)
        return (raw0 + jit).to(DEVICE).requires_grad_(True)

    # ---- physical CZ objective for an ancilla-coupler gate -------------
    def _cz_data_vz_setup(self):
        """Precompute constants for the ``cz_data_virtualz`` objective: the CZ on the
        two data (target) qubits with the ancilla coupler idle in ``|0>``, single-qubit
        Z free. The optimizer's default target is the strict ``CZ (x) I`` over all
        elements in ``{0,1}``, which penalizes coupler-EXCITED inputs that never occur
        in the real gate -- counterproductive for a tunable coupler. This objective
        scores only the physically-reachable data subspace (coupler ``|0>``)."""
        nl = self.profile.n_levels
        comp = (self._comp_idx.detach().cpu().numpy() if torch.is_tensor(self._comp_idx)
                else np.asarray(self._comp_idx))
        lev = lambda idx, e: (idx // nl ** (self.N - 1 - e)) % nl
        tq = self.target_qubits
        if len(tq) != 2:
            raise ValueError("cz_data_virtualz expects exactly two data (target) qubits.")
        coupler = [e for e in range(self.N) if e not in tq]
        if len(coupler) != 1:
            raise ValueError("cz_data_virtualz expects exactly one ancilla element (coupler).")
        d_idx = [k for k in range(self._dcomp) if lev(comp[k], coupler[0]) == 0]
        if len(d_idx) != 4:
            raise ValueError(f"cz_data_virtualz expects 4 coupler-idle data states, got {len(d_idx)}.")
        di = torch.tensor(d_idx, device=DEVICE)
        return {"d_idx": di,
                "q0": torch.tensor([lev(comp[k], tq[0]) for k in d_idx], device=DEVICE, dtype=self.rdtype),
                "q1": torch.tensor([lev(comp[k], tq[1]) for k in d_idx], device=DEVICE, dtype=self.rdtype),
                "czd": torch.diagonal(self.u_target)[di],          # CZ phases on the 4 data states
                "grid": torch.linspace(0, 2 * math.pi, 49, device=DEVICE, dtype=self.rdtype)}

    def _cz_data_vz_fidelity(self, rho_choi, setup, return_phases=False):
        """Process fidelity of a CZ on the two data qubits (coupler idle ``|0>``),
        maximized over single-qubit virtual-Z (free frame shifts on hardware -- exactly
        what the device's native CZ applies via ``shift_phase``). Differentiable in the
        pulse; the optimal Z phases are found on a DETACHED grid (envelope theorem), so
        the gradient flows only through the channel, not the argmax."""
        ci, dc = self._comp_idx, self._dcomp
        di, q0, q1, czd, grid = (setup["d_idx"], setup["q0"], setup["q1"],
                                 setup["czd"], setup["grid"])
        C = rho_choi[:, :, ci, :][:, :, :, ci].reshape(-1, dc, dc, dc, dc)[0]   # [i,j,a,c]
        G = C[di[:, None], di[None, :], di[:, None], di[None, :]]               # 4x4 (live)
        with torch.no_grad():
            e0 = torch.exp(1j * grid[:, None] * q0[None, :])                    # [P,4]
            e1 = torch.exp(1j * grid[:, None] * q1[None, :])
            V = czd[None, None, :].to(e0.dtype) * e0[:, None, :] * e1[None, :, :]   # [P,P,4]
            Fg = torch.einsum('abk,kl,abl->ab', V.conj(), G.detach(), V).real / 16.0
            P = grid.numel(); flat = int(Fg.argmax().item())
            p0, p1 = grid[flat // P], grid[flat % P]
        Vb = czd * torch.exp(1j * (p0 * q0 + p1 * q1))                          # live, optimal phases
        F = (Vb.conj() @ G @ Vb).real / 16.0
        return (F, (float(p0), float(p1))) if return_phases else F

    def optimize(self, n_slices: int = 200, dt_ns: float = 1.0,
                 iterations: int = 250, n_seeds: int = 2, lr: float = 0.05,
                 leak_weight: float = 1.0, seed0: int = 0,
                 fidelity: str = "choi", n_states: int = 32, state_seed: int = 0,
                 grad_clip: float = 1e3, checkpoint_segments: int = 0,
                 warm_start=None, edge_rest_slices: int = 0,
                 verbose: bool = False) -> dict:
        """Autodiff GRAPE. Loss = (1 - F_proc) + leak_weight * leakage, where
        F_proc rewards the gate on `target_qubits` AND identity on every spectator
        -- so suppressing crosstalk/collision is part of the objective.

        ``fidelity`` selects the open-system objective:
          * ``"choi"`` (default) -- exact entanglement fidelity via the full 4**N
            Choi stack. Use for N up to ~4 (memory exact).
          * ``"state_transfer"`` -- a memory-light estimate from ``n_states``
            Haar-random input states (O(n_states) propagated operators instead of
            4**N), which lets the open-system optimizer reach larger N at the cost
            of Monte-Carlo variance ~1/n_states. ``state_seed`` fixes the states for
            a deterministic objective. (Ignored when ``open_system=False``: the
            unitary path is already memory-cheap.)

        ``checkpoint_segments=S>1`` gradient-checkpoints the open-system slice loop
        (autograd memory ~O(Nt/S + S) instead of O(Nt), ~2x forward compute) -- the
        memory-side lever for reaching larger registers, complementary to the
        compute-side ``state_transfer`` estimator. Same optimum, lower peak memory.

        ``warm_start`` seeds GRAPE from a known-good control envelope instead of
        random init -- decisive for hard 27-D open-system models (e.g. the tunable-
        coupler CZ), where random init plateaus in leaky local optima but seeding the
        physically-correct adiabatic shape converges toward the coherence limit. Pass
        a ``[n_slices, n_channels]`` array (or a list of them, one per seed) in the
        smoothed-control convention: values in ``[0,1]`` where ``0.5`` is rest for a
        bipolar flux channel (``u = 2*x - 1``), so a raised-cosine activation bump is
        ``0.5 + 0.5*A*cos_bump``. Resampled to ``n_slices`` if lengths differ; seeds
        past the provided warm-start(s) fall back to random, so warm and random seeds
        compete and the best wins. Default ``None`` preserves the random-init path.

        ``edge_rest_slices=r>0`` forces every control to its rest value (``x=0.5``,
        ``u=0``) at the first/last ``r`` slices via a raised-cosine ramp, so the pulse
        starts and ends with the coupler/drives idle -- a valid *composable* gate (one
        that can be chained, as interleaved RB does, without leaving the coupler
        detuned). Nothing pins the endpoints otherwise, so an unconstrained optimum can
        drift them off rest and produce a pulse that corrupts any gate after it. Default
        ``0`` preserves the unconstrained path byte-for-byte; use it for any pulse you
        will actually play in a sequence on hardware.

        ``fidelity="cz_data_virtualz"`` (open system; exactly two target qubits + one
        ancilla coupler) optimizes the PHYSICAL CZ objective: the gate on the two data
        qubits with the coupler idle in ``|0>``, maximized over single-qubit virtual-Z
        (free frame shifts on hardware -- what the device's native CZ applies via
        ``shift_phase``). The default ``"choi"`` target is the strict ``CZ (x) I`` over
        all elements, which penalizes coupler-EXCITED inputs that never occur and is
        counterproductive for a tunable coupler. With this mode the result also carries
        ``virtual_z_phases`` ``(phi0, phi2)`` to apply on the two data qubits.

        Returns {best_fidelity, best_waveform (smoothed [n_slices, n_channels]),
        best_raw_param (pre-smoothing logits), leakage, history, fidelity_mode}
        (+ ``virtual_z_phases`` for ``cz_data_virtualz``).
        """
        st = (fidelity == "state_transfer") and self.open_system
        cz_dvz = (fidelity == "cz_data_virtualz")
        if fidelity not in ("choi", "state_transfer", "cz_data_virtualz"):
            raise ValueError("fidelity must be 'choi', 'state_transfer', or 'cz_data_virtualz'.")
        if cz_dvz and not self.open_system:
            raise ValueError("cz_data_virtualz requires open_system=True.")
        d = 4.0 if cz_dvz else float(self._dcomp)
        kernel = self._smoother(n_slices, dt_ns)
        _erw = self._edge_rest_window(n_slices, edge_rest_slices)

        def _smooth_ctrl(raw):
            xs = self._smooth(torch.sigmoid(raw), kernel)
            return xs if _erw is None else 0.5 + (xs - 0.5) * _erw

        _dvz = self._cz_data_vz_setup() if cz_dvz else None
        if st:
            states, targets = self._haar_comp_states(n_states, state_seed)
            rho0 = torch.einsum('ki,kj->kij', states, states.conj()).unsqueeze(0)

        def _eval(xs):
            """Return (F_proc, leakage) tensors for the smoothed controls xs."""
            if cz_dvz:
                rho = self._propagate_choi(xs, dt_ns,
                                           checkpoint_segments=checkpoint_segments)
                return self._cz_data_vz_fidelity(rho, _dvz), self._leakage_choi(rho)[0]
            if st:
                rho = self._propagate_choi(xs, dt_ns, rho0=rho0,
                                           checkpoint_segments=checkpoint_segments)
                favg = self._favg_from_states(rho, targets)[0]
                return ((d + 1.0) * favg - 1.0) / d, self._leakage_from_states(rho)[0]
            if self.open_system:
                rho = self._propagate_choi(xs, dt_ns,
                                           checkpoint_segments=checkpoint_segments)
                return self._process_fidelity_choi(rho)[0], self._leakage_choi(rho)[0]
            U = self._propagate_unitary(xs, dt_ns)
            return (self._process_fidelity_unitary(U)[0],
                    torch.zeros((), device=DEVICE, dtype=self.rdtype))

        best = {"best_fidelity": -1.0}
        n_nonfinite = 0
        last_grad_norm = float("nan")
        for s in range(n_seeds):
            g = torch.Generator(device="cpu").manual_seed(seed0 + s)
            raw = self._init_raw(n_slices, warm_start, s, g)
            opt = torch.optim.Adam([raw], lr=lr)
            last_good = raw.detach().clone()
            hist, seed_best = [], -1.0
            for it in range(iterations):
                opt.zero_grad()
                xs = _smooth_ctrl(raw)
                f, leak = _eval(xs)
                loss = (1.0 - f) + leak_weight * leak
                # ---- divergence guard: roll back non-finite loss/grad steps ----
                if not torch.isfinite(loss):
                    n_nonfinite += 1
                    opt.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        raw.copy_(last_good)
                else:
                    loss.backward()
                    gnorm = torch.nn.utils.clip_grad_norm_([raw], max_norm=grad_clip)
                    if torch.isfinite(gnorm):
                        opt.step()
                        last_grad_norm = float(gnorm)
                        with torch.no_grad():
                            last_good = raw.detach().clone()
                    else:
                        n_nonfinite += 1
                        opt.zero_grad(set_to_none=True)
                        with torch.no_grad():
                            raw.copy_(last_good)
                fval = f.item()
                if fval == fval and fval > seed_best:   # not NaN and improved
                    seed_best = fval
                if verbose and it % 50 == 0:
                    print(f"  seed {s} it {it:4d}  F={fval:.5f}  leak={leak.item():.2e}")
                hist.append(seed_best)
            with torch.no_grad():
                xs = _smooth_ctrl(raw)
                f_t, leak_t = _eval(xs)
                f_final, leak_final = float(f_t), float(leak_t)
            if f_final > best["best_fidelity"]:
                best = {"best_fidelity": f_final,
                        "best_waveform": xs[0].detach().cpu().numpy(),
                        "best_raw_param": raw[0].detach().cpu().numpy(),
                        "leakage": leak_final, "history": hist,
                        "n_qubits": self.N, "target_qubits": self.target_qubits,
                        "fidelity_mode": fidelity}
        best["F_avg"] = (d * best["best_fidelity"] + 1.0) / (d + 1.0)
        if cz_dvz and best["best_fidelity"] > -1.0:
            with torch.no_grad():
                xs_b = torch.as_tensor(best["best_waveform"], dtype=self.rdtype,
                                       device=DEVICE).unsqueeze(0)
                rho_b = self._propagate_choi(xs_b, dt_ns)
                _, best["virtual_z_phases"] = self._cz_data_vz_fidelity(
                    rho_b, _dvz, return_phases=True)   # (phi0, phi2) rad on target_qubits
        # ---- convergence diagnostics (winning seed's trajectory) ----
        bh = best.get("history", [])
        window = max(10, iterations // 5)
        best["converged"] = bool(len(bh) >= window and bh[-1] - bh[-window] < 1e-5)
        best["final_grad_norm"] = last_grad_norm
        best["n_nonfinite_steps"] = n_nonfinite
        if n_nonfinite > 0:
            print(f" [gradpulse] divergence guard: rolled back {n_nonfinite} "
                  f"non-finite step(s); best result is finite and unaffected.")
        return best
