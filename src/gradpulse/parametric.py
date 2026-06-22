"""gradpulse.parametric - parametric-coupler CZ GRAPE optimizer (tunable-transmon parametric-coupler architecture).

Models a tunable-transmon + parametric-coupling architecture. A fixed-frequency,
dispersively-coupled architecture -- fixed-frequency transmons coupled through a
bus resonator -- is a different physical model and is out of scope here.

==================================================================
HAMILTONIAN (9D Hilbert space, 3 levels per transmon, no bus mode)
==================================================================

The parametric coupler is dispersively eliminated under Schrieffer-Wolff,
yielding an effective two-qubit Hamiltonian whose interaction term
is *off* by default and *activated* by parametric flux modulation:

  H(t) = H_drift
       + Omega_max * u1(t) * X1
       + Omega_max * u2(t) * X2
       + g_max    * u3(t) * (a1+ a2 + a1 a2+)        <-- parametric

  H_drift = Delta * n2                               (q2 detuning in q1 frame)
          + (alpha1/2) * a1+ a1+ a1 a1               (q1 anharmonicity)
          + (alpha2/2) * a2+ a2+ a2 a2               (q2 anharmonicity)

where:
  X_i      = a_i + a_i+      (drive operator on transmon i)
  Delta    = 2pi * (f_q2 - f_q1)
  alpha_i  = 2pi * anharm_i  (negative for transmons)
  g_max    = 2pi * g_max_mhz (effective parametric coupling rate, ~12 MHz)
  Omega_max= 2pi * 50 MHz    (drive saturation Rabi rate)

Control channels:
  u1, u2, u3 in [0,1]; centered to [-1,+1] inside simulate_gradient_batch.
  ch0 = drive q1, ch1 = drive q2, ch2 = parametric coupling activation.

==================================================================
VS A FIXED-FREQUENCY DISPERSIVE MODEL (NOT IMPLEMENTED)
==================================================================

| Feature               | Fixed-freq dispersive         | Parametric (this model)
|-----------------------|-------------------------------|--------------------
| Hilbert space         | 3 x 3 x 5 = 45 (qubit*qubit*bus) | 3 x 3 = 9 (no bus)
| Coupling operator     | n1 * n2 (always-on ZZ)        | a1+a2 + a1 a2+ (parametric)
| Coupling control u3   | scales ZZ rate                | scales coupling on/off
| Native 2Q             | dispersive CZ via Stark shift | parametric iSWAP-family
| Transmons             | fixed-frequency               | tunable-via-flux

==================================================================
HONEST CAVEATS THIS OPTIMIZER DOES *NOT* RESOLVE
==================================================================

- Anharmonicity, T1/T2, qubit frequencies default to representative published-
  typical values, not measurements from your assigned qubits (real values vary
  pair-to-pair by ~10-30%). `ParametricCouplerProfile.from_braket_calibration`
  loads the measured T1/T2 and native-CZ fidelity straight from a Braket
  `standardized_gate_model_qpu_device_properties` JSON; that schema carries no
  frequency or anharmonicity, so pass those explicitly if you have them.
- The "parametric activation" model places the coupler drive at the
  |11>-|02> resonance by default. The drive frequency can be made a control:
  `coupler_phase_mode='frequency'` (with n_channels>=4) turns channel 4 into an
  instantaneous drive detuning whose running integral is the coupler phase, so a
  static frequency offset becomes optimizable. The coupling magnitude's falloff
  with detuning is optional (off by default): `coupler_g_linewidth_mhz=kappa`
  scales it by g/sqrt(1+(delta/kappa)^2) for a user-calibrated linewidth kappa.
- Integration runs in complex64/float32 by default; `precision='double'` selects
  complex128/float64 to resolve fine-dt behavior below the single-precision floor.
- Drive saturation Omega_max = 50 MHz is conservative. Some devices
  support higher; a Rabi calibration on the device gives the exact number.
- Fock truncation is 3 levels/transmon (qutrit) by default; `profile.n_levels`
  raises it. For THIS gate 3 is converged: the coupler-activated CZ keeps its
  single-qubit drives near-quiet, so re-scoring a 3-level pulse at n_levels=4 moves
  F_proc by only ~2e-4 (below the decoherence floor). This is unlike the strong-
  drive cross-resonance gate, which is NOT converged at 3 (~3% overstatement -- see
  crossresonance.py); a leakage study of any drive-dominated regime should confirm
  with n_levels=4. Every operator and the QuTiP cross-check rebuild from n_levels.
- Drives enter as a+a^dag in the rotating frame (the RWA), dropping each drive's
  counter-rotating partner. The CZ's coupler-activated drives are near-quiet so this
  is tiny here; the measured beyond-RWA check lives on the cross-resonance optimizer
  (`counter_rotating_fidelity`), whose single drive frame makes the term exact and
  whose strong drive makes it matter. The parametric frame is doubly-rotating, so
  the counter-rotating term is not cleanly defined here -- hence the check is there,
  not here.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Re-exported from sibling modules so `from gradpulse.parametric import ...` stays
# valid; absolute fallback supports running this file directly.
try:
    from .diagnostics import channel_unitarity, pauli_transfer_matrix
    from .profiles import ParametricCouplerProfile, RepresentativeDefaultsWarning
    from .basis import FourierBasis
    from .analysis import ParametricCZAnalysisMixin
    from ._device import DEVICE
except ImportError:  # pragma: no cover - direct-script execution
    from diagnostics import channel_unitarity, pauli_transfer_matrix
    from profiles import ParametricCouplerProfile, RepresentativeDefaultsWarning
    from basis import FourierBasis
    from analysis import ParametricCZAnalysisMixin
    from _device import DEVICE

DTYPE = torch.complex64


# ---- Operator builder ------------------------------------------------------

def _build_coupler_ops(profile: ParametricCouplerProfile,
                       dtype: torch.dtype = DTYPE) -> dict:
    """Construct 9D Hamiltonian + Lindblad operators for the parametric-coupler CZ.

    dtype: complex dtype for all operators (complex64 default; complex128 for
    the optimizer's precision='double' path). Every derived operator inherits
    it from the ladder operator a3.

    Hilbert space basis (n_levels per qubit; n_levels=3 shown):
      idx 0: |00>   idx 1: |01>   idx 2: |02>
      idx 3: |10>   idx 4: |11>   idx 5: |12>
      idx 6: |20>   idx 7: |21>   idx 8: |22>

    Computational subspace = indices [0, 1, n_levels, n_levels+1]
    (|00>, |01>, |10>, |11>). With the default n_levels=3 that is [0, 1, 3, 4].
    The single-transmon dimension n_levels is read from ``profile.n_levels`` so a
    truncation-convergence check (n_levels=4+) rebuilds every operator from it.
    """
    n_levels = int(getattr(profile, "n_levels", 3))
    if n_levels < 3:
        raise ValueError(
            f"n_levels must be >= 3 (need |2> for the leakage/|02> CZ physics), "
            f"got {n_levels}")
    # Single-transmon ladder operator a truncated at n_levels Fock states:
    # a|k> = sqrt(k)|k-1>, i.e. sqrt(1..n_levels-1) on the first super-diagonal.
    _sub = torch.tensor([math.sqrt(k) for k in range(1, n_levels)],
                        dtype=dtype, device=DEVICE)
    a3 = torch.diag(_sub, 1).contiguous()
    ad3 = a3.conj().t().contiguous()
    n3 = (ad3 @ a3).contiguous()
    i3 = torch.eye(n_levels, dtype=dtype, device=DEVICE)

    def kron2(A, B):
        return torch.kron(A.contiguous(), B.contiguous())

    # Energy scales (rad/ns)
    alpha1 = float(profile.anharm_ghz_q1) * 2 * math.pi
    alpha2 = float(profile.anharm_ghz_q2) * 2 * math.pi
    delta = (float(profile.freq_ghz_q2) - float(profile.freq_ghz_q1)) * 2 * math.pi

    anh1 = 0.5 * alpha1 * (ad3 @ ad3 @ a3 @ a3)  # (alpha/2) a+a+aa
    anh2 = 0.5 * alpha2 * (ad3 @ ad3 @ a3 @ a3)

    # Drift: q2 detuning + anharmonicities + parasitic ZZ (q1's rotating frame).
    # chi_zz_mhz models idle-spectator dressing of |11> as ordinary coherent
    # error the optimizer can pre-compensate; zero by default (term vanishes).
    chi_zz = float(profile.chi_zz_mhz) * 2 * math.pi / 1000.0   # rad/ns
    n1n2 = kron2(n3, i3) @ kron2(i3, n3)
    h_drift = (
        delta * kron2(i3, n3)
        + kron2(anh1, i3)
        + kron2(i3, anh2)
        + chi_zz * n1n2
    )

    # Drive operators: X_i = a_i + a_i+ (in-phase),  Y_i = i(a_i+ - a_i) (quadrature, for DRAG)
    x_tr = a3 + ad3
    y_tr = 1j * (ad3 - a3)
    x1 = kron2(x_tr, i3)
    x2 = kron2(i3, x_tr)
    y1 = kron2(y_tr, i3)
    y2 = kron2(i3, y_tr)

    # Parametric coupling, I/Q-decomposed at the |11>-|02> resonance:
    #   H = g * u3 * (cos(phi) * coupling_x + sin(phi) * coupling_y)
    #   coupling_x = a1+ a2 + a1 a2+ (I);  coupling_y = i(a1+ a2 - a1 a2+) (Q)
    # 3-channel mode fixes phi=0 (coupling_x only); 4-channel adds u4(t)->phi(t).
    coupling_x = kron2(ad3, a3) + kron2(a3, ad3)
    coupling_y = 1j * (kron2(ad3, a3) - kron2(a3, ad3))
    # Backwards-compatible alias used by the 3-channel code path
    coupling = coupling_x

    # Lindblad (jump) operators
    a_q1 = kron2(a3, i3).contiguous()
    a_q2 = kron2(i3, a3).contiguous()
    ad_q1 = kron2(ad3, i3).contiguous()
    ad_q2 = kron2(i3, ad3).contiguous()
    n_q1 = kron2(n3, i3).contiguous()
    n_q2 = kron2(i3, n3).contiguous()

    def _t_phi(t1: float, t2: float) -> float:
        # 1/T_phi = 1/T2 - 1/(2 T1)  (Markovian decomposition)
        rate = 1.0 / max(t2, 1.0) - 1.0 / max(2.0 * t1, 1.0)
        return 1.0 / max(rate, 1e-9)

    t_phi_q1 = _t_phi(profile.t1_ns_q1, profile.t2_ns_q1)
    t_phi_q2 = _t_phi(profile.t1_ns_q2, profile.t2_ns_q2)

    # Finite-T amplitude damping: relaxation -> (1+n_th)/T1 plus an excitation
    # jump at n_th/T1. n_th=0 (default) recovers the cold-bath 1/T1 exactly.
    n_th_q1 = max(0.0, float(profile.n_thermal_q1))
    n_th_q2 = max(0.0, float(profile.n_thermal_q2))
    rate_t1_q1 = math.sqrt((1.0 + n_th_q1) / profile.t1_ns_q1)
    rate_t1_q2 = math.sqrt((1.0 + n_th_q2) / profile.t1_ns_q2)
    rate_phi_q1 = math.sqrt(2.0 / t_phi_q1)
    rate_phi_q2 = math.sqrt(2.0 / t_phi_q2)

    L_t1_q1 = (rate_t1_q1 * a_q1).contiguous()
    L_t1_q2 = (rate_t1_q2 * a_q2).contiguous()
    L_phi_q1 = (rate_phi_q1 * n_q1).contiguous()
    L_phi_q2 = (rate_phi_q2 * n_q2).contiguous()
    # Thermal-excitation collapse operators (None for a cold bath, so the
    # simulator skips them and the default dynamics are untouched).
    L_th_q1 = ((math.sqrt(n_th_q1 / profile.t1_ns_q1) * ad_q1).contiguous()
               if n_th_q1 > 0.0 else None)
    L_th_q2 = ((math.sqrt(n_th_q2 / profile.t1_ns_q2) * ad_q2).contiguous()
               if n_th_q2 > 0.0 else None)

    # Anticommutator term: 0.5 * sum_k L_k+ L_k (thermal jumps included if any).
    _loss_terms = [L_t1_q1, L_t1_q2, L_phi_q1, L_phi_q2]
    if L_th_q1 is not None:
        _loss_terms.append(L_th_q1)
    if L_th_q2 is not None:
        _loss_terms.append(L_th_q2)
    L_loss_sum = (0.5 * sum(L.conj().t() @ L for L in _loss_terms)).contiguous()

    # Computational subspace indices: |00>=0, |01>=1, |10>=n_levels, |11>=n_levels+1
    comp_indices = torch.tensor([0, 1, n_levels, n_levels + 1],
                                dtype=torch.long, device=DEVICE)

    # N_Q1/N_Q2 double as the Stark/Z drive operators (ch4/ch5): H += alpha_stark *
    # u_i(t) * N_i, so the optimizer can pre-compensate AC Stark shifts from the coupler.

    return {
        "H_DRIFT":     h_drift.contiguous(),
        "X1":          x1.contiguous(),
        "X2":          x2.contiguous(),
        "Y1":          y1.contiguous(),
        "Y2":          y2.contiguous(),
        "COUPLING":    coupling.contiguous(),
        "COUPLING_X":  coupling_x.contiguous(),
        "COUPLING_Y":  coupling_y.contiguous(),
        "N_Q1":        n_q1,
        "N_Q2":        n_q2,
        "L_T1_Q1":     L_t1_q1,
        "L_T1_Q2":     L_t1_q2,
        "L_PHI_Q1":    L_phi_q1,
        "L_PHI_Q2":    L_phi_q2,
        "L_TH_Q1":     L_th_q1,
        "L_TH_Q2":     L_th_q2,
        "L_LOSS_SUM":  L_loss_sum,
        "I9":          torch.eye(n_levels ** 2, dtype=dtype, device=DEVICE),
        "comp_indices": comp_indices,
        "n_levels":    n_levels,
        "dim":         n_levels ** 2,
        "alpha1":      alpha1,
        "alpha2":      alpha2,
        "delta":       delta,
    }


# ---- Optimizer --------------------------------------------------------------

class ParametricCZOptimizer(ParametricCZAnalysisMixin):
    """GRAPE optimizer for a parametric-coupler CZ gate.

    Default 3-channel control: drives on q1, q2, plus parametric coupling
    activation (see ``n_channels`` for the 4- and 6-channel modes). Target =
    CZ in the 4D computational subspace. Output pulse format:
    [n_slices, n_channels] real envelope in [0,1].
    """
    def __init__(self, profile: Optional[ParametricCouplerProfile] = None,
                 bandwidth_mhz: float = 80.0,
                 use_drag: bool = False,
                 drag_order: int = 2,
                 n_channels: int = 3,
                 smoother_type: str = "gaussian",
                 activation: str = "clamp",
                 step_order: int = 1,
                 coupler_phase_mode: str = "phase",
                 delta_max_mhz: float = 30.0,
                 coupler_g_linewidth_mhz: Optional[float] = None,
                 line_response=None,
                 target_gate: str = "cz",
                 precision: str = "single"):
        """drag_order:
            0 - off (no quadrature correction)
            1 - 1st-order DRAG: v_y = -du/dt / alpha
            2 - 2nd-order: order-1 quadrature + in-phase amplitude correction
                Omega_x = Omega - Omega^3/(4 alpha^2) (standard transmon DRAG)
        n_channels:
            3 - legacy: q1_drive, q2_drive, coupler_envelope (θ=0 fixed)
            4 - + parametric coupler control channel u4(t) (see coupler_phase_mode)
                so coupling = g·u3·(cos(θ)·C_x + sin(θ)·C_y)
            6 - 4-channel + per-qubit Stark/Z drives (ch4=q1 N, ch5=q2 N)
                so H += α_stark·(u4·N1 + u5·N2). Lets the optimizer pre-
                compensate AC Stark shifts induced by the parametric drive
                on the qubits' bare frequencies.
        coupler_phase_mode: interpretation of channel 4 (needs n_channels >= 4)
            'phase'     - u4 → θ = π·u4 ∈ [-π, π], an absolute drive phase; the
                          coupler drive stays at the |11>-|02> resonance.
            'frequency' - u4 → instantaneous detuning δ = DELTA_MAX·u4 of the
                          parametric drive from resonance, with phase θ(t) its
                          running integral. Makes the drive frequency a real
                          control parameter (it can hold a static offset), which
                          'phase' (bounded per slice) cannot represent.
        delta_max_mhz: drive-detuning saturation for coupler_phase_mode='frequency'
            (±MHz). Conservative default; set to your pump's chirp range.
        step_order: master-equation integration order
            1 - Trotter split: exact unitary + first-order (Euler) Lindblad
                step. Default; matches gradpulse.validate's scheme.
            2 - symmetric (Strang) split with a 2nd-order dissipator substep,
                global error O(dt^2). At dt=1 ns it sits essentially at the
                dt→0 limit (see dt_convergence); same per-step cost order as 1.
        smoother_type:
            'gaussian' - soft Gaussian filter (default, ~6dB/oct rolloff)
            'firbrick' - windowed-sinc FIR brick-wall (~80dB stopband, sharper)
        coupler_g_linewidth_mhz: optional phenomenological linewidth (MHz) for
            the effective parametric coupling's falloff with drive detuning.
            Only active with coupler_phase_mode='frequency' (ignored otherwise,
            since 'phase' mode has no detuning). When set, the coupling magnitude
            is scaled by a single-pole Lorentzian of the instantaneous detuning
            delta(t): g_eff = g_max / sqrt(1 + (delta/kappa)^2), with kappa the
            given linewidth in rad/ns -- i.e. kappa is the detuning at which the
            coupling falls to 1/sqrt(2) of its on-resonance peak. This is a
            user-calibrated model, not a measurement; default None leaves the
            coupling at its peak value regardless of detuning.
        line_response: optional AWG/transmission-line impulse response convolved
            with the band-limited control, so the optimizer pre-compensates the
            distortion the qubit actually sees (finite settling, cable response).
            None (default) is a pristine line and leaves all behaviour unchanged.
            Pass {"type": "exponential", "tau_ns": t} for a single-pole settling
            tail (built at the sim dt), or an array-like causal impulse response
            sampled at the working dt -- e.g. your MEASURED response. Normalised
            to unit DC gain, so held amplitudes are preserved and only transitions
            are reshaped. This is infrastructure for hardware-realistic studies;
            the built-in exponential is illustrative, not a device measurement.
        precision: master-equation integration precision.
            'single' - complex64 / float32 (default; fast, ~1e-6 noise floor on
                       the accumulated density matrix -- ample for optimization).
            'double' - complex128 / float64 (~10x slower) -- drops the integrator
                       noise floor by orders of magnitude so dt_convergence shows
                       a clean O(dt^2) trend well below the single-precision floor.
        """
        self.profile = profile or ParametricCouplerProfile()
        self.bandwidth_mhz = float(bandwidth_mhz)
        self.use_drag = bool(use_drag)
        self.drag_order = int(drag_order) if use_drag else 0
        self.n_channels = int(n_channels)
        self.smoother_type = str(smoother_type).lower()
        if self.smoother_type not in ("gaussian", "firbrick"):
            raise ValueError(f"smoother_type must be 'gaussian' or 'firbrick', got {smoother_type!r}")
        if self.n_channels not in (3, 4, 6):
            raise ValueError(f"n_channels must be 3, 4, or 6, got {n_channels}")
        # Activation maps the raw parameter to [0,1]. 'clamp': hard clip -- simple,
        # but post-smoothing clip events leak broadband spectral content past the
        # bandwidth penalty. 'sigmoid': smooth saturation, no clip artifacts, so
        # bandwidth-limiting stays faithful.
        self.activation = str(activation).lower()
        if self.activation not in ("clamp", "sigmoid"):
            raise ValueError(f"activation must be 'clamp' or 'sigmoid', got {activation!r}")
        self.step_order = int(step_order)
        if self.step_order not in (1, 2):
            raise ValueError(f"step_order must be 1 or 2, got {step_order}")
        self.coupler_phase_mode = str(coupler_phase_mode).lower()
        if self.coupler_phase_mode not in ("phase", "frequency"):
            raise ValueError(
                f"coupler_phase_mode must be 'phase' or 'frequency', got {coupler_phase_mode!r}")
        if self.coupler_phase_mode == "frequency" and self.n_channels < 4:
            raise ValueError(
                "coupler_phase_mode='frequency' needs n_channels>=4 "
                "(channel 4 carries the drive detuning).")
        self.delta_max_mhz = float(delta_max_mhz)
        # Integration precision: complex/real dtype pair used everywhere in the
        # master-equation evolution. 'single' is the default.
        self.precision = str(precision).lower()
        if self.precision == "single":
            self.cdtype, self.rdtype = torch.complex64, torch.float32
        elif self.precision == "double":
            self.cdtype, self.rdtype = torch.complex128, torch.float64
        else:
            raise ValueError(f"precision must be 'single' or 'double', got {precision!r}")
        # Optional coupling-vs-detuning rolloff linewidth (rad/ns), used only in
        # coupler_phase_mode='frequency'. None => coupling held at peak.
        self.coupler_g_linewidth_mhz = coupler_g_linewidth_mhz
        if coupler_g_linewidth_mhz is not None:
            lw = float(coupler_g_linewidth_mhz)
            if lw <= 0:
                raise ValueError(
                    f"coupler_g_linewidth_mhz must be > 0, got {coupler_g_linewidth_mhz}")
            self.G_LINEWIDTH = 2 * math.pi * (lw / 1000.0)
        else:
            self.G_LINEWIDTH = None
        self._smoother_kernels: dict = {}
        # Optional AWG/line impulse response after band-limiting, so the optimizer
        # pre-compensates real distortion. None (default) = pristine line.
        self.line_response = self._normalize_line_response(line_response)
        self._line_kernels: dict = {}

        ops = _build_coupler_ops(self.profile, dtype=self.cdtype)
        self._H_DRIFT     = ops["H_DRIFT"]
        self._X1          = ops["X1"]
        self._X2          = ops["X2"]
        self._Y1          = ops["Y1"]
        self._Y2          = ops["Y2"]
        self._COUPLING    = ops["COUPLING"]
        self._COUPLING_X  = ops["COUPLING_X"]
        self._COUPLING_Y  = ops["COUPLING_Y"]
        self._N_Q1        = ops["N_Q1"]
        self._N_Q2        = ops["N_Q2"]
        self._alpha1      = float(ops["alpha1"])
        self._alpha2      = float(ops["alpha2"])
        self._L_T1_Q1     = ops["L_T1_Q1"]
        self._L_T1_Q2     = ops["L_T1_Q2"]
        self._L_PHI_Q1    = ops["L_PHI_Q1"]
        self._L_PHI_Q2    = ops["L_PHI_Q2"]
        self._L_TH_Q1     = ops["L_TH_Q1"]    # thermal excitation (None if cold)
        self._L_TH_Q2     = ops["L_TH_Q2"]
        self._L_LOSS_SUM  = ops["L_LOSS_SUM"]
        self._comp_idx    = ops["comp_indices"]
        # Every operator stack allocates from self._dim, so n_levels=4 (a
        # truncation-convergence check) needs no other change.
        self.n_levels     = int(ops["n_levels"])
        self._dim         = int(ops["dim"])             # = n_levels ** 2

        # Saturation rates (rad/ns)
        self.OMEGA_MAX = 2 * math.pi * (self.profile.omega_max_mhz / 1000.0)
        self.G_MAX     = 2 * math.pi * (self.profile.g_max_mhz / 1000.0)
        # Stark-shift compensation max: ±20 MHz of effective Z-rotation is
        # conservative given how much off-resonant power avoids populating the qubit.
        self.STARK_MAX = 2 * math.pi * (20.0 / 1000.0)   # 20 MHz
        # Drive-detuning saturation for coupler_phase_mode='frequency' (rad/ns).
        self.DELTA_MAX = 2 * math.pi * (self.delta_max_mhz / 1000.0)

        # target_gate: named gate or custom 4x4 unitary (see _build_target_unitary).
        if isinstance(target_gate, str):
            self.target_gate = target_gate.lower()
        else:
            self.target_gate = "custom"
        self.u_target_4x4 = self._build_target_unitary(target_gate)

    def _build_target_unitary(self, target_gate) -> torch.Tensor:
        """Target unitary in the 4D computational basis {|00>,|01>,|10>,|11>}.

        The optimizer is gate-agnostic: only u_target_4x4 distinguishes one
        two-qubit gate from another (both _process_fidelity and
        _avg_state_fidelity read it, nothing else). The parametric coupler is an
        exchange interaction g(t)*(a1^dag a2 + a1 a2^dag), so the iSWAP family is
        its NATIVE gate -- a population swap in the {|01>,|10>} subspace -- while
        CZ is synthesized indirectly via the |11>-|02> avoided crossing.

        ``target_gate`` is one of the named gates below OR a custom 4x4 unitary
        (numpy array, nested list, or torch tensor) in the {|00>,|01>,|10>,|11>}
        basis -- validated to be unitary so a typo can't silently define a
        non-physical target. Named gates:

          'cz'         - controlled-Z, diag(1, 1, 1, -1).
          'iswap'      - full iSWAP: |01> <-> |10> with a +i phase.
          'sqrt_iswap' - sqrt(iSWAP), the native half-swap and a perfect
                         entangler (two of them plus single-qubit gates compile
                         a CNOT).
        """
        # Custom matrix path: accept any array-like 4x4 and verify it is unitary.
        if not isinstance(target_gate, str):
            U = torch.as_tensor(np.asarray(target_gate, dtype=complex),
                                dtype=self.cdtype, device=DEVICE)
            if U.shape != (4, 4):
                raise ValueError(
                    f"custom target_gate must be a 4x4 unitary in the "
                    f"computational basis, got shape {tuple(U.shape)}")
            err = torch.linalg.norm(
                U.conj().t() @ U - torch.eye(4, dtype=self.cdtype, device=DEVICE)).item()
            if err > 1e-6:
                raise ValueError(
                    f"custom target_gate is not unitary (||U^dag U - I|| = {err:.2e})")
            return U
        target_gate = target_gate.lower()
        s = 1.0 / math.sqrt(2.0)
        gates = {
            "cz":         [[1, 0, 0,  0],
                           [0, 1, 0,  0],
                           [0, 0, 1,  0],
                           [0, 0, 0, -1]],
            "iswap":      [[1, 0,  0,  0],
                           [0, 0,  1j, 0],
                           [0, 1j, 0,  0],
                           [0, 0,  0,  1]],
            "sqrt_iswap": [[1, 0,    0,    0],
                           [0, s,    1j*s, 0],
                           [0, 1j*s, s,    0],
                           [0, 0,    0,    1]],
        }
        if target_gate not in gates:
            raise ValueError(
                f"target_gate must be one of {sorted(gates)} or a custom 4x4 "
                f"unitary, got {target_gate!r}")
        return torch.tensor(gates[target_gate], dtype=self.cdtype, device=DEVICE)

    # ---- Bandwidth limiter --------------------------------------------------
    def _build_smoother_kernel(self, dt_ns: float = 1.0):
        """Build the bandwidth-limiting filter kernel.

        gaussian: soft cutoff (~6dB/oct rolloff). Cheap, smooth, but content
                  above cutoff is only attenuated by ~6-20 dB at the device
                  filter edge; optimizer can quietly accumulate energy there.

        firbrick: windowed-sinc FIR with Hamming window. ~80 dB stopband
                  attenuation past cutoff. Slightly more expensive but
                  prevents the optimizer from "cheating" by putting energy
                  in the soft tail of the Gaussian.
        """
        if self.bandwidth_mhz <= 0:
            return None
        if self.smoother_type == "gaussian":
            sigma_t = 1.0 / (2.0 * math.pi * (self.bandwidth_mhz / 1000.0))
            sigma = max(sigma_t / dt_ns, 0.5)
            half = int(math.ceil(4.0 * sigma))
            ts = torch.arange(-half, half + 1, dtype=self.rdtype, device=DEVICE)
            k = torch.exp(-0.5 * (ts / sigma) ** 2)
            k = k / k.sum()
            return k.view(1, 1, -1)
        else:  # firbrick: windowed-sinc lowpass
            # Cutoff frequency normalized to Nyquist (which is 0.5/dt_ns in MHz)
            nyquist_mhz = 0.5 / (dt_ns * 1e-3)   # = 500 MHz at dt=1ns
            cutoff_norm = self.bandwidth_mhz / nyquist_mhz
            # Rule of thumb: L > 4/transition for steep Hamming-FIR rolloff;
            # transition ~0.1*cutoff_norm gives ~80 dB stopband.
            transition = max(0.05, 0.1 * cutoff_norm)
            length = int(math.ceil(4.0 / transition))
            length = length + (1 - length % 2)   # force odd length (linear phase)
            length = max(length, 21)              # minimum length
            half = length // 2
            ts = torch.arange(-half, half + 1, dtype=self.rdtype, device=DEVICE)
            # Sinc lowpass at cutoff_norm
            sinc = torch.where(
                ts == 0,
                torch.tensor(cutoff_norm, dtype=self.rdtype, device=DEVICE),
                torch.sin(math.pi * cutoff_norm * ts) / (math.pi * ts.clamp_min(1e-12)),
            )
            # Hamming window for stopband attenuation
            window = 0.54 - 0.46 * torch.cos(2 * math.pi *
                                              (torch.arange(length, dtype=self.rdtype,
                                                            device=DEVICE) / (length - 1)))
            k = sinc * window
            k = k / k.sum()
            return k.view(1, 1, -1)

    def _smooth(self, u, kernel):
        if kernel is None:
            return u
        # Kernels are cached in self.rdtype; if a caller hands in a pulse of a
        # different real precision (e.g. float32 into a precision='double'
        # optimizer), follow the input's dtype rather than erroring in conv1d.
        if kernel.dtype != u.dtype:
            kernel = kernel.to(u.dtype)
        B, N, C = u.shape
        x = u.permute(0, 2, 1).reshape(B * C, 1, N)
        K = kernel.shape[-1]
        pad = K // 2
        x = F.pad(x, (pad, pad), mode="replicate")
        x = F.conv1d(x, kernel)
        return x.reshape(B, C, N).permute(0, 2, 1)

    # ---- AWG / transmission-line response -----------------------------------
    def _normalize_line_response(self, spec):
        """Validate the line_response spec into None | dict | list.

        Forms:
          None             - pristine line (no distortion).
          {"type": "exponential", "tau_ns": t}
                           - a single-pole settling tail h(s)=exp(-s/t), the
                             dominant first-order AWG/cable response. Built at the
                             simulation dt so it is step-size consistent.
          array-like       - an explicit causal impulse response sampled at the
                             working dt (e.g. a MEASURED step/impulse response);
                             stored as a list so it serialises into pulse metadata.
        The kernel is normalised to unit DC gain at use time, so a held amplitude
        is preserved and only transitions are reshaped.
        """
        if spec is None:
            return None
        if isinstance(spec, dict):
            kind = str(spec.get("type", "exponential")).lower()
            if kind != "exponential":
                raise ValueError(
                    f"line_response type must be 'exponential', got {kind!r}")
            tau = float(spec.get("tau_ns", 0.0))
            if tau <= 0.0:
                raise ValueError("line_response 'exponential' needs tau_ns > 0")
            return {"type": "exponential", "tau_ns": tau}
        arr = [float(v) for v in spec]
        if not arr:
            raise ValueError("line_response array must be non-empty")
        return arr

    def _line_kernel(self, dt: float):
        """Causal, unit-DC-gain impulse-response kernel [1,1,K] at this dt (cached)."""
        if self.line_response is None:
            return None
        key = round(float(dt), 6)
        if key in self._line_kernels:
            return self._line_kernels[key]
        spec = self.line_response
        if isinstance(spec, dict):                       # parametric settling tail
            tau = spec["tau_ns"]
            n = max(2, int(math.ceil(6.0 * tau / dt)))   # 6 time-constants of support
            t = torch.arange(n, dtype=self.rdtype, device=DEVICE) * dt
            h = torch.exp(-t / tau)
        else:                                            # explicit measured response
            h = torch.as_tensor(spec, dtype=self.rdtype, device=DEVICE)
        h = h / h.sum().clamp_min(1e-12)                 # unit DC gain
        k = h.view(1, 1, -1)
        self._line_kernels[key] = k
        return k

    def _apply_line_response(self, u_smooth, dt: float):
        """Convolve each control channel with the causal line response.

        None => identity (the default), so the pristine-line path is unchanged.
        Otherwise the qubit sees distort(smooth(x)); because the convolution is
        differentiable the optimizer learns to pre-compensate it. The saved
        envelope (smoothed_waveform) is the PRE-distortion AWG output, and the
        QuTiP validator re-applies this same response, so the cross-check stays
        apples-to-apples.
        """
        kernel = self._line_kernel(dt)
        if kernel is None:
            return u_smooth
        B, N, C = u_smooth.shape
        K = kernel.shape[-1]
        x = u_smooth.permute(0, 2, 1).reshape(B * C, 1, N)
        x = F.pad(x, (K - 1, 0), mode="replicate")       # causal: pad the past only
        w = kernel.flip(-1).to(x.dtype)                  # conv1d is cross-corr; flip => sum_j h[j] x[n-j]
        y = F.conv1d(x, w)
        return y.reshape(B, C, N).permute(0, 2, 1)

    def smoothed_waveform(self, x_raw, dt: float = 1.0):
        """Return the bandwidth-smoothed pulse in [0,1] format.

        Mirrors the activation choice (clamp vs sigmoid) from
        simulate_gradient_batch so the saved pulse matches what the simulator
        evaluated. With sigmoid activation the output stays smoothly within
        (0, 1) without hard clamping artifacts.
        """
        if x_raw.dim() == 2:
            x_raw = x_raw.unsqueeze(0)        # add batch dim if missing
            squeeze = True
        else:
            squeeze = False
        if self.activation == "sigmoid":
            u01 = torch.sigmoid(x_raw)
        else:
            u01 = x_raw
        u_signed = 2.0 * u01 - 1.0            # [B, N, C]
        key = round(float(dt), 6)
        if key not in self._smoother_kernels:
            self._smoother_kernels[key] = self._build_smoother_kernel(dt_ns=dt)
        u_smooth = self._smooth(u_signed, self._smoother_kernels[key])
        # Re-center to [0,1]. With sigmoid the clamp is a no-op safety net (no
        # artifacts); with clamp activation, the clamp itself is the HF-artifact source.
        out = ((u_smooth + 1.0) * 0.5).clamp(0.0, 1.0)
        return out.squeeze(0) if squeeze else out

    # ---- Complete I/Q export (DRAG quadrature baked in) -------------------
    def iq_waveform(self, x_raw, dt: float = 1.0) -> dict:
        """The COMPLETE per-channel drive the simulator actually applied,
        including the DRAG quadrature -- the hardware-export-ready pulse.

        ``smoothed_waveform`` returns only the real, in-phase [0,1] envelope. With
        ``use_drag`` the simulator ALSO drives the quadrature (Y) operator with a
        derived Motzoi tone ``v = -d/dt(Omega)/alpha``, which that envelope does
        not carry -- so an exported pulse built from it is missing the imaginary
        part. ``iq_waveform`` returns the full complex envelope per drive channel,
        in physical rad/ns, so the exported pulse is a complete description and the
        receiver does not have to re-derive anything.

        Conventions (channel coefficient that multiplies its operator in H(t)):
          q1 drive : Omega1(t) = u1_eff*OMEGA_MAX + i*v1   (X1 + i Y1, rad/ns)
          q2 drive : Omega2(t) = u2_eff*OMEGA_MAX + i*v2
          coupler  : g(t) = u3*G_MAX*g_scale * exp(i*theta)  (theta=0 in 3-channel
                     mode, so the coupler is purely real there)
          Stark    : real uz*STARK_MAX on N (6-channel only)
        With ``use_drag=False`` every imaginary part is identically zero and the
        real parts equal ``smoothed_waveform`` rescaled to rad/ns, so this reduces
        to the existing (already complete) real export.

        Returns a dict with one complex numpy array per channel plus labels, units,
        and the physical peak |.| per channel (the absolute Rabi scale a device
        frame calibration must match). Feed a channel array straight to
        ``braket_bridge.to_braket_waveform`` / ``openpulse.to_openpulse_program``
        (both accept complex I/Q).
        """
        x = torch.as_tensor(x_raw, device=DEVICE, dtype=self.rdtype)
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        with torch.no_grad():
            c = self._smoothed_controls(x, dt)
            B = x.shape[0]
            chans, labels = [], []
            # q1, q2 drives: in-phase * OMEGA_MAX + i * DRAG quadrature.
            for k, (ueff, vq, lab) in enumerate((
                    (c["u1_eff"], c["v1"], "q1_drive"),
                    (c["u2_eff"], c["v2"], "q2_drive"))):
                inphase = ueff * self.OMEGA_MAX
                quad = vq if vq is not None else torch.zeros_like(inphase)
                chans.append((inphase + 1j * quad))
                labels.append(lab)
            # coupler: real magnitude * exp(i theta); theta carried by cos/sin_phi.
            gmag = c["u3"] * self.G_MAX
            if c["g_scale"] is not None:
                gmag = gmag * c["g_scale"]
            if c["cos_phi"] is not None:
                coupler = gmag * (c["cos_phi"] + 1j * c["sin_phi"])
            else:
                coupler = gmag.to(self.cdtype)
            chans.append(coupler)
            labels.append("coupler")
            # Stark drives (6-channel): real.
            if c["uz1"] is not None:
                chans.append((c["uz1"] * self.STARK_MAX).to(self.cdtype))
                chans.append((c["uz2"] * self.STARK_MAX).to(self.cdtype))
                labels += ["q1_stark", "q2_stark"]
            stack = torch.stack(chans, dim=-1).cpu().numpy()    # [B, n_slices, n_ch] complex
        if squeeze:
            stack = stack[0]
        peak = np.max(np.abs(stack), axis=-2)                   # per-channel |.| peak
        return {"iq": stack, "labels": labels, "dt_ns": float(dt),
                "units": "rad/ns", "peak": peak,
                "n_channels": stack.shape[-1]}

    # ---- Control preprocessing (shared) ----------------------------------
    def _smoothed_controls(self, u_stack, dt: float = 1.0) -> dict:
        """Map raw [B, n_slices, n_ch] parameters to per-slice control envelopes.

        The single source of truth for everything between the raw optimizer
        parameter and the per-slice drive coefficients: activation, the bandwidth
        smoother, AWG/line response, the coupler phase/frequency mode (and its
        optional g-rolloff), DRAG orders 1-2, and the 6-channel Stark drives.
        simulate_gradient_batch consumes this and assembles H(t) inline (its hot
        loop is unchanged); resonant_collision_fidelity reuses it so the collision
        diagnostic sees the EXACT same gate the optimizer/simulator does, with no
        risk of the smoothing/DRAG logic drifting between the two paths.

        Returns the smoothed drive amplitudes u1_eff/u2_eff (DRAG in-phase), the
        coupler envelope u3, the XY-phase factors cos_phi/sin_phi (None in
        3-channel mode), the coupling rolloff g_scale (None unless frequency mode +
        linewidth), the DRAG quadratures v1/v2 (None unless use_drag), and the
        Stark drives uz1/uz2 (None unless 6-channel).
        """
        n_ch = u_stack.shape[2]
        if self.activation == "sigmoid":
            u01 = torch.sigmoid(u_stack)
        else:
            u01 = u_stack
        u_signed = 2.0 * u01 - 1.0

        key = round(float(dt), 6)
        if key not in self._smoother_kernels:
            self._smoother_kernels[key] = self._build_smoother_kernel(dt_ns=dt)
        u_smooth = self._smooth(u_signed, self._smoother_kernels[key])
        u_smooth = self._apply_line_response(u_smooth, dt)

        u1 = u_smooth[:, :, 0]
        u2 = u_smooth[:, :, 1]
        u3 = u_smooth[:, :, 2]
        g_scale = None
        if n_ch >= 4:
            if self.coupler_phase_mode == "frequency":
                delta_t = self.DELTA_MAX * u_smooth[:, :, 3]
                phase_inc = delta_t * dt
                theta = torch.cumsum(phase_inc, dim=1) - 0.5 * phase_inc
                if self.G_LINEWIDTH is not None:
                    g_scale = torch.rsqrt(1.0 + (delta_t / self.G_LINEWIDTH) ** 2)
            else:
                theta = math.pi * u_smooth[:, :, 3]
            cos_phi = torch.cos(theta)
            sin_phi = torch.sin(theta)
        else:
            cos_phi = sin_phi = None
        if n_ch == 6:
            uz1 = u_smooth[:, :, 4]
            uz2 = u_smooth[:, :, 5]
        else:
            uz1 = uz2 = None

        if self.use_drag and self.drag_order > 0:
            def _ddt(u):
                u_pad = F.pad(u.unsqueeze(1), (1, 1), mode="replicate").squeeze(1)
                return (u_pad[:, 2:] - u_pad[:, :-2]) / (2.0 * dt)
            du1, du2 = _ddt(u1), _ddt(u2)
            v1 = -du1 * self.OMEGA_MAX / self._alpha1
            v2 = -du2 * self.OMEGA_MAX / self._alpha2
            if self.drag_order >= 2:
                c1 = self.OMEGA_MAX ** 2 / (4.0 * self._alpha1 ** 2)
                c2 = self.OMEGA_MAX ** 2 / (4.0 * self._alpha2 ** 2)
                u1_eff = u1 - c1 * (u1 ** 3)
                u2_eff = u2 - c2 * (u2 ** 3)
            else:
                u1_eff, u2_eff = u1, u2
        else:
            v1 = v2 = None
            u1_eff, u2_eff = u1, u2

        return {"u1_eff": u1_eff, "u2_eff": u2_eff, "u3": u3,
                "cos_phi": cos_phi, "sin_phi": sin_phi, "g_scale": g_scale,
                "v1": v1, "v2": v2, "uz1": uz1, "uz2": uz2}

    # ---- Core simulator ---------------------------------------------------
    def simulate_gradient_batch(self, u_stack, dt: float = 1.0,
                                diss_scale: float = 1.0, rho0=None,
                                detuning_offset=0.0, detuning_traj=None,
                                checkpoint_segments: int = 0):
        """Trotter-split master equation evolution.

        u_stack: [B, n_slices, n_channels] in [0,1]
            ch0/ch1: drive amplitudes for q1/q2
            ch2:     coupler envelope amplitude
            ch3:     (4-channel mode only) coupler-phase control, interpreted
                     per coupler_phase_mode ('phase': θ=π·u4 ∈ [-π,π];
                     'frequency': detuning δ=DELTA_MAX·u4 integrated to θ(t),
                     and -- if coupler_g_linewidth_mhz is set -- also scaling the
                     coupling magnitude by g/sqrt(1+(δ/κ)²))
            ch4/ch5: (6-channel mode only) per-qubit Stark/Z drives
        diss_scale: scalar multiplier on the Lindblad dissipative increment.
            Scaling BOTH T1 and T2 by a common factor m scales every collapse
            operator by 1/sqrt(m), hence the whole dissipator by 1/m, so
            diss_scale = 1/m reproduces coherence times m× the profile's,
            exactly and with no operator rebuild. 1.0 = profile as-is. The
            robust-optimization loop uses this to perturb T1/T2.
        rho0: optional initial operator stack [B, M, 9, 9]. The evolution is
            linear in rho, so M is arbitrary; None (default) builds the 4
            X-basis density matrices below. simulate_choi_batch passes the 16
            |i><j| operators here for the exact process fidelity.
        detuning_offset: static qubit-frequency offset added to the drift for
            the whole gate (rad/ns). A scalar applies a common δ·(N1+N2); a
            (δ1, δ2) pair applies per-qubit δ1·N1 + δ2·N2. The default 0.0 adds
            nothing and leaves the hot path unchanged. Used by the
            robustness sweep (drive-frequency miscalibration), the robust-
            optimization frequency axis, and the quasi-static-dephasing average,
            all of which feed a fixed offset (quasi-static = constant over the
            gate, varying shot-to-shot).
        Returns rho_final: [B, M, 9, 9] - the evolved operator stack. With the
            default rho0 (None) M=4: one density matrix per (batch_seed, X-basis
            input), the 4 inputs being |++>, |+->, |-+>, |-->.
        """
        # Coerce external numpy/CPU input onto DEVICE to match the internal operators;
        # a no-op (same object, autograd intact) when already a DEVICE tensor.
        u_stack = torch.as_tensor(u_stack, device=DEVICE)
        B, n_slices, n_ch = u_stack.shape
        if n_ch not in (3, 4, 6):
            raise ValueError(f"u_stack must have 3, 4, or 6 channels, got {n_ch}")

        # Extracted to _smoothed_controls so resonant_collision_fidelity sees the
        # EXACT same gate; the per-slice H assembly below is unchanged.
        _c = self._smoothed_controls(u_stack, dt)
        u3 = _c["u3"]
        cos_phi, sin_phi, g_scale = _c["cos_phi"], _c["sin_phi"], _c["g_scale"]
        uz1, uz2 = _c["uz1"], _c["uz2"]
        v1, v2 = _c["v1"], _c["v2"]
        u1_eff, u2_eff = _c["u1_eff"], _c["u2_eff"]

        if rho0 is not None:
            rho = rho0  # caller-supplied stack; evolution is linear in rho so M is arbitrary
        else:
            # Initialize in X-basis: |++>, |+->, |-+>, |--> (chosen for
            # sensitivity to CZ's diagonal phase structure)
            V = 0.5 * torch.tensor([[1,  1,  1,  1],
                                    [1, -1,  1, -1],
                                    [1,  1, -1, -1],
                                    [1, -1, -1,  1]], dtype=self.cdtype, device=DEVICE)
            rho = torch.zeros((B, 4, self._dim, self._dim),
                              device=DEVICE, dtype=self.cdtype)
            for k in range(4):
                psi_0 = torch.zeros(self._dim, dtype=self.cdtype, device=DEVICE)
                for j in range(4):
                    psi_0[self._comp_idx[j]] = V[j, k]
                rho_0 = torch.outer(psi_0, psi_0.conj())
                rho[:, k] = rho_0

        L1, L1d = self._L_T1_Q1,  self._L_T1_Q1.conj().t().contiguous()
        L2, L2d = self._L_T1_Q2,  self._L_T1_Q2.conj().t().contiguous()
        L3, L3d = self._L_PHI_Q1, self._L_PHI_Q1.conj().t().contiguous()
        L4, L4d = self._L_PHI_Q2, self._L_PHI_Q2.conj().t().contiguous()
        # Empty for a cold bath -> loops below skip them, default unchanged.
        Lth = [(L, L.conj().t().contiguous())
               for L in (self._L_TH_Q1, self._L_TH_Q2) if L is not None]

        if self.step_order == 2:
            # 2nd-order half-step propagator: exp((dt/2)*L_diss)*rho ~= rho + tau*L +
            # (tau^2/2)*L^2, tau=dt/2. diss_scale folds into L as in order 1.
            def _Ldiss(r):
                jump = (L1 @ r @ L1d) + (L2 @ r @ L2d) + \
                       (L3 @ r @ L3d) + (L4 @ r @ L4d)
                for Lk, Lkd in Lth:
                    jump = jump + (Lk @ r @ Lkd)
                anti = (self._L_LOSS_SUM @ r) + (r @ self._L_LOSS_SUM)
                return diss_scale * (jump - anti)

            def _diss_half(r):
                tau = 0.5 * dt
                Lr = _Ldiss(r)
                return r + tau * Lr + (0.5 * tau * tau) * _Ldiss(Lr)

        # Static frequency-offset term added to the drift (rad/ns); default 0
        # leaves the nominal path untouched.
        H_det = None
        if detuning_offset is not None:
            if isinstance(detuning_offset, (tuple, list)):
                d1, d2 = float(detuning_offset[0]), float(detuning_offset[1])
            else:
                d1 = d2 = float(detuning_offset)
            if d1 != 0.0 or d2 != 0.0:
                H_det = d1 * self._N_Q1 + d2 * self._N_Q2

        # Colored-noise per-qubit detuning trajectory [B, n_slices, 2] rad/ns; None
        # on the nominal path. Lets colored_noise_fidelity batch many realizations.
        dt_traj = None
        if detuning_traj is not None:
            dt_traj = torch.as_tensor(detuning_traj, dtype=self.rdtype, device=DEVICE)

        # Vectorized per-slice Hamiltonian H_all[B, n_slices, 9, 9], built once so the
        # matrix exponential is ONE batched call instead of n_slices sequential ones.
        # Byte-for-byte identical; ~3x faster on CPU and far more
        # on GPU, where each per-slice call was its own kernel launch. The control
        # envelopes are already [B, n_slices], so this is pure broadcasting. The
        # X drive uses the DRAG-corrected in-phase amplitude u*_eff (identical to u*
        # unless drag_order==2).
        def _bc(x):
            return x.view(B, n_slices, 1, 1)
        H_all = (self._H_DRIFT
                 + _bc(u1_eff * self.OMEGA_MAX) * self._X1
                 + _bc(u2_eff * self.OMEGA_MAX) * self._X2)
        # Coupling Hamiltonian. 3-channel: g·u3·C_x. 4-channel: also rotates by
        # phase φ = π·u4 in the XY plane via cos(φ)·C_x + sin(φ)·C_y.
        if cos_phi is not None:
            gu3 = _bc(u3 * self.G_MAX)
            if g_scale is not None:                 # detuning-dependent rolloff
                gu3 = gu3 * _bc(g_scale)
            H_all = H_all + gu3 * _bc(cos_phi) * self._COUPLING_X
            H_all = H_all + gu3 * _bc(sin_phi) * self._COUPLING_Y
        else:
            H_all = H_all + _bc(u3 * self.G_MAX) * self._COUPLING
        if v1 is not None:                          # DRAG quadrature
            H_all = H_all + _bc(v1) * self._Y1 + _bc(v2) * self._Y2
        if uz1 is not None:                         # 6-channel per-qubit Stark/Z
            H_all = H_all + _bc(uz1 * self.STARK_MAX) * self._N_Q1
            H_all = H_all + _bc(uz2 * self.STARK_MAX) * self._N_Q2
        if H_det is not None:                        # static offset (broadcasts)
            H_all = H_all + H_det
        if dt_traj is not None:                      # colored-noise per-slice detuning
            H_all = H_all + _bc(dt_traj[:, :, 0]) * self._N_Q1 \
                          + _bc(dt_traj[:, :, 1]) * self._N_Q2
        # .contiguous(): per-slice envelopes are strided slices that stay strided
        # through scalar multiplies, so H_all can end up non-contiguous -- which
        # matrix_exp's batched reshape rejects for some stride patterns (e.g. the
        # einsum-built envelope from optimize_spectral). No-op on the already-
        # contiguous piecewise path.
        U_all = torch.linalg.matrix_exp(-1j * H_all.contiguous() * dt)  # [B, n_slices, 9, 9]
        Ud_all = U_all.conj().transpose(-2, -1)

        def _advance(rho, i):
            """Evolve the operator stack one slice with the pre-computed per-slice
            propagator U_all[:, i]. The single per-slice body, called directly in the
            default loop and grouped under checkpointing -- so both paths run
            identical math (no chance of drift)."""
            U = U_all[:, i].unsqueeze(1)                 # [B, 1, 9, 9] (bcast over ops)
            Ud = Ud_all[:, i].unsqueeze(1)

            if self.step_order == 1:
                # Lie-Trotter: exact unitary then a first-order (Euler) Lindblad step.
                rho = U @ rho @ Ud
                jump = (L1 @ rho @ L1d) + (L2 @ rho @ L2d) + \
                       (L3 @ rho @ L3d) + (L4 @ rho @ L4d)
                for Lk, Lkd in Lth:
                    jump = jump + (Lk @ rho @ Lkd)
                anti = (self._L_LOSS_SUM @ rho) + (rho @ self._L_LOSS_SUM)
                rho = rho + dt * diss_scale * (jump - anti)
            else:
                # Strang split: half-step dissipator, exact unitary, half-step
                # dissipator -- symmetric composition gives global error O(dt^2).
                # The unitary factor is exact at any dt, so all the dt_convergence
                # dependence lives in the dissipator splitting.
                rho = _diss_half(rho)
                rho = U @ rho @ Ud
                rho = _diss_half(rho)
            return rho

        # checkpoint_segments=S>1 splits the n_slices steps into S segments, storing
        # only the rho at each boundary and recomputing the inside during backprop:
        # autograd memory ~O(n_slices/S)+O(S) instead of O(n_slices), at ~2x forward
        # compute. Default (0/1) is the plain loop.
        n_ckpt = int(checkpoint_segments or 0)
        if n_ckpt > 1 and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            def _segment(rho_in, lo, hi):
                r = rho_in
                for i in range(int(lo), int(hi)):
                    r = _advance(r, i)
                return r

            bounds = [round(k * n_slices / n_ckpt) for k in range(n_ckpt + 1)]
            for s in range(n_ckpt):
                lo, hi = bounds[s], bounds[s + 1]
                if hi <= lo:
                    continue
                rho = checkpoint(_segment, rho, lo, hi, use_reentrant=False)
        else:
            for i in range(n_slices):
                rho = _advance(rho, i)

        return rho

    # ---- Process-tomography (Choi) evolution -----------------------------
    def _choi_basis_rho0(self, B: int):
        """Initial operator stack for the d^2=16 computational-basis operators.

        Returns [B, 16, 9, 9] where operator m = i*4 + j is |i><j| with i, j the
        four computational levels (|00>, |01>, |10>, |11> at comp_idx). The
        off-diagonal m (i != j) are not density matrices, but the master-equation
        step is linear, so evolving them gives the channel's action on a full
        operator basis -- i.e. its Choi matrix -- for the exact gate fidelity.
        """
        ci = self._comp_idx
        rho0 = torch.zeros((B, 16, self._dim, self._dim),
                           device=DEVICE, dtype=self.cdtype)
        for i in range(4):
            for j in range(4):
                rho0[:, i * 4 + j, ci[i], ci[j]] = 1.0
        return rho0

    def simulate_choi_batch(self, u_stack, dt: float = 1.0,
                            diss_scale: float = 1.0, detuning_offset=0.0,
                            detuning_traj=None, checkpoint_segments: int = 0):
        """Evolve the 16 |i><j| computational-basis operators through the channel.

        Same pulse/integration as simulate_gradient_batch, but seeded with the
        d^2=16 operator basis instead of the 4 X-basis states, so the output
        fully characterizes the (leakage-aware) channel restricted to the
        computational subspace -- the input to the exact _process_fidelity.
        Returns [B, 16, 9, 9]; operator m = i*4 + j is the channel applied to
        |i><j|. detuning_traj ([B, n_slices, 2] rad/ns, optional) applies a
        time-dependent per-qubit detuning (colored-noise trajectories).
        checkpoint_segments>1 gradient-checkpoints the slice loop (lower autograd
        memory; see simulate_gradient_batch).
        """
        B = u_stack.shape[0]
        return self.simulate_gradient_batch(
            u_stack, dt=dt, diss_scale=diss_scale, rho0=self._choi_basis_rho0(B),
            detuning_offset=detuning_offset, detuning_traj=detuning_traj,
            checkpoint_segments=checkpoint_segments)

    # ---- Fidelity --------------------------------------------------------
    def _avg_state_fidelity(self, rho_final):
        """Avg over 4 X-basis inputs of <psi_k | rho_final_k | psi_k>.

        A cheap basis-average surrogate (4 X-basis inputs only, not a full
        operator basis). _process_fidelity is the rigorous metric -- the exact
        entanglement fidelity over the d^2 Choi basis; prefer it for the reported
        number and this for a fast optimization objective (use_process_fidelity).
        """
        V = 0.5 * torch.tensor([[1,  1,  1,  1],
                                [1, -1,  1, -1],
                                [1,  1, -1, -1],
                                [1, -1, -1,  1]], dtype=self.cdtype, device=DEVICE)
        target_subspace = self.u_target_4x4 @ V  # [4, 4]
        target_states_9 = torch.zeros((4, self._dim), device=DEVICE, dtype=self.cdtype)
        for k in range(4):
            for j in range(4):
                target_states_9[k, self._comp_idx[j]] = target_subspace[j, k]
        # target_rhos[k, i, j] = psi_k[i] * psi_k[j]^*  (pure-state projector)
        target_rhos = torch.einsum('ki,kj->kij',
                                    target_states_9, target_states_9.conj())
        # F_k = Tr(target_rho_k . rho_final_k) = sum_ij T[i,j] R[j,i]
        F = torch.einsum('kij,Bkji->Bk', target_rhos, rho_final).real
        return F.mean(dim=1)

    def _bandwidth_violation(self, x_clamped, dt_ns: float, filter_mhz: float):
        """Spectral energy above the device baseband filter (per batch).

        Returns shape [B], the sum (per channel, per batch) of squared FFT
        magnitudes for frequencies > filter_mhz, normalized by total spectral
        energy. The optimizer's smoother (bandwidth_mhz, soft Gaussian)
        attenuates but does not zero out content above its cutoff, so a
        polish run can quietly accumulate energy in the 200-300 MHz tail
        that the hardware AWG silently drops, producing a pulse that looks
        better in simulation than it plays on hardware. This penalty makes
        such cheating costly directly in the loss.
        """
        # x_clamped: [B, n_slices, n_channels] in [0, 1]
        # Use the SMOOTHED signed waveform; it's what the simulator/hardware
        # would actually see.
        u_signed = 2.0 * x_clamped - 1.0
        key = round(float(dt_ns), 6)
        if key not in self._smoother_kernels:
            self._smoother_kernels[key] = self._build_smoother_kernel(dt_ns=dt_ns)
        u_smooth = self._smooth(u_signed, self._smoother_kernels[key])

        B, N, C = u_smooth.shape
        # Real FFT along time axis; treat all channels as real-valued (which
        # is true for our envelope outputs; phase channel u4 is also real
        # in the saved pulse format).
        spec = torch.fft.rfft(u_smooth, dim=1)               # [B, N//2+1, C]
        freqs_mhz = torch.fft.rfftfreq(N, d=dt_ns * 1e-9,
                                        device=u_smooth.device) * 1e-6
        mag2 = (spec.real ** 2 + spec.imag ** 2)              # [B, N//2+1, C]
        mask = (freqs_mhz > filter_mhz).view(1, -1, 1).to(mag2.dtype)  # [1, N//2+1, 1]
        # Normalize by total energy so penalty is scale-invariant in pulse amplitude
        total_energy = mag2.sum(dim=(1, 2)).clamp_min(1e-12)  # [B]
        violation = (mag2 * mask).sum(dim=(1, 2)) / total_energy  # [B], in [0, 1]
        return violation

    def _leakage(self, rho_final):
        """Average population that leaked OUT of the 4D computational subspace.

        Accepts either operator stack the simulator produces:
          - [B, 4, 9, 9]  the 4 X-basis density matrices (simulate_gradient_batch)
          - [B, 16, 9, 9] the Choi stack (simulate_choi_batch); only the 4
            diagonal operators |i><i| (m in {0, 5, 10, 15}) carry populations, so
            leakage is averaged over the computational-basis inputs |i>.
        Population in the computational subspace = sum of rho diagonal at
        comp_idx; leakage = 1 - that. Used as an explicit regularizer; penalizing
        it directly favors pulses that minimize |2*> excitation (the dominant
        transmon error mechanism). Returns shape [B].
        """
        if rho_final.shape[1] == 16:
            rho_pop = rho_final[:, [0, 5, 10, 15]]          # diagonal |i><i| inputs
        else:
            rho_pop = rho_final                              # 4 X-basis states
        diag = rho_pop.diagonal(dim1=-2, dim2=-1).real      # [B, 4, 9]
        comp_pop = diag[..., self._comp_idx].sum(dim=-1)    # [B, 4]
        return (1.0 - comp_pop).mean(dim=1).clamp(0.0, 1.0)  # [B]

    def _process_fidelity(self, rho_choi):
        """Exact entanglement (process) fidelity of the channel to the target.

        Consumes the 16-operator Choi stack from simulate_choi_batch:
        ``rho_choi[:, m]`` with m = i*4 + j is the channel applied to |i><j| over
        the four computational levels. Projecting each evolved operator to the
        4-D computational subspace gives the channel's Choi matrix; the process
        (entanglement) fidelity to target U is the exact

            F_proc = (1/d^2) * sum_{i,j} <i| U^dag Phi(|i><j|) U |j>,   d = 4,

        a genuine 2-design (Haar) average rather than the 4-state basis-average
        proxy (which survives as _avg_state_fidelity). The corresponding average
        gate fidelity is F_avg = (d*F_proc + 1)/(d + 1). The metric is
        leakage-aware: population that leaves the computational subspace lowers
        the projected Choi trace and hence F_proc, with no special-casing. The
        QuTiP cross-check evaluates this identical quantity. Returns F_proc
        clamped to [0, 1], shape [B].

        Refs: Horodecki et al., PRA 60, 1888 (1999); Nielsen, PLA 303, 249 (2002).
        """
        ci = self._comp_idx
        B = rho_choi.shape[0]
        # Project each evolved operator to the computational subspace:
        # proj[b, m, a, c] = <a| Phi(|i><j|) |c> with m = i*4 + j (a, c in 0..3).
        proj = rho_choi[:, :, ci, :][:, :, :, ci]        # [B, 16, 4, 4]
        C = proj.reshape(B, 4, 4, 4, 4)                   # [B, i, j, a, c]
        U = self.u_target_4x4                            # [4, 4]
        # F_proc = (1/d^2) sum_{i,j,a,c} conj(U[a,i]) C[i,j,a,c] U[c,j].
        F_proc = torch.einsum('ai,zijac,cj->z', U.conj(), C, U).real / 16.0
        # Clamp to avoid pathological gradients near edge cases.
        return F_proc.clamp(0.0, 1.0)

    # ---- Error budget / coherence diagnostics ----------------------------
    def _comp_superop_from_choi(self, rho_choi) -> np.ndarray:
        """[16, 16] computational-subspace superoperator from a (B=1) Choi stack.

        Column m = vec(Phi(E_m)) with E_m = |i><j| (m = i*4 + j) projected to the
        4-D computational subspace; vec is the row-major reshape, matching the
        basis convention of pauli_transfer_matrix.
        """
        ci = self._comp_idx
        proj = rho_choi[0][:, ci][:, :, ci]              # [16, 4, 4]
        proj = proj.detach().cpu().numpy()
        return np.stack([proj[m].reshape(-1) for m in range(16)], axis=1)


    # ---- Warm-start ------------------------------------------------------
    def _warm_start(self, n_slices: int, mode: str = "parametric_cz",
                    rng: Optional[torch.Generator] = None,
                    n_channels: Optional[int] = None):
        """Initial pulse envelope for GRAPE search.

        modes:
          parametric_cz - quiet drives (0.5), Gaussian envelope on coupler ch2
          echo          - parametric_cz + π-pulse echo on both qubits at midpoint
                          (refocuses dephasing during gate; helps when T2-limited)
          flat          - everything at 0.5 (signed midpoint)
          random        - uniform [0.2, 0.8]

        n_channels: 3 or 4 (with phase mod). If None, uses self.n_channels.
        For 4-channel modes, ch3 is initialized at 0.5 (signed = 0 → φ = 0)
        which reproduces the 3-channel behavior exactly at start.
        """
        nc = n_channels if n_channels is not None else self.n_channels
        ts = torch.linspace(0, 1, n_slices, device=DEVICE)
        if mode == "parametric_cz":
            env = 0.5 + 0.4 * torch.exp(-((ts - 0.5) / 0.2) ** 2)
            quiet = torch.full_like(env, 0.5)
            out = torch.stack([quiet, quiet, env], dim=-1)
        elif mode == "echo":
            # Coupler envelope: split into two halves with a wider gap mid-gate
            # so the X*X echo can act on bare qubits (no parametric coupling).
            # Both lobes and the π-pulse are wider here than in earlier versions
            # so the FFT stays under the device's 250 MHz baseband filter even
            # when smoother bandwidth is set to 100-120 MHz.
            half_l = 0.5 + 0.4 * torch.exp(-((ts - 0.225) / 0.13) ** 2)
            half_r = 0.5 + 0.4 * torch.exp(-((ts - 0.775) / 0.13) ** 2)
            in_gap = (ts > 0.42) & (ts < 0.58)
            coupler = torch.where(in_gap, torch.full_like(ts, 0.5),
                                          torch.maximum(half_l, half_r))
            # Drives: X*X π-pulse during the [0.42, 0.58] gap, wider Gaussian
            # (sigma=0.05 of gate), keeps spectrum under device filter.
            pi_pulse = 0.5 + 0.45 * torch.exp(-((ts - 0.5) / 0.05) ** 2)
            out = torch.stack([pi_pulse, pi_pulse, coupler], dim=-1)
        elif mode == "flat":
            out = torch.full((n_slices, 3), 0.5, device=DEVICE)
        elif mode == "random":
            rng = rng or torch.Generator(device=DEVICE).manual_seed(0)
            out = 0.2 + 0.6 * torch.rand((n_slices, 3),
                                          generator=rng, device=DEVICE)
        else:
            raise ValueError(f"Unknown warm-start: {mode!r}")
        out = out.clamp(0.05, 0.95)
        # Pad to 4 or 6 channels with neutral defaults (0.5 → signed 0).
        # Initializing all extra channels at 0.5 means the higher-channel
        # optimizer starts EXACTLY where the 3-channel one starts, so any
        # improvement during search is real expressivity gain.
        if nc == 4:
            phase_ch = torch.full((n_slices, 1), 0.5, device=DEVICE)
            out = torch.cat([out, phase_ch], dim=-1)
        elif nc == 6:
            extra = torch.full((n_slices, 3), 0.5, device=DEVICE)  # phase + 2 Stark
            out = torch.cat([out, extra], dim=-1)
        return out

    # ---- L-BFGS polish ---------------------------------------------------
    def _dephasing_avg_choi(self, x_clamped, dt, sigma_mhz, n_nodes,
                            diss_scale: float = 1.0, checkpoint_segments: int = 0):
        """Choi channel averaged over quasi-static (Gaussian RMS ``sigma_mhz``)
        per-qubit frequency offsets, using the SAME deterministic Gauss-Hermite
        quadrature ``quasi_static_fidelity`` scores with. Minimising
        ``1 - F_proc`` of this channel hardens the gate against exactly the slow
        1/f dephasing the scorer later measures -- optimize-against and
        measure-against aligned, the way the robust ``*_jitter`` axes mirror
        ``robustness_sweep``.

        It averages the CHANNEL (the physically correct incoherent shot-to-shot
        mixture), not the per-offset fidelities -- a constant-weighted sum of
        ``simulate_choi_batch`` outputs, so it stays differentiable for GRAPE.
        ``n_nodes`` Gauss-Hermite nodes per qubit (``n_nodes**2`` evolutions); 3
        steers the gradient well and the final pulse is re-scored at the 5-node
        ``quasi_static_fidelity`` default.
        """
        t_nodes, w = np.polynomial.hermite_e.hermegauss(int(n_nodes))
        p = w / math.sqrt(2.0 * math.pi)
        scale = 2.0 * math.pi * (float(sigma_mhz) / 1000.0)   # rad/ns per unit node
        avg = None
        for ti, pi in zip(t_nodes, p):
            for tj, pj in zip(t_nodes, p):
                choi = self.simulate_choi_batch(
                    x_clamped, dt=dt, diss_scale=diss_scale,
                    detuning_offset=(scale * float(ti), scale * float(tj)),
                    checkpoint_segments=checkpoint_segments)
                term = (float(pi) * float(pj)) * choi
                avg = term if avg is None else avg + term
        return avg

    def _lbfgs_refine(self, x_init, lbfgs_iters: int = 30, n_slices: int = 150,
                      dt_ns: float = 1.0, use_process_fidelity: bool = True,
                      leakage_penalty: float = 0.0,
                      bandwidth_penalty: float = 0.0,
                      bandwidth_filter_mhz: float = 250.0,
                      checkpoint_segments: int = 0,
                      diss_scale: float = 1.0,
                      robust_dephasing_sigma_mhz: float = 0.0,
                      robust_dephasing_nodes: int = 3,
                      robust_filter_sigma_mhz: float = 0.0,
                      robust_filter_alpha: float = 1.0,
                      robust_filter_band_mhz: tuple = (1e-3, 5.0),
                      robust_filter_n_freq: int = 96):
        """Refine a single waveform with L-BFGS (better convergence near F→1).

        ``diss_scale`` and ``robust_dephasing_*``/``robust_filter_*`` mirror
        ``optimize_multi_seed`` so the polish optimises the SAME objective Adam did
        (a coherent-only, dephasing-robust, or filter-robust Adam result is not
        silently re-polished toward the nominal open-system fidelity)."""
        x = x_init.clone().detach().requires_grad_(True)
        opt = torch.optim.LBFGS([x], lr=0.5, max_iter=lbfgs_iters,
                                 history_size=20, line_search_fn="strong_wolfe")

        last_fid = [0.0]
        def closure():
            opt.zero_grad()
            # With sigmoid activation, the simulator already maps unbounded x
            # to [0, 1] internally, so no need to clamp. With clamp activation,
            # we still pre-clamp here.
            x_clamped = x if self.activation == "sigmoid" else x.clamp(0.0, 1.0)
            x_clamped = x_clamped.unsqueeze(0)
            if robust_dephasing_sigma_mhz > 0.0:
                rho_final = self._dephasing_avg_choi(
                    x_clamped, dt_ns, robust_dephasing_sigma_mhz,
                    robust_dephasing_nodes, diss_scale=diss_scale,
                    checkpoint_segments=checkpoint_segments)
                fids = self._process_fidelity(rho_final)
            elif robust_filter_sigma_mhz > 0.0:
                rho_final = self.simulate_choi_batch(
                    x_clamped, dt=dt_ns, diss_scale=diss_scale,
                    checkpoint_segments=checkpoint_segments)
                lo, hi = robust_filter_band_mhz
                fids = self._process_fidelity(rho_final) - self._filter_dephasing_infidelity(
                    x_clamped, dt_ns, robust_filter_sigma_mhz, robust_filter_alpha,
                    lo, hi, robust_filter_n_freq)
            elif use_process_fidelity:
                rho_final = self.simulate_choi_batch(
                    x_clamped, dt=dt_ns, diss_scale=diss_scale,
                    checkpoint_segments=checkpoint_segments)
                fids = self._process_fidelity(rho_final)
            else:
                rho_final = self.simulate_gradient_batch(
                    x_clamped, dt=dt_ns, diss_scale=diss_scale,
                    checkpoint_segments=checkpoint_segments)
                fids = self._avg_state_fidelity(rho_final)
            loss = 1.0 - fids.mean()
            if leakage_penalty > 0.0:
                loss = loss + leakage_penalty * self._leakage(rho_final).mean()
            if bandwidth_penalty > 0.0:
                viol = self._bandwidth_violation(x_clamped, dt_ns, bandwidth_filter_mhz)
                loss = loss + bandwidth_penalty * viol.mean()
            loss.backward()
            last_fid[0] = float(fids.item())
            return loss

        opt.step(closure)
        # Return the RAW optimizer parameter (the activation's input space), not
        # sigmoid(x): callers round-trip this through smoothed_waveform /
        # simulate_gradient_batch / dt_convergence to reproduce the polished pulse
        # exactly. Returning sigmoid(x) used to double-apply the sigmoid when a
        # caller saved smoothed_waveform(refined_x) -- fixed by returning raw x.
        if self.activation == "sigmoid":
            return x.detach(), last_fid[0]
        return x.clamp(0.0, 1.0).detach(), last_fid[0]

    # ---- Multi-seed driver -----------------------------------------------
    def optimize_multi_seed(self, label: str = "parametric_grape",
                            n_seeds: int = 4, iterations: int = 200,
                            n_slices: int = 150,
                            warm_start_mode: str = "parametric_cz",
                            lr: float = 0.01,
                            lbfgs_polish: bool = True,
                            lbfgs_iters: int = 50,
                            dt_ns: float = 1.0,
                            use_process_fidelity: bool = True,
                            lr_schedule: str = "cosine",
                            lr_min_factor: float = 0.05,
                            warmup_frac: float = 0.05,
                            leakage_penalty: float = 0.0,
                            bandwidth_penalty: float = 0.0,
                            bandwidth_filter_mhz: float = 250.0,
                            robust_g_jitter: float = 0.0,
                            robust_t12_jitter: float = 0.0,
                            robust_amp_jitter: float = 0.0,
                            robust_freq_jitter_mhz: float = 0.0,
                            robust_dephasing_sigma_mhz: float = 0.0,
                            robust_dephasing_nodes: int = 3,
                            robust_filter_sigma_mhz: float = 0.0,
                            robust_filter_alpha: float = 1.0,
                            robust_filter_band_mhz: tuple = (1e-3, 5.0),
                            robust_filter_n_freq: int = 96,
                            diss_scale: float = 1.0,
                            warm_start_pulse: Optional[np.ndarray] = None,
                            grad_clip: float = 1e3,
                            checkpoint_segments=0,
                            rng: Optional[torch.Generator] = None):
        """Multi-seed GRAPE. Each seed gets a perturbed warm-start.

        lr_schedule:
          'cosine'   - linear warmup then cosine decay to lr_min_factor*lr
          'constant' - flat lr

        diss_scale: scalar on the in-loop Lindblad dissipator. 1.0 (default)
          optimises the true open-system process fidelity; ``0.0`` makes the
          objective the COHERENT (unitary + leakage) fidelity -- the
          "optimise-coherent, multiply-by-e^{-t/T} afterward" recipe this package
          argues against. Reused by ``headtohead`` to run that recipe honestly.
        robust_dephasing_sigma_mhz: if > 0, optimise the process fidelity of the
          channel AVERAGED over quasi-static (Gaussian RMS sigma) per-qubit
          frequency offsets, via the same Gauss-Hermite grid ``quasi_static_fidelity``
          scores with -- i.e. put the slow 1/f dephasing INSIDE the gradient instead
          of only scoring it afterward. Requires use_process_fidelity=True and is
          standalone (not combinable with the robust_*_jitter axes). With it on,
          ``best_fidelity`` is the dephasing-averaged process fidelity (the
          objective), not the nominal one. Costs ``robust_dephasing_nodes**2``
          evolutions per step (default 3 -> 9).
        robust_filter_sigma_mhz: if > 0, add the first-order 1/f^alpha dephasing
          infidelity from the leakage-inclusive FILTER FUNCTION to the objective:
          ``loss = (1 - F_nominal) + sigma_rad^2 * <F>_band``. Unlike
          robust_dephasing_sigma_mhz (which hardens only F(0), the slow limit), this
          hardens the gate across the whole noise band ``robust_filter_band_mhz``
          (default 1e-3..5 MHz) with PSD exponent ``robust_filter_alpha`` (1 = 1/f),
          using the SAME estimator ``filter_function_fidelity`` scores with -- so the
          mid-band sensitivity is optimised, not just measured. The two infidelities
          are both first order in their noise and independent, so the implicit weight
          is 1 (no hyperparameter). Requires use_process_fidelity=True; standalone
          (not combinable with robust_dephasing or the robust_*_jitter axes, which
          would double-count the slow band). With it on, ``best_fidelity`` is the
          combined (nominal minus filter) estimate. Costs one nominal sim plus
          ``n_seeds`` toggling-frame builds per step.
        """
        rng = rng or torch.Generator(device=DEVICE).manual_seed(42)
        # checkpoint_segments="auto": pick the sqrt(n_slices) split that minimises
        # autograd memory x recompute, but only once the slice loop is long enough to
        # be worth the recompute (short pulses keep the plain, fastest path). Resolved
        # to an int here, before any sim call -- exact gradients either way; this only
        # trades a little recompute for lower peak memory on long pulses / big models.
        if checkpoint_segments == "auto":
            checkpoint_segments = round(n_slices ** 0.5) if n_slices >= 64 else 0
        checkpoint_segments = int(checkpoint_segments)
        if robust_dephasing_sigma_mhz > 0.0:
            if not use_process_fidelity:
                raise ValueError(
                    "robust_dephasing_sigma_mhz requires use_process_fidelity=True "
                    "(it averages the Choi channel over the dephasing distribution).")
            if (robust_g_jitter or robust_t12_jitter or robust_amp_jitter
                    or robust_freq_jitter_mhz):
                raise ValueError(
                    "robust_dephasing_sigma_mhz is a standalone objective; do not "
                    "combine it with the robust_*_jitter axes -- the cost multiplies "
                    "and the channel-average (here) vs loss-average (jitter) "
                    "semantics differ.")
        if robust_filter_sigma_mhz > 0.0:
            if not use_process_fidelity:
                raise ValueError(
                    "robust_filter_sigma_mhz requires use_process_fidelity=True "
                    "(it adds the filter-function process infidelity to F_proc).")
            if (robust_dephasing_sigma_mhz or robust_g_jitter or robust_t12_jitter
                    or robust_amp_jitter or robust_freq_jitter_mhz):
                raise ValueError(
                    "robust_filter_sigma_mhz is a standalone objective; do not "
                    "combine it with robust_dephasing_sigma_mhz (double-counts the "
                    "slow band -- the filter function already covers F(0)) or the "
                    "robust_*_jitter axes.")
            lo, hi = robust_filter_band_mhz
            if not (0.0 < lo < hi):
                raise ValueError(
                    f"robust_filter_band_mhz must be 0 < f_low < f_high, got "
                    f"{robust_filter_band_mhz}.")

        # In sigmoid mode, the optimizer parameter x is unbounded and
        # sigmoid(x) gives the [0, 1] envelope. Warm-starts in [0.05, 0.95]
        # are converted via logit so sigmoid(logit(u)) = u, preserving the
        # initial pulse shape.
        def _to_param_space(u):
            if self.activation == "sigmoid":
                u = u.clamp(1e-4, 1 - 1e-4)
                return torch.log(u / (1.0 - u))
            return u

        x = torch.zeros((n_seeds, n_slices, self.n_channels),
                        dtype=self.rdtype, device=DEVICE, requires_grad=True)
        with torch.no_grad():
            base = self._warm_start(n_slices, mode=warm_start_mode, rng=rng)
            base_seed = int(rng.initial_seed()) if rng is not None else 12345
            if warm_start_pulse is not None:
                wp = torch.as_tensor(warm_start_pulse, dtype=self.rdtype, device=DEVICE)
                if wp.shape[0] != n_slices or wp.shape[1] not in (3, 4, 6):
                    raise ValueError(
                        f"warm_start_pulse shape {tuple(wp.shape)} must be "
                        f"(n_slices={n_slices}, 3/4/6)"
                    )
                if wp.shape[1] < self.n_channels:
                    pad_n = self.n_channels - wp.shape[1]
                    pad = torch.full((n_slices, pad_n), 0.5, device=DEVICE)
                    wp = torch.cat([wp, pad], dim=-1)
                elif wp.shape[1] > self.n_channels:
                    wp = wp[:, :self.n_channels]
                base = wp.clone()
            for i in range(n_seeds):
                g_i = torch.Generator(device=DEVICE).manual_seed(
                    (base_seed & 0x7fffffff) * 1009 + i
                )
                if warm_start_pulse is not None:
                    if i == 0:
                        u_init = base.clone()
                    else:
                        noise = 0.05 * (torch.rand((n_slices, self.n_channels),
                                                   generator=g_i, device=DEVICE) - 0.5)
                        u_init = torch.clamp(base + noise, 0.05, 0.95)
                else:
                    rand_warm = self._warm_start(n_slices, mode="random", rng=g_i,
                                                  n_channels=self.n_channels)
                    u_init = torch.clamp(0.7 * base + 0.3 * rand_warm, 0.05, 0.95)
                x[i] = _to_param_space(u_init)

        opt = torch.optim.Adam([x], lr=lr)
        best_fid = torch.zeros(n_seeds, device=DEVICE)
        # Divergence guards + convergence diagnostics: roll back to the last
        # finite parameter on any non-finite loss/gradient (so one blown-up seed
        # can never poison the batch or silently corrupt best_fidelity), and
        # record the convergence history for plotting / "did it converge?".
        last_good_x = x.detach().clone()
        n_nonfinite = 0
        last_grad_norm = float("nan")
        history = []
        best_wf = torch.zeros_like(x)
        best_raw = torch.zeros_like(x)  # raw param per seed; see _lbfgs_refine's return comment

        loss_label = "F_proc" if use_process_fidelity else "F_state"
        gate_dur_ns = n_slices * dt_ns
        sched_label = lr_schedule if lr_schedule == "constant" else f"cosine(min={lr_min_factor:.2f})"
        print(f" [gradpulse] Starting parametric-CZ GRAPE ({n_seeds} seeds, "
              f"{n_slices} slices @ {dt_ns} ns = {gate_dur_ns:.1f} ns gate, "
              f"{iterations} iters, loss={loss_label}, lr={lr:.4f}/{sched_label}, "
              f"drag_order={self.drag_order})...")
        if diss_scale == 0.0:
            print("    (coherent-only objective: diss_scale=0, decoherence OFF "
                  "in the loop -- the optimise-coherent-then-multiply baseline)")
        if robust_dephasing_sigma_mhz > 0.0:
            print(f"    (dephasing-robust objective: sigma="
                  f"{robust_dephasing_sigma_mhz} MHz over "
                  f"{robust_dephasing_nodes}^2 Gauss-Hermite nodes)")
        if robust_filter_sigma_mhz > 0.0:
            print(f"    (filter-function-robust objective: sigma="
                  f"{robust_filter_sigma_mhz} MHz, 1/f^{robust_filter_alpha} over "
                  f"{robust_filter_band_mhz[0]}-{robust_filter_band_mhz[1]} MHz, "
                  f"{robust_filter_n_freq} freqs -- whole-band, not just F(0))")

        warmup_iters = max(1, int(warmup_frac * iterations))
        for it in range(iterations):
            # LR schedule: linear warmup then cosine decay to lr_min_factor*lr
            if lr_schedule == "cosine":
                if it < warmup_iters:
                    lr_now = lr * (it + 1) / warmup_iters
                else:
                    progress = (it - warmup_iters) / max(1, iterations - warmup_iters)
                    cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                    lr_now = lr * (lr_min_factor + (1.0 - lr_min_factor) * cos_factor)
                for pg in opt.param_groups:
                    pg["lr"] = lr_now

            opt.zero_grad()
            # Sigmoid: simulator handles the [0,1] mapping; pass raw x.
            # Clamp:  pre-clamp here.
            x_clamped = x if self.activation == "sigmoid" else x.clamp(0.0, 1.0)

            # Robust optimization: average the loss over +/- perturbations of g,
            # T1/T2, drive amplitude, and detuning -- the SAME axes robustness_sweep
            # measures, so optimize-for-robustness and measure-robustness align.
            any_jitter = (robust_g_jitter > 0.0 or robust_t12_jitter > 0.0
                          or robust_amp_jitter > 0.0 or robust_freq_jitter_mhz > 0.0)
            if robust_dephasing_sigma_mhz > 0.0:
                # fids IS the dephasing-averaged robust objective here, so it drives
                # both the loss and best-tracking (not the nominal F_proc).
                rho_final = self._dephasing_avg_choi(
                    x_clamped, dt_ns, robust_dephasing_sigma_mhz,
                    robust_dephasing_nodes, diss_scale=diss_scale,
                    checkpoint_segments=checkpoint_segments)
                fids = self._process_fidelity(rho_final)
                loss = 1.0 - fids.mean()
            elif robust_filter_sigma_mhz > 0.0:
                # rho_final stays the true open-system channel (so leakage_penalty
                # sees the real gate); fids = nominal - filter infidelity drives
                # both the loss and best-tracking.
                rho_final = self.simulate_choi_batch(
                    x_clamped, dt=dt_ns, diss_scale=diss_scale,
                    checkpoint_segments=checkpoint_segments)
                f_nom = self._process_fidelity(rho_final)
                lo, hi = robust_filter_band_mhz
                f_infid = self._filter_dephasing_infidelity(
                    x_clamped, dt_ns, robust_filter_sigma_mhz, robust_filter_alpha,
                    lo, hi, robust_filter_n_freq)
                fids = f_nom - f_infid
                loss = 1.0 - fids.mean()
            elif any_jitter:
                o_orig, g_orig, s_orig = self.OMEGA_MAX, self.G_MAX, self.STARK_MAX
                losses_per_perturb = []
                fids_for_best = None  # nominal-eval F drives best-tracking
                # (g_mult, t12_mult, amp_mult, det_rad); nominal first, then only
                # the enabled axes (no duplicate nominal evals).
                perturbations = [(1.0, 1.0, 1.0, 0.0)]
                if robust_g_jitter > 0.0:
                    perturbations += [(1.0 + robust_g_jitter, 1.0, 1.0, 0.0),
                                      (1.0 - robust_g_jitter, 1.0, 1.0, 0.0)]
                if robust_t12_jitter > 0.0:
                    perturbations += [(1.0, 1.0 - robust_t12_jitter, 1.0, 0.0),
                                      (1.0, 1.0 + robust_t12_jitter, 1.0, 0.0)]
                if robust_amp_jitter > 0.0:
                    perturbations += [(1.0, 1.0, 1.0 + robust_amp_jitter, 0.0),
                                      (1.0, 1.0, 1.0 - robust_amp_jitter, 0.0)]
                if robust_freq_jitter_mhz > 0.0:
                    dd = 2.0 * math.pi * (robust_freq_jitter_mhz / 1000.0)
                    perturbations += [(1.0, 1.0, 1.0, dd), (1.0, 1.0, 1.0, -dd)]
                try:
                    for g_mult, t12_mult, amp_mult, det_rad in perturbations:
                        # amp_mult scales all drives (AWG gain); T1/T2 jitter is exact
                        # via diss_scale=1/t12_mult (no operator rebuild).
                        self.OMEGA_MAX = o_orig * amp_mult
                        self.STARK_MAX = s_orig * amp_mult
                        self.G_MAX = g_orig * g_mult * amp_mult
                        if use_process_fidelity:
                            rho_p = self.simulate_choi_batch(
                                x_clamped, dt=dt_ns, diss_scale=diss_scale / t12_mult,
                                detuning_offset=det_rad,
                                checkpoint_segments=checkpoint_segments)
                            fids_p = self._process_fidelity(rho_p)
                        else:
                            rho_p = self.simulate_gradient_batch(
                                x_clamped, dt=dt_ns, diss_scale=diss_scale / t12_mult,
                                detuning_offset=det_rad,
                                checkpoint_segments=checkpoint_segments)
                            fids_p = self._avg_state_fidelity(rho_p)
                        losses_per_perturb.append(1.0 - fids_p.mean())
                        if fids_for_best is None:
                            fids_for_best = fids_p   # nominal eval (first one)
                finally:
                    self.OMEGA_MAX, self.G_MAX, self.STARK_MAX = o_orig, g_orig, s_orig
                loss = sum(losses_per_perturb) / len(losses_per_perturb)
                fids = fids_for_best
                # Use the LAST forward pass for leakage/bandwidth penalty
                # (these don't depend on physics params, just on the pulse).
                rho_final = rho_p
            else:
                if use_process_fidelity:
                    rho_final = self.simulate_choi_batch(
                        x_clamped, dt=dt_ns, diss_scale=diss_scale,
                        checkpoint_segments=checkpoint_segments)
                    fids = self._process_fidelity(rho_final)
                else:
                    rho_final = self.simulate_gradient_batch(
                        x_clamped, dt=dt_ns, diss_scale=diss_scale,
                        checkpoint_segments=checkpoint_segments)
                    fids = self._avg_state_fidelity(rho_final)
                loss = 1.0 - fids.mean()
            if leakage_penalty > 0.0:
                loss = loss + leakage_penalty * self._leakage(rho_final).mean()
            if bandwidth_penalty > 0.0:
                viol = self._bandwidth_violation(x_clamped, dt_ns, bandwidth_filter_mhz)
                loss = loss + bandwidth_penalty * viol.mean()

            # divergence guard: a non-finite loss/gradient would poison every seed
            # via the shared scalar loss, so roll back to the last finite state.
            if not torch.isfinite(loss):
                n_nonfinite += 1
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    x.copy_(last_good_x)
            else:
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_([x], max_norm=grad_clip)
                if torch.isfinite(gnorm):
                    opt.step()
                    last_grad_norm = float(gnorm)
                    with torch.no_grad():
                        last_good_x = x.detach().clone()
                else:
                    n_nonfinite += 1
                    opt.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        x.copy_(last_good_x)

            with torch.no_grad():
                better = fids > best_fid     # NaN fids compare False -> never win
                best_fid = torch.where(better, fids, best_fid)
                # Save the SMOOTHED pulse, not the raw control; the simulator
                # evaluated the smoothed version, so that's what hardware should see.
                if better.any():
                    smoothed = self.smoothed_waveform(x_clamped.detach(), dt=dt_ns)
                    for i in range(n_seeds):
                        if better[i]:
                            best_wf[i] = smoothed[i]
                            best_raw[i] = x_clamped[i].detach()
                history.append(float(best_fid.max().item()))

            if it % 25 == 0 or it == iterations - 1:
                cur_lr = opt.param_groups[0]["lr"]
                print(f"    Step {it:04d} | Max Fid: {fids.max().item():.7f}  "
                      f"| Mean: {fids.mean().item():.7f}  | lr: {cur_lr:.5f}")

        best_idx = int(best_fid.argmax().item())
        adam_best_fid = float(best_fid[best_idx].item())
        best_wf_final = best_wf[best_idx]
        best_raw_final = best_raw[best_idx]
        polished = False

        # ---- L-BFGS polish on the best Adam result ----
        if lbfgs_polish:
            print(f"\n [gradpulse] Adam plateau: {loss_label}={adam_best_fid:.7f}. "
                  f"Switching to L-BFGS polish ({lbfgs_iters} iters)...")
            try:
                # Seed L-BFGS from the RAW Adam parameter (not the smoothed
                # envelope) so it continues from exactly where Adam stopped in
                # the optimizer's own coordinate space.
                refined_x, polished_f = self._lbfgs_refine(
                    best_raw[best_idx], lbfgs_iters=lbfgs_iters, n_slices=n_slices,
                    dt_ns=dt_ns, use_process_fidelity=use_process_fidelity,
                    leakage_penalty=leakage_penalty,
                    bandwidth_penalty=bandwidth_penalty,
                    bandwidth_filter_mhz=bandwidth_filter_mhz,
                    checkpoint_segments=checkpoint_segments,
                    diss_scale=diss_scale,
                    robust_dephasing_sigma_mhz=robust_dephasing_sigma_mhz,
                    robust_dephasing_nodes=robust_dephasing_nodes,
                    robust_filter_sigma_mhz=robust_filter_sigma_mhz,
                    robust_filter_alpha=robust_filter_alpha,
                    robust_filter_band_mhz=robust_filter_band_mhz,
                    robust_filter_n_freq=robust_filter_n_freq,
                )
                if polished_f > adam_best_fid:
                    print(f"    L-BFGS:  {loss_label}={polished_f:.7f}  "
                          f"(+{polished_f - adam_best_fid:.5f})")
                    # refined_x is the raw param; smoothed_waveform applies the
                    # activation+smoother once, matching what L-BFGS evaluated.
                    best_wf_final = self.smoothed_waveform(
                        refined_x.unsqueeze(0), dt=dt_ns).squeeze(0)
                    best_raw_final = refined_x
                    adam_best_fid = polished_f
                    polished = True
                else:
                    print(f"    L-BFGS:  {loss_label}={polished_f:.7f}  (no improvement)")
            except Exception as e:
                print(f"    L-BFGS:  failed ({e}); using Adam result.")

        # ---- convergence diagnostics ----
        # "converged" = running-best fidelity plateaued (gain below tol) over the
        # last window; answers "did this converge, or need more iterations?".
        window = max(10, iterations // 5)
        if len(history) >= window:
            recent_gain = history[-1] - history[-window]
            converged = bool(recent_gain < 1e-5)
        else:
            recent_gain = float("nan")
            converged = False
        if n_nonfinite > 0:
            print(f" [gradpulse] divergence guard: rolled back {n_nonfinite} "
                  f"non-finite step(s); best result is finite and unaffected.")

        return {
            "best_fidelity":   adam_best_fid,
            "history":         history,
            "converged":       converged,
            "final_grad_norm": last_grad_norm,
            "recent_gain":     recent_gain,
            "n_nonfinite_steps": n_nonfinite,
            "best_waveform":   best_wf_final.cpu().numpy(),
            "best_raw_param":  best_raw_final.cpu().numpy(),  # raw param; see _lbfgs_refine
            "all_fidelities":  best_fid.cpu().numpy(),
            "best_seed_idx":   best_idx,
            "lbfgs_polished":  polished,
        }

    # ---- Band-limited (Fourier/CRAB) spectral optimization ---------------
    def _spectral_forward_host(self):
        """A forward-only twin of this optimizer with NO bandwidth smoother and a
        pass-through ('clamp') activation, so a band-limited envelope synthesized
        from the Fourier basis is fed to the master equation EXACTLY as given (the
        basis already enforces the band limit -- a second smoother would distort it,
        and a sigmoid would add out-of-band harmonics). Everything else (profile,
        channels, DRAG, step order, coupler mode, line response, precision) matches,
        so the physics is identical -- only the control parameterization differs."""
        return ParametricCZOptimizer(
            profile=self.profile, bandwidth_mhz=0.0, use_drag=self.use_drag,
            drag_order=self.drag_order, n_channels=self.n_channels,
            activation="clamp", step_order=self.step_order,
            coupler_phase_mode=self.coupler_phase_mode, delta_max_mhz=self.delta_max_mhz,
            coupler_g_linewidth_mhz=self.coupler_g_linewidth_mhz,
            line_response=self.line_response,
            target_gate=self.u_target_4x4.detach().cpu().numpy(),
            precision=self.precision)

    def out_of_band_fraction(self, envelope, dt_ns: float = 1.0,
                             f_max_mhz: float = None) -> float:
        """Fraction of the per-channel AC spectral energy that sits ABOVE f_max --
        the measured check that a pulse really is band-limited (0 = perfectly).

        Uses the rFFT of each channel's AC part (DC removed), averaged over channels.
        f_max defaults to self.bandwidth_mhz."""
        u = np.asarray(envelope, dtype=float)
        if u.ndim == 1:
            u = u[:, None]
        fmax = float(self.bandwidth_mhz if f_max_mhz is None else f_max_mhz)
        n = u.shape[0]
        freqs_mhz = np.fft.rfftfreq(n, d=dt_ns) * 1000.0     # MHz
        num = den = 0.0
        for c in range(u.shape[1]):
            ac = u[:, c] - u[:, c].mean()
            p = np.abs(np.fft.rfft(ac)) ** 2
            den += p.sum()
            num += p[freqs_mhz > fmax].sum()
        return float(num / den) if den > 0 else 0.0

    def optimize_spectral(self, n_harmonics: int = None, f_max_mhz: float = None,
                          n_slices: int = 150, dt_ns: float = 1.0, n_seeds: int = 4,
                          iterations: int = 300, lr: float = 0.05,
                          warm_start_mode: str = "parametric_cz",
                          use_process_fidelity: bool = True, lbfgs_polish: bool = True,
                          lbfgs_iters: int = 50, leakage_penalty: float = 0.0,
                          amp_penalty: float = 20.0,
                          coeff_jitter: float = 0.03, grad_clip: float = 1e3,
                          seed: int = 42, verbose: bool = True) -> dict:
        """GRAPE in a band-limited Fourier (CRAB-style) basis instead of per-slice.

        The control on each channel is a sum of sinusoids at harmonics of 1/T up to
        ``f_max_mhz`` (default = self.bandwidth_mhz), so it is band-limited BY
        CONSTRUCTION -- no post-hoc smoother, no anti-cheating FFT penalty, and far
        fewer parameters (~2*f_max*T per channel vs n_slices). The optimizer searches
        the Fourier coefficients; the synthesized [0,1] envelope is fed to a smoother-
        free forward twin (see _spectral_forward_host) so what is optimized is exactly
        what runs. The returned ``out_of_band_fraction`` MEASURES the residual energy
        above f_max (tiny -- only the [0,1] clamp can introduce any), so the
        band-limiting is verified rather than asserted.

        Returns a dict shaped like optimize_multi_seed -- best_fidelity, best_waveform
        ([n_slices, n_channels] in [0,1], the exact control), history, converged --
        plus 'best_coeffs', 'basis', 'n_params' (coeffs vs the piecewise count), and
        'out_of_band_fraction'. best_waveform cross-checks against QuTiP with NO
        smoother (gradpulse.validate.qutip_f_proc consumes the envelope directly)."""
        fmax = float(self.bandwidth_mhz if f_max_mhz is None else f_max_mhz)
        basis = FourierBasis(n_slices, dt_ns, f_max_mhz=fmax, n_harmonics=n_harmonics,
                             dtype=self.rdtype, device=DEVICE)
        host = self._spectral_forward_host()
        rng = torch.Generator(device=DEVICE).manual_seed(int(seed))

        # Seed each coefficient with the least-squares fit of the warm-start
        # envelope onto the basis (then perturb per seed).
        base_env = self._warm_start(n_slices, mode=warm_start_mode).to(self.rdtype)  # [n,C]
        Phi = basis.Phi.to(self.rdtype)
        coeff0 = torch.linalg.lstsq(Phi, base_env).solution                          # [B,C]
        coeffs = torch.zeros((n_seeds, basis.n_basis, self.n_channels),
                             dtype=self.rdtype, device=DEVICE)
        for s in range(n_seeds):
            jit = coeff_jitter * torch.randn((basis.n_basis, self.n_channels),
                                             generator=rng, device=DEVICE, dtype=self.rdtype)
            coeffs[s] = coeff0 + (0.0 if s == 0 else jit)
        coeffs.requires_grad_(True)

        n_params = int(basis.n_basis * self.n_channels)
        piecewise = int(n_slices * self.n_channels)
        if verbose:
            print(f" [gradpulse] Spectral GRAPE ({n_seeds} seeds, basis={basis.n_basis} "
                  f"coeffs/ch x {self.n_channels} ch = {n_params} params vs "
                  f"{piecewise} piecewise; f_max~{basis.f_max_mhz:.0f} MHz, "
                  f"{n_slices} slices @ {dt_ns} ns)...")

        opt = torch.optim.Adam([coeffs], lr=lr)
        best_fid = torch.full((n_seeds,), -1.0, device=DEVICE)
        best_coeffs = coeffs.detach().clone()
        last_good = coeffs.detach().clone()
        history, n_nonfinite = [], 0

        def _forward(cf):
            # The synthesized signal is band-limited; clamp only guards physicality.
            # amp_penalty keeps the optimizer off the clamp boundary -- saturating it
            # is the only way to reintroduce out-of-band energy (see out_of_band_fraction).
            synth = basis.synthesize(cf)                      # [S, n, C] band-limited
            env = synth.clamp(0.0, 1.0)
            if use_process_fidelity:
                rho = host.simulate_choi_batch(env, dt=dt_ns)
                fids = host._process_fidelity(rho)
            else:
                rho = host.simulate_gradient_batch(env, dt=dt_ns)
                fids = host._avg_state_fidelity(rho)
            over = (F.relu(synth - 1.0) ** 2 + F.relu(-synth) ** 2).mean()
            return fids, rho, over

        for it in range(iterations):
            opt.zero_grad()
            fids, rho, over = _forward(coeffs)
            loss = 1.0 - fids.mean()
            if leakage_penalty > 0.0:
                loss = loss + leakage_penalty * host._leakage(rho).mean()
            if amp_penalty > 0.0:
                loss = loss + amp_penalty * over
            if not torch.isfinite(loss):
                n_nonfinite += 1
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    coeffs.copy_(last_good)
            else:
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_([coeffs], max_norm=grad_clip)
                if torch.isfinite(gnorm):
                    opt.step()
                    with torch.no_grad():
                        last_good = coeffs.detach().clone()
                else:
                    n_nonfinite += 1
                    with torch.no_grad():
                        coeffs.copy_(last_good)
            with torch.no_grad():
                better = fids > best_fid
                best_fid = torch.where(better, fids, best_fid)
                if better.any():
                    for s in range(n_seeds):
                        if better[s]:
                            best_coeffs[s] = coeffs[s].detach()
                history.append(float(best_fid.max().item()))
            if verbose and (it % 50 == 0 or it == iterations - 1):
                print(f"    Step {it:04d} | Max Fid: {fids.max().item():.7f}")

        best_idx = int(best_fid.argmax().item())
        best_cf = best_coeffs[best_idx]
        adam_best = float(best_fid[best_idx].item())
        polished = False

        if lbfgs_polish:
            cf = best_cf.clone().detach().requires_grad_(True)
            lopt = torch.optim.LBFGS([cf], lr=0.3, max_iter=lbfgs_iters,
                                     history_size=20, line_search_fn="strong_wolfe")
            last = [adam_best]

            def closure():
                lopt.zero_grad()
                fids, rho, over = _forward(cf.unsqueeze(0))
                loss = 1.0 - fids.mean()
                if leakage_penalty > 0.0:
                    loss = loss + leakage_penalty * host._leakage(rho).mean()
                if amp_penalty > 0.0:
                    loss = loss + amp_penalty * over
                loss.backward()
                last[0] = float(fids.item())
                return loss
            try:
                lopt.step(closure)
                if last[0] > adam_best:
                    best_cf = cf.detach()
                    adam_best = last[0]
                    polished = True
            except Exception as e:  # pragma: no cover - polish is best-effort
                if verbose:
                    print(f"    L-BFGS:  failed ({e}); using Adam result.")

        with torch.no_grad():
            best_synth = basis.synthesize(best_cf)                 # pre-clamp control
            best_env = best_synth.clamp(0.0, 1.0)                  # [n, C], the control
        best_env_np = best_env.cpu().numpy()
        # How far the band-limited signal ran outside [0,1] (clamp activity); the amp
        # penalty keeps this ~0, so clamp is inactive and the control stays band-limited.
        synth_np = best_synth.cpu().numpy()
        max_overshoot = float(max(0.0, synth_np.max() - 1.0, -synth_np.min()))
        oob = self.out_of_band_fraction(best_env_np, dt_ns=dt_ns, f_max_mhz=fmax)
        window = max(10, iterations // 5)
        converged = (len(history) >= window and history[-1] - history[-window] < 1e-5)
        if verbose:
            print(f" [gradpulse] spectral best F={adam_best:.6f}  "
                  f"(out-of-band energy fraction {oob:.2e}, max overshoot {max_overshoot:.1e}; "
                  f"{n_params} params)")
        return {
            "best_fidelity": adam_best,
            "best_waveform": best_env_np,
            "best_coeffs": best_cf.cpu().numpy(),
            "basis": basis,
            "n_params": n_params,
            "n_params_piecewise": piecewise,
            "out_of_band_fraction": oob,
            "max_overshoot": max_overshoot,
            "f_max_mhz": fmax,
            "history": history,
            "converged": bool(converged),
            "n_nonfinite_steps": n_nonfinite,
            "all_fidelities": best_fid.cpu().numpy(),
            "lbfgs_polished": polished,
        }


# ---- Self-test --------------------------------------------------------------

def _selftest():
    """Verify the operator builder + simulator produce sensible results."""
    print("=" * 60)
    print("  GRADPULSE SELF-TEST")
    print("=" * 60)
    profile = ParametricCouplerProfile()
    print(f"  GPU: {DEVICE}")
    print(f"  Profile: anharm={profile.anharm_ghz_q1*1000:.0f} MHz, "
          f"f1={profile.freq_ghz_q1}, f2={profile.freq_ghz_q2}, "
          f"g_max={profile.g_max_mhz} MHz")

    opt = ParametricCZOptimizer(profile)

    # Test 1: zero-pulse should give identity (no drift in computational subspace)
    print("\n  TEST 1: zero pulse + identity-only drift")
    u_zero = torch.full((1, 10, 3), 0.5, device=DEVICE)  # 0.5 -> signed = 0
    rho = opt.simulate_gradient_batch(u_zero, dt=1.0)
    fid = opt._avg_state_fidelity(rho)
    print(f"    avg state fidelity (vs CZ target): {fid.item():.5f}")
    print(f"    NOTE: nonzero because q2 detuning drives free phase evolution.")

    # Test 2: short optimization, verify GRAPE converges
    print("\n  TEST 2: 50-iter GRAPE convergence on 100ns")
    result = opt.optimize_multi_seed(n_seeds=2, iterations=50, n_slices=100)
    print(f"    final best F: {result['best_fidelity']:.5f}")
    print(f"    waveform shape: {result['best_waveform'].shape}")
    print(f"    waveform range: [{result['best_waveform'].min():.3f}, "
          f"{result['best_waveform'].max():.3f}]")

    if result["best_fidelity"] > 0.5:
        print("\n  [PASS] GRAPE is making progress on the parametric-coupler Hamiltonian.")
    else:
        print("\n  [WARN] GRAPE not converging - check Hamiltonian construction.")


if __name__ == "__main__":
    _selftest()
