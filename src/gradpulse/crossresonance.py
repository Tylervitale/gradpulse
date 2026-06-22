"""gradpulse.crossresonance - cross-resonance (fixed-frequency) ZX gate optimizer.

A SECOND architecture, complementary to ``gradpulse.parametric``. Where the
parametric model has *tunable* transmons entangled by a flux-activated coupler,
this models two *fixed-frequency* transmons with an always-on exchange coupling
J, entangled by the **cross-resonance** (CR) effect: drive the control qubit at
the *target* qubit's frequency and the static coupling turns that drive into an
effective ZX interaction whose sign is conditioned on the control state. ZX(pi/2)
is locally equivalent to CNOT, so this is a native entangling gate for the
fixed-frequency architecture used by, e.g., IBM-style devices.

This is also the regime where DRAG *could* earn its keep. The CR tone is a strong,
off-resonant drive on the control transmon, so it readily excites the control's |1>-|2>
transition -- a leakage channel that the coupler-activated CZ, with its near-quiet
single-qubit drives, simply does not have. The optimizer therefore exposes the
derived-quadrature DRAG correction (Motzoi et al.) on the CR drive (``use_drag``);
how much it helps is regime-dependent and reported, not assumed (see
``examples/optimize_cross_resonance.py``) -- expected to matter most at strong drive
and small control |1>-|2> detuning, while deep in the dispersive regime the in-phase
pulse shaping already controls leakage (measured there: DRAG gives little, and can
slightly hurt, since the off-resonant drive is not the resonant case Motzoi DRAG
assumes).

CONVERGENCE / VALIDITY CHECKS (two knobs that turn the standing caveats into
measured numbers, both reachable from the optimizer):
  * Fock truncation -- ``profile.n_levels`` (default 4, CONVERGED). The strong CR
    drive makes |2>->|3> leakage real, so the 3-level (qutrit) truncation is NOT
    converged -- a 3-level-optimized pulse delivers only ~0.953 F_proc when honestly
    scored at 4 levels (~3.5% leakage it is blind to and cannot suppress). At 4
    levels the gate is converged: re-scoring a 4-level-optimal pulse at 5 moves F_proc
    by ~1e-4 (below the decoherence floor, comparable to the residual beyond-RWA term),
    and the achievable fidelity is flat across 4/5/6 -- hence the default is 4 (the QuTiP
    cross-check rebuilds with it). Drop to 3 only for quick relative studies.
  * Beyond-RWA error -- ``counter_rotating_fidelity()`` restores each drive's
    counter-rotating partner (the Bloch-Siegert-type term the rotating frame drops)
    and reports how far F_proc moves. CR's single, unambiguous drive frame makes
    that term exact, and CR is the strong-drive gate where it matters most.

==================================================================
HAMILTONIAN (n_levels per transmon, default 4 -> 16D; frame rotating at drive = f_target)
==================================================================

Working in the frame rotating at the drive frequency omega_d = omega_target makes
the whole drift static (the only time dependence is the control envelope):

  H(t) = H_drift
       + Omega_max * u_c(t) * X_c              <-- the CR drive (on control)
       + Omega_max * u_t(t) * X_t              <-- optional target cancellation
       + (DRAG quadratures on Y_c, Y_t)        <-- derived, when use_drag

  H_drift = Delta_c * n_c                       (control detuning; target is resonant)
          + (alpha_c/2) a_c+ a_c+ a_c a_c       (control anharmonicity)
          + (alpha_t/2) a_t+ a_t+ a_t a_t       (target anharmonicity)
          + J (a_c+ a_t + a_c a_t+)             (always-on exchange; static here)
          [+ chi_zz n_c n_t]                    (optional static ZZ)

where Delta_c = 2*pi*(f_control - f_target). The ZX interaction is NOT inserted
by hand -- it emerges from simulating this full model, so the optimizer discovers
the pulse that realises ZX(pi/2). The entangling content can only come from J
(a target-only tone is single-qubit), so an optional active-cancellation tone on
the target merely nulls the unwanted classical-crosstalk IX term; it cannot fake
the gate.

Single-qubit Z rotations are free on hardware (virtual-Z / frame changes), so the
target ZX(pi/2) is matched *up to* a single-qubit-Z frame: the two frame angles
(phi_control, phi_target) are optimised jointly with the pulse and reported, and
the QuTiP cross-check applies the same frame. This is exactly how CR-based CNOTs
are calibrated on real devices.

Units match ``gradpulse.parametric`` exactly (GHz/MHz -> rad/ns, time in ns) so
the two architectures share the same conventions and the same cross-check ethos.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .diagnostics import channel_unitarity
    from ._device import DEVICE
except ImportError:  # pragma: no cover - direct-script execution
    from diagnostics import channel_unitarity
    from _device import DEVICE

DTYPE = torch.complex64


# ---- Device profile --------------------------------------------------------

@dataclass
class CrossResonanceProfile:
    """Fixed-frequency transmon pair for a cross-resonance ZX gate.

    Defaults are representative published-typical fixed-frequency-transmon values
    (IBM-style: higher coherence, weak static coupling), not measurements of any
    specific device. Override via constructor kwargs with your calibrated values.
    The control qubit is driven at the target's frequency.
    """
    qubit_pair: tuple = (0, 1)   # (control, target)

    # Fock levels kept per transmon (Hilbert space = n_levels**2); default 4 is the
    # converged truncation for this strong-drive gate -- see the module docstring's
    # CONVERGENCE / VALIDITY CHECKS section. >= 3.
    n_levels: int = 4
    # Qubit frequencies in GHz. The control is the driven qubit; the drive sits
    # at the target frequency, so only their difference (the CR detuning) matters.
    freq_ghz_control: float = 5.00
    freq_ghz_target: float = 4.85
    # Anharmonicity in GHz (negative). Fixed-frequency transmons run near -0.33.
    anharm_ghz_control: float = -0.33
    anharm_ghz_target: float = -0.33
    # Always-on exchange coupling rate J in MHz (static, transverse). Dispersive
    # regime |Delta_c| >> J: the bare exchange is virtual and mediates the CR ZX.
    j_coupling_mhz: float = 3.0
    # Drive amplitude saturation Rabi rate in MHz (coefficient of X_c).
    omega_max_mhz: float = 60.0
    # Optional static parasitic ZZ in MHz (0 disables).
    chi_zz_mhz: float = 0.0

    # T1 (energy relaxation) and T2 (dephasing) in ns. Fixed-frequency transmons
    # typically have longer coherence than tunable ones.
    t1_ns_control: float = 150_000.0
    t1_ns_target: float = 150_000.0
    t2_ns_control: float = 120_000.0
    t2_ns_target: float = 120_000.0

    # Native CNOT reference (for comparison reporting only; not used in opt).
    native_cnot_fidelity: float = 0.990
    native_cnot_duration_ns: float = 300.0

    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Pure dephasing requires T2 <= 2*T1; warn rather than silently floor the
        # rate to ~0 (usually mixed T1/T2 units in a calibration file).
        import warnings
        for tag, t1, t2 in (("control", self.t1_ns_control, self.t2_ns_control),
                            ("target", self.t1_ns_target, self.t2_ns_target)):
            if t2 > 2.0 * t1:
                warnings.warn(
                    f"CrossResonanceProfile {tag}: T2={t2:g} ns > 2*T1={2.0 * t1:g} ns "
                    f"is unphysical for pure dephasing (1/T_phi < 0); the dephasing rate "
                    f"will be floored to ~0. Check your T1/T2 calibration units.",
                    stacklevel=2)


# ---- Operator builder ------------------------------------------------------

def _build_cr_ops(profile: CrossResonanceProfile, dtype: torch.dtype = DTYPE) -> dict:
    """Construct the (n_levels**2)-D Hamiltonian + Lindblad operators for the CR ZX gate.

    Hilbert space (n_levels per transmon; n_levels=3 shown for a compact layout),
    basis |control,target>:
      idx 0: |00>   idx 1: |01>   idx 2: |02>
      idx 3: |10>   idx 4: |11>   idx 5: |12>
      idx 6: |20>   idx 7: |21>   idx 8: |22>
    Computational subspace = indices [0, 1, n_levels, n_levels+1]
    (= [0, 1, 4, 5] at the default n_levels=4; [0, 1, 3, 4] at n_levels=3). n_levels
    is read from ``profile.n_levels`` so a truncation check rebuilds every operator.

    Everything is in the frame rotating at the drive frequency f_target, so the
    target detuning is zero and the drift is static.
    """
    n_levels = int(getattr(profile, "n_levels", 4))
    if n_levels < 3:
        raise ValueError(
            f"n_levels must be >= 3 (need |2> for leakage physics), got {n_levels}")
    # Single-transmon ladder a|k> = sqrt(k)|k-1>, truncated at n_levels.
    _sub = torch.tensor([math.sqrt(k) for k in range(1, n_levels)],
                        dtype=dtype, device=DEVICE)
    a3 = torch.diag(_sub, 1).contiguous()
    ad3 = a3.conj().t().contiguous()
    n3 = (ad3 @ a3).contiguous()
    i3 = torch.eye(n_levels, dtype=dtype, device=DEVICE)

    def kron2(A, B):
        return torch.kron(A.contiguous(), B.contiguous())

    # Energy scales (rad/ns)
    alpha_c = float(profile.anharm_ghz_control) * 2 * math.pi
    alpha_t = float(profile.anharm_ghz_target) * 2 * math.pi
    delta_c = (float(profile.freq_ghz_control)
               - float(profile.freq_ghz_target)) * 2 * math.pi   # control detuning
    j_rate = float(profile.j_coupling_mhz) * 2 * math.pi / 1000.0
    chi_zz = float(profile.chi_zz_mhz) * 2 * math.pi / 1000.0

    anh_c = 0.5 * alpha_c * (ad3 @ ad3 @ a3 @ a3)
    anh_t = 0.5 * alpha_t * (ad3 @ ad3 @ a3 @ a3)

    n_c = kron2(n3, i3)
    n_t = kron2(i3, n3)
    # Always-on exchange coupling, static in the common rotating frame.
    exchange = kron2(ad3, a3) + kron2(a3, ad3)

    h_drift = (
        delta_c * n_c
        + kron2(anh_c, i3)
        + kron2(i3, anh_t)
        + j_rate * exchange
        + chi_zz * (n_c @ n_t)
    )

    # Drive operators: X = a + a+ (in-phase), Y = i(a+ - a) (quadrature, DRAG)
    x_tr = a3 + ad3
    y_tr = 1j * (ad3 - a3)
    x_c = kron2(x_tr, i3)
    x_t = kron2(i3, x_tr)
    y_c = kron2(y_tr, i3)
    y_t = kron2(i3, y_tr)

    # Lindblad (jump) operators -- same Markovian T1/T2 model as the parametric
    # architecture (amplitude damping + pure dephasing on each transmon).
    a_c = kron2(a3, i3).contiguous()
    a_t = kron2(i3, a3).contiguous()

    def _t_phi(t1: float, t2: float) -> float:
        rate = 1.0 / max(t2, 1.0) - 1.0 / max(2.0 * t1, 1.0)
        return 1.0 / max(rate, 1e-9)

    t_phi_c = _t_phi(profile.t1_ns_control, profile.t2_ns_control)
    t_phi_t = _t_phi(profile.t1_ns_target, profile.t2_ns_target)

    L_t1_c = (math.sqrt(1.0 / profile.t1_ns_control) * a_c).contiguous()
    L_t1_t = (math.sqrt(1.0 / profile.t1_ns_target) * a_t).contiguous()
    L_phi_c = (math.sqrt(2.0 / t_phi_c) * n_c).contiguous()
    L_phi_t = (math.sqrt(2.0 / t_phi_t) * n_t).contiguous()
    _loss_terms = [L_t1_c, L_t1_t, L_phi_c, L_phi_t]
    L_loss_sum = (0.5 * sum(L.conj().t() @ L for L in _loss_terms)).contiguous()

    comp_indices = torch.tensor([0, 1, n_levels, n_levels + 1],
                                dtype=torch.long, device=DEVICE)

    # Ideal echo pi-pulse on the control's {|0>,|1>} subspace, X_pi = [[0,-i],[-i,0]]
    # (identity on |2>+ and target). Used by echo=True; see __init__'s echo docstring.
    xpi_1 = torch.eye(n_levels, dtype=dtype, device=DEVICE)
    xpi_1[0, 0] = 0.0; xpi_1[1, 1] = 0.0
    xpi_1[0, 1] = -1j; xpi_1[1, 0] = -1j
    xpi_c = kron2(xpi_1, i3).contiguous()

    return {
        "H_DRIFT":     h_drift.contiguous(),
        "XPI_C":       xpi_c,
        "X_C":         x_c.contiguous(),
        "X_T":         x_t.contiguous(),
        "Y_C":         y_c.contiguous(),
        "Y_T":         y_t.contiguous(),
        "N_C":         n_c.contiguous(),
        "N_T":         n_t.contiguous(),
        "A_C":         (kron2(a3, i3)).contiguous(),
        "A_T":         (kron2(i3, a3)).contiguous(),
        "L_T1_C":      L_t1_c,
        "L_T1_T":      L_t1_t,
        "L_PHI_C":     L_phi_c,
        "L_PHI_T":     L_phi_t,
        "L_LOSS_SUM":  L_loss_sum,
        "I9":          torch.eye(n_levels ** 2, dtype=dtype, device=DEVICE),
        "comp_indices": comp_indices,
        "n_levels":    n_levels,
        "dim":         n_levels ** 2,
        "alpha_c":     alpha_c,
        "alpha_t":     alpha_t,
        "delta_c":     delta_c,
    }


def zx90_target() -> np.ndarray:
    """ZX(pi/2) = exp(-i (pi/4) Z(x)X) as a 4x4 matrix, basis |00>,|01>,|10>,|11>.

    Locally equivalent to CNOT. Control |0> rotates the target by Rx(+pi/2),
    control |1> by Rx(-pi/2) -- the conditional rotation that makes it entangling.
    """
    inv = 1.0 / math.sqrt(2.0)
    return np.array([
        [inv,      -1j * inv, 0,        0],
        [-1j * inv, inv,      0,        0],
        [0,         0,        inv,      1j * inv],
        [0,         0,        1j * inv, inv],
    ], dtype=complex)


# ---- Optimizer --------------------------------------------------------------

class CrossResonanceZXOptimizer:
    """Autodiff GRAPE optimizer for a cross-resonance ZX(pi/2) gate.

    Mirrors ``ParametricCZOptimizer`` (open-system Lindblad simulation,
    bandwidth-limited controls, leakage-aware exact process fidelity), specialised
    to the fixed-frequency CR architecture. Controls (tanh-bounded to [-1, 1]):

      ch0: control in-phase CR drive (X_c)   -- the entangling drive
      ch1: target  in-phase tone   (X_t)     -- optional active IX cancellation

    With ``use_drag=True`` the derived-quadrature (Motzoi) DRAG correction is
    applied to each driven qubit (Y_c, Y_t), suppressing |1>-|2> leakage on the
    strongly-driven control. The target ZX(pi/2) is matched up to a single-qubit-Z
    frame (two angles optimised jointly and reported).
    """

    def __init__(self, profile: Optional[CrossResonanceProfile] = None,
                 bandwidth_mhz: float = 60.0,
                 use_drag: bool = True,
                 use_target_cancel: bool = True,
                 echo: bool = False,
                 precision: str = "single"):
        """echo: if True, run the *echoed* cross-resonance sequence -- two CR
        half-pulses with the control drive sign-flipped in the second half,
        separated by ideal pi-pulses on the control (at the midpoint and the end).
        The echo refocuses every term that anticommutes with X_c: the classical IX
        crosstalk, the control Stark shift ZI, and -- crucially -- the static ZZ,
        which a post-gate virtual-Z frame cannot remove (echo turns it into a
        removable single-qubit IZ). This is how CR CNOTs are calibrated on hardware;
        default False keeps the single-pulse (active-cancellation) gate. With echo on,
        ``use_target_cancel`` is optional since the echo already nulls IX."""
        self.profile = profile or CrossResonanceProfile()
        self.bandwidth_mhz = float(bandwidth_mhz)
        self.use_drag = bool(use_drag)
        self.use_target_cancel = bool(use_target_cancel)
        self.echo = bool(echo)
        self.n_channels = 2 if self.use_target_cancel else 1

        if precision not in ("single", "double"):
            raise ValueError("precision must be 'single' or 'double'")
        self.precision = precision
        self.cdtype = torch.complex128 if precision == "double" else torch.complex64
        self.rdtype = torch.float64 if precision == "double" else torch.float32

        ops = _build_cr_ops(self.profile, dtype=self.cdtype)
        self._H_DRIFT = ops["H_DRIFT"]
        self._XPI_C = ops["XPI_C"]                       # ideal echo pi-pulse (control)
        self._X_C, self._X_T = ops["X_C"], ops["X_T"]
        self._Y_C, self._Y_T = ops["Y_C"], ops["Y_T"]
        self._N_C, self._N_T = ops["N_C"], ops["N_T"]
        self._L_T1_C, self._L_T1_T = ops["L_T1_C"], ops["L_T1_T"]
        self._L_PHI_C, self._L_PHI_T = ops["L_PHI_C"], ops["L_PHI_T"]
        self._L_LOSS_SUM = ops["L_LOSS_SUM"]
        self._comp_idx = ops["comp_indices"]
        self._A_C, self._A_T = ops["A_C"], ops["A_T"]   # annihilation (counter-rotating)
        self._alpha_c = ops["alpha_c"]
        self._alpha_t = ops["alpha_t"]
        # Single-transmon truncation + full Hilbert-space dimension (n_levels**2);
        # every operator-stack allocation reads self._dim, so n_levels=4 just works.
        self.n_levels = int(ops["n_levels"])
        self._dim = int(ops["dim"])
        # Drive/frame angular frequency (rad/ns); counter-rotating terms oscillate
        # at 2*omega_d, needed only by counter_rotating_fidelity.
        self._omega_d = 2.0 * math.pi * float(self.profile.freq_ghz_target)

        self.OMEGA_MAX = float(self.profile.omega_max_mhz) * 2 * math.pi / 1000.0

        # Target ZX(pi/2) in the 4D computational subspace.
        self.u_target_4x4 = torch.tensor(zx90_target(), dtype=self.cdtype, device=DEVICE)
        # Control/target excitation bits of the computational basis (00,01,10,11),
        # used to build the virtual-Z frame.
        self._cbits = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=self.rdtype, device=DEVICE)
        self._tbits = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=self.rdtype, device=DEVICE)

        self._smoother_kernels = {}

    # ---- Bandwidth smoother (Gaussian, matches the parametric default) -------
    def _build_smoother_kernel(self, dt_ns: float = 1.0):
        if self.bandwidth_mhz <= 0:
            return None
        sigma_t = 1.0 / (2.0 * math.pi * (self.bandwidth_mhz / 1000.0))
        sigma = max(sigma_t / dt_ns, 0.5)
        half = int(math.ceil(4.0 * sigma))
        ts = torch.arange(-half, half + 1, dtype=self.rdtype, device=DEVICE)
        k = torch.exp(-0.5 * (ts / sigma) ** 2)
        k = k / k.sum()
        return k.view(1, 1, -1)

    def _smooth(self, u, kernel):
        if kernel is None:
            return u
        if kernel.dtype != u.dtype:
            kernel = kernel.to(u.dtype)
        B, N, C = u.shape
        x = u.permute(0, 2, 1).reshape(B * C, 1, N)
        K = kernel.shape[-1]
        pad = K // 2
        x = F.pad(x, (pad, pad), mode="replicate")
        x = F.conv1d(x, kernel)
        return x.reshape(B, C, N).permute(0, 2, 1)

    def smoothed_waveform(self, x_raw, dt: float = 1.0):
        """Bandwidth-smoothed bipolar envelope in [-1, 1], shape [n_slices, n_ch]."""
        if x_raw.dim() == 2:
            x_raw = x_raw.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        u_signed = torch.tanh(x_raw)
        key = round(float(dt), 6)
        if key not in self._smoother_kernels:
            self._smoother_kernels[key] = self._build_smoother_kernel(dt_ns=dt)
        u_smooth = self._smooth(u_signed, self._smoother_kernels[key])
        return u_smooth.squeeze(0) if squeeze else u_smooth

    def iq_waveform(self, x_raw, dt: float = 1.0) -> dict:
        """The COMPLETE complex CR drive the simulator applied, DRAG quadrature
        baked in -- the hardware-export-ready pulse.

        ``smoothed_waveform`` returns only the real in-phase envelope; with
        ``use_drag`` the simulator also drives the quadrature (Y) operator with the
        derived Motzoi tone ``v = -d/dt(Omega)/alpha``, which that envelope omits.
        Here each drive channel is returned as the full complex coefficient
        (in-phase*OMEGA_MAX + i*quadrature) in physical rad/ns:

          control tone : Omega_c(t) = uc*OMEGA_MAX + i*vc   (X_C + i Y_C)
          target tone  : Omega_t(t) = ut*OMEGA_MAX + i*vt   (1-channel: control only)

        With ``use_drag=False`` the imaginary parts are identically zero. The CR
        gate also carries a post-gate virtual-Z frame (``result['virtual_z']``)
        that this envelope does NOT include -- it is a frame change applied to the
        following single-qubit gates, not part of the played pulse, so a complete
        export is (this I/Q envelope, that virtual-Z) together. Returns a dict
        matching ``ParametricCZOptimizer.iq_waveform``.
        """
        x = torch.as_tensor(x_raw, device=DEVICE, dtype=self.rdtype)
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        with torch.no_grad():
            c = self._smoothed_controls(x, dt)
            chans, labels = [], []
            for ueff, vq, lab in ((c["uc"], c["vc"], "control_drive"),
                                  (c["ut"], c["vt"], "target_drive")):
                if ueff is None:
                    continue
                inphase = ueff * self.OMEGA_MAX
                quad = vq if vq is not None else torch.zeros_like(inphase)
                chans.append(inphase + 1j * quad)
                labels.append(lab)
            stack = torch.stack(chans, dim=-1).cpu().numpy()
        if squeeze:
            stack = stack[0]
        peak = np.max(np.abs(stack), axis=-2)
        return {"iq": stack, "labels": labels, "dt_ns": float(dt),
                "units": "rad/ns", "peak": peak, "n_channels": stack.shape[-1]}

    # ---- Control preprocessing (shared) --------------------------------------
    def _smoothed_controls(self, u_stack, dt: float = 1.0) -> dict:
        """Map raw [B, n_slices, n_ch] params to per-slice CR drive envelopes.

        Single source of truth between the raw optimizer parameter and the drive
        coefficients: tanh activation, the bandwidth smoother, and the derived-
        quadrature (Motzoi) DRAG tones. simulate_gradient_batch consumes this and
        assembles H(t) inline (its hot loop is unchanged); resonant_collision_fidelity
        reuses it so the collision diagnostic sees the EXACT same gate, with no risk
        of the smoothing/DRAG logic drifting between the two paths.

        Returns the control/target in-phase drives uc/ut (ut None for 1-channel) and
        the DRAG quadratures vc/vt (None unless use_drag).
        """
        n_ch = u_stack.shape[2]
        u_signed = torch.tanh(u_stack)
        key = round(float(dt), 6)
        if key not in self._smoother_kernels:
            self._smoother_kernels[key] = self._build_smoother_kernel(dt_ns=dt)
        u_smooth = self._smooth(u_signed, self._smoother_kernels[key])

        uc = u_smooth[:, :, 0]
        ut = u_smooth[:, :, 1] if n_ch >= 2 else None
        if self.use_drag:
            def _ddt(u):
                u_pad = F.pad(u.unsqueeze(1), (1, 1), mode="replicate").squeeze(1)
                return (u_pad[:, 2:] - u_pad[:, :-2]) / (2.0 * dt)
            vc = -_ddt(uc) * self.OMEGA_MAX / self._alpha_c
            vt = (-_ddt(ut) * self.OMEGA_MAX / self._alpha_t) if ut is not None else None
        else:
            vc = vt = None
        return {"uc": uc, "ut": ut, "vc": vc, "vt": vt}

    # ---- Core simulator ------------------------------------------------------
    def simulate_gradient_batch(self, u_stack, dt: float = 1.0,
                                diss_scale: float = 1.0, rho0=None,
                                detuning_offset=0.0):
        """Trotter-split (Lie-Trotter, 1st-order) open-system evolution.

        u_stack: [B, n_slices, n_channels] raw params (tanh-bounded internally).
            ch0 = control CR drive, ch1 = target cancellation tone (if enabled).
        diss_scale: multiplier on the dissipative increment (0 => closed system,
            for the error-budget ablation; 1 => the profile's T1/T2).
        rho0: optional initial operator stack [B, M, 9, 9]; None builds the 16
            Choi-basis operators (process tomography).
        detuning_offset: static qubit-frequency offset added to the drift for the
            whole gate (rad/ns). A scalar applies a common delta*(N_c+N_t); a
            (delta_c, delta_t) pair applies per-qubit delta_c*N_c + delta_t*N_t.
            The default 0.0 adds nothing and leaves the hot path unchanged.
            Mirrors ParametricCZOptimizer's primitive; used by the
            spectator-ZZ analysis (an idle neighbour shifts a gate qubit's
            frequency).
        Returns the evolved stack [B, M, 9, 9].
        """
        B, n_slices, n_ch = u_stack.shape
        if n_ch != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {n_ch}")

        # Extracted to _smoothed_controls so resonant_collision_fidelity sees the
        # EXACT same gate; the per-slice H assembly below is unchanged.
        _c = self._smoothed_controls(u_stack, dt)
        uc, ut, vc, vt = _c["uc"], _c["ut"], _c["vc"], _c["vt"]

        if rho0 is None:
            rho0 = self._choi_basis_rho0(B)
        rho = rho0

        L1, L1d = self._L_T1_C,  self._L_T1_C.conj().t().contiguous()
        L2, L2d = self._L_T1_T,  self._L_T1_T.conj().t().contiguous()
        L3, L3d = self._L_PHI_C, self._L_PHI_C.conj().t().contiguous()
        L4, L4d = self._L_PHI_T, self._L_PHI_T.conj().t().contiguous()

        # Static frequency-offset term (rad/ns); default 0 leaves the nominal path unchanged.
        H_det = None
        if detuning_offset is not None:
            if isinstance(detuning_offset, (tuple, list)):
                dc, dt_off = float(detuning_offset[0]), float(detuning_offset[1])
            else:
                dc = dt_off = float(detuning_offset)
            if dc != 0.0 or dt_off != 0.0:
                H_det = dc * self._N_C + dt_off * self._N_T

        # Echoed-CR sequence: the control CR drive sign-flips in the second half and
        # an ideal pi-pulse on the control is applied at the midpoint and the end.
        # echo=False leaves the single-pulse gate unchanged.
        echo = self.echo
        mid = n_slices // 2
        if echo:
            XPI = self._XPI_C
            XPId = XPI.conj().t().contiguous()

        # Vectorized H_all -> ONE batched matrix_exp instead of n_slices sequential
        # calls. The echo's 2nd-half sign-flip applies to the envelopes before
        # broadcasting; the ideal control pi-pulses stay in the loop below.
        uc_s, vc_s = uc, vc
        if echo:
            uc_s = uc.clone(); uc_s[:, mid:] = -uc_s[:, mid:]
            if vc is not None:
                vc_s = vc.clone(); vc_s[:, mid:] = -vc_s[:, mid:]

        def _bc(x):
            return x.view(B, n_slices, 1, 1)
        H_all = self._H_DRIFT + _bc(uc_s * self.OMEGA_MAX) * self._X_C
        if ut is not None:
            H_all = H_all + _bc(ut * self.OMEGA_MAX) * self._X_T
        if vc is not None:
            H_all = H_all + _bc(vc_s) * self._Y_C
            if vt is not None:
                H_all = H_all + _bc(vt) * self._Y_T
        if H_det is not None:
            H_all = H_all + H_det
        U_all = torch.linalg.matrix_exp(-1j * H_all * dt)      # [B, n_slices, 9, 9]
        Ud_all = U_all.conj().transpose(-2, -1)

        for i in range(n_slices):
            U = U_all[:, i].unsqueeze(1)                       # [B,1,9,9]
            Ud = Ud_all[:, i].unsqueeze(1)
            rho = U @ rho @ Ud
            jump = (L1 @ rho @ L1d) + (L2 @ rho @ L2d) + \
                   (L3 @ rho @ L3d) + (L4 @ rho @ L4d)
            anti = (self._L_LOSS_SUM @ rho) + (rho @ self._L_LOSS_SUM)
            rho = rho + dt * diss_scale * (jump - anti)

            if echo and (i == mid - 1 or i == n_slices - 1):   # echo pi-pulse on control
                rho = XPI @ rho @ XPId

        return rho

    def _choi_basis_rho0(self, B: int):
        """[B, 16, dim, dim] stack of |i><j| over the 4 computational levels
        (dim = n_levels**2; 16 for the default 4-level pair)."""
        ci = self._comp_idx
        rho0 = torch.zeros((B, 16, self._dim, self._dim),
                           device=DEVICE, dtype=self.cdtype)
        for i in range(4):
            for j in range(4):
                rho0[:, i * 4 + j, ci[i], ci[j]] = 1.0
        return rho0

    def simulate_choi_batch(self, u_stack, dt: float = 1.0, diss_scale: float = 1.0,
                            detuning_offset=0.0):
        B = u_stack.shape[0]
        return self.simulate_gradient_batch(
            u_stack, dt=dt, diss_scale=diss_scale, rho0=self._choi_basis_rho0(B),
            detuning_offset=detuning_offset)

    # ---- Fidelity ------------------------------------------------------------
    def _vz_target(self, vz):
        """Target ZX(pi/2) composed with a single-qubit-Z frame (post-gate).

        vz = (phi_control, phi_target); free virtual-Z is applied as a diagonal
        phase D = diag(exp(i*(phi_c*c + phi_t*t))) on the |c,t> computational
        basis, returning D @ ZX(pi/2). The optimiser tunes (phi_c, phi_t) so the
        reported fidelity is "up to virtual-Z", the physically free operation.

        Vectorised over the seed/batch dim: a single ``vz`` of shape [2] returns
        the [4, 4] framed target (the analysis/cross-check path), while a per-seed
        ``vz`` of shape [B, 2] returns a [B, 4, 4] stack of framed targets (the
        batched multi-seed optimiser, where every seed carries its own frame).
        """
        batched = vz.dim() == 2
        vz_b = vz if batched else vz.unsqueeze(0)             # [B, 2]
        theta = vz_b[:, 0:1] * self._cbits + vz_b[:, 1:2] * self._tbits   # [B, 4]
        D = torch.diag_embed(torch.exp(1j * theta.to(self.cdtype)))       # [B, 4, 4]
        U = D @ self.u_target_4x4                              # [B, 4, 4]
        return U if batched else U[0]

    def _process_fidelity(self, rho_choi, vz):
        """Exact leakage-aware entanglement (process) fidelity to ZX(pi/2).

        Same estimator as ParametricCZOptimizer._process_fidelity:
            F_proc = (1/d^2) sum_{i,j} <i| U^dag Phi(|i><j|) U |j>,  d = 4,
        with U the virtual-Z-framed target. F_avg = (d*F_proc + 1)/(d + 1).
        """
        ci = self._comp_idx
        B = rho_choi.shape[0]
        proj = rho_choi[:, :, ci, :][:, :, :, ci]        # [B,16,4,4]
        C = proj.reshape(B, 4, 4, 4, 4)                   # [B,i,j,a,c]
        U = self._vz_target(vz)
        if U.dim() == 3:
            # Per-seed virtual-Z frame: U is [B,4,4], contract each batch element
            # against its own frame (the batched multi-seed optimiser path).
            F_proc = torch.einsum('zai,zijac,zcj->z', U.conj(), C, U).real / 16.0
        else:
            # Single shared frame: U is [4,4] (analysis / cross-check path).
            F_proc = torch.einsum('ai,zijac,cj->z', U.conj(), C, U).real / 16.0
        return F_proc.clamp(0.0, 1.0)

    def _leakage(self, rho_choi):
        """Average population leaked out of the computational subspace, shape [B]."""
        rho_pop = rho_choi[:, [0, 5, 10, 15]]            # diagonal |i><i| inputs
        diag = rho_pop.diagonal(dim1=-2, dim2=-1).real   # [B,4,9]
        comp_pop = diag[..., self._comp_idx].sum(dim=-1)  # [B,4]
        return (1.0 - comp_pop).mean(dim=1).clamp(0.0, 1.0)

    # ---- Warm start + optimization ------------------------------------------
    def _warm_start(self, n_slices: int, generator=None):
        """Cosine flat-top on the control drive, near-zero elsewhere (+ noise)."""
        x = 0.02 * torch.randn((n_slices, self.n_channels), generator=generator,
                               device=DEVICE, dtype=self.rdtype)
        # Smooth flat-top envelope with cosine ramps over the first/last 20%.
        ramp = max(1, n_slices // 5)
        env = torch.ones(n_slices, device=DEVICE, dtype=self.rdtype)
        edge = 0.5 * (1.0 - torch.cos(
            math.pi * torch.arange(1, ramp + 1, device=DEVICE, dtype=self.rdtype) / ramp))
        env[:ramp] = edge
        env[-ramp:] = edge.flip(0)
        x[:, 0] = x[:, 0] + 0.7 * env       # control: moderate CR amplitude
        return x

    def optimize(self, n_slices: int = 300, dt_ns: float = 1.0,
                 iterations: int = 400, n_seeds: int = 3, lr: float = 0.05,
                 leak_weight: float = 2.0, seed0: int = 0, grad_clip: float = 1e3,
                 verbose: bool = False, diss_scale: float = 1.0):
        """Multi-seed Adam GRAPE toward ZX(pi/2). Returns a result dict.

        ``diss_scale`` (default 1.0 = full open-system, the true objective) scales the
        dissipator inside the optimized forward pass. Set 0.0 to optimize the *coherent*
        objective (decoherence off), e.g. for the decoherence-in-loop head-to-head:
        optimizing coherent-then-scoring-open vs optimizing in-loop. The default keeps
        every existing call byte-identical.

        All ``n_seeds`` random restarts optimise TOGETHER in one batched
        forward/backward (parameter tensor ``x`` of shape [n_seeds, n_slices,
        n_channels], frame ``vz`` of shape [n_seeds, 2]), mirroring
        ``ParametricCZOptimizer.optimize_multi_seed`` -- the seeds share one
        autograd graph and one Adam optimiser, so a run is ~n_seeds x faster than
        the old sequential per-seed loop, with results equivalent up to optimizer
        noise. Per-seed warm starts (seed ``seed0 + s``) are unchanged, so each
        seed starts from exactly the same pulse it did before.
        """
        # Per-seed warm starts stacked along the batch dim; identical seeding to
        # the old sequential path so each seed begins from the same pulse.
        x = torch.zeros((n_seeds, n_slices, self.n_channels),
                        device=DEVICE, dtype=self.rdtype)
        with torch.no_grad():
            for s in range(n_seeds):
                gen = torch.Generator(device=DEVICE).manual_seed(seed0 + s)
                x[s] = self._warm_start(n_slices, generator=gen)
        x.requires_grad_(True)
        vz = torch.zeros((n_seeds, 2), device=DEVICE, dtype=self.rdtype,
                         requires_grad=True)
        opt = torch.optim.Adam([x, vz], lr=lr)

        # Per-seed running-best trackers; NaN compares False so a blown-up seed
        # never wins. best_x/best_vz hold each seed's best iterate.
        best_fid = torch.zeros(n_seeds, device=DEVICE)
        best_leak_t = torch.ones(n_seeds, device=DEVICE)
        best_x = x.detach().clone()
        best_vz = vz.detach().clone()
        seed_histories = [[] for _ in range(n_seeds)]
        # Divergence guard: the shared scalar loss means one non-finite seed would
        # poison every seed's parameters, so roll the batch back to the last finite state.
        last_good_x = x.detach().clone()
        last_good_vz = vz.detach().clone()
        n_nonfinite = 0
        last_grad_norm = float("nan")

        for it in range(iterations):
            opt.zero_grad()
            rho = self.simulate_choi_batch(x, dt=dt_ns, diss_scale=diss_scale)  # [n_seeds,16,..]
            Fp = self._process_fidelity(rho, vz)                 # [n_seeds]
            leak = self._leakage(rho)                            # [n_seeds]
            loss = (1.0 - Fp).mean() + leak_weight * leak.mean()
            # ---- divergence guard: roll back non-finite loss/grad steps ----
            if not torch.isfinite(loss):
                n_nonfinite += 1
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    x.copy_(last_good_x); vz.copy_(last_good_vz)
            else:
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_([x, vz], max_norm=grad_clip)
                if torch.isfinite(gnorm):
                    opt.step()
                    last_grad_norm = float(gnorm)
                    with torch.no_grad():
                        last_good_x = x.detach().clone()
                        last_good_vz = vz.detach().clone()
                else:
                    n_nonfinite += 1
                    opt.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        x.copy_(last_good_x); vz.copy_(last_good_vz)
            with torch.no_grad():
                better = Fp > best_fid       # NaN fids compare False -> never win
                best_fid = torch.where(better, Fp, best_fid)
                best_leak_t = torch.where(better, leak, best_leak_t)
                if better.any():
                    idx = better.nonzero(as_tuple=False).flatten().tolist()
                    for s in idx:
                        best_x[s] = x[s].detach()
                        best_vz[s] = vz[s].detach()
                for s in range(n_seeds):
                    seed_histories[s].append(float(best_fid[s].item()))
            if verbose and (it % 50 == 0 or it == iterations - 1):
                print(f"  it {it:4d}  F_proc(max)={float(Fp.max().item()):.5f}  "
                      f"leak(mean)={float(leak.mean().item()):.2e}", flush=True)

        # ---- pick the best seed at the end ----
        best_s = int(best_fid.argmax().item())
        best_f = float(best_fid[best_s].item())
        best_leak = float(best_leak_t[best_s].item())
        best_x_s = best_x[best_s]
        best_vz_s = best_vz[best_s]
        all_f = [float(f) for f in best_fid.cpu().numpy()]
        waveform = self.smoothed_waveform(best_x_s, dt=dt_ns).detach().cpu().numpy()
        # ---- convergence diagnostics (on the winning seed's trajectory) ----
        history = seed_histories[best_s]
        window = max(10, iterations // 5)
        converged = bool(len(history) >= window
                         and history[-1] - history[-window] < 1e-5)
        if n_nonfinite > 0:
            print(f" [gradpulse] divergence guard: rolled back {n_nonfinite} "
                  f"non-finite step(s); best result is finite and unaffected.")
        return {
            "best_fidelity": best_f,
            "best_fidelity_avg": (4.0 * best_f + 1.0) / 5.0,
            "best_leakage": best_leak,
            "best_waveform": waveform,                          # [n_slices, n_ch] in [-1,1]
            "best_raw_param": best_x_s.cpu().numpy(),
            "virtual_z": [float(best_vz_s[0]), float(best_vz_s[1])],
            "all_fidelities": all_f,
            "history": history,
            "converged": converged,
            "final_grad_norm": last_grad_norm,
            "n_nonfinite_steps": n_nonfinite,
            "n_slices": n_slices,
            "dt_ns": dt_ns,
            "echo": self.echo,
        }

    # ---- Analysis ------------------------------------------------------------
    def error_budget(self, x_raw, dt: float = 1.0, vz=None) -> dict:
        """Split the ZX(pi/2) infidelity into coherent (control/leakage) vs the
        decoherence floor, plus the channel unitarity (coherent-vs-incoherent
        diagnostic). Mirrors ParametricCZOptimizer.error_budget.

        x_raw: the raw pulse parameter [n_slices, n_ch].
        vz: (phi_c, phi_t) virtual-Z frame (e.g. result['virtual_z']); 0 if None.
        """
        if not torch.is_tensor(x_raw):
            x_raw = torch.tensor(x_raw, device=DEVICE, dtype=self.rdtype)
        if vz is None:
            vz = torch.zeros(2, device=DEVICE, dtype=self.rdtype)
        elif not torch.is_tensor(vz):
            vz = torch.tensor(vz, device=DEVICE, dtype=self.rdtype)
        u = x_raw.unsqueeze(0)
        with torch.no_grad():
            rho_full = self.simulate_choi_batch(u, dt=dt, diss_scale=1.0)
            rho_closed = self.simulate_choi_batch(u, dt=dt, diss_scale=0.0)
            f_full = float(self._process_fidelity(rho_full, vz).item())
            f_closed = float(self._process_fidelity(rho_closed, vz).item())
            S = self._comp_superop_from_choi(rho_closed)
            u_unit = channel_unitarity(S)
        r_total = 1.0 - f_full
        r_coherent = 1.0 - f_closed           # error with decoherence turned off
        r_decoherence = max(0.0, r_total - r_coherent)
        # Coherent-vs-incoherent from unitarity: r >= (d-1)/d (1 - sqrt(u)).
        d = 4.0
        r_incoh_floor = (d - 1.0) / d * (1.0 - math.sqrt(max(0.0, min(1.0, u_unit))))
        return {
            "f_proc": f_full,
            "r_total": r_total,
            "r_coherent": r_coherent,
            "r_decoherence": r_decoherence,
            "unitarity": u_unit,
            "r_incoherent_floor_from_unitarity": r_incoh_floor,
            "r_coherent_from_unitarity": max(0.0, r_total - r_incoh_floor),
        }

    def _comp_superop_from_choi(self, rho_choi) -> np.ndarray:
        """[16,16] computational-subspace superoperator from a (closed-system)
        Choi stack, column m = vec(Phi(E_m)), E_m=|i><j|, m=i*4+j (row-major)."""
        ci = self._comp_idx
        proj = rho_choi[0][:, ci, :][:, :, ci]           # [16,4,4]
        return proj.reshape(16, 16).t().cpu().numpy()

    # ---- Counter-rotating (beyond-RWA) validity check ------------------------
    def counter_rotating_fidelity(self, x_raw, dt: float = 1.0, vz=None,
                                  substeps: int = 200,
                                  diss_scale: float = 0.0) -> dict:
        """Measure the rotating-wave-approximation (RWA) error of a CR pulse.

        The optimizer works in the frame rotating at the drive frequency
        omega_d = 2*pi*f_target, where the in-phase / quadrature drives are static
        (X = a + a^dag, Y = i(a^dag - a)) -- the standard RWA, which drops each
        drive's counter-rotating partner (oscillating at 2*omega_d). For the
        STRONG, off-resonant CR tone that partner is the leading neglected term (a
        Bloch-Siegert-type shift). This method re-simulates the SAME saved pulse
        with the counter-rotating terms restored and reports how far the process
        fidelity moves -- turning the RWA caveat into a measured number rather than
        an assumption. (CR uses a single, unambiguous drive frame, so the
        counter-rotating term is exact; the parametric-coupler CZ has a
        doubly-rotating frame where it is not cleanly defined, which is why this
        check lives on the CR architecture -- also the gate whose strong drive
        makes RWA error matter most.)

        For a drive with in-phase amplitude Omega*u(t) and quadrature v(t) on
        transmon ch, the restored term is exactly
            H_cr(t) = (Omega*u + i v) a_ch e^{-2 i omega_d t}
                    + (Omega*u - i v) a_ch^dag e^{+2 i omega_d t}.
        It oscillates at ~2*omega_d (~10 GHz, period ~0.1 ns), so each control
        slice is integrated with ``substeps`` fine midpoint sub-steps (the control
        is held constant across the slice; only the fast phase advances). The RWA
        reference uses the SAME sub-stepped integrator WITHOUT the counter-rotating
        term, so the reported delta is purely the counter-rotating physics, not an
        integrator-scheme artifact.

        x_raw: raw pulse parameter [n_slices, n_ch] (e.g. result['best_raw_param']).
        vz: (phi_c, phi_t) virtual-Z frame (e.g. result['virtual_z']); 0 if None.
        substeps: fine sub-steps per slice (resolves the 2*omega_d integral);
            ~150-300 is ample at 1 ns slices. Cost is one extra forward pass.
        diss_scale: dissipator multiplier; default 0.0 isolates the *coherent* RWA
            error (decoherence is identical in both runs). Set 1.0 to include T1/T2.

        Returns a dict:
          * f_proc_rwa           RWA reference (sub-stepped, no counter-rotating).
          * f_proc_counter_rot   same pulse WITH counter-rotating terms restored.
          * delta_r_counter_rot  f_proc_rwa - f_proc_counter_rot: the process
                                 infidelity the RWA omits for THIS pulse (the
                                 measured size of the caveat; can be either sign).
          * f_avg_rwa / f_avg_counter_rot   average-gate-fidelity versions.
          * omega_d_ghz, substeps, dt_fine_ns   provenance echoes.
        """
        if not torch.is_tensor(x_raw):
            x_raw = torch.tensor(x_raw, device=DEVICE, dtype=self.rdtype)
        if vz is None:
            vz = torch.zeros(2, device=DEVICE, dtype=self.rdtype)
        elif not torch.is_tensor(vz):
            vz = torch.tensor(vz, device=DEVICE, dtype=self.rdtype)
        x = x_raw.unsqueeze(0) if x_raw.dim() == 2 else x_raw   # [1, n_slices, n_ch]
        n_sub = max(1, int(substeps))
        dt_fine = float(dt) / n_sub
        wd = float(self._omega_d)

        with torch.no_grad():
            # Control prep identical to simulate_gradient_batch (tanh -> smoother
            # -> Motzoi DRAG quadrature), so the RWA reference reproduces the
            # nominal pulse the optimizer evolved.
            u_signed = torch.tanh(x)
            key = round(float(dt), 6)
            if key not in self._smoother_kernels:
                self._smoother_kernels[key] = self._build_smoother_kernel(dt_ns=dt)
            u_smooth = self._smooth(u_signed, self._smoother_kernels[key])
            n_slices, n_ch = u_smooth.shape[1], u_smooth.shape[2]
            uc = u_smooth[:, :, 0]
            ut = u_smooth[:, :, 1] if n_ch >= 2 else None
            if self.use_drag:
                def _ddt(u):
                    u_pad = F.pad(u.unsqueeze(1), (1, 1), mode="replicate").squeeze(1)
                    return (u_pad[:, 2:] - u_pad[:, :-2]) / (2.0 * dt)
                vc = -_ddt(uc) * self.OMEGA_MAX / self._alpha_c
                vt = (-_ddt(ut) * self.OMEGA_MAX / self._alpha_t) if ut is not None else None
            else:
                vc = vt = None

            # This measures a ~1e-5-scale effect over ~10^5 fine-substep propagators;
            # single-precision round-off (~1e-3) would swamp it, so this integration
            # runs in double regardless of the optimizer's working precision.
            cdt = torch.complex128
            H_DRIFT = self._H_DRIFT.to(cdt)
            X_C = self._X_C.to(cdt)
            X_T = self._X_T.to(cdt) if ut is not None else None
            Y_C, Y_T = self._Y_C.to(cdt), self._Y_T.to(cdt)
            Ac, Act = self._A_C.to(cdt), self._A_C.to(cdt).conj().t().contiguous()
            At, Att = self._A_T.to(cdt), self._A_T.to(cdt).conj().t().contiguous()
            L1, L1d = self._L_T1_C.to(cdt),  self._L_T1_C.to(cdt).conj().t().contiguous()
            L2, L2d = self._L_T1_T.to(cdt),  self._L_T1_T.to(cdt).conj().t().contiguous()
            L3, L3d = self._L_PHI_C.to(cdt), self._L_PHI_C.to(cdt).conj().t().contiguous()
            L4, L4d = self._L_PHI_T.to(cdt), self._L_PHI_T.to(cdt).conj().t().contiguous()
            L_LOSS = self._L_LOSS_SUM.to(cdt)
            ucd = uc.to(torch.float64)
            utd = ut.to(torch.float64) if ut is not None else None
            vcd = vc.to(torch.float64) if vc is not None else None
            vtd = vt.to(torch.float64) if vt is not None else None
            OMEGA = float(self.OMEGA_MAX)
            cz = torch.zeros((), dtype=cdt, device=DEVICE)

            def _evolve(include_cr: bool):
                rho = self._choi_basis_rho0(1)[0].to(cdt)      # [16, dim, dim]
                for i in range(n_slices):
                    H0 = H_DRIFT + (ucd[0, i] * OMEGA) * X_C
                    if utd is not None:
                        H0 = H0 + (utd[0, i] * OMEGA) * X_T
                    if vcd is not None:
                        H0 = H0 + vcd[0, i] * Y_C
                        if vtd is not None:
                            H0 = H0 + vtd[0, i] * Y_T
                    if include_cr:
                        # complex drive amplitudes Omega*u + i v on each transmon
                        ec = (ucd[0, i] * OMEGA).to(cdt) + \
                             (1j * vcd[0, i].to(cdt) if vcd is not None else cz)
                        et = None
                        if utd is not None:
                            et = (utd[0, i] * OMEGA).to(cdt) + \
                                 (1j * vtd[0, i].to(cdt) if vtd is not None else cz)
                    for s in range(n_sub):
                        H = H0
                        if include_cr:
                            t_mid = (i + (s + 0.5) / n_sub) * float(dt)
                            ph = torch.exp(torch.tensor(-1j * 2.0 * wd * t_mid,
                                                        dtype=cdt, device=DEVICE))
                            Hcr = ec * ph * Ac + (ec.conj() * ph.conj()) * Act
                            if et is not None:
                                Hcr = Hcr + et * ph * At + (et.conj() * ph.conj()) * Att
                            H = H0 + Hcr
                        U = torch.linalg.matrix_exp(-1j * H * dt_fine)
                        Ud = U.conj().t()
                        rho = U @ rho @ Ud
                        if diss_scale != 0.0:
                            jump = (L1 @ rho @ L1d) + (L2 @ rho @ L2d) + \
                                   (L3 @ rho @ L3d) + (L4 @ rho @ L4d)
                            anti = (L_LOSS @ rho) + (rho @ L_LOSS)
                            rho = rho + dt_fine * diss_scale * (jump - anti)
                return rho.unsqueeze(0)                         # [1, 16, dim, dim]

            # vz-framed target rebuilt exactly in double (not up-cast) so the small-effect
            # contraction stays clean; same estimator as _process_fidelity.
            ci = self._comp_idx
            cb, tb = self._cbits.to(torch.float64), self._tbits.to(torch.float64)
            theta = vz[0].to(torch.float64) * cb + vz[1].to(torch.float64) * tb
            Utgt = torch.diag(torch.exp(1j * theta.to(cdt))) @ \
                torch.tensor(zx90_target(), dtype=cdt, device=DEVICE)

            def _fproc(rho_choi):
                proj = rho_choi[:, :, ci, :][:, :, :, ci]      # [B, 16, 4, 4]
                C = proj.reshape(rho_choi.shape[0], 4, 4, 4, 4)
                return torch.einsum('ai,zijac,cj->z', Utgt.conj(), C, Utgt).real / 16.0

            f_rwa = float(_fproc(_evolve(False)).item())
            f_cr = float(_fproc(_evolve(True)).item())

        d = 4.0
        return {
            "f_proc_rwa": f_rwa,
            "f_proc_counter_rot": f_cr,
            "delta_r_counter_rot": f_rwa - f_cr,
            "f_avg_rwa": (d * f_rwa + 1.0) / (d + 1.0),
            "f_avg_counter_rot": (d * f_cr + 1.0) / (d + 1.0),
            "omega_d_ghz": wd / (2.0 * math.pi),
            "substeps": n_sub,
            "dt_fine_ns": dt_fine,
        }

    def refine_beyond_rwa(self, x_raw, vz=None, dt_ns: float = 1.0,
                          iterations: int = 40, substeps: int = 40,
                          lr: float = 0.01, diss_scale: float = 0.0,
                          verbose: bool = False) -> dict:
        """Polish an RWA-optimized CR pulse against the FULL time-dependent
        Hamiltonian -- counter-rotating terms included *inside* the gradient loop --
        so the beyond-RWA residual is **removed from the pulse** rather than only
        measured. This is the optimization counterpart of `counter_rotating_fidelity`
        (which diagnoses the residual): start from a pulse the RWA optimizer found,
        then descend on the beyond-RWA process infidelity, co-optimizing the
        in-phase/quadrature controls and the virtual-Z frame.

        Because the counter-rotating term oscillates at 2*omega_d (period ~0.1 ns),
        each 1 ns control slice is integrated with ``substeps`` fine midpoint
        sub-steps; ``substeps>=40`` resolves the integral. The RWA pulse is already
        near-optimal, so few iterations suffice (the term is a small perturbation).

        x_raw : starting raw parameter (e.g. result['best_raw_param']).
        vz    : starting virtual-Z (phi_c, phi_t) (e.g. result['virtual_z']).
        diss_scale : dissipator multiplier; default 0.0 isolates the coherent RWA
            error (the term decoherence cannot mask).

        Returns {best_raw_param, virtual_z, f_proc_before, f_proc_after,
        delta_removed, substeps, iterations} where the fidelities are the
        beyond-RWA process fidelity of the input vs the refined pulse.
        """
        if not torch.is_tensor(x_raw):
            x_raw = torch.tensor(x_raw, device=DEVICE, dtype=self.rdtype)
        x = (x_raw.detach().clone().unsqueeze(0) if x_raw.dim() == 2
             else x_raw.detach().clone())
        x = x.to(DEVICE).to(self.rdtype).requires_grad_(True)
        if vz is None:
            vz = torch.zeros(2, device=DEVICE, dtype=self.rdtype)
        elif not torch.is_tensor(vz):
            vz = torch.tensor(vz, device=DEVICE, dtype=self.rdtype)
        vz = vz.detach().clone().to(DEVICE).to(self.rdtype).requires_grad_(True)

        n_sub = max(1, int(substeps))
        dt_fine = float(dt_ns) / n_sub
        wd = float(self._omega_d)
        key = round(float(dt_ns), 6)
        if key not in self._smoother_kernels:
            self._smoother_kernels[key] = self._build_smoother_kernel(dt_ns=dt_ns)
        kernel = self._smoother_kernels[key]
        cz = torch.zeros((), dtype=self.cdtype, device=DEVICE)
        L1, L1d = self._L_T1_C,  self._L_T1_C.conj().t().contiguous()
        L2, L2d = self._L_T1_T,  self._L_T1_T.conj().t().contiguous()
        L3, L3d = self._L_PHI_C, self._L_PHI_C.conj().t().contiguous()
        L4, L4d = self._L_PHI_T, self._L_PHI_T.conj().t().contiguous()
        Ac, Act = self._A_C, self._A_C.conj().t().contiguous()
        At, Att = self._A_T, self._A_T.conj().t().contiguous()

        def _controls():
            u_signed = torch.tanh(x)
            u_smooth = self._smooth(u_signed, kernel)
            n_slices, n_ch = u_smooth.shape[1], u_smooth.shape[2]
            uc = u_smooth[:, :, 0]
            ut = u_smooth[:, :, 1] if n_ch >= 2 else None
            if self.use_drag:
                def _ddt(u):
                    u_pad = F.pad(u.unsqueeze(1), (1, 1), mode="replicate").squeeze(1)
                    return (u_pad[:, 2:] - u_pad[:, :-2]) / (2.0 * dt_ns)
                vc = -_ddt(uc) * self.OMEGA_MAX / self._alpha_c
                vt = (-_ddt(ut) * self.OMEGA_MAX / self._alpha_t) if ut is not None else None
            else:
                vc = vt = None
            return uc, ut, vc, vt, n_slices

        def _phase(i, s, n_slices):
            t_mid = (i + (s + 0.5) / n_sub) * float(dt_ns)
            return torch.exp(torch.tensor(-1j * 2.0 * wd * t_mid,
                                          dtype=self.cdtype, device=DEVICE))

        def _evolve_cr():
            uc, ut, vc, vt, n_slices = _controls()
            rho = self._choi_basis_rho0(1)[0]                  # [16, dim, dim]
            for i in range(n_slices):
                H0 = self._H_DRIFT + (uc[0, i] * self.OMEGA_MAX) * self._X_C
                if ut is not None:
                    H0 = H0 + (ut[0, i] * self.OMEGA_MAX) * self._X_T
                if vc is not None:
                    H0 = H0 + vc[0, i] * self._Y_C
                    if vt is not None:
                        H0 = H0 + vt[0, i] * self._Y_T
                ec = (uc[0, i] * self.OMEGA_MAX).to(self.cdtype) + \
                     (1j * vc[0, i].to(self.cdtype) if vc is not None else cz)
                et = None
                if ut is not None:
                    et = (ut[0, i] * self.OMEGA_MAX).to(self.cdtype) + \
                         (1j * vt[0, i].to(self.cdtype) if vt is not None else cz)
                for s in range(n_sub):
                    p = _phase(i, s, n_slices)
                    Hcr = ec * p * Ac + (ec.conj() * p.conj()) * Act
                    if et is not None:
                        Hcr = Hcr + et * p * At + (et.conj() * p.conj()) * Att
                    U = torch.linalg.matrix_exp(-1j * (H0 + Hcr) * dt_fine)
                    rho = U @ rho @ U.conj().transpose(-2, -1)
                    if diss_scale != 0.0:
                        jump = (L1 @ rho @ L1d) + (L2 @ rho @ L2d) + \
                               (L3 @ rho @ L3d) + (L4 @ rho @ L4d)
                        anti = (self._L_LOSS_SUM @ rho) + (rho @ self._L_LOSS_SUM)
                        rho = rho + dt_fine * diss_scale * (jump - anti)
            return rho.unsqueeze(0)

        # _evolve_cr (single precision) is adequate for descending, but at the ~1e-5
        # floor it's round-off-noisy, so the *reported* numbers use the double-
        # precision counter_rotating_fidelity instead of a single-precision re-score.
        with torch.no_grad():
            f_before = self.counter_rotating_fidelity(
                x.detach(), dt=dt_ns, vz=vz.detach(), substeps=n_sub,
                diss_scale=diss_scale)["f_proc_counter_rot"]

        opt = torch.optim.Adam([x, vz], lr=lr)
        for it in range(int(iterations)):
            opt.zero_grad()
            f = self._process_fidelity(_evolve_cr(), vz)
            f = f if f.dim() == 0 else f[0]
            (1.0 - f).backward()
            opt.step()
            if verbose and it % 10 == 0:
                print(f"  [beyond-RWA] it {it:3d}  F_cr={float(f.detach()):.6f}")

        with torch.no_grad():
            f_after = self.counter_rotating_fidelity(
                x.detach(), dt=dt_ns, vz=vz.detach(), substeps=n_sub,
                diss_scale=diss_scale)["f_proc_counter_rot"]

        return {
            "best_raw_param": x[0].detach().cpu().numpy(),
            "virtual_z": vz.detach().cpu().numpy(),
            "f_proc_before": f_before,
            "f_proc_after": f_after,
            "delta_removed": f_after - f_before,
            "substeps": n_sub,
            "iterations": int(iterations),
        }

    # ---- Spectator (always-on ZZ) crosstalk ----------------------------------
    def spectator_fidelity(self, x_raw, dt: float = 1.0, vz=None, zeta_mhz=0.1,
                           spectator_pop: float = 0.5) -> dict:
        """ZX(pi/2) fidelity penalty from an always-on ZZ to an idle neighbour.

        Cross-resonance counterpart of ParametricCZOptimizer.spectator_fidelity
        (see it for the physics and the returned keys): an off-resonant neighbour
        frozen in state s shifts a gate qubit's frequency by zeta*s, applied via the
        detuning_offset primitive; averaging over an *unmeasured* neighbour's state
        gives a dephasing channel whose conservative nominal-frame cost is
        delta_r_spectator (virtual-Z re-tuning for the mean shift removes part of it).
        Resonant exchange / frequency collisions are the complementary regime --
        see ``resonant_collision_fidelity`` (the one that matters most here, since
        fixed-frequency lattices cannot tune away from a collision).

        x_raw: raw pulse parameter [n_slices, n_ch] (e.g. result['best_raw_param']).
        vz: (phi_c, phi_t) virtual-Z frame (e.g. result['virtual_z']); 0 if None.
        zeta_mhz, spectator_pop: as in the parametric method.
        """
        if not torch.is_tensor(x_raw):
            x_raw = torch.tensor(x_raw, device=DEVICE, dtype=self.rdtype)
        if vz is None:
            vz = torch.zeros(2, device=DEVICE, dtype=self.rdtype)
        elif not torch.is_tensor(vz):
            vz = torch.tensor(vz, device=DEVICE, dtype=self.rdtype)
        u = x_raw.unsqueeze(0) if x_raw.dim() == 2 else x_raw
        if isinstance(zeta_mhz, (tuple, list)):
            zc, zt = float(zeta_mhz[0]), float(zeta_mhz[1])
        else:
            zc = zt = float(zeta_mhz)
        d = 4.0
        p = float(spectator_pop)

        def _rad(mhz):
            return 2.0 * math.pi * (mhz / 1000.0)               # MHz -> rad/ns

        ax_c = [(0, 1.0 - p), (1, p)] if zc != 0.0 else [(0, 1.0)]
        ax_t = [(0, 1.0 - p), (1, p)] if zt != 0.0 else [(0, 1.0)]

        with torch.no_grad():
            configs = {}
            for sc, _ in ax_c:
                for st, _ in ax_t:
                    if (sc, st) not in configs:
                        configs[(sc, st)] = self.simulate_choi_batch(
                            u, dt=dt, detuning_offset=(_rad(zc) * sc, _rad(zt) * st))
            avg_choi = None
            for sc, wc in ax_c:
                for st, wt in ax_t:
                    term = (wc * wt) * configs[(sc, st)]
                    avg_choi = term if avg_choi is None else avg_choi + term
            exc_key = (1 if zc != 0.0 else 0, 1 if zt != 0.0 else 0)
            f_idle = float(self._process_fidelity(configs[(0, 0)], vz).item())
            f_exc = float(self._process_fidelity(configs[exc_key], vz).item())
            f_avg_choi = float(self._process_fidelity(avg_choi, vz).item())

        f_avg_gate = (d * f_avg_choi + 1.0) / (d + 1.0)
        f_avg_idle = (d * f_idle + 1.0) / (d + 1.0)
        t_gate_ns = u.shape[1] * float(dt)
        zz_phase = max(abs(_rad(zc)), abs(_rad(zt))) * t_gate_ns
        return {
            "f_proc_idle": f_idle,
            "f_proc_excited": f_exc,
            "f_proc_spectator_avg": f_avg_choi,
            "f_avg_idle": f_avg_idle,
            "f_avg_spectator_avg": f_avg_gate,
            "delta_r_spectator": f_avg_idle - f_avg_gate,
            "zz_phase_rad": zz_phase,
            "zeta_mhz": (zc, zt),
            "spectator_pop": p,
            "n_evals": len(configs),
        }

    def multi_spectator_fidelity(self, x_raw, neighbours, dt: float = 1.0,
                                 vz=None) -> dict:
        """Always-on-ZZ penalty from an ARBITRARY set of idle neighbours (CR).

        Cross-resonance counterpart of ParametricCZOptimizer.multi_spectator_fidelity:
        each frozen off-resonant neighbour adds a static detuning on the gate qubit it
        couples to (control=0, target=1); detunings on the same qubit SUM, and an
        unmeasured ensemble averages the channel over every neighbour-state
        combination. Same exact additive reduction as spectator_fidelity, applied to N
        neighbours; cross-checked against an explicit multi-transmon QuTiP sim in
        tests/test_spectators.py. Resonant/collision regime: resonant_collision_fidelity.

        neighbours: list of ``(gate_qubit, zeta_mhz, pop)`` with gate_qubit in
            {0 (control), 1 (target)}. vz: (phi_c, phi_t) virtual-Z frame; 0 if None.
        """
        from itertools import product
        if not torch.is_tensor(x_raw):
            x_raw = torch.tensor(x_raw, device=DEVICE, dtype=self.rdtype)
        if vz is None:
            vz = torch.zeros(2, device=DEVICE, dtype=self.rdtype)
        elif not torch.is_tensor(vz):
            vz = torch.tensor(vz, device=DEVICE, dtype=self.rdtype)
        u = x_raw.unsqueeze(0) if x_raw.dim() == 2 else x_raw
        d = 4.0

        def _rad(mhz):
            return 2.0 * math.pi * (mhz / 1000.0)

        nb = [(int(q), float(z), float(p)) for (q, z, p) in neighbours if float(z) != 0.0]
        N = len(nb)
        configs = {}

        def _choi(det):
            key = (round(det[0], 9), round(det[1], 9))   # round only for cache dedup
            if key not in configs:
                with torch.no_grad():
                    configs[key] = self.simulate_choi_batch(u, dt=dt, detuning_offset=det)
            return configs[key]

        idle = _choi((0.0, 0.0))
        avg_choi = None
        for states in product((0, 1), repeat=N):
            w, d0, d1 = 1.0, 0.0, 0.0
            for (q, z, p), s in zip(nb, states):
                w *= (p if s == 1 else 1.0 - p)
                if s == 1:
                    if q == 0:
                        d0 += _rad(z)
                    else:
                        d1 += _rad(z)
            term = w * _choi((d0, d1))
            avg_choi = term if avg_choi is None else avg_choi + term
        if avg_choi is None:
            avg_choi = idle

        f_idle = float(self._process_fidelity(idle, vz).item())
        f_avg_choi = float(self._process_fidelity(avg_choi, vz).item())
        f_avg_gate = (d * f_avg_choi + 1.0) / (d + 1.0)
        f_avg_idle = (d * f_idle + 1.0) / (d + 1.0)
        return {
            "f_proc_idle": f_idle,
            "f_proc_spectator_avg": f_avg_choi,
            "f_avg_idle": f_avg_idle,
            "f_avg_spectator_avg": f_avg_gate,
            "delta_r_spectator": f_avg_idle - f_avg_gate,
            "n_neighbours": N,
            "n_configs": len(configs),
        }

    # ---- Resonant / frequency-collision crosstalk ----------------------------
    def resonant_collision_fidelity(self, x_raw, dt: float = 1.0, vz=None,
                                    detuning_mhz=0.0, j_mhz: float = 8.0,
                                    couples_to: str = "control",
                                    diss_scale: float = 1.0):
        """ZX(pi/2) fidelity vs. a NEAR-RESONANT spectator that exchanges population.

        Cross-resonance counterpart of ParametricCZOptimizer.resonant_collision_fidelity
        -- and the regime that matters MOST here, since fixed-frequency lattices cannot
        tune away from a frequency collision. The complement of spectator_fidelity:
        that freezes an off-resonant neighbour into a static ZZ; this models a
        spectator whose frequency approaches a gate qubit's, where the always-on-style
        exchange J(a_g+ a_s + a_g a_s+) becomes RESONANT and population coherently
        swaps into the spectator during the gate. Explicitly evolves a third transmon
        in the full (n_levels**3)-D open system (64-D at the default n_levels=4), the
        gate Hamiltonian taken from the SAME _smoothed_controls path the optimizer
        uses. Cross-checked against an independent QuTiP simulation in
        tests/test_collision.py; the J=0 limit reproduces the bare-gate F_proc to
        machine precision.

        x_raw: raw pulse parameter [n_slices, n_ch] (e.g. result['best_raw_param']).
        vz: (phi_c, phi_t) virtual-Z frame (e.g. result['virtual_z']); 0 if None.
        detuning_mhz: spectator detuning FROM the coupled gate qubit (MHz); 0 = an
            exact collision, large = far off-resonant (recovers the bare gate). A
            scalar evaluates one point; an array/list evaluates the whole collision
            curve in ONE batched call.
        j_mhz: transverse exchange to the coupled gate qubit (MHz).
        couples_to: which gate qubit the spectator neighbours -- "control" or "target".
        diss_scale: scales the gate-pair Lindblad rates (0.0 isolates the coherent
            collision error). The spectator is modelled as coherent (no decoherence).

        Returns the same keys as the parametric method (scalars for a scalar detuning,
        lists for a sweep): detuning_mhz, f_proc/f_avg, f_proc_isolated/f_avg_isolated,
        delta_r_collision, spectator_leakage, j_mhz, couples_to.
        """
        if not torch.is_tensor(x_raw):
            x_raw = torch.tensor(x_raw, device=DEVICE, dtype=self.rdtype)
        if vz is None:
            vz = torch.zeros(2, device=DEVICE, dtype=self.rdtype)
        elif not torch.is_tensor(vz):
            vz = torch.tensor(vz, device=DEVICE, dtype=self.rdtype)
        u = (x_raw.unsqueeze(0) if x_raw.dim() == 2 else x_raw)[:1]
        n_slices = u.shape[1]
        nl = self.n_levels
        d2 = self._dim
        d3 = nl ** 3
        cdt = self.cdtype
        ct = str(couples_to).lower()
        if ct in ("control", "c", "0"):
            to_control = True
        elif ct in ("target", "t", "1"):
            to_control = False
        else:
            raise ValueError("couples_to must be 'control' or 'target'")

        det_arr = np.atleast_1d(np.asarray(detuning_mhz, dtype=float)).ravel()
        B = det_arr.shape[0]
        det_rad = torch.as_tensor(2.0 * math.pi * det_arr / 1000.0,
                                  dtype=cdt, device=DEVICE)

        _sub = torch.tensor([math.sqrt(k) for k in range(1, nl)],
                            dtype=cdt, device=DEVICE)
        a = torch.diag(_sub, 1)
        I = torch.eye(nl, dtype=cdt, device=DEVICE)
        I9 = torch.eye(d2, dtype=cdt, device=DEVICE)

        def lift(op9):
            return torch.kron(op9.contiguous(), I)

        a_s = torch.kron(I9, a)
        ad_s = a_s.conj().t()
        n_s = ad_s @ a_s
        # Frame rotates at f_target: target sits at 0, control at delta_c; a
        # spectator's collision (detuning 0) is with whichever qubit it neighbours.
        delta_c = (self.profile.freq_ghz_control
                   - self.profile.freq_ghz_target) * 2.0 * math.pi
        anh_s = (self.profile.anharm_ghz_control if to_control
                 else self.profile.anharm_ghz_target) * 2.0 * math.pi
        base = delta_c if to_control else 0.0
        a_g9 = torch.kron(a, I) if to_control else torch.kron(I, a)
        a_g = lift(a_g9)
        j = 2.0 * math.pi * (float(j_mhz) / 1000.0)
        H_exch = j * (a_g.conj().t() @ a_s + a_g @ ad_s)
        H_s_const = base * n_s + 0.5 * anh_s * (ad_s @ ad_s @ a_s @ a_s)

        Ls = [(lift(L), lift(L).conj().t().contiguous())
              for L in (self._L_T1_C, self._L_T1_T,
                        self._L_PHI_C, self._L_PHI_T)]
        L_loss = lift(self._L_LOSS_SUM)

        ci3 = (self._comp_idx * nl).tolist()
        rho = torch.zeros((B, 16, d3, d3), dtype=cdt, device=DEVICE)
        for i in range(4):
            for jx in range(4):
                rho[:, i * 4 + jx, ci3[i], ci3[jx]] = 1.0

        ctrl = self._smoothed_controls(u, dt)
        uc, ut, vc, vt = ctrl["uc"], ctrl["ut"], ctrl["vc"], ctrl["vt"]

        with torch.no_grad():
            for i in range(n_slices):
                # Gate-pair H(t): identical assembly to simulate_gradient_batch.
                H9 = self._H_DRIFT + (uc[0, i] * self.OMEGA_MAX) * self._X_C
                if ut is not None:
                    H9 = H9 + (ut[0, i] * self.OMEGA_MAX) * self._X_T
                if vc is not None:
                    H9 = H9 + vc[0, i] * self._Y_C
                    if vt is not None:
                        H9 = H9 + vt[0, i] * self._Y_T
                H_common = lift(H9) + H_exch + H_s_const
                H_b = H_common.unsqueeze(0) + det_rad.view(B, 1, 1) * n_s
                U = torch.linalg.matrix_exp(-1j * H_b * dt).unsqueeze(1)
                Ud = U.conj().transpose(-2, -1)
                rho = U @ rho @ Ud
                jump = sum(L @ rho @ Ld for (L, Ld) in Ls)
                anti = (L_loss @ rho) + (rho @ L_loss)
                rho = rho + dt * diss_scale * (jump - anti)

            r6 = rho.reshape(B, 16, d2, nl, d2, nl)
            choi9 = torch.einsum('bmipjp->bmij', r6)       # partial trace over spectator
            f_proc = self._process_fidelity(choi9, vz)     # [B]
            diag = rho.diagonal(dim1=-2, dim2=-1).real.reshape(B, 16, d2, nl)
            p_s0 = diag[..., 0].sum(dim=-1)                # P(spectator in |0>)
            spec_leak = (1.0 - p_s0[:, [0, 5, 10, 15]]).mean(dim=1).clamp(0.0, 1.0)
            f_iso = float(self._process_fidelity(
                self.simulate_choi_batch(u, dt=dt, diss_scale=diss_scale), vz).item())

        dd = 4.0
        f_proc_np = f_proc.detach().cpu().numpy()
        f_avg_np = (dd * f_proc_np + 1.0) / (dd + 1.0)
        f_avg_iso = (dd * f_iso + 1.0) / (dd + 1.0)
        scalar = det_arr.size == 1

        def _s(arr):
            return float(arr[0]) if scalar else arr.tolist()

        return {
            "detuning_mhz": float(det_arr[0]) if scalar else det_arr.tolist(),
            "f_proc": _s(f_proc_np),
            "f_avg": _s(f_avg_np),
            "f_proc_isolated": f_iso,
            "f_avg_isolated": f_avg_iso,
            "delta_r_collision": _s(f_avg_iso - f_avg_np),
            "spectator_leakage": _s(spec_leak.detach().cpu().numpy()),
            "j_mhz": float(j_mhz),
            "couples_to": "control" if to_control else "target",
        }
