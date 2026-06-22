"""gradpulse.liouville - a THIRD, QuTiP-free independent solver for F_proc.

Why this module exists
----------------------
The package's whole discipline is "no number ships unless a second, independent
solver reproduces it." Until now the independent referee was QuTiP
(:mod:`gradpulse.validate`): the matched piecewise-constant cross-check and the
adaptive ``mesolve`` unbiasedness check. But *both* of those are QuTiP, so two
solvers agreeing rules out a transcription bug in either operator build yet
cannot, even in principle, rule out a QuTiP-specific artifact (an operator
convention, its ``.expm()``, its ODE backend) shared by both paths.

This module is the third leg, deliberately independent on three axes at once:

1. **Library.** It imports neither QuTiP nor PyTorch, nor even SciPy - only NumPy
   (its matrix exponential is a self-contained Pade approximant, ``_expm``). It runs
   without the ``[validate]`` extra installed, so the cross-check is available to
   every user, not just those with QuTiP.
2. **Representation.** It does not step the density matrix as a matrix. It builds
   the full Lindbladian *superoperator* (the Liouvillian) and propagates the
   column-stacked ``vec(rho)`` through it. Different math, different code.
3. **Splitting.** The optimizer and the QuTiP cross-check both use the *matched*
   scheme - an exact unitary step followed by a first-order Lindblad dissipator
   (a Lie-Trotter split). This module takes the exact matrix exponential of the
   *full* generator (Hamiltonian + dissipator together) per slice. So it does not
   merely re-confirm the operators: a match independently bounds the Trotter
   *splitting error*, the one thing the matched QuTiP check structurally cannot
   probe.

The evolved quantity is the SAME exact entanglement/process fidelity the rest of
the package reports: each of the 16 computational-basis operators |i><j| is
propagated through the channel (linear in rho, so they need not be states),
projected to the computational subspace to form the channel's Choi matrix, and
contracted with the target -- F_proc = (1/16) sum_ij <i| U^dag Phi(|i><j|) U |j>.

Scope: the parametric-coupler architecture (CZ / iSWAP family, :func:`liouville_f_proc`)
and the cross-resonance ZX(pi/2) gate (:func:`liouville_cr_f_proc`) -- so both pair
architectures now carry a *library-independent* third solver, not just a second
QuTiP-based one. The N-qubit register is cross-checked the same way for small cases
via :func:`liouville_nqubit_closed_f_proc`. Each shares no operator-construction or
matrix-exponential code with the optimizer or with QuTiP.
"""
from __future__ import annotations

import math
from itertools import product

import numpy as np


def _expm(A: np.ndarray) -> np.ndarray:
    """Dense matrix exponential, scaling-and-squaring with a [13/13] Pade
    approximant (Higham 2005) -- a self-contained NumPy implementation.

    Deliberately depends on NOTHING but NumPy: not SciPy, not PyTorch's
    ``matrix_exp`` (the optimizer's routine), not QuTiP's ``.expm()``. So the
    third solver shares no exponential code with any other path in the package,
    which is the point -- it cannot inherit a library-specific artifact from
    either solver it cross-checks. It is the same algorithm SciPy and MATLAB use,
    so it carries their robustness; correctness is confirmed end-to-end by the
    cross-solver agreement test (a wrong expm could not reproduce the QuTiP and
    PyTorch fidelities to ~1e-7).
    """
    A = np.asarray(A, dtype=complex)
    n = A.shape[0]
    ident = np.eye(n, dtype=complex)
    # Pade-13 coefficients b_0..b_13.
    b = (64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
         1187353796428800.0, 129060195264000.0, 10559470521600.0,
         670442572800.0, 33522128640.0, 1323241920.0, 40840800.0,
         960960.0, 16380.0, 182.0, 1.0)
    norm1 = float(np.max(np.sum(np.abs(A), axis=0)))     # 1-norm
    if norm1 == 0.0:
        return ident
    theta13 = 5.371920351148152                          # double-precision threshold
    s = max(0, int(math.ceil(math.log2(norm1 / theta13))))
    A = A / (2.0 ** s)
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A4 @ A2
    U = A @ (A6 @ (b[13] * A6 + b[11] * A4 + b[9] * A2)
             + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = (A6 @ (b[12] * A6 + b[10] * A4 + b[8] * A2)
         + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident)
    R = np.linalg.solve(V - U, V + U)                    # (V-U)^{-1}(V+U)
    for _ in range(s):
        R = R @ R
    return R


def _target_unitary(target_gate: str) -> np.ndarray:
    """4x4 target in the computational basis {|00>,|01>,|10>,|11>}.

    Re-derived here (not imported from :mod:`gradpulse.validate`, which pulls in
    QuTiP) so the third solver shares no code with the QuTiP path - an
    independent re-statement of the target is part of the independence.
    """
    s = 1.0 / math.sqrt(2.0)
    gates = {
        "cz": np.diag([1, 1, 1, -1]).astype(complex),
        "iswap": np.array([[1, 0,  0,  0],
                           [0, 0,  1j, 0],
                           [0, 1j, 0,  0],
                           [0, 0,  0,  1]], dtype=complex),
        "sqrt_iswap": np.array([[1, 0,    0,    0],
                                [0, s,    1j*s, 0],
                                [0, 1j*s, s,    0],
                                [0, 0,    0,    1]], dtype=complex),
    }
    g = str(target_gate).lower()
    if g not in gates:
        raise ValueError(
            f"target_gate must be one of {sorted(gates)}, got {target_gate!r}")
    return gates[g]


def _apply_line_response(u_smooth: np.ndarray, spec, dt: float) -> np.ndarray:
    """Causal, unit-DC-gain line-response convolution.

    Independent re-implementation of the same operation the PyTorch simulator and
    the QuTiP cross-check apply, so all three evaluate the SAME post-line control.
    spec: None (identity), {"type": "exponential", "tau_ns": t}, or a causal
    impulse response sampled at the working dt. u_smooth is [n_samples, n_ch].
    """
    if spec is None:
        return u_smooth
    if isinstance(spec, dict):
        tau = float(spec["tau_ns"])
        n = max(2, int(math.ceil(6.0 * tau / dt)))
        h = np.exp(-(np.arange(n) * dt) / tau)
    else:
        h = np.asarray(spec, dtype=float)
    h = h / max(float(h.sum()), 1e-12)
    K = len(h)
    out = np.empty_like(u_smooth)
    for c in range(u_smooth.shape[1]):
        x = u_smooth[:, c]
        xp = np.concatenate([np.full(K - 1, x[0]), x])
        out[:, c] = np.convolve(xp, h, mode="valid")
    return out


# Same fallback profile as gradpulse.validate.DEFAULT_QUTIP_PROFILE, re-stated so
# this module needs no import from the QuTiP path.
DEFAULT_PROFILE = {
    "n_levels":      3,
    "freq_ghz_q1":   4.85,
    "freq_ghz_q2":   5.05,
    "anharm_ghz_q1": -0.200,
    "anharm_ghz_q2": -0.200,
    "t1_ns_q1":      30000.0,
    "t1_ns_q2":      30000.0,
    "t2_ns_q1":      25000.0,
    "t2_ns_q2":      25000.0,
    "g_max_mhz":     12.0,
    "omega_max_mhz": 50.0,
    "chi_zz_mhz":    0.0,
    "n_thermal_q1":  0.0,
    "n_thermal_q2":  0.0,
}


def _build_numpy_ops(profile: dict) -> dict:
    """Build the two-transmon Hamiltonian/Lindblad operators in pure NumPy.

    Mirrors gradpulse.validate._build_qutip_ops exactly (same physics, same
    rad/ns conventions) but with plain ndarrays, so the only thing shared with
    the QuTiP path is the physics being modelled - not a line of code.
    """
    nl = int(profile.get("n_levels", 3))
    a = np.diag(np.sqrt(np.arange(1, nl, dtype=float)), 1).astype(complex)
    ad = a.conj().T
    n = ad @ a
    I = np.eye(nl, dtype=complex)

    def kron2(A, B):
        return np.kron(A, B)

    alpha1 = profile["anharm_ghz_q1"] * 2 * math.pi
    alpha2 = profile["anharm_ghz_q2"] * 2 * math.pi
    delta = (profile["freq_ghz_q2"] - profile["freq_ghz_q1"]) * 2 * math.pi
    omega_max = 2 * math.pi * (profile["omega_max_mhz"] / 1000.0)
    g_max = 2 * math.pi * (profile["g_max_mhz"] / 1000.0)

    anh1 = 0.5 * alpha1 * (ad @ ad @ a @ a)
    anh2 = 0.5 * alpha2 * (ad @ ad @ a @ a)
    chi_zz = float(profile.get("chi_zz_mhz", 0.0)) * 2 * math.pi / 1000.0
    H_drift = (
        delta * kron2(I, n)
        + kron2(anh1, I)
        + kron2(I, anh2)
        + chi_zz * (kron2(n, I) @ kron2(I, n))
    )
    X1 = kron2(a + ad, I)
    X2 = kron2(I, a + ad)
    Cx = kron2(ad, a) + kron2(a, ad)
    Cy = 1j * (kron2(ad, a) - kron2(a, ad))
    N1 = kron2(n, I)
    N2 = kron2(I, n)

    a_q1 = kron2(a, I)
    a_q2 = kron2(I, a)
    n_q1 = kron2(n, I)
    n_q2 = kron2(I, n)
    t1_q1 = profile["t1_ns_q1"]
    t1_q2 = profile["t1_ns_q2"]
    t2_q1 = profile["t2_ns_q1"]
    t2_q2 = profile["t2_ns_q2"]

    def t_phi(t1, t2):
        rate = 1.0 / max(t2, 1.0) - 1.0 / max(2.0 * t1, 1.0)
        return 1.0 / max(rate, 1e-9)

    n_th_q1 = max(0.0, float(profile.get("n_thermal_q1", 0.0)))
    n_th_q2 = max(0.0, float(profile.get("n_thermal_q2", 0.0)))
    L_ops = [
        math.sqrt((1.0 + n_th_q1) / t1_q1) * a_q1,
        math.sqrt((1.0 + n_th_q2) / t1_q2) * a_q2,
        math.sqrt(2.0 / t_phi(t1_q1, t2_q1)) * n_q1,
        math.sqrt(2.0 / t_phi(t1_q2, t2_q2)) * n_q2,
    ]
    if n_th_q1 > 0.0:
        L_ops.append(math.sqrt(n_th_q1 / t1_q1) * a_q1.conj().T)
    if n_th_q2 > 0.0:
        L_ops.append(math.sqrt(n_th_q2 / t1_q2) * a_q2.conj().T)

    return {
        "H_drift": H_drift, "X1": X1, "X2": X2,
        "Cx": Cx, "Cy": Cy, "N1": N1, "N2": N2,
        "L_ops": L_ops,
        "omega_max": omega_max, "g_max": g_max,
    }


def _lindbladian(H: np.ndarray, L_ops, dim: int) -> np.ndarray:
    """Column-stacking Lindbladian superoperator L acting on vec(rho).

    Uses the identity vec(A rho B) = (B^T (x) A) vec(rho) for the column-stacked
    vec (NumPy order='F'). For drho/dt = -i[H,rho] + sum_k (L rho L^dag
    - 1/2 {L^dag L, rho}):

        L = -i ( I (x) H  -  H^T (x) I )
            + sum_k [ conj(L_k) (x) L_k
                      - 1/2 ( I (x) L_k^dag L_k )
                      - 1/2 ( (L_k^dag L_k)^T (x) I ) ]

    so vec(rho(t+dt)) = expm(L dt) vec(rho(t)). Taking the exact exponential of
    this FULL generator (no unitary/dissipator split) is what makes this an
    independent check of the matched scheme's Trotter splitting.
    """
    Id = np.eye(dim, dtype=complex)
    Lsuper = -1j * (np.kron(Id, H) - np.kron(H.T, Id))
    for L in L_ops:
        LdL = L.conj().T @ L
        Lsuper = Lsuper + (
            np.kron(L.conj(), L)
            - 0.5 * np.kron(Id, LdL)
            - 0.5 * np.kron(LdL.T, Id)
        )
    return Lsuper


def liouville_f_proc(profile, waveform: np.ndarray, target_gate: str = "cz",
                     dt_ns: float = 1.0, line_response=None,
                     detuning_offset=0.0) -> float:
    """Independent (QuTiP-free) exact F_proc via the Liouvillian superoperator.

    Drop-in analogue of :func:`gradpulse.validate.qutip_f_proc`: same SAVED [0,1]
    envelope, same profile (dataclass or dict), same exact 16-operator Choi
    process fidelity - computed through a wholly separate solver (NumPy/SciPy,
    superoperator representation, exact full-generator exponential).

    Parameters mirror ``qutip_f_proc``. ``detuning_offset`` adds a static
    qubit-frequency offset (rad/ns; scalar -> common delta*(N1+N2), or (d1, d2)
    per qubit) so the robustness sweep / quasi-static / spectator-ZZ paths can be
    cross-checked here too. Returns F_proc clipped to [0, 1].
    """
    if hasattr(profile, "__dataclass_fields__"):
        from dataclasses import asdict
        pdict = asdict(profile)
    else:
        pdict = dict(profile)
    prof = dict(DEFAULT_PROFILE)
    prof.update({k: pdict[k] for k in prof if k in pdict})

    u = np.asarray(waveform, dtype=float)
    n_samples, n_ch = u.shape
    nl = int(prof.get("n_levels", 3))
    dim = nl * nl
    ops = _build_numpy_ops(prof)

    if isinstance(detuning_offset, (tuple, list)):
        d1, d2 = float(detuning_offset[0]), float(detuning_offset[1])
    else:
        d1 = d2 = float(detuning_offset)
    H_det = (d1 * ops["N1"] + d2 * ops["N2"]) if (d1 != 0.0 or d2 != 0.0) else None

    # The saved envelope is ALREADY smoothed+clamped: only center to [-1,1] and
    # re-apply the line response (identical preprocessing to the other solvers).
    u_smooth = 2.0 * u - 1.0
    u_smooth = _apply_line_response(u_smooth, line_response, dt_ns)

    omega = ops["omega_max"]
    g = ops["g_max"]
    stark = 2 * math.pi * (20.0 / 1000.0)

    # Phi = prod_i expm(L_i dt), shared across all 16 Choi basis operators.
    Phi = np.eye(dim * dim, dtype=complex)
    for i in range(n_samples):
        H = ops["H_drift"].copy()
        H = H + (u_smooth[i, 0] * omega) * ops["X1"]
        H = H + (u_smooth[i, 1] * omega) * ops["X2"]
        if n_ch == 3:
            H = H + (u_smooth[i, 2] * g) * ops["Cx"]
        else:
            phi = math.pi * u_smooth[i, 3]
            H = H + (u_smooth[i, 2] * math.cos(phi) * g) * ops["Cx"]
            H = H + (u_smooth[i, 2] * math.sin(phi) * g) * ops["Cy"]
        if n_ch == 6:
            H = H + (u_smooth[i, 4] * stark) * ops["N1"]
            H = H + (u_smooth[i, 5] * stark) * ops["N2"]
        if H_det is not None:
            H = H + H_det
        S = _expm(_lindbladian(H, ops["L_ops"], dim) * dt_ns)
        Phi = S @ Phi

    comp = [0, 1, nl, nl + 1]
    U4 = _target_unitary(target_gate)
    F_acc = 0.0 + 0.0j
    for ii in range(4):
        for jj in range(4):
            M0 = np.zeros((dim, dim), dtype=complex)
            M0[comp[ii], comp[jj]] = 1.0
            v_out = Phi @ M0.reshape(-1, order="F")        # vec(rho_out), col-stack
            rho_out = v_out.reshape((dim, dim), order="F")
            M = rho_out[np.ix_(comp, comp)]                # 4x4 projected Phi(|i><j|)
            F_acc += (U4.conj().T @ M @ U4)[ii, jj]
    return max(0.0, min(1.0, float(np.real(F_acc / 16.0))))


# ===================================================================== #
#  Cross-resonance ZX(pi/2): the SAME third-solver discipline           #
# --------------------------------------------------------------------- #
#  Until now the cross-resonance architecture's only independent referee #
#  was QuTiP (gradpulse.validate.cr_cross_check). That left the CR       #
#  headline a *double*-solver check whose two non-optimizer legs share a #
#  library, so a QuTiP-specific artifact could pass undetected. The      #
#  functions below extend the library-independent NumPy Liouvillian to   #
#  CR, so "three solvers sharing no code" holds for this architecture    #
#  too -- not just the parametric CZ.                                    #
# ===================================================================== #

# Fallback CR profile (mirrors crossresonance.CrossResonanceProfile defaults),
# re-stated so this module needs no import from the torch/QuTiP CR paths.
DEFAULT_CR_PROFILE = {
    "n_levels":           4,
    "freq_ghz_control":   5.00,
    "freq_ghz_target":    4.85,
    "anharm_ghz_control": -0.33,
    "anharm_ghz_target":  -0.33,
    "j_coupling_mhz":     3.0,
    "omega_max_mhz":      60.0,
    "chi_zz_mhz":         0.0,
    "t1_ns_control":      150000.0,
    "t1_ns_target":       150000.0,
    "t2_ns_control":      120000.0,
    "t2_ns_target":       120000.0,
}


def _zx90_target() -> np.ndarray:
    """ZX(pi/2) target in the comp basis {|00>,|01>,|10>,|11>}.

    Independent re-derivation of crossresonance.zx90_target (not imported, which
    would pull in torch) -- restating the target is part of the independence.
    """
    inv = 1.0 / math.sqrt(2.0)
    return np.array([
        [inv,       -1j * inv, 0,         0],
        [-1j * inv, inv,       0,         0],
        [0,         0,         inv,       1j * inv],
        [0,         0,         1j * inv,  inv],
    ], dtype=complex)


def _build_numpy_cr_ops(profile: dict) -> dict:
    """Pure-NumPy cross-resonance Hamiltonian/Lindblad operators.

    Mirrors crossresonance._build_cr_ops and validate._build_qutip_cr_ops exactly
    (same physics, same rad/ns conventions, frame rotating at the drive = target
    frequency) but with plain ndarrays -- the only thing shared with the other CR
    paths is the physics, not a line of code.
    """
    nl = int(profile.get("n_levels", 4))
    a = np.diag(np.sqrt(np.arange(1, nl, dtype=float)), 1).astype(complex)
    ad = a.conj().T
    n = ad @ a
    I = np.eye(nl, dtype=complex)

    def kron2(A, B):
        return np.kron(A, B)

    alpha_c = profile["anharm_ghz_control"] * 2 * math.pi
    alpha_t = profile["anharm_ghz_target"] * 2 * math.pi
    delta_c = (profile["freq_ghz_control"]
               - profile["freq_ghz_target"]) * 2 * math.pi   # control detuning
    omega_max = 2 * math.pi * (profile["omega_max_mhz"] / 1000.0)
    j_rate = 2 * math.pi * (profile["j_coupling_mhz"] / 1000.0)
    chi_zz = 2 * math.pi * (float(profile.get("chi_zz_mhz", 0.0)) / 1000.0)

    anh_c = 0.5 * alpha_c * (ad @ ad @ a @ a)
    anh_t = 0.5 * alpha_t * (ad @ ad @ a @ a)
    n_c = kron2(n, I)
    n_t = kron2(I, n)
    exchange = kron2(ad, a) + kron2(a, ad)                  # always-on, static
    H_drift = (delta_c * n_c + kron2(anh_c, I) + kron2(I, anh_t)
               + j_rate * exchange + chi_zz * (n_c @ n_t))

    X_c = kron2(a + ad, I)
    X_t = kron2(I, a + ad)
    Y_c = kron2(1j * (ad - a), I)
    Y_t = kron2(I, 1j * (ad - a))

    a_c = kron2(a, I)
    a_t = kron2(I, a)

    def t_phi(t1, t2):
        rate = 1.0 / max(t2, 1.0) - 1.0 / max(2.0 * t1, 1.0)
        return 1.0 / max(rate, 1e-9)

    L_ops = [
        math.sqrt(1.0 / profile["t1_ns_control"]) * a_c,
        math.sqrt(1.0 / profile["t1_ns_target"]) * a_t,
        math.sqrt(2.0 / t_phi(profile["t1_ns_control"], profile["t2_ns_control"])) * n_c,
        math.sqrt(2.0 / t_phi(profile["t1_ns_target"], profile["t2_ns_target"])) * n_t,
    ]

    # Ideal echo pi-pulse on the control {|0>,|1>} subspace (== _build_cr_ops XPI_C).
    xpi = np.eye(nl, dtype=complex)
    xpi[0, 0] = 0.0; xpi[1, 1] = 0.0
    xpi[0, 1] = -1j; xpi[1, 0] = -1j
    XPI_c = kron2(xpi, I)

    return {
        "H_drift": H_drift, "X_c": X_c, "X_t": X_t, "Y_c": Y_c, "Y_t": Y_t,
        "XPI_c": XPI_c, "L_ops": L_ops,
        "omega_max": omega_max, "alpha_c": alpha_c, "alpha_t": alpha_t,
    }


def liouville_cr_f_proc(profile, waveform: np.ndarray, vz=(0.0, 0.0),
                        echo: bool = False, use_drag: bool = True,
                        dt_ns: float = 1.0) -> float:
    """Independent (QuTiP-free) exact F_proc for a cross-resonance ZX(pi/2) pulse.

    The CR analogue of :func:`liouville_f_proc`. It computes the SAME exact,
    leakage-aware entanglement fidelity that
    ``CrossResonanceZXOptimizer._process_fidelity`` and
    ``gradpulse.validate.cr_cross_check`` compute -- through the third,
    library-independent solver (NumPy-only Liouvillian superoperator, exact
    full-generator exponential, no Trotter split). Its agreement with the
    optimizer therefore bounds the CR splitting error the same way the parametric
    check does, and -- unlike the QuTiP cross-check -- it shares no library with
    the optimizer, closing the "both referees are QuTiP" gap for this architecture.

    ``waveform`` is the SMOOTHED, signed drive stack (``result['best_waveform']``):
    column 0 the control CR drive, column 1 the optional target tone. ``vz`` is the
    saved virtual-Z frame (control, target); ``echo`` runs the echoed sequence
    (second-half CR sign-flip + ideal control pi-pulse at the midpoint and end);
    ``use_drag`` re-derives the Motzoi quadratures from the smoothed drives with
    the optimizer's formula. Returns F_proc clipped to [0, 1].
    """
    if hasattr(profile, "__dataclass_fields__"):
        from dataclasses import asdict
        pdict = asdict(profile)
    else:
        pdict = dict(profile)
    prof = dict(DEFAULT_CR_PROFILE)
    prof.update({k: pdict[k] for k in prof if k in pdict})

    u = np.asarray(waveform, dtype=float)
    if u.ndim == 1:
        u = u[:, None]
    n_samples, n_ch = u.shape
    nl = int(prof.get("n_levels", 4))
    dim = nl * nl
    ops = _build_numpy_cr_ops(prof)

    uc = u[:, 0]
    ut = u[:, 1] if n_ch >= 2 else None

    # Motzoi DRAG quadratures, re-derived from the smoothed signed drives with the
    # optimizer's replicate-padded central difference -- so the DRAG-on pulse is
    # cross-checked apples-to-apples.
    def _ddt(x):
        xp = np.concatenate([x[:1], x, x[-1:]])
        return (xp[2:] - xp[:-2]) / (2.0 * dt_ns)

    if use_drag:
        vc = -_ddt(uc) * ops["omega_max"] / ops["alpha_c"]
        vt = (-_ddt(ut) * ops["omega_max"] / ops["alpha_t"]) if ut is not None else None
    else:
        vc = vt = None

    omega = ops["omega_max"]
    mid = n_samples // 2
    # Echo pi-pulse as a superoperator: vec(XPI rho XPI^dag) = (conj(XPI) (x) XPI) vec.
    xpi_super = np.kron(ops["XPI_c"].conj(), ops["XPI_c"])

    # Phi = ordered product of per-slice exact-generator exponentials, with the
    # ideal control pi-pulse superoperator interleaved at the midpoint and the end
    # when echoing -- the same operation order as the optimizer's hot loop.
    Phi = np.eye(dim * dim, dtype=complex)
    for i in range(n_samples):
        uci = -uc[i] if (echo and i >= mid) else uc[i]
        H = ops["H_drift"] + (uci * omega) * ops["X_c"]
        if ut is not None:
            H = H + (ut[i] * omega) * ops["X_t"]
        if vc is not None:
            vci = -vc[i] if (echo and i >= mid) else vc[i]
            H = H + vci * ops["Y_c"]
            if vt is not None:
                H = H + vt[i] * ops["Y_t"]
        S = _expm(_lindbladian(H, ops["L_ops"], dim) * dt_ns)
        Phi = S @ Phi
        if echo and (i == mid - 1 or i == n_samples - 1):
            Phi = xpi_super @ Phi

    # Saved virtual-Z frame on the target (post-gate single-qubit Z), applied to
    # the target exactly as the optimizer and QuTiP cross-check do.
    U4 = _zx90_target()
    cbits = np.array([0.0, 0.0, 1.0, 1.0])
    tbits = np.array([0.0, 1.0, 0.0, 1.0])
    theta = vz[0] * cbits + vz[1] * tbits
    U4 = np.diag(np.exp(1j * theta)) @ U4

    comp = [0, 1, nl, nl + 1]
    F_acc = 0.0 + 0.0j
    for ii in range(4):
        for jj in range(4):
            M0 = np.zeros((dim, dim), dtype=complex)
            M0[comp[ii], comp[jj]] = 1.0
            v_out = Phi @ M0.reshape(-1, order="F")
            rho_out = v_out.reshape((dim, dim), order="F")
            M = rho_out[np.ix_(comp, comp)]
            F_acc += (U4.conj().T @ M @ U4)[ii, jj]
    return max(0.0, min(1.0, float(np.real(F_acc / 16.0))))


# ===================================================================== #
#  General N-qubit register: a library-independent check for the        #
#  closed-system path -- the novel kernel of this architecture (a gate  #
#  on a subset with identity on the rest, and crosstalk/collisions       #
#  present in the drift). The open-system Choi path keeps its QuTiP      #
#  cross-check; its Lindblad dissipator is the same form as the          #
#  well-tested pair models, while the unitary propagation and the        #
#  multi-qubit subset target are what this leg independently checks.     #
# ===================================================================== #

def _ladder_np(d: int) -> np.ndarray:
    """Truncated annihilation operator a|k> = sqrt(k)|k-1> on d Fock levels."""
    return np.diag(np.sqrt(np.arange(1, d, dtype=float)), 1).astype(complex)


def _embed_np(op1: np.ndarray, q: int, N: int, d: int) -> np.ndarray:
    """Embed a single-transmon operator on qubit q into the N-transmon space."""
    out = op1 if q == 0 else np.eye(d, dtype=complex)
    for k in range(1, N):
        out = np.kron(out, op1 if k == q else np.eye(d, dtype=complex))
    return out


def liouville_nqubit_closed_f_proc(optimizer, waveform: np.ndarray,
                                   dt_ns: float = 1.0) -> float:
    """Independent (NumPy-only) closed-system F_proc for the N-qubit architecture.

    Mirrors ``MultiQubitOptimizer._propagate_unitary`` +
    ``_process_fidelity_unitary`` in pure NumPy -- an independent operator build,
    the self-contained :func:`_expm`, and an independently reconstructed
    multi-qubit subset target -- so the general N-qubit register carries a
    library-independent third solver for its distinctive content: a gate on a
    subset with identity on the rest, with crosstalk and frequency collisions
    living in the drift (the always-on couplings) exactly as the optimizer sees
    them.

    The open-system Choi path keeps its QuTiP cross-check; its Lindblad
    dissipator is the same well-tested form as the pair models, so the part that
    is genuinely new here -- the unitary propagation and the subset target -- is
    what this leg pins down. ``waveform`` is the already-smoothed [0,1] control the
    optimizer evolved (``result['best_waveform']`` or any smoothed pulse), consumed
    verbatim like ``MultiQubitOptimizer.process_fidelity``. Returns
    F_proc = |Tr(U_target^dag U_comp)|^2 / dc^2 over the 2**N computational subspace.
    """
    prof = optimizer.profile
    N, d = int(optimizer.N), int(optimizer.d)
    D = d ** N
    f_ref = float(prof.f_ref_ghz)
    two_pi = 2.0 * math.pi

    a = [_embed_np(_ladder_np(d), q, N, d) for q in range(N)]
    ad = [op.conj().T for op in a]
    nop = [ad[q] @ a[q] for q in range(N)]

    # Drift: per-qubit detuning + anharmonicity + always-on (fixed-edge) exchange.
    H_drift = np.zeros((D, D), dtype=complex)
    for q in range(N):
        delta = two_pi * (float(prof.freqs_ghz[q]) - f_ref)
        alpha = two_pi * float(prof.anharm_mhz[q]) / 1000.0
        H_drift = H_drift + delta * nop[q] + 0.5 * alpha * (ad[q] @ ad[q] @ a[q] @ a[q])
    for (i, j) in optimizer.fixed_edges:
        g = two_pi * float(prof.couplings[(i, j)]) / 1000.0
        H_drift = H_drift + g * (ad[i] @ a[j] + a[i] @ ad[j])

    X = {q: (a[q] + ad[q]) for q in optimizer.drive_qubits}
    Y = {q: (1j * (ad[q] - a[q])) for q in optimizer.drive_qubits}
    Cop = {tuple(e): (ad[e[0]] @ a[e[1]] + a[e[0]] @ ad[e[1]])
           for e in optimizer.tunable_edges}
    Fop = {q: nop[q] for q in optimizer.freq_control_qubits}

    omega, gmax, dmax = (float(optimizer.OMEGA_MAX), float(optimizer.G_MAX),
                         float(optimizer.DELTA_MAX))
    use_drag = bool(optimizer.use_drag)

    x = np.asarray(waveform, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    Nt = x.shape[0]
    u = 2.0 * np.clip(x, 0.0, 1.0) - 1.0                  # signed [-1, 1]

    # Independent reconstruction of the 2**N subset target (mirrors _build_target):
    # each spec's gate on its disjoint qubit group, identity on the rest.
    comp_states = list(product((0, 1), repeat=N))
    index = {s: r for r, s in enumerate(comp_states)}
    dc = 2 ** N
    U_t = np.zeros((dc, dc), dtype=complex)
    for r, s in enumerate(comp_states):
        partials = [(1.0 + 0j, list(s))]
        for G, qubits in optimizer.target_specs:
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
            U_t[index[tuple(sout)], r] += amp

    # Propagate the full unitary slice by slice with the exact per-slice expm,
    # assembling H exactly as _hamiltonian_slice does (channel order: drives [+DRAG],
    # tunable couplings, frequency-control elements).
    U = np.eye(D, dtype=complex)
    for i in range(Nt):
        H = H_drift.copy()
        ch = 0
        for q in optimizer.drive_qubits:
            H = H + (u[i, ch] * omega) * X[q]
            if use_drag:
                alpha = two_pi * float(prof.anharm_mhz[q]) / 1000.0
                du = ((u[i + 1, ch] - u[i - 1, ch]) * omega / (2.0 * dt_ns)
                      if 0 < i < Nt - 1 else 0.0)
                H = H + (-du / alpha) * Y[q]
            ch += 1
        for e in optimizer.tunable_edges:
            H = H + (u[i, ch] * gmax) * Cop[tuple(e)]
            ch += 1
        for q in optimizer.freq_control_qubits:
            H = H + (u[i, ch] * dmax) * Fop[q]
            ch += 1
        U = _expm(-1j * H * dt_ns) @ U

    # Project to the 2**N computational subspace (each qubit in {0,1}) and contract.
    ci = []
    for bits in comp_states:
        kk = 0
        for b in bits:
            kk = kk * d + b
        ci.append(kk)
    Uc = U[np.ix_(ci, ci)]
    M = np.sum(U_t.conj() * Uc)                           # Tr(U_t^dag U_comp)
    return float((M.real ** 2 + M.imag ** 2) / (dc * dc))
