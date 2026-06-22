"""gradpulse.validate - Independent cross-check of our PyTorch simulator using QuTiP.

QuTiP is a standard, well-tested open-quantum-systems library used for over a decade
in published quantum-computing research. By feeding the SAME pulse and the SAME
Hamiltonian/decoherence parameters into an independent integrator built from QuTiP
operators (matched piecewise-constant master-equation stepping, not mesolve; see
the note in cross_check), we cross-check that the F_proc we report isn't an artifact
of our own simulator.

Two independent QuTiP paths, validating different things:
  - the matched piecewise-constant integrator (cross_check / qutip_f_proc) shares the
    optimizer's *scheme* but not its code, so it catches an implementation/transcription
    bug in either operator build;
  - ``mesolve_zoh_fproc`` runs QuTiP's *adaptive* solver on a zero-order-hold staircase
    of the same pulse, a different numerical METHOD, so it catches the one thing the
    matched scheme structurally cannot: a consistent-but-biased stepping scheme (one
    that converges smoothly under dt-refinement, but to the wrong continuous-time
    limit). See that function's docstring.

Both architectures are supported: the cross-check auto-detects the pulse's
``architecture`` field ("parametric_cz", the default, or "cross_resonance") and
rebuilds the matching QuTiP model. For cross-resonance it re-derives the DRAG
quadrature from the saved waveform and applies the optimized virtual-Z frame, so a
DRAG-on ZX(pi/2) pulse cross-checks apples-to-apples.

Usage:
  python -m gradpulse.validate                              # cross-check ./cz_pulse.json
  python -m gradpulse.validate --pulse path/to/pulse.json   # specific pulse (any architecture)

QuTiP recomputes the exact entanglement (process) fidelity the way our simulator
does: it evolves the 16 computational-basis operators |i><j| through the channel
(linear evolution, so they need not be density matrices), projects each output to
the computational subspace to form the channel's Choi matrix, and contracts it with
the target to get F_proc = (1/d^2) sum_ij <i| U^dag Phi(|i><j|) U |j>, d=4, a
genuine 2-design (Haar) average, leakage-aware. If QuTiP's F_proc agrees with our
reported F_proc to ≤ 0.001, we have independent confirmation. Larger discrepancies
indicate a bug in one (or both) implementations.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import qutip as qt


def _target_unitary(target_gate: str) -> np.ndarray:
    """4x4 target in the computational basis {|00>,|01>,|10>,|11>}, mirroring
    gradpulse.parametric.ParametricCZOptimizer._build_target_unitary.

    The cross-check is gate-agnostic: only this matrix changes between CZ and the
    iSWAP family (the coupler's native exchange gate). 'cz' is the default so a
    pulse JSON written before target_gate existed still cross-checks as CZ.
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


def _zx90_target() -> np.ndarray:
    """ZX(pi/2) = exp(-i (pi/4) Z(x)X) in the comp basis {|00>,|01>,|10>,|11>}.

    Independent re-derivation of gradpulse.crossresonance.zx90_target for the CR
    cross-check (locally equivalent to CNOT).
    """
    s = 1.0 / math.sqrt(2.0)
    return np.array([
        [s,      -1j * s, 0,      0],
        [-1j * s, s,      0,      0],
        [0,       0,      s,      1j * s],
        [0,       0,      1j * s, s],
    ], dtype=complex)


def _apply_line_response(u_smooth: np.ndarray, spec, dt: float) -> np.ndarray:
    """Causal, unit-DC-gain line-response convolution, mirroring
    ParametricCZOptimizer._apply_line_response so the cross-check evaluates the
    SAME post-line control the PyTorch simulator did.

    spec: None (identity), {"type": "exponential", "tau_ns": t}, or an array-like
    causal impulse response sampled at the working dt. u_smooth is [n_samples, n_ch].
    """
    if spec is None:
        return u_smooth
    if isinstance(spec, dict):
        tau = float(spec["tau_ns"])
        n = max(2, int(math.ceil(6.0 * tau / dt)))
        h = np.exp(-(np.arange(n) * dt) / tau)
    else:
        h = np.asarray(spec, dtype=float)
    h = h / max(float(h.sum()), 1e-12)                # unit DC gain
    K = len(h)
    out = np.empty_like(u_smooth)
    for c in range(u_smooth.shape[1]):
        x = u_smooth[:, c]
        xp = np.concatenate([np.full(K - 1, x[0]), x])  # causal replicate pad
        out[:, c] = np.convolve(xp, h, mode="valid")    # out[n] = sum_j h[j] x[n-j]
    return out


def _build_qutip_ops(profile: dict):
    """Build the same Hamiltonian/Lindblad operators in QuTiP for an
    (n_levels**2)-D two-transmon system. n_levels (default 3) is read from the
    profile so a model optimized at n_levels=4 cross-checks at n_levels=4."""
    nl = int(profile.get("n_levels", 3))
    a = qt.destroy(nl)
    ad = a.dag()
    n = ad * a
    I3 = qt.qeye(nl)

    def kron2(A, B):
        return qt.tensor(A, B)

    alpha1 = profile["anharm_ghz_q1"] * 2 * math.pi
    alpha2 = profile["anharm_ghz_q2"] * 2 * math.pi
    delta = (profile["freq_ghz_q2"] - profile["freq_ghz_q1"]) * 2 * math.pi
    omega_max = 2 * math.pi * (profile["omega_max_mhz"] / 1000.0)
    g_max = 2 * math.pi * (profile["g_max_mhz"] / 1000.0)

    anh1 = 0.5 * alpha1 * (ad * ad * a * a)
    anh2 = 0.5 * alpha2 * (ad * ad * a * a)
    # Static parasitic ZZ (chi_zz * n1 n2), mirroring the PyTorch drift so the
    # cross-check stays apples-to-apples when chi_zz_mhz != 0 (default 0).
    chi_zz = float(profile.get("chi_zz_mhz", 0.0)) * 2 * math.pi / 1000.0
    H_drift = (
        delta * kron2(I3, n)
        + kron2(anh1, I3)
        + kron2(I3, anh2)
        + chi_zz * (kron2(n, I3) * kron2(I3, n))
    )
    X1 = kron2(a + ad, I3)
    X2 = kron2(I3, a + ad)
    Cx = kron2(ad, a) + kron2(a, ad)
    Cy = 1j * (kron2(ad, a) - kron2(a, ad))
    N1 = kron2(n, I3)
    N2 = kron2(I3, n)

    # Lindblad jump operators (same construction as PyTorch sim)
    a_q1 = kron2(a, I3)
    a_q2 = kron2(I3, a)
    n_q1 = kron2(n, I3)
    n_q2 = kron2(I3, n)
    t1_q1 = profile["t1_ns_q1"]
    t1_q2 = profile["t1_ns_q2"]
    t2_q1 = profile["t2_ns_q1"]
    t2_q2 = profile["t2_ns_q2"]
    def t_phi(t1, t2):
        rate = 1.0 / max(t2, 1.0) - 1.0 / max(2.0 * t1, 1.0)
        return 1.0 / max(rate, 1e-9)
    # Mirrors the PyTorch builder: relaxation -> (1+n_th)/T1 plus an excitation
    # jump at n_th/T1; n_th=0 (default) recovers the cold-bath original.
    n_th_q1 = max(0.0, float(profile.get("n_thermal_q1", 0.0)))
    n_th_q2 = max(0.0, float(profile.get("n_thermal_q2", 0.0)))
    L_t1_q1 = math.sqrt((1.0 + n_th_q1) / t1_q1) * a_q1
    L_t1_q2 = math.sqrt((1.0 + n_th_q2) / t1_q2) * a_q2
    L_phi_q1 = math.sqrt(2.0 / t_phi(t1_q1, t2_q1)) * n_q1
    L_phi_q2 = math.sqrt(2.0 / t_phi(t1_q2, t2_q2)) * n_q2
    L_ops = [L_t1_q1, L_t1_q2, L_phi_q1, L_phi_q2]
    if n_th_q1 > 0.0:
        L_ops.append(math.sqrt(n_th_q1 / t1_q1) * a_q1.dag())
    if n_th_q2 > 0.0:
        L_ops.append(math.sqrt(n_th_q2 / t1_q2) * a_q2.dag())

    return {
        "H_drift": H_drift, "X1": X1, "X2": X2,
        "Cx": Cx, "Cy": Cy, "N1": N1, "N2": N2,
        "L_ops": L_ops,
        "omega_max": omega_max, "g_max": g_max, "alpha1": alpha1, "alpha2": alpha2,
    }


# Fallback profile (matches examples/optimize_cz.py); used when a pulse JSON
# carries no "profile" block, and as the key set for the QuTiP model.
DEFAULT_QUTIP_PROFILE = {
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


def _qutip_evolve_fproc(profile: dict, u: np.ndarray, target_gate: str,
                        dt_ns: float, line_response=None,
                        detuning_offset=0.0) -> float:
    """Core QuTiP cross-check evolution: a SAVED [0,1] envelope -> exact F_proc.

    The numerical heart of ``cross_check``, factored out so other code (notably
    ``gradpulse.hardware.QuTiPDeviceBackend``) can run the SAME independent
    integrator. Matched piecewise-constant master equation: exact unitary step +
    first-order Lindblad dissipator per dt slice (mirrors simulate_gradient_batch),
    NOT mesolve (which would interpolate the coefficient arrays into a different
    effective pulse).

    detuning_offset mirrors simulate_gradient_batch's primitive (scalar -> common
    delta*(N1+N2); (d1, d2) -> per-qubit, rad/ns); default 0 leaves the evolution
    unchanged. Lets the independent engine cross-check the robustness sweep, the
    quasi-static average and the spectator-ZZ model -- all of which feed a static
    detuning that the cross-check otherwise never exercised.
    """
    n_samples, n_ch = u.shape
    nl = int(profile.get("n_levels", 3))
    dim = nl * nl
    ops = _build_qutip_ops(profile)
    # Static qubit-frequency offset (rad/ns), added to the drift each slice.
    if isinstance(detuning_offset, (tuple, list)):
        d1, d2 = float(detuning_offset[0]), float(detuning_offset[1])
    else:
        d1 = d2 = float(detuning_offset)
    H_det = (d1 * ops["N1"] + d2 * ops["N2"]) if (d1 != 0.0 or d2 != 0.0) else None
    # The saved .npy is ALREADY the smoothed+clamped envelope -- re-smoothing here
    # would attenuate features twice. Center to [-1,+1] and re-apply line response.
    u_smooth = 2.0 * u - 1.0
    u_smooth = _apply_line_response(u_smooth, line_response, dt_ns)

    comp = [0, 1, nl, nl + 1]   # |00> |01> |10> |11> in the (nl**2)-D space
    omega = ops["omega_max"]
    g = ops["g_max"]
    L_anti = sum(L.dag() * L for L in ops["L_ops"]) * 0.5

    # Per-slice propagator is shared across all 16 Choi basis operators.
    U_list = []
    for i in range(n_samples):
        H = ops["H_drift"]
        H = H + (u_smooth[i, 0] * omega) * ops["X1"]
        H = H + (u_smooth[i, 1] * omega) * ops["X2"]
        if n_ch == 3:
            H = H + (u_smooth[i, 2] * g) * ops["Cx"]
        else:
            phi = math.pi * u_smooth[i, 3]
            H = H + (u_smooth[i, 2] * math.cos(phi) * g) * ops["Cx"]
            H = H + (u_smooth[i, 2] * math.sin(phi) * g) * ops["Cy"]
        if n_ch == 6:
            stark = 2 * math.pi * (20.0 / 1000.0)
            H = H + (u_smooth[i, 4] * stark) * ops["N1"]
            H = H + (u_smooth[i, 5] * stark) * ops["N2"]
        if H_det is not None:
            H = H + H_det
        U_list.append((-1j * H * dt_ns).expm())

    U4 = _target_unitary(target_gate)                # target in comp subspace
    F_acc = 0.0 + 0.0j
    for ii in range(4):
        for jj in range(4):
            M0 = np.zeros((dim, dim), dtype=complex)
            M0[comp[ii], comp[jj]] = 1.0
            rho = qt.Qobj(M0, dims=[[nl, nl], [nl, nl]])
            for i in range(n_samples):
                U = U_list[i]
                rho = U * rho * U.dag()
                jump = sum(L * rho * L.dag() for L in ops["L_ops"])
                rho = rho + dt_ns * (jump - L_anti * rho - rho * L_anti)
            M = rho.full()[np.ix_(comp, comp)]        # 4x4 projected Phi(|i><j|)
            F_acc += (U4.conj().T @ M @ U4)[ii, jj]
    return max(0.0, min(1.0, float(np.real(F_acc / 16.0))))


def qutip_f_proc(profile, waveform: np.ndarray, target_gate: str = "cz",
                 dt_ns: float = 1.0, line_response=None,
                 detuning_offset=0.0) -> float:
    """Independent (QuTiP) exact F_proc for a saved [0,1] envelope and ANY
    ``ParametricCouplerProfile`` (dataclass or dict).

    Same integrator the CLI ship-gate cross-check uses. ``gradpulse.hardware``'s
    independent-engine device backend calls this so the calibration loop's
    "measurement" comes from a DIFFERENT code path than the optimizer's own
    simulator. The profile may carry physics the optimizer's model omits (extra
    static-ZZ, finite temperature, shorter coherence) -- those keys flow straight
    into the QuTiP Hamiltonian/collapse operators here.
    """
    if hasattr(profile, "__dataclass_fields__"):
        from dataclasses import asdict
        pdict = asdict(profile)
    else:
        pdict = dict(profile)
    prof = dict(DEFAULT_QUTIP_PROFILE)
    prof.update({k: pdict[k] for k in prof if k in pdict})
    u = np.asarray(waveform, dtype=float)
    return _qutip_evolve_fproc(prof, u, str(target_gate).lower(), float(dt_ns),
                               line_response, detuning_offset)


def mesolve_zoh_fproc(profile, waveform: np.ndarray, target_gate: str = "cz",
                      dt_ns: float = 1.0, line_response=None,
                      atol: float = 1e-12, rtol: float = 1e-10) -> float:
    """Exact F_proc from QuTiP's ADAPTIVE master-equation solver (``mesolve``) on a
    zero-order-hold (ZOH) staircase of the saved pulse -- a SECOND, method-independent
    cross-check that complements :func:`qutip_f_proc`.

    Why this exists. :func:`qutip_f_proc` reimplements the optimizer's *matched*
    piecewise-constant scheme (exact unitary + first-order Lindblad step) in QuTiP, so
    it proves the two operator builds agree but shares the integration scheme; and
    ``dt_convergence`` shows that scheme converges smoothly as dt->0. Neither, on its
    own, can tell whether that dt->0 limit is the *true* continuous-time Lindblad
    solution: a consistent-but-biased stepper converges beautifully to the wrong
    answer and passes both. ``mesolve`` is a genuinely different numerical method
    (adaptive Runge-Kutta/Adams with error control), so agreement here promotes
    "the scheme self-converges" to "an independent adaptive solver confirms the scheme
    is unbiased" -- the stronger claim a careful referee wants.

    Pulse fidelity matters: the saved envelope is a ZOH staircase (one constant
    control value per dt-slice). The evolution is run *interval by interval* -- one
    ``mesolve`` per slice with that slice's constant Hamiltonian, threading the state
    across boundaries -- rather than as one run over a function-coefficient grid. That
    restart at every slice edge is deliberate: an adaptive step of size dt samples the
    *next* cell at its final stage, smearing the staircase into a slightly different
    (partly-interpolated) pulse and contaminating the comparison with ~1e-7 of
    pulse-shape error instead of measuring scheme bias. With the per-slice restart the
    solver integrates the identical staircase the optimizer's loop applies, so the only
    thing left between the two is the splitting/Lindblad-step error -- exactly what we
    want to bound. The evolved quantity is the SAME 16-operator Choi process fidelity
    (each |i><j| evolved linearly under the Liouvillian, projected to the computational
    subspace, contracted with the target).
    """
    if hasattr(profile, "__dataclass_fields__"):
        from dataclasses import asdict
        pdict = asdict(profile)
    else:
        pdict = dict(profile)
    prof = dict(DEFAULT_QUTIP_PROFILE)
    prof.update({k: pdict[k] for k in prof if k in pdict})

    u = np.asarray(waveform, dtype=float)
    n_samples, n_ch = u.shape
    nl = int(prof.get("n_levels", 3))
    ops = _build_qutip_ops(prof)
    omega, g = ops["omega_max"], ops["g_max"]

    # Same pulse preprocessing as _qutip_evolve_fproc.
    u_smooth = _apply_line_response(2.0 * u - 1.0, line_response, dt_ns)

    # Precompute the constant per-slice Hamiltonian once, reused across all 16 Choi
    # operators; same rad/ns amplitudes _qutip_evolve_fproc's per-slice H uses.
    stark = 2 * math.pi * (20.0 / 1000.0)
    H_slices = []
    for i in range(n_samples):
        H = (ops["H_drift"]
             + (u_smooth[i, 0] * omega) * ops["X1"]
             + (u_smooth[i, 1] * omega) * ops["X2"])
        if n_ch == 3:
            H = H + (u_smooth[i, 2] * g) * ops["Cx"]
        else:
            phi = math.pi * u_smooth[i, 3]
            H = H + (u_smooth[i, 2] * math.cos(phi) * g) * ops["Cx"]
            H = H + (u_smooth[i, 2] * math.sin(phi) * g) * ops["Cy"]
        if n_ch == 6:
            H = H + (u_smooth[i, 4] * stark) * ops["N1"]
            H = H + (u_smooth[i, 5] * stark) * ops["N2"]
        H_slices.append(H)

    comp = [0, 1, nl, nl + 1]
    U4 = _target_unitary(target_gate)
    opts = {"atol": atol, "rtol": rtol, "nsteps": 200_000}

    F_acc = 0.0 + 0.0j
    for ii in range(4):
        for jj in range(4):
            M0 = np.zeros((nl * nl, nl * nl), dtype=complex)
            M0[comp[ii], comp[jj]] = 1.0
            rho = qt.Qobj(M0, dims=[[nl, nl], [nl, nl]])
            for i in range(n_samples):
                rho = qt.mesolve(H_slices[i], rho, [0.0, dt_ns],
                                 c_ops=ops["L_ops"], options=opts).states[-1]
            M = rho.full()[np.ix_(comp, comp)]
            F_acc += (U4.conj().T @ M @ U4)[ii, jj]
    return max(0.0, min(1.0, float(np.real(F_acc / 16.0))))


def cross_check(pulse_json, profile_overrides: dict | None = None):
    # Coerce str/PathLike to Path once -- a bare string used to die with a
    # confusing "'str' object has no attribute 'read_text'".
    pulse_json = Path(pulse_json)
    meta = json.loads(pulse_json.read_text())
    # Default = the original parametric-coupler CZ, so pulse JSONs written
    # before this field existed still cross-check as before.
    arch = str(meta.get("architecture", "parametric_cz")).lower()
    if arch in ("cross_resonance", "cr", "zx"):
        return _cross_check_cr(meta, pulse_json, profile_overrides)
    npy_path = pulse_json.parent / Path(meta["pulse_npy"]).name
    u = np.load(npy_path)                            # [n_samples, n_channels]
    n_samples, n_ch = u.shape
    dt_ns = float(meta["pulse_dt_ns"])
    bw_mhz = float(meta.get("bandwidth_mhz", 80.0))
    smoother_type = meta.get("smoother_type", "gaussian")
    n_channels_meta = int(meta.get("n_channels") or n_ch)
    target_gate = str(meta.get("target_gate", "cz")).lower()

    profile = dict(DEFAULT_QUTIP_PROFILE)
    # Device params saved alongside the pulse take precedence over the defaults,
    # so the QuTiP model matches the profile the optimizer actually used.
    if isinstance(meta.get("profile"), dict):
        profile.update({k: meta["profile"][k] for k in profile if k in meta["profile"]})
    if profile_overrides:
        profile.update(profile_overrides)

    print(f"  Pulse: {pulse_json.name} ({n_samples} samples x {n_ch} ch @ {dt_ns} ns)")
    print(f"  Target gate: {target_gate}")
    print(f"  Pulse F_proc (PyTorch sim, reported): {meta.get('grape_f', 0):.5f}")
    print(f"  Smoother:  {smoother_type} @ {bw_mhz} MHz")

    # Shared with gradpulse.hardware.QuTiPDeviceBackend: matched piecewise-constant
    # master equation, exact F_proc over the 16-op Choi basis.
    F_proc = _qutip_evolve_fproc(profile, u, target_gate, dt_ns,
                                 meta.get("line_response"))
    d = 4.0
    F_avg = (d * F_proc + 1.0) / (d + 1.0)            # average gate fidelity

    reported = float(meta.get("grape_f", 0.0))
    delta = F_proc - reported
    status = "PASS" if abs(delta) < 0.001 else "WARN" if abs(delta) < 0.005 else "FAIL"

    print("\n  QuTiP independent cross-check (matched piecewise-constant master eq):")
    print(f"    F_proc (exact entanglement fidelity, 16-op Choi): {F_proc:.5f}")
    print(f"    F_avg  (avg gate fidelity, (d*F_proc+1)/(d+1)):   {F_avg:.5f}")
    print(f"\n  delta_F = QuTiP F_proc - reported F_proc = {delta:+.5f}")
    _msg = {
        "PASS": "[PASS] Within +/-0.001 -- simulators agree, F_sim is real not artifact.",
        "WARN": "[WARN] Within +/-0.005 -- small discrepancy, plausible numerical-method differences.",
        "FAIL": "[FAIL] > +/-0.005 disagreement -- possible bug in one simulator.",
    }[status]
    print(f"\n  {_msg}")
    return {
        "target_gate": target_gate, "F_proc": F_proc, "F_avg": F_avg,
        "reported": reported, "delta": delta, "status": status,
        "n_samples": n_samples, "n_channels": n_ch, "dt_ns": dt_ns,
    }


def _build_qutip_cr_ops(profile: dict):
    """Independent QuTiP build of the cross-resonance 9D Hamiltonian/Lindblad ops.

    Mirrors gradpulse.crossresonance._build_cr_ops: frame rotating at the drive
    (= target) frequency, so the drift is static (control detuning + both
    anharmonicities + always-on exchange [+ optional ZZ]). n_levels (default 3)
    is read from the profile so an n_levels=4 CR model cross-checks at n_levels=4.
    """
    nl = int(profile.get("n_levels", 3))
    a = qt.destroy(nl)
    ad = a.dag()
    n = ad * a
    I3 = qt.qeye(nl)

    def kron2(A, B):
        return qt.tensor(A, B)

    alpha_c = profile["anharm_ghz_control"] * 2 * math.pi
    alpha_t = profile["anharm_ghz_target"] * 2 * math.pi
    delta_c = (profile["freq_ghz_control"]
               - profile["freq_ghz_target"]) * 2 * math.pi
    omega_max = 2 * math.pi * (profile["omega_max_mhz"] / 1000.0)
    j_rate = 2 * math.pi * (profile["j_coupling_mhz"] / 1000.0)
    chi_zz = 2 * math.pi * (float(profile.get("chi_zz_mhz", 0.0)) / 1000.0)

    anh_c = 0.5 * alpha_c * (ad * ad * a * a)
    anh_t = 0.5 * alpha_t * (ad * ad * a * a)
    n_c = kron2(n, I3)
    n_t = kron2(I3, n)
    exchange = kron2(ad, a) + kron2(a, ad)
    H_drift = (delta_c * n_c + kron2(anh_c, I3) + kron2(I3, anh_t)
               + j_rate * exchange + chi_zz * (n_c * n_t))

    X_c = kron2(a + ad, I3)
    X_t = kron2(I3, a + ad)
    Y_c = kron2(1j * (ad - a), I3)
    Y_t = kron2(I3, 1j * (ad - a))

    a_c = kron2(a, I3)
    a_t = kron2(I3, a)

    def t_phi(t1, t2):
        rate = 1.0 / max(t2, 1.0) - 1.0 / max(2.0 * t1, 1.0)
        return 1.0 / max(rate, 1e-9)

    L_ops = [
        math.sqrt(1.0 / profile["t1_ns_control"]) * a_c,
        math.sqrt(1.0 / profile["t1_ns_target"]) * a_t,
        math.sqrt(2.0 / t_phi(profile["t1_ns_control"], profile["t2_ns_control"])) * n_c,
        math.sqrt(2.0 / t_phi(profile["t1_ns_target"], profile["t2_ns_target"])) * n_t,
    ]
    # Ideal echo pi-pulse on the control {|0>,|1>} subspace (mirrors
    # crossresonance._build_cr_ops XPI_C): X_pi = exp(-i pi/2 sigma_x), id on |2>+.
    xpi = np.eye(nl, dtype=complex)
    xpi[0, 0] = 0.0; xpi[1, 1] = 0.0
    xpi[0, 1] = -1j; xpi[1, 0] = -1j
    XPI_c = qt.tensor(qt.Qobj(xpi), I3)

    return {
        "H_drift": H_drift, "X_c": X_c, "X_t": X_t, "Y_c": Y_c, "Y_t": Y_t,
        "a_c": a_c, "a_t": a_t, "XPI_c": XPI_c,
        "L_ops": L_ops, "omega_max": omega_max,
        "alpha_c": alpha_c, "alpha_t": alpha_t,
    }


def _qutip_cr_fproc(ops, uc, ut, vc, vt, vz, dt_ns, nl, echo=False):
    """Independent QuTiP exact F_proc for a CR pulse (echo-aware).

    The single QuTiP code path for the cross-resonance gate -- shared by the
    file/CLI cross-check (``_cross_check_cr``) and the in-process ``cr_cross_check``
    so the optimizer-vs-validator logic lives in ONE place. Mirrors
    ``CrossResonanceZXOptimizer.simulate_gradient_batch`` + ``_process_fidelity``:
    when ``echo`` is set the control CR drive sign-flips in the second half and an
    ideal control pi-pulse is applied at the midpoint and the end (``ops['XPI_c']``).
    uc/ut are the smoothed signed drives; vc/vt the Motzoi DRAG quadratures (or None).
    """
    omega = ops["omega_max"]
    n_samples = len(uc)
    mid = n_samples // 2
    L_anti = sum(L.dag() * L for L in ops["L_ops"]) * 0.5
    XPI = ops["XPI_c"]

    U_list = []
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
        U_list.append((-1j * H * dt_ns).expm())

    # Saved virtual-Z frame on the target (post-gate single-qubit Z).
    U4 = _zx90_target()
    cbits = np.array([0.0, 0.0, 1.0, 1.0])
    tbits = np.array([0.0, 1.0, 0.0, 1.0])
    theta = vz[0] * cbits + vz[1] * tbits
    U4 = np.diag(np.exp(1j * theta)) @ U4

    dim = nl * nl
    comp = [0, 1, nl, nl + 1]
    F_acc = 0.0 + 0.0j
    for ii in range(4):
        for jj in range(4):
            M0 = np.zeros((dim, dim), dtype=complex)
            M0[comp[ii], comp[jj]] = 1.0
            rho = qt.Qobj(M0, dims=[[nl, nl], [nl, nl]])
            for i in range(n_samples):
                U = U_list[i]
                rho = U * rho * U.dag()
                jump = sum(L * rho * L.dag() for L in ops["L_ops"])
                rho = rho + dt_ns * (jump - L_anti * rho - rho * L_anti)
                if echo and (i == mid - 1 or i == n_samples - 1):
                    rho = XPI * rho * XPI.dag()
            M = rho.full()[np.ix_(comp, comp)]
            F_acc += (U4.conj().T @ M @ U4)[ii, jj]
    return max(0.0, min(1.0, float(np.real(F_acc / 16.0))))


def cr_cross_check(optimizer, waveform, vz=None, echo=None, dt_ns: float = 1.0) -> float:
    """Independent (QuTiP) exact F_proc for a cross-resonance pulse, in-process.

    The in-memory analogue of the file-based CR cross-check: pass a
    ``CrossResonanceZXOptimizer`` and its smoothed signed ``waveform``
    (``result['best_waveform']``) and get the exact process fidelity from the
    independent QuTiP integrator. ``echo`` defaults to the optimizer's own setting,
    so an echoed gate is always cross-checked with the echo applied. Used by tests
    and quick checks without writing a pulse to disk.
    """
    from dataclasses import asdict
    p = asdict(optimizer.profile)
    keys = ("n_levels", "freq_ghz_control", "freq_ghz_target", "anharm_ghz_control",
            "anharm_ghz_target", "j_coupling_mhz", "omega_max_mhz", "chi_zz_mhz",
            "t1_ns_control", "t1_ns_target", "t2_ns_control", "t2_ns_target")
    profile = {k: p[k] for k in keys if k in p}
    ops = _build_qutip_cr_ops(profile)
    u = np.asarray(waveform, dtype=float)
    if u.ndim == 1:
        u = u[:, None]
    n_ch = u.shape[1]
    uc = u[:, 0]
    ut = u[:, 1] if n_ch >= 2 else None

    def _ddt(x):
        xp = np.concatenate([x[:1], x, x[-1:]])
        return (xp[2:] - xp[:-2]) / (2.0 * dt_ns)

    if bool(getattr(optimizer, "use_drag", True)):
        vc = -_ddt(uc) * ops["omega_max"] / ops["alpha_c"]
        vt = (-_ddt(ut) * ops["omega_max"] / ops["alpha_t"]) if ut is not None else None
    else:
        vc = vt = None
    if vz is None:
        vz = [0.0, 0.0]
    if echo is None:
        echo = bool(getattr(optimizer, "echo", False))
    nl = int(profile.get("n_levels", 3))
    return _qutip_cr_fproc(ops, uc, ut, vc, vt, list(vz), float(dt_ns), nl, bool(echo))


def _cross_check_cr(meta: dict, pulse_json: Path, profile_overrides: dict | None = None):
    """QuTiP cross-check for a cross-resonance ZX(pi/2) pulse.

    Re-derives the DRAG quadrature from the saved (smoothed, signed) waveform with
    the same Motzoi formula the optimizer used, so the cross-check validates the
    DRAG-on pulse apples-to-apples, and applies the saved virtual-Z frame to the
    target before contracting the channel's Choi matrix.
    """
    npy_path = pulse_json.parent / Path(meta["pulse_npy"]).name
    u = np.load(npy_path)                            # [n_samples, n_ch], in [-1, 1]
    n_samples, n_ch = u.shape
    dt_ns = float(meta["pulse_dt_ns"])
    use_drag = bool(meta.get("use_drag", True))
    vz = list(meta.get("virtual_z", [0.0, 0.0]))

    # Fallback CR device profile (matches CrossResonanceProfile defaults).
    profile = {
        "n_levels":           3,
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
    if isinstance(meta.get("profile"), dict):
        profile.update({k: meta["profile"][k] for k in profile if k in meta["profile"]})
    if profile_overrides:
        profile.update(profile_overrides)

    print(f"  Pulse: {pulse_json.name} ({n_samples} samples x {n_ch} ch @ {dt_ns} ns)")
    print(f"  Architecture: cross_resonance (ZX90, up to virtual-Z {np.round(vz, 4).tolist()})")
    print(f"  DRAG: {'on' if use_drag else 'off'}")
    print(f"  Pulse F_proc (PyTorch sim, reported): {meta.get('grape_f', 0):.5f}")

    ops = _build_qutip_cr_ops(profile)
    omega = ops["omega_max"]
    # The saved waveform is already the smoothed, SIGNED envelope -- use directly
    # (no [0,1]->[-1,1] recentering, unlike the parametric CZ pulses).
    uc = u[:, 0]
    ut = u[:, 1] if n_ch >= 2 else None

    # Re-derive the Motzoi DRAG quadrature from the saved waveform (deterministic):
    #   Omega_y = -d(Omega)/dt / alpha. Central difference with replicate padding,
    # matching CrossResonanceZXOptimizer._ddt exactly.
    def _ddt(x):
        xp = np.concatenate([x[:1], x, x[-1:]])
        return (xp[2:] - xp[:-2]) / (2.0 * dt_ns)
    if use_drag:
        vc = -_ddt(uc) * omega / ops["alpha_c"]
        vt = (-_ddt(ut) * omega / ops["alpha_t"]) if ut is not None else None
    else:
        vc = vt = None

    echo = bool(meta.get("echo", False))
    print(f"  Echo: {'on (echoed CR)' if echo else 'off (single-pulse CR)'}")
    nl = int(profile.get("n_levels", 3))
    F_proc = _qutip_cr_fproc(ops, uc, ut, vc, vt, vz, dt_ns, nl, echo)
    d = 4.0
    F_avg = (d * F_proc + 1.0) / (d + 1.0)

    reported = float(meta.get("grape_f", 0.0))
    delta = F_proc - reported
    status = "PASS" if abs(delta) < 0.001 else "WARN" if abs(delta) < 0.005 else "FAIL"

    print("\n  QuTiP independent cross-check (matched piecewise-constant master eq):")
    print(f"    F_proc (exact entanglement fidelity, 16-op Choi): {F_proc:.5f}")
    print(f"    F_avg  (avg gate fidelity, (d*F_proc+1)/(d+1)):   {F_avg:.5f}")
    print(f"\n  delta_F = QuTiP F_proc - reported F_proc = {delta:+.5f}")
    _msg = {
        "PASS": "[PASS] Within +/-0.001 -- simulators agree, F_sim is real not artifact.",
        "WARN": "[WARN] Within +/-0.005 -- small discrepancy, plausible numerical-method differences.",
        "FAIL": "[FAIL] > +/-0.005 disagreement -- possible bug in one simulator.",
    }[status]
    print(f"\n  {_msg}")
    return {
        "architecture": "cross_resonance", "target_gate": "zx90",
        "F_proc": F_proc, "F_avg": F_avg, "reported": reported,
        "delta": delta, "status": status,
        "n_samples": n_samples, "n_channels": n_ch, "dt_ns": dt_ns,
    }


def cr_counter_rotating_cross_check(profile: dict, waveform: np.ndarray, vz,
                                    dt_ns: float, *, use_drag: bool = True,
                                    omega_d_ghz: float | None = None,
                                    max_step_ns: float = 0.02) -> dict:
    """Independent QuTiP check of ``CrossResonanceZXOptimizer.counter_rotating_fidelity``.

    The torch method restores each drive's counter-rotating partner (oscillating at
    2*omega_d) via fixed midpoint sub-stepping. Here we build the SAME time-dependent
    Hamiltonian in QuTiP and integrate it with QuTiP's ADAPTIVE propagator
    (sesolve Runge-Kutta) -- a genuinely different integrator AND library. Agreement
    of the beyond-RWA F_proc shift (delta_r_counter_rot) holds the RWA number to the
    same independent-solver bar as the rest of the package and, because QuTiP resolves
    the 2*omega_d oscillation adaptively rather than by our fixed sub-steps, also
    confirms the sub-stepping scheme itself is converged.

    Coherent only (matches the method's default ``diss_scale=0``): the counter-rotating
    term is a coherent shift and decoherence is identical with/without it, so the
    unitary comparison is the clean one.

    waveform: the saved [n_samples, n_ch] SIGNED, smoothed envelope (as for
    ``cross_check``). vz: (phi_c, phi_t) virtual-Z. Returns the QuTiP analogue of the
    torch dict: ``{f_proc_rwa, f_proc_counter_rot, delta_r_counter_rot, omega_d_ghz}``.
    """
    ops = _build_qutip_cr_ops(profile)
    omega = ops["omega_max"]
    nl = int(profile.get("n_levels", 3))
    dim = nl * nl
    comp = [0, 1, nl, nl + 1]
    n_samples, n_ch = waveform.shape
    if omega_d_ghz is None:
        omega_d_ghz = float(profile["freq_ghz_target"])
    wd = 2.0 * math.pi * omega_d_ghz
    T = n_samples * dt_ns

    uc = waveform[:, 0]
    ut = waveform[:, 1] if n_ch >= 2 else None

    # Re-derive the Motzoi DRAG quadrature exactly as the optimizer / cross_check do.
    def _ddt(x):
        xp = np.concatenate([x[:1], x, x[-1:]])
        return (xp[2:] - xp[:-2]) / (2.0 * dt_ns)
    if use_drag:
        vc = -_ddt(uc) * omega / ops["alpha_c"]
        vt = (-_ddt(ut) * omega / ops["alpha_t"]) if ut is not None else None
    else:
        vc = vt = None

    def _slc(t):
        return min(int(t / dt_ns), n_samples - 1)

    # Static (RWA) control coefficients -- piecewise-constant per slice.
    def c_uc(t, args=None): return omega * float(uc[_slc(t)])
    H_rwa = [ops["H_drift"], [ops["X_c"], c_uc]]
    if ut is not None:
        def c_ut(t, args=None): return omega * float(ut[_slc(t)])
        H_rwa.append([ops["X_t"], c_ut])
    if vc is not None:
        def c_vc(t, args=None): return float(vc[_slc(t)])
        H_rwa.append([ops["Y_c"], c_vc])
        if vt is not None:
            def c_vt(t, args=None): return float(vt[_slc(t)])
            H_rwa.append([ops["Y_t"], c_vt])

    # Counter-rotating partner: ec(t) e^{-2i wd t} a + h.c., ec = Omega*u + i v.
    def _with_cr(H):
        H = list(H)
        def c_ac(t, args=None):
            ec = omega * float(uc[_slc(t)]) + 1j * (float(vc[_slc(t)]) if vc is not None else 0.0)
            return ec * np.exp(-1j * 2.0 * wd * t)
        def c_acd(t, args=None):
            ec = omega * float(uc[_slc(t)]) + 1j * (float(vc[_slc(t)]) if vc is not None else 0.0)
            return np.conj(ec) * np.exp(1j * 2.0 * wd * t)
        H += [[ops["a_c"], c_ac], [ops["a_c"].dag(), c_acd]]
        if ut is not None:
            def c_at(t, args=None):
                et = omega * float(ut[_slc(t)]) + 1j * (float(vt[_slc(t)]) if vt is not None else 0.0)
                return et * np.exp(-1j * 2.0 * wd * t)
            def c_atd(t, args=None):
                et = omega * float(ut[_slc(t)]) + 1j * (float(vt[_slc(t)]) if vt is not None else 0.0)
                return np.conj(et) * np.exp(1j * 2.0 * wd * t)
            H += [[ops["a_t"], c_at], [ops["a_t"].dag(), c_atd]]
        return H

    opts = {"max_step": max_step_ns, "nsteps": 500000, "atol": 1e-10, "rtol": 1e-8}

    # vz-framed ZX(pi/2) target (post-gate single-qubit Z), as in _cross_check_cr.
    U4 = _zx90_target()
    cbits = np.array([0.0, 0.0, 1.0, 1.0]); tbits = np.array([0.0, 1.0, 0.0, 1.0])
    U4 = np.diag(np.exp(1j * (vz[0] * cbits + vz[1] * tbits))) @ U4

    def _f_proc(H):
        U = qt.propagator(H, T, [], options=opts)
        Uf = U.full() if hasattr(U, "full") else np.asarray(U)
        F_acc = 0.0 + 0.0j
        for ii in range(4):
            for jj in range(4):
                M0 = np.zeros((dim, dim), dtype=complex); M0[comp[ii], comp[jj]] = 1.0
                M = (Uf @ M0 @ Uf.conj().T)[np.ix_(comp, comp)]
                F_acc += (U4.conj().T @ M @ U4)[ii, jj]
        return max(0.0, min(1.0, float(np.real(F_acc / 16.0))))

    f_rwa = _f_proc(H_rwa)
    f_cr = _f_proc(_with_cr(H_rwa))
    return {
        "f_proc_rwa": f_rwa,
        "f_proc_counter_rot": f_cr,
        "delta_r_counter_rot": f_rwa - f_cr,
        "omega_d_ghz": omega_d_ghz,
    }


def cr_collision_cross_check(profile: dict, waveform: np.ndarray, vz,
                             dt_ns: float, detuning_mhz: float, j_mhz: float,
                             couples_to: str = "control", *,
                             use_drag: bool = True) -> dict:
    """Full 3-transmon QuTiP F_proc for the CR gate PAIR next to a NEAR-RESONANT,
    EVOLVING spectator coupled to a gate qubit by transverse exchange J(a_g+ a_s +
    a_g a_s+), the spectator starting idle in |0>.

    The independent check behind CrossResonanceZXOptimizer.resonant_collision_fidelity.
    Unlike the frozen off-resonant neighbour of spectator_cross_check (a static ZZ),
    here the spectator is dynamical: as its detuning from the coupled gate qubit -> 0
    the exchange becomes resonant and population coherently SWAPS into it. Matched
    piecewise-constant (order-1) master-equation stepping lifted to (n_levels**3)-D
    (64-D at the default n_levels=4), DRAG re-derived exactly as the optimizer does,
    vz-framed ZX(pi/2) target; no spectator decoherence. A machine-precision match
    with the PyTorch resonant_collision_fidelity confirms the collision dynamics are
    faithful in both code paths.

    couples_to: "control" or "target" (which gate qubit the spectator neighbours).
    detuning_mhz: spectator detuning from that qubit (0 = exact collision). Returns
    {"f_proc","spectator_leakage"} (leakage = population swapped into the spectator).
    """
    ops = _build_qutip_cr_ops(profile)
    nl = int(profile.get("n_levels", 3))
    dim = nl * nl
    comp = [0, 1, nl, nl + 1]
    omega = ops["omega_max"]
    n_samples, n_ch = waveform.shape
    to_control = str(couples_to).lower() in ("control", "c", "0")
    vzf = (float(vz[0]), float(vz[1]))

    a = qt.destroy(nl)
    I3 = qt.qeye(nl)
    delta_c = (profile["freq_ghz_control"] - profile["freq_ghz_target"]) * 2.0 * math.pi
    anh = (profile["anharm_ghz_control"] if to_control
           else profile["anharm_ghz_target"]) * 2.0 * math.pi
    base = delta_c if to_control else 0.0
    dn = 2.0 * math.pi * (float(detuning_mhz) / 1000.0)
    jx = 2.0 * math.pi * (float(j_mhz) / 1000.0)

    a_s = qt.tensor(qt.qeye(nl), qt.qeye(nl), a)
    n_s = a_s.dag() * a_s
    H_s = (base + dn) * n_s + 0.5 * anh * (a_s.dag() * a_s.dag() * a_s * a_s)
    a_g = (qt.tensor(a, qt.qeye(nl), qt.qeye(nl)) if to_control
           else qt.tensor(qt.qeye(nl), a, qt.qeye(nl)))
    H_x = jx * (a_g.dag() * a_s + a_g * a_s.dag())

    def lift(op9):
        return qt.tensor(op9, I3)

    uc = waveform[:, 0]
    ut = waveform[:, 1] if n_ch >= 2 else None

    def _ddt(x):
        xp = np.concatenate([x[:1], x, x[-1:]])
        return (xp[2:] - xp[:-2]) / (2.0 * dt_ns)
    if use_drag:
        vc = -_ddt(uc) * omega / ops["alpha_c"]
        vt = (-_ddt(ut) * omega / ops["alpha_t"]) if ut is not None else None
    else:
        vc = vt = None

    L_ops = [lift(L) for L in ops["L_ops"]]
    L_anti = sum(L.dag() * L for L in L_ops) * 0.5

    U_list = []
    for i in range(n_samples):
        H9 = ops["H_drift"] + (omega * float(uc[i])) * ops["X_c"]
        if ut is not None:
            H9 = H9 + (omega * float(ut[i])) * ops["X_t"]
        if vc is not None:
            H9 = H9 + float(vc[i]) * ops["Y_c"]
            if vt is not None:
                H9 = H9 + float(vt[i]) * ops["Y_t"]
        U_list.append((-1j * (lift(H9) + H_s + H_x) * dt_ns).expm())

    U4 = _zx90_target()
    cbits = np.array([0.0, 0.0, 1.0, 1.0])
    tbits = np.array([0.0, 1.0, 0.0, 1.0])
    U4 = np.diag(np.exp(1j * (vzf[0] * cbits + vzf[1] * tbits))) @ U4
    ket0 = qt.basis(nl, 0)
    spec0 = ket0 * ket0.dag()
    F_acc = 0.0 + 0.0j
    leak_acc = 0.0
    for ii in range(4):
        for jj in range(4):
            M0 = np.zeros((dim, dim), dtype=complex)
            M0[comp[ii], comp[jj]] = 1.0
            rho = qt.tensor(qt.Qobj(M0, dims=[[nl, nl], [nl, nl]]), spec0)
            for U in U_list:
                rho = U * rho * U.dag()
                jump = sum(L * rho * L.dag() for L in L_ops)
                rho = rho + dt_ns * (jump - L_anti * rho - rho * L_anti)
            M = rho.ptrace([0, 1]).full()[np.ix_(comp, comp)]
            F_acc += (U4.conj().T @ M @ U4)[ii, jj]
            if ii == jj:
                leak_acc += 1.0 - float(rho.ptrace([2]).full()[0, 0].real)
    return {"f_proc": max(0.0, min(1.0, float(np.real(F_acc / 16.0)))),
            "spectator_leakage": float(leak_acc / 4.0)}


def spectator_cross_check_3transmon(profile: dict, waveform: np.ndarray,
                                    target_gate: str, dt_ns: float,
                                    zeta_mhz: float, couples_to: int = 1) -> float:
    """Full 3-transmon (27-D) QuTiP F_proc for the gate PAIR with one extra
    spectator qutrit coupled to a gate qubit by an always-on ZZ (zeta_mhz), the
    spectator frozen idle in |1>.

    The independent check behind ParametricCZOptimizer.spectator_fidelity: it
    simulates the genuine three-body system (no spectator drive, no spectator
    decoherence) and partial-traces the spectator out, so a match with the 9-D
    effective detuning model (``qutip_f_proc(..., detuning_offset=zeta on the
    coupled qubit)``) confirms the always-on-ZZ -> static-detuning reduction is
    faithful rather than assumed. Same matched piecewise-constant master-equation
    stepping as the rest of this module, lifted to the 27-D tensor space.

    couples_to: 1 or 2 -- which gate qubit the spectator's ZZ acts on.
    Returns the gate pair's exact F_proc over the computational subspace.
    """
    prof = dict(DEFAULT_QUTIP_PROFILE)
    if isinstance(profile, dict):
        prof.update({k: profile[k] for k in prof if k in profile})
    u = np.asarray(waveform, dtype=float)
    n_samples, n_ch = u.shape
    nl = int(prof.get("n_levels", 3))
    dim = nl * nl
    ops = _build_qutip_ops(prof)

    a = qt.destroy(nl)
    n = a.dag() * a
    I3 = qt.qeye(nl)
    zeta = 2.0 * math.pi * (float(zeta_mhz) / 1000.0)           # rad/ns
    # Spectator ZZ on the chosen gate qubit: zeta * n_gateq (x) n_spectator
    # (nl**3-D tensor space; 27-D for the default nl=3).
    if int(couples_to) == 1:
        H_zz = zeta * qt.tensor(n, I3, n)
    else:
        H_zz = zeta * qt.tensor(I3, n, n)

    def lift(op9):
        return qt.tensor(op9, I3)                               # (nl**2)-D (x) I_spectator

    u_smooth = 2.0 * u - 1.0
    omega = ops["omega_max"]
    g = ops["g_max"]
    L_ops = [lift(L) for L in ops["L_ops"]]
    L_anti = sum(L.dag() * L for L in L_ops) * 0.5

    U_list = []
    for i in range(n_samples):
        H9 = ops["H_drift"]
        H9 = H9 + (u_smooth[i, 0] * omega) * ops["X1"]
        H9 = H9 + (u_smooth[i, 1] * omega) * ops["X2"]
        if n_ch == 3:
            H9 = H9 + (u_smooth[i, 2] * g) * ops["Cx"]
        else:
            phi = math.pi * u_smooth[i, 3]
            H9 = H9 + (u_smooth[i, 2] * math.cos(phi) * g) * ops["Cx"]
            H9 = H9 + (u_smooth[i, 2] * math.sin(phi) * g) * ops["Cy"]
        if n_ch == 6:
            stark = 2 * math.pi * (20.0 / 1000.0)
            H9 = H9 + (u_smooth[i, 4] * stark) * ops["N1"]
            H9 = H9 + (u_smooth[i, 5] * stark) * ops["N2"]
        U_list.append((-1j * (lift(H9) + H_zz) * dt_ns).expm())

    U4 = _target_unitary(target_gate)
    comp = [0, 1, nl, nl + 1]
    ket1 = qt.basis(nl, 1)
    spec0 = ket1 * ket1.dag()                                   # |1><1|_S, idle
    F_acc = 0.0 + 0.0j
    for ii in range(4):
        for jj in range(4):
            M0 = np.zeros((dim, dim), dtype=complex)
            M0[comp[ii], comp[jj]] = 1.0
            rho = qt.tensor(qt.Qobj(M0, dims=[[nl, nl], [nl, nl]]), spec0)
            for i in range(n_samples):
                U = U_list[i]
                rho = U * rho * U.dag()
                jump = sum(L * rho * L.dag() for L in L_ops)
                rho = rho + dt_ns * (jump - L_anti * rho - rho * L_anti)
            M = rho.ptrace([0, 1]).full()[np.ix_(comp, comp)]   # trace out spectator
            F_acc += (U4.conj().T @ M @ U4)[ii, jj]
    return max(0.0, min(1.0, float(np.real(F_acc / 16.0))))


def spectator_cross_check_multi(profile: dict, waveform: np.ndarray,
                                target_gate: str, dt_ns: float,
                                neighbours) -> float:
    """Full multi-transmon QuTiP F_proc for the gate PAIR with N frozen 2-level
    spectators, each coupled to a gate qubit by an always-on ZZ, all idle in |1>.

    The independent check behind ``multi_spectator_fidelity``'s additive-detuning
    reduction: several frozen, off-resonant neighbours on a gate qubit should be
    exactly one static detuning equal to the SUM of their ZZ rates (diagonal shifts
    commute and add). Comparing this explicit ((n_levels**2) * 2**N)-D simulation to
    the effective model with the summed detuning confirms additivity rather than
    assuming it. Spectators are 2-level: idle and off-resonant, they never leave
    {|0>,|1>}, so |2> would be unpopulated -- this is exact and keeps the lifted space
    small (36-D for two spectators at the default n_levels=3).

    neighbours: list of ``(gate_qubit, zeta_mhz)`` with gate_qubit in {0, 1}; every
        spectator is frozen idle in |1>. Returns the gate pair's exact F_proc.
    """
    prof = dict(DEFAULT_QUTIP_PROFILE)
    if isinstance(profile, dict):
        prof.update({k: profile[k] for k in prof if k in profile})
    u = np.asarray(waveform, dtype=float)
    n_samples, n_ch = u.shape
    nl = int(prof.get("n_levels", 3))
    dim = nl * nl
    ops = _build_qutip_ops(prof)

    a = qt.destroy(nl)
    n = a.dag() * a
    I3 = qt.qeye(nl)
    I2 = qt.qeye(2)
    n2 = qt.num(2)                                  # |1><1| on a 2-level spectator
    nb = [(int(q), float(z)) for (q, z) in neighbours]
    N = len(nb)
    n_g = [qt.tensor(n, I3), qt.tensor(I3, n)]      # gate-qubit number ops (9-D)

    def lift(op9):
        return qt.tensor(op9, *([I2] * N)) if N else op9

    H_zz = 0
    for k, (q, z) in enumerate(nb):
        zeta = 2.0 * math.pi * (z / 1000.0)
        spec_ops = [n2 if j == k else I2 for j in range(N)]
        H_zz = H_zz + zeta * qt.tensor(n_g[q], *spec_ops)

    u_smooth = 2.0 * u - 1.0
    omega = ops["omega_max"]
    g = ops["g_max"]
    L_ops = [lift(L) for L in ops["L_ops"]]
    L_anti = sum(L.dag() * L for L in L_ops) * 0.5

    U_list = []
    for i in range(n_samples):
        H9 = ops["H_drift"]
        H9 = H9 + (u_smooth[i, 0] * omega) * ops["X1"]
        H9 = H9 + (u_smooth[i, 1] * omega) * ops["X2"]
        if n_ch == 3:
            H9 = H9 + (u_smooth[i, 2] * g) * ops["Cx"]
        else:
            phi = math.pi * u_smooth[i, 3]
            H9 = H9 + (u_smooth[i, 2] * math.cos(phi) * g) * ops["Cx"]
            H9 = H9 + (u_smooth[i, 2] * math.sin(phi) * g) * ops["Cy"]
        if n_ch == 6:
            stark = 2 * math.pi * (20.0 / 1000.0)
            H9 = H9 + (u_smooth[i, 4] * stark) * ops["N1"]
            H9 = H9 + (u_smooth[i, 5] * stark) * ops["N2"]
        H = lift(H9) + (H_zz if N else 0)
        U_list.append((-1j * H * dt_ns).expm())

    U4 = _target_unitary(target_gate)
    comp = [0, 1, nl, nl + 1]
    ket1 = qt.basis(2, 1)
    spec1 = ket1 * ket1.dag()                       # each spectator frozen idle in |1>
    F_acc = 0.0 + 0.0j
    for ii in range(4):
        for jj in range(4):
            M0 = np.zeros((dim, dim), dtype=complex)
            M0[comp[ii], comp[jj]] = 1.0
            rho_g = qt.Qobj(M0, dims=[[nl, nl], [nl, nl]])
            rho = qt.tensor(rho_g, *([spec1] * N)) if N else rho_g
            for i in range(n_samples):
                U = U_list[i]
                rho = U * rho * U.dag()
                jump = sum(L * rho * L.dag() for L in L_ops)
                rho = rho + dt_ns * (jump - L_anti * rho - rho * L_anti)
            red = rho.ptrace([0, 1]) if N else rho   # trace out all spectators
            M = red.full()[np.ix_(comp, comp)]
            F_acc += (U4.conj().T @ M @ U4)[ii, jj]
    return max(0.0, min(1.0, float(np.real(F_acc / 16.0))))


def collision_cross_check(profile: dict, waveform: np.ndarray,
                          target_gate: str, dt_ns: float,
                          detuning_mhz: float, j_mhz: float,
                          couples_to: int = 2) -> dict:
    """Full 3-transmon (27-D) QuTiP F_proc for the gate PAIR next to a
    NEAR-RESONANT, EVOLVING spectator coupled to a gate qubit by a transverse
    exchange J(a_g+ a_s + a_g a_s+), the spectator starting idle in |0>.

    The independent check behind ParametricCZOptimizer.resonant_collision_fidelity.
    Unlike spectator_cross_check_3transmon -- where the off-resonant neighbour is
    FROZEN and reduces to a static ZZ -- here the spectator is dynamical: as its
    detuning from the coupled gate qubit -> 0 the exchange becomes resonant and
    population coherently SWAPS into it during the gate, which no static detuning
    can capture. A match with the PyTorch resonant_collision_fidelity therefore
    confirms that the (n_levels**3)-D collision dynamics are implemented faithfully
    in BOTH code paths. Same matched piecewise-constant (order-1) master-equation
    stepping as the rest of this module, lifted to the (n_levels**3)-D space; no
    spectator decoherence (the conservative coherent-spectator convention).

    detuning_mhz: spectator detuning FROM the coupled gate qubit (0 = exact
        collision). j_mhz: transverse exchange rate. couples_to: 1 (q1) or 2 (q2).
    Returns {"f_proc", "spectator_leakage"} -- the latter the population that
    swapped into the spectator (1 - P(spectator |0>), averaged over the 4
    computational inputs), the smoking gun of a collision.
    """
    prof = dict(DEFAULT_QUTIP_PROFILE)
    if isinstance(profile, dict):
        prof.update({k: profile[k] for k in prof if k in profile})
    u = np.asarray(waveform, dtype=float)
    n_samples, n_ch = u.shape
    nl = int(prof.get("n_levels", 3))
    dim = nl * nl
    ops = _build_qutip_ops(prof)

    a = qt.destroy(nl)
    I3 = qt.qeye(nl)
    delta = (prof["freq_ghz_q2"] - prof["freq_ghz_q1"]) * 2.0 * math.pi
    anh = (prof["anharm_ghz_q2"] if int(couples_to) == 2
           else prof["anharm_ghz_q1"]) * 2.0 * math.pi
    base = delta if int(couples_to) == 2 else 0.0      # coupled-qubit freq, q1 frame
    dn = 2.0 * math.pi * (float(detuning_mhz) / 1000.0)
    jx = 2.0 * math.pi * (float(j_mhz) / 1000.0)

    a_s = qt.tensor(qt.qeye(nl), qt.qeye(nl), a)        # spectator annihilation
    n_s = a_s.dag() * a_s
    H_s = (base + dn) * n_s + 0.5 * anh * (a_s.dag() * a_s.dag() * a_s * a_s)
    a_g = (qt.tensor(qt.qeye(nl), a, qt.qeye(nl)) if int(couples_to) == 2
           else qt.tensor(a, qt.qeye(nl), qt.qeye(nl)))
    H_x = jx * (a_g.dag() * a_s + a_g * a_s.dag())

    def lift(op9):
        return qt.tensor(op9, I3)

    u_smooth = 2.0 * u - 1.0
    omega = ops["omega_max"]
    g = ops["g_max"]
    L_ops = [lift(L) for L in ops["L_ops"]]
    L_anti = sum(L.dag() * L for L in L_ops) * 0.5

    U_list = []
    for i in range(n_samples):
        H9 = ops["H_drift"]
        H9 = H9 + (u_smooth[i, 0] * omega) * ops["X1"]
        H9 = H9 + (u_smooth[i, 1] * omega) * ops["X2"]
        if n_ch == 3:
            H9 = H9 + (u_smooth[i, 2] * g) * ops["Cx"]
        else:
            phi = math.pi * u_smooth[i, 3]
            H9 = H9 + (u_smooth[i, 2] * math.cos(phi) * g) * ops["Cx"]
            H9 = H9 + (u_smooth[i, 2] * math.sin(phi) * g) * ops["Cy"]
        if n_ch == 6:
            stark = 2 * math.pi * (20.0 / 1000.0)
            H9 = H9 + (u_smooth[i, 4] * stark) * ops["N1"]
            H9 = H9 + (u_smooth[i, 5] * stark) * ops["N2"]
        U_list.append((-1j * (lift(H9) + H_s + H_x) * dt_ns).expm())

    U4 = _target_unitary(target_gate)
    comp = [0, 1, nl, nl + 1]
    ket0 = qt.basis(nl, 0)
    spec0 = ket0 * ket0.dag()                           # spectator idle in |0>
    F_acc = 0.0 + 0.0j
    leak_acc = 0.0
    for ii in range(4):
        for jj in range(4):
            M0 = np.zeros((dim, dim), dtype=complex)
            M0[comp[ii], comp[jj]] = 1.0
            rho = qt.tensor(qt.Qobj(M0, dims=[[nl, nl], [nl, nl]]), spec0)
            for U in U_list:
                rho = U * rho * U.dag()
                jump = sum(L * rho * L.dag() for L in L_ops)
                rho = rho + dt_ns * (jump - L_anti * rho - rho * L_anti)
            M = rho.ptrace([0, 1]).full()[np.ix_(comp, comp)]   # trace out spectator
            F_acc += (U4.conj().T @ M @ U4)[ii, jj]
            if ii == jj:                                # diagonal inputs -> populations
                leak_acc += 1.0 - float(rho.ptrace([2]).full()[0, 0].real)
    return {"f_proc": max(0.0, min(1.0, float(np.real(F_acc / 16.0)))),
            "spectator_leakage": float(leak_acc / 4.0)}


def tls_defect_cross_check(profile: dict, waveform: np.ndarray,
                           target_gate: str, dt_ns: float,
                           detuning_mhz: float, g_mhz: float,
                           t1_tls_ns: float, couples_to: int = 1) -> dict:
    """Full (nl**2 * 2)-D QuTiP F_proc for the gate PAIR next to one explicit LOSSY
    two-level-system defect: a transverse exchange g(a_g+ sigma_- + a_g sigma_+) to a
    gate qubit, the defect's own frequency, and -- the load-bearing ingredient -- its
    OWN T1 collapse operator. The independent check behind
    ParametricCZOptimizer.tls_defect_fidelity.

    Unlike collision_cross_check (a COHERENT evolving transmon), the defect here is a
    2-level mode that also DECAYS, so a match with the PyTorch path confirms both the
    coherent qubit-TLS exchange AND the TLS dissipator are implemented faithfully in two
    independent code paths. Same matched order-1 master-equation stepping as the rest of
    this module. Returns {"f_proc", "tls_excitation"} (the latter P(TLS in |1>) averaged
    over the 4 computational inputs).
    """
    prof = dict(DEFAULT_QUTIP_PROFILE)
    if isinstance(profile, dict):
        prof.update({k: profile[k] for k in prof if k in profile})
    u = np.asarray(waveform, dtype=float)
    n_samples, n_ch = u.shape
    nl = int(prof.get("n_levels", 3))
    dim = nl * nl
    ops = _build_qutip_ops(prof)

    a = qt.destroy(nl)
    a2 = qt.destroy(2)
    I2 = qt.qeye(2)
    delta = (prof["freq_ghz_q2"] - prof["freq_ghz_q1"]) * 2.0 * math.pi
    base = delta if int(couples_to) == 2 else 0.0      # coupled-qubit freq, q1 frame
    dn = 2.0 * math.pi * (float(detuning_mhz) / 1000.0)
    gg = 2.0 * math.pi * (float(g_mhz) / 1000.0)

    a_t = qt.tensor(qt.qeye(nl), qt.qeye(nl), a2)       # 2-level defect
    n_t = a_t.dag() * a_t
    H_t = (base + dn) * n_t
    a_g = (qt.tensor(qt.qeye(nl), a, I2) if int(couples_to) == 2
           else qt.tensor(a, qt.qeye(nl), I2))
    H_x = gg * (a_g.dag() * a_t + a_g * a_t.dag())

    def lift(op9):
        return qt.tensor(op9, I2)

    u_smooth = 2.0 * u - 1.0
    omega = ops["omega_max"]
    g = ops["g_max"]
    L_ops = [lift(L) for L in ops["L_ops"]]
    L_ops.append(math.sqrt(1.0 / max(float(t1_tls_ns), 1e-9)) * a_t)   # TLS T1
    L_anti = sum(L.dag() * L for L in L_ops) * 0.5

    U_list = []
    for i in range(n_samples):
        H9 = ops["H_drift"]
        H9 = H9 + (u_smooth[i, 0] * omega) * ops["X1"]
        H9 = H9 + (u_smooth[i, 1] * omega) * ops["X2"]
        if n_ch == 3:
            H9 = H9 + (u_smooth[i, 2] * g) * ops["Cx"]
        else:
            phi = math.pi * u_smooth[i, 3]
            H9 = H9 + (u_smooth[i, 2] * math.cos(phi) * g) * ops["Cx"]
            H9 = H9 + (u_smooth[i, 2] * math.sin(phi) * g) * ops["Cy"]
        if n_ch == 6:
            stark = 2 * math.pi * (20.0 / 1000.0)
            H9 = H9 + (u_smooth[i, 4] * stark) * ops["N1"]
            H9 = H9 + (u_smooth[i, 5] * stark) * ops["N2"]
        U_list.append((-1j * (lift(H9) + H_t + H_x) * dt_ns).expm())

    U4 = _target_unitary(target_gate)
    comp = [0, 1, nl, nl + 1]
    tls0 = qt.basis(2, 0) * qt.basis(2, 0).dag()        # defect idle in |0>
    F_acc = 0.0 + 0.0j
    exc_acc = 0.0
    for ii in range(4):
        for jj in range(4):
            M0 = np.zeros((dim, dim), dtype=complex)
            M0[comp[ii], comp[jj]] = 1.0
            rho = qt.tensor(qt.Qobj(M0, dims=[[nl, nl], [nl, nl]]), tls0)
            for U in U_list:
                rho = U * rho * U.dag()
                jump = sum(L * rho * L.dag() for L in L_ops)
                rho = rho + dt_ns * (jump - L_anti * rho - rho * L_anti)
            M = rho.ptrace([0, 1]).full()[np.ix_(comp, comp)]   # trace out the TLS
            F_acc += (U4.conj().T @ M @ U4)[ii, jj]
            if ii == jj:
                exc_acc += float(rho.ptrace([2]).full()[1, 1].real)
    return {"f_proc": max(0.0, min(1.0, float(np.real(F_acc / 16.0)))),
            "tls_excitation": float(exc_acc / 4.0)}


def coupler_elimination_cross_check(freq_ghz: float = 4.8, anharm_ghz: float = -0.30,
                                    coupler_detuning_mhz: float = 1200.0,
                                    gc_mhz: float = 80.0, n_levels: int = 3,
                                    n_times: int = 500) -> dict:
    """Validate the Schrieffer-Wolff coupler elimination behind the parametric model.

    The parametric optimizer eliminates the tunable coupler under Schrieffer-Wolff,
    replacing the physical transmon-COUPLER-transmon system with a direct effective
    exchange ``J*(a1+ a2 + a1 a2+)`` (the optimizer's COUPLING operator). This builds
    the explicit 3-body system (two transmons + a dispersively-detuned coupler mode,
    each ``n_levels``) and the 2-body effective model with
    ``J = (gc^2/2)(1/D1 + 1/D2)``, evolves the single-excitation swap
    ``|100> -> |010>`` in both, and reports:

      * ``J_eff_mhz``       -- the effective exchange the coupler mediates,
      * ``max_coupler_pop`` -- peak population of the coupler (the eliminated DOF), and
      * ``max_traj_diff``   -- peak swap-trajectory difference, 3-body vs 2-body.

    Both residuals are ``O((gc/Delta)^2)`` -- the elimination's small parameter -- so
    they vanish quadratically as the coupler is detuned (``coupler_cross_check`` at
    Delta and 2*Delta shows the error QUARTERS), the signature that the reduction is
    correct to leading order rather than assumed. The exchange validated here is the
    same operator the CZ uses for its |11>-|02> resonance; the flux *activation* that
    turns it on (modulating the coupler) is a standard parametric-drive result on top,
    with the activated rate entering the optimizer as ``g_max``.

    Returns {"J_eff_mhz","max_coupler_pop","max_traj_diff","sw_param","detuning_mhz"}.
    """
    two_pi = 2.0 * math.pi
    nl = int(n_levels)
    w = two_pi * freq_ghz
    anh = two_pi * anharm_ghz
    wc = w + two_pi * (coupler_detuning_mhz / 1000.0)
    gc = two_pi * (gc_mhz / 1000.0)
    D1 = D2 = w - wc
    J = (gc ** 2 / 2.0) * (1.0 / D1 + 1.0 / D2)

    a = qt.destroy(nl)
    I = qt.qeye(nl)

    def kerr(x):
        return 0.5 * anh * (x.dag() * x.dag() * x * x)

    def op3(o, s):
        return qt.tensor(*[o if i == s else I for i in range(3)])

    a1, ac, a2 = op3(a, 0), op3(a, 1), op3(a, 2)
    H3 = (w * a1.dag() * a1 + wc * ac.dag() * ac + w * a2.dag() * a2
          + kerr(a1) + kerr(ac) + kerr(a2)
          + gc * (a1.dag() * ac + a1 * ac.dag())
          + gc * (a2.dag() * ac + a2 * ac.dag()))

    b1 = qt.tensor(a, I)
    b2 = qt.tensor(I, a)
    H2 = (w * b1.dag() * b1 + w * b2.dag() * b2 + kerr(b1) + kerr(b2)
          + J * (b1.dag() * b2 + b1 * b2.dag()))

    T = math.pi / (2.0 * abs(J))
    tl = np.linspace(0.0, 2.0 * T, int(n_times))
    e3 = qt.sesolve(H3, qt.tensor(qt.basis(nl, 1), qt.basis(nl, 0), qt.basis(nl, 0)),
                    tl, e_ops=[a2.dag() * a2, ac.dag() * ac])
    e2 = qt.sesolve(H2, qt.tensor(qt.basis(nl, 1), qt.basis(nl, 0)),
                    tl, e_ops=[b2.dag() * b2])
    return {
        "J_eff_mhz": abs(J) / two_pi * 1000.0,
        "max_coupler_pop": float(np.max(e3.expect[1])),
        "max_traj_diff": float(np.max(np.abs(e3.expect[0] - e2.expect[0]))),
        "sw_param": float((gc / D1) ** 2),
        "detuning_mhz": float(coupler_detuning_mhz),
    }


def multiqubit_cross_check(optimizer, waveform, dt_ns: float = 1.0) -> dict:
    """Independent QuTiP cross-check of a ``MultiQubitOptimizer`` (open system).

    Rebuilds the N-transmon Hamiltonian + Lindblad operators from scratch in QuTiP
    (a different library), evolves the saved smoothed [0,1] envelope through the
    SAME matched piecewise-constant master-equation scheme the other cross-checks
    use, and recomputes F_proc over the 2**N computational subspace. Returns
    ``{"f_qutip", "f_torch", "delta"}``; agreement to ~1e-6 confirms the
    general-N optimizer realizes the model it claims. Validates the default
    (``use_drag=False``) open-system optimizer; the dynamics, operators and
    integrator are rebuilt independently (only the target/index *definitions* are
    shared, exactly as ``_target_unitary`` is in the pair cross-check).
    """
    prof = optimizer.profile
    N, nl = prof.n_qubits, prof.n_levels
    a1 = qt.destroy(nl); ad1 = a1.dag(); n1 = ad1 * a1; I = qt.qeye(nl)

    def emb(op, q):
        return qt.tensor([op if k == q else I for k in range(N)])

    a = [emb(a1, q) for q in range(N)]
    ad = [emb(ad1, q) for q in range(N)]
    nop = [emb(n1, q) for q in range(N)]

    f_ref = prof.f_ref_ghz
    H_drift = 0
    for q in range(N):
        delta = 2 * math.pi * (prof.freqs_ghz[q] - f_ref)
        alpha = 2 * math.pi * prof.anharm_mhz[q] / 1000.0
        H_drift = H_drift + delta * nop[q] + 0.5 * alpha * (ad[q] * ad[q] * a[q] * a[q])
    for (i, j) in optimizer.fixed_edges:
        g = 2 * math.pi * prof.couplings[(i, j)] / 1000.0
        H_drift = H_drift + g * (ad[i] * a[j] + a[i] * ad[j])

    Xop = {q: (a[q] + ad[q]) for q in optimizer.drive_qubits}
    Cop = {e: (ad[e[0]] * a[e[1]] + a[e[0]] * ad[e[1]]) for e in optimizer.tunable_edges}
    Fop = {q: nop[q] for q in getattr(optimizer, "freq_control_qubits", [])}

    L_ops = []
    for q in range(N):
        t1, t2 = prof.t1_ns[q], prof.t2_ns[q]
        rate_t1 = math.sqrt(1.0 / max(t1, 1.0))
        inv = 1.0 / max(t2, 1.0) - 1.0 / max(2.0 * t1, 1.0)
        rate_phi = math.sqrt(2.0 / max(1.0 / max(inv, 1e-9), 1e-9))
        L_ops += [rate_t1 * a[q], rate_phi * nop[q]]
    L_anti = sum(L.dag() * L for L in L_ops) * 0.5

    omega, gmax = optimizer.OMEGA_MAX, optimizer.G_MAX
    dmax = getattr(optimizer, "DELTA_MAX", 0.0)
    u = np.asarray(waveform, dtype=float)
    n_slices = u.shape[0]
    u_s = 2.0 * u - 1.0

    U_list = []
    for i in range(n_slices):
        H = H_drift
        ch = 0
        for q in optimizer.drive_qubits:
            H = H + (u_s[i, ch] * omega) * Xop[q]; ch += 1
        for e in optimizer.tunable_edges:
            H = H + (u_s[i, ch] * gmax) * Cop[e]; ch += 1
        for q in getattr(optimizer, "freq_control_qubits", []):
            H = H + (u_s[i, ch] * dmax) * Fop[q]; ch += 1
        U_list.append((-1j * H * dt_ns).expm())

    ci = optimizer._comp_idx.cpu().numpy()
    dc = optimizer._dcomp
    Utar = optimizer.u_target.cpu().numpy()
    dim = nl ** N
    dims = [[nl] * N, [nl] * N]
    F_acc = 0.0 + 0.0j
    for ii in range(dc):
        for jj in range(dc):
            M0 = np.zeros((dim, dim), dtype=complex)
            M0[ci[ii], ci[jj]] = 1.0
            rho = qt.Qobj(M0, dims=dims)
            for i in range(n_slices):
                U = U_list[i]
                rho = U * rho * U.dag()
                jump = sum(L * rho * L.dag() for L in L_ops)
                rho = rho + dt_ns * (jump - L_anti * rho - rho * L_anti)
            M = rho.full()[np.ix_(ci, ci)]
            F_acc += (Utar.conj().T @ M @ Utar)[ii, jj]
    f_q = max(0.0, min(1.0, float(np.real(F_acc / (dc * dc)))))
    f_t = optimizer.process_fidelity(waveform, dt_ns=dt_ns)
    return {"f_qutip": f_q, "f_torch": f_t, "delta": abs(f_q - f_t)}


def _mesolve_cli(pulse_json: Path):
    """Run the adaptive-solver (mesolve-ZOH) cross-check on a parametric-coupler pulse
    and print F_proc alongside the reported value. Complements the matched-scheme
    cross_check: a different integration METHOD, so it confirms the scheme is unbiased.
    """
    meta = json.loads(pulse_json.read_text())
    arch = str(meta.get("architecture", "parametric_cz")).lower()
    if arch not in ("parametric_cz",):
        print(f"  --mesolve currently supports parametric_cz pulses only (got {arch!r}).")
        return
    u = np.load(pulse_json.parent / Path(meta["pulse_npy"]).name)
    dt_ns = float(meta["pulse_dt_ns"])
    target_gate = str(meta.get("target_gate", "cz")).lower()
    profile = dict(DEFAULT_QUTIP_PROFILE)
    if isinstance(meta.get("profile"), dict):
        profile.update({k: meta["profile"][k] for k in profile if k in meta["profile"]})
    reported = float(meta.get("grape_f", 0.0))
    print("\n  QuTiP adaptive-solver cross-check (mesolve on ZOH staircase, "
          "interval-by-interval):")
    fp = mesolve_zoh_fproc(profile, u, target_gate, dt_ns, meta.get("line_response"))
    print(f"    F_proc (adaptive mesolve, exact 16-op Choi): {fp:.5f}")
    print(f"    delta_F = mesolve F_proc - reported F_proc = {fp - reported:+.2e}")
    print("    A different numerical METHOD agreeing here means the piecewise-constant")
    print("    scheme is unbiased, not merely self-consistent under dt-refinement.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulse", default=None,
                    help="Path to a pulse JSON written by examples/optimize_cz.py. "
                         "Default: ./cz_pulse.json.")
    ap.add_argument("--mesolve", action="store_true",
                    help="Also run the adaptive-solver (mesolve-ZOH) cross-check, a "
                         "different integration method that confirms the scheme is "
                         "unbiased (parametric_cz pulses).")
    args = ap.parse_args()

    if args.pulse:
        pulse = Path(args.pulse)
    else:
        pulse = Path("cz_pulse.json")
        if not pulse.exists():
            print("No --pulse given and ./cz_pulse.json not found.")
            print("Generate it first:  python examples/optimize_cz.py")
            print("then re-run:         python -m gradpulse.validate")
            sys.exit(1)

    print(f"  Cross-checking: {pulse}")
    print()
    cross_check(pulse)
    if args.mesolve:
        _mesolve_cli(pulse)


if __name__ == "__main__":
    main()
