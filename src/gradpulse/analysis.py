"""gradpulse.analysis - forward-pass diagnostics for ParametricCZOptimizer.

Post-optimization analysis methods -- error budget & channel unitarity, robustness
sweeps, slow/colored/filter-function dephasing models, spectator- and collision-ZZ
crosstalk, and dt-convergence -- factored out of ParametricCZOptimizer into a mixin so
parametric.py stays focused on the simulator + GRAPE core. Every method here runs
forward passes on an already-optimized pulse and reaches the model through the
optimizer's own attributes and methods via ``self``; none run inside the gradient loop.
Mixed into ParametricCZOptimizer (gradpulse.parametric); not instantiated on its own.

This module deliberately has no import-time dependency on parametric: the dependency is
one-way (parametric imports this mixin). ``DEVICE`` comes from the shared ``_device`` leaf
module (the same object parametric uses), so tensors built here land on the same physical
device as the optimizer's operators.
"""
from __future__ import annotations

import math

import numpy as np
import torch

try:
    from .diagnostics import channel_unitarity
    from ._device import DEVICE
except ImportError:  # pragma: no cover - direct-script execution
    from diagnostics import channel_unitarity
    from _device import DEVICE


class ParametricCZAnalysisMixin:
    """Forward-pass diagnostics mixed into :class:`~gradpulse.parametric.ParametricCZOptimizer`.

    Not meant to be instantiated directly; it assumes the host class has built the
    operator stack and metadata (``self._H_DRIFT``, ``self._comp_idx``, ``self.cdtype``,
    the ``simulate_*`` / ``_process_fidelity`` / ``_leakage`` methods, etc.) in ``__init__``.
    """
    def error_budget(self, u_stack, dt: float = 1.0) -> dict:
        """Decompose the gate infidelity into a control/leakage part and a
        decoherence floor, and report the channel unitarity.

        Runs two forward simulations of the SAME pulse:
          * full open system (diss_scale=1) -> F_avg and total infidelity r_total,
          * closed system (diss_scale=0, T1/T_phi off) -> r_control_leakage, the
            coherent + leakage error a better or longer pulse could remove.
        The decoherence floor set by T1/T_phi is
        r_decoherence = r_total - r_control_leakage (an approximate additive
        split; coherent, leakage and incoherent errors do not add exactly).

        The channel unitarity u (Wallman et al. 2015) gives an independent,
        RB-style view of the SAME split: the infidelity a purely-stochastic
        channel of this unitarity would have is r_incoherent = (d-1)/d
        (1 - sqrt(u)), and coherent_excess = r_total - r_incoherent is the part
        attributable to coherent control error. The ablation (r_decoherence) and
        the unitarity (r_incoherent) are independent estimates of the incoherent
        floor and should tell a consistent story; leakage is neither cleanly
        coherent nor unital, so exact agreement is not expected.

        All quantities are for the d=4 computational subspace. Returns a dict.
        """
        u = torch.as_tensor(u_stack, dtype=self.rdtype, device=DEVICE)
        if u.dim() == 2:
            u = u.unsqueeze(0)
        with torch.no_grad():
            choi_full = self.simulate_choi_batch(u, dt=dt, diss_scale=1.0)
            choi_closed = self.simulate_choi_batch(u, dt=dt, diss_scale=0.0)
            f_proc = float(self._process_fidelity(choi_full)[0])
            f_proc_closed = float(self._process_fidelity(choi_closed)[0])
            leak = float(self._leakage(choi_full)[0])
            S = self._comp_superop_from_choi(choi_full)
        d = 4.0
        f_avg = (d * f_proc + 1.0) / (d + 1.0)
        f_avg_closed = (d * f_proc_closed + 1.0) / (d + 1.0)
        r_total = 1.0 - f_avg
        r_control = 1.0 - f_avg_closed
        u_unit = channel_unitarity(S)
        r_incoherent = (d - 1.0) / d * (1.0 - math.sqrt(max(u_unit, 0.0)))
        return {
            "F_proc": f_proc, "F_avg": f_avg,
            "F_proc_closed": f_proc_closed,
            "r_total": r_total,
            "r_control_leakage": r_control,
            "r_decoherence": r_total - r_control,
            "leakage": leak,
            "unitarity": u_unit,
            "r_incoherent_unitarity": r_incoherent,
            "coherent_excess": r_total - r_incoherent,
        }

    # ---- Robustness / miscalibration sweep -------------------------------
    def robustness_sweep(self, u_stack, dt: float = 1.0,
                         amp_fracs=None, freq_mhz=None) -> dict:
        """Gate fidelity vs control miscalibration, the question any
        experimentalist asks first: how tight must calibration be?

        Pure forward simulations of the fixed (already-optimized) pulse -- no
        re-optimization -- perturbed along the two axes a tune-up actually
        calibrates and that this rotating-frame model represents faithfully:
          * amplitude: all drive/coupler amplitudes scaled by (1 + frac)
            (an AWG-gain / Rabi-calibration error), via OMEGA_MAX/G_MAX/STARK_MAX.
          * frequency: a static common-mode drive-frequency offset (MHz),
            applied through the detuning_offset primitive.
        Each axis's zero point reproduces the nominal fidelity exactly. Returns
        {axis: {"x", "unit", "F_proc", "F_avg"}} with F_avg=(d*F_proc+1)/(d+1).

        Notes / caveats:
          * Ranges. A static detuning is a coherent Z-rotation whose effect is
            *periodic* in (detuning x gate-duration) with period 1/T (~6.7 MHz
            for a 150 ns gate), so a sweep wider than ~1/(2T) aliases across
            phase-wraps and stops being monotonic -- keep freq_mhz inside it.
          * The frequency axis is the RAW (un-recalibrated) detuning sensitivity:
            a common-mode detuning is, on the computational subspace, almost
            exactly a local Z(x)Z rotation, which hardware removes by virtual-Z
            recalibration. So this curve is a conservative worst case; the
            practical tolerance after re-tuning virtual-Z is looser. Amplitude
            (gain) error is not Z-like and has no such caveat -- it is the clean,
            directly-quotable calibration tolerance.
          * A pulse-length/timing axis is deliberately omitted: in a rotating
            frame with a large static qubit detuning (here ~200 MHz on q2)
            scaling dt is dominated by the same uncompensated free-evolution
            phase, so it would overstate timing fragility for the same reason.
        """
        u = torch.as_tensor(u_stack, dtype=self.rdtype, device=DEVICE)
        if u.dim() == 2:
            u = u.unsqueeze(0)
        if amp_fracs is None:
            amp_fracs = np.linspace(-0.10, 0.10, 11)
        if freq_mhz is None:
            freq_mhz = np.linspace(-1.0, 1.0, 11)
        d = 4.0

        def _fid(choi):
            fp = float(self._process_fidelity(choi)[0])
            return fp, (d * fp + 1.0) / (d + 1.0)

        out = {}
        with torch.no_grad():
            # Amplitude: scale every physical drive amplitude together.
            xs, fps, fas = [], [], []
            o0, g0, s0 = self.OMEGA_MAX, self.G_MAX, self.STARK_MAX
            try:
                for fr in amp_fracs:
                    scale = 1.0 + float(fr)
                    self.OMEGA_MAX, self.G_MAX, self.STARK_MAX = \
                        o0 * scale, g0 * scale, s0 * scale
                    fp, fa = _fid(self.simulate_choi_batch(u, dt=dt))
                    xs.append(float(fr)); fps.append(fp); fas.append(fa)
            finally:
                self.OMEGA_MAX, self.G_MAX, self.STARK_MAX = o0, g0, s0
            out["amplitude"] = {"x": xs, "unit": "fractional amplitude error",
                                "F_proc": fps, "F_avg": fas}

            # Frequency: common-mode static drive detuning (MHz -> rad/ns).
            xs, fps, fas = [], [], []
            for mhz in freq_mhz:
                delta = 2.0 * math.pi * (float(mhz) / 1000.0)
                fp, fa = _fid(self.simulate_choi_batch(u, dt=dt, detuning_offset=delta))
                xs.append(float(mhz)); fps.append(fp); fas.append(fa)
            out["frequency"] = {"x": xs, "unit": "MHz drive detuning",
                                "F_proc": fps, "F_avg": fas}
        return out

    # ---- Quasi-static (1/f-like) dephasing -------------------------------
    def quasi_static_fidelity(self, u_stack, dt: float = 1.0, sigma_mhz: float = 0.3,
                              n_nodes: int = 5, include_decoherence: bool = True) -> dict:
        """Process fidelity under quasi-static (slow, 1/f-like) dephasing.

        Low-frequency flux / frequency noise is ~constant over one gate but
        random shot-to-shot, so it is NOT captured by the Markovian Lindblad
        T_phi (which models white noise). Here each qubit's frequency is offset
        by a static delta drawn from a zero-mean Gaussian of width sigma_mhz, the
        gate is evolved at that fixed offset, and the resulting CHANNELS are
        averaged over the noise distribution -- the correct incoherent shot-to-
        shot mixture -- before computing the fidelity. The Gaussian average uses
        deterministic Gauss-Hermite quadrature (no RNG, fully reproducible),
        tensored over the two qubits (n_nodes^2 channel evaluations).

        sigma_mhz: per-qubit quasi-static frequency std dev (MHz).
        n_nodes:   Gauss-Hermite nodes per qubit (5 integrates a Gaussian well).
        include_decoherence: keep the Markovian T1/T_phi too (diss_scale=1);
            False isolates the quasi-static contribution (diss_scale=0).
        Returns {"F_proc", "F_avg", "F_proc_nominal", "sigma_mhz", "n_evals"}.
        """
        u = torch.as_tensor(u_stack, dtype=self.rdtype, device=DEVICE)
        if u.dim() == 2:
            u = u.unsqueeze(0)
        diss = 1.0 if include_decoherence else 0.0
        d = 4.0
        # Probabilists' Gauss-Hermite: nodes/weights for E_{N(0,1)}[f] with
        # normalized weights p_i = w_i / sqrt(2*pi) (sum to 1).
        t_nodes, w = np.polynomial.hermite_e.hermegauss(int(n_nodes))
        p = w / math.sqrt(2.0 * math.pi)
        scale = 2.0 * math.pi * (float(sigma_mhz) / 1000.0)     # rad/ns per unit t
        with torch.no_grad():
            f_nominal = float(self._process_fidelity(
                self.simulate_choi_batch(u, dt=dt, diss_scale=diss))[0])
            avg_choi = None
            for ti, pi in zip(t_nodes, p):
                for tj, pj in zip(t_nodes, p):
                    choi = self.simulate_choi_batch(
                        u, dt=dt, diss_scale=diss,
                        detuning_offset=(scale * float(ti), scale * float(tj)))
                    term = (pi * pj) * choi
                    avg_choi = term if avg_choi is None else avg_choi + term
            f_proc = float(self._process_fidelity(avg_choi)[0])
        return {
            "F_proc": f_proc, "F_avg": (d * f_proc + 1.0) / (d + 1.0),
            "F_proc_nominal": f_nominal, "sigma_mhz": float(sigma_mhz),
            "n_evals": int(n_nodes) ** 2,
        }

    # ---- Colored (1/f^alpha) dephasing across the full band --------------
    def colored_noise_fidelity(self, u_stack, dt: float = 1.0, sigma_mhz: float = 0.3,
                               alpha: float = 1.0, f_low_mhz: float = 1e-3,
                               f_high_mhz: float = 5.0, n_traj: int = 128,
                               n_tones: int = 40, seed: int = 0,
                               include_decoherence: bool = True,
                               correlation: float = 0.0) -> dict:
        """Process fidelity under 1/f^alpha frequency noise across the FULL band --
        the intermediate band between quasi-static (``quasi_static_fidelity``, the
        slow limit) and white (the Markovian Lindblad T_phi, the fast limit), which
        neither of those captures on its own.

        Direct colored-noise Monte-Carlo: for each of ``n_traj`` realizations a
        zero-mean per-qubit frequency trajectory
        ``delta(t) = sum_k a_k cos(2 pi f_k t + phi_k)`` is synthesized with tone
        powers ``a_k^2 ~ 1/f_k^alpha`` over ``[f_low, f_high]`` (log-spaced),
        normalized so ``Var[delta] = sigma_mhz^2``. The gate's Choi channel is evolved
        under that time-dependent detuning and the channels are AVERAGED -- the correct
        incoherent shot-to-shot mixture -- before the fidelity. Being a full simulation
        it captures every band: ``f_high*T << 1`` reproduces ``quasi_static_fidelity``
        and raising ``f_high`` toward ``~1/dt`` approaches the white/Markovian limit
        (fast noise motionally narrows). All trajectories run in parallel over the batch.

        The two qubits' trajectories are drawn with a tunable spatial ``correlation``:
        0 = independent (per-qubit noise), +1 = fully common-mode (a shared field, e.g.
        global flux noise -- which a difference-frequency gate is partly immune to),
        -1 = fully differential (anti-correlated). Intermediate values interpolate
        (``delta_2 = sqrt|rho| delta_common +/- sqrt(1-|rho|) delta_indep``), so
        cross-qubit-correlated noise -- which independent per-qubit draws cannot
        represent -- is now in the model. Each trajectory keeps unit (sigma) variance
        regardless of ``correlation``.

        sigma_mhz: total per-qubit RMS frequency deviation (MHz). alpha: PSD exponent
            (1 = 1/f, 0 = white over the band, 2 = Brownian). f_low_mhz/f_high_mhz:
            band edges (MHz); keep f_high below the slice Nyquist ~1/(2*dt). n_traj:
            realizations (batch); n_tones: sinusoids per trajectory; seed: RNG seed.
        include_decoherence: keep Markovian T1/T_phi too (diss_scale=1); False isolates
            the colored-noise contribution (compare to quasi_static at f_high*T<<1).
        correlation: cross-qubit noise correlation in [-1, 1] (0 = independent).
        Returns {"F_proc","F_avg","F_proc_nominal","sigma_mhz","alpha","n_traj",
                 "f_low_mhz","f_high_mhz","correlation"}.
        """
        if not -1.0 <= float(correlation) <= 1.0:
            raise ValueError("correlation must be in [-1, 1].")
        u = torch.as_tensor(u_stack, dtype=self.rdtype, device=DEVICE)
        if u.dim() == 2:
            u = u.unsqueeze(0)
        u = u[:1]                                # one pulse, replicated over trajectories
        n_slices = u.shape[1]
        diss = 1.0 if include_decoherence else 0.0
        d = 4.0
        B = int(n_traj)

        # Log-spaced tones with 1/f^alpha power, normalized so Var = sigma^2.
        rng = np.random.default_rng(int(seed))
        f_cyc = np.logspace(math.log10(f_low_mhz / 1000.0),
                            math.log10(f_high_mhz / 1000.0), int(n_tones))  # cycles/ns
        wts = f_cyc ** (-float(alpha))
        amp = np.sqrt(2.0 * wts / wts.sum())     # sum(amp^2 / 2) = 1  (unit variance)
        sigma_rad = 2.0 * math.pi * (float(sigma_mhz) / 1000.0)
        t = (np.arange(n_slices) + 0.5) * float(dt)            # slice-midpoint times (ns)

        def _traj():
            ph = rng.uniform(0.0, 2.0 * math.pi, size=(B, int(n_tones), 1))
            arg = 2.0 * math.pi * f_cyc[None, :, None] * t[None, None, :] + ph
            return sigma_rad * (amp[None, :, None] * np.cos(arg)).sum(axis=1)  # [B,n_slices]

        def _correlated_pair(rho):
            # delta_1, delta_2 each unit (sigma) variance, with Cov -> rho.
            if rho == 0.0:
                return np.stack([_traj(), _traj()], axis=-1)
            r = abs(float(rho))
            common, e1, e2 = _traj(), _traj(), _traj()
            sign = 1.0 if rho > 0 else -1.0
            d1 = math.sqrt(r) * common + math.sqrt(1.0 - r) * e1
            d2 = sign * math.sqrt(r) * common + math.sqrt(1.0 - r) * e2
            return np.stack([d1, d2], axis=-1)

        with torch.no_grad():
            f_nominal = float(self._process_fidelity(
                self.simulate_choi_batch(u, dt=dt, diss_scale=diss))[0])
            if sigma_rad == 0.0:
                f_proc = f_nominal                        # no noise -> exactly nominal
            else:
                det = _correlated_pair(float(correlation))   # [B, n_slices, 2]
                det_t = torch.as_tensor(det, dtype=self.rdtype, device=DEVICE)
                u_b = u.expand(B, -1, -1).contiguous()
                choi = self.simulate_choi_batch(u_b, dt=dt, diss_scale=diss,
                                                detuning_traj=det_t)      # [B,16,9,9]
                f_proc = float(self._process_fidelity(choi.mean(dim=0, keepdim=True))[0])
        return {
            "F_proc": f_proc, "F_avg": (d * f_proc + 1.0) / (d + 1.0),
            "F_proc_nominal": f_nominal, "sigma_mhz": float(sigma_mhz),
            "alpha": float(alpha), "n_traj": B,
            "f_low_mhz": float(f_low_mhz), "f_high_mhz": float(f_high_mhz),
            "correlation": float(correlation),
        }

    # ---- Filter-function (analytic) dephasing sensitivity ----------------
    def _toggling_frame_ops(self, u_stack, dt: float):
        """Toggling-frame dephasing operators Λ_q(t_i) (4x4, computational
        subspace) for each qubit's frequency-noise coupling A_q = N_q.

        Closed-system: build the noiseless control propagators U_c(t) from the SAME
        per-slice Hamiltonian the simulator uses (drift + drives + coupling + DRAG +
        Stark; no dissipation, no noise), take their value at slice centers, and
        rotate A_q into that frame: Λ_q(t_i) = P_c U_c(t_i)^dag N_q U_c(t_i) P_c.
        Returns (lam, t_centers): lam[q] is [n_slices, 4, dim] complex (the 4
        computational ROWS of the full toggling-frame operator -- the full columns
        are kept so the leakage contribution P_c Lambda^2 P_c is not discarded),
        t_centers [n_slices]. These are the building blocks of the first-order
        dephasing filter function.
        """
        u = torch.as_tensor(u_stack, dtype=self.rdtype, device=DEVICE)
        if u.dim() == 2:
            u = u.unsqueeze(0)
        u = u[:1]                                   # one pulse
        c = self._smoothed_controls(u, dt)
        u1_eff, u2_eff, u3 = c["u1_eff"][0], c["u2_eff"][0], c["u3"][0]
        cos_phi, sin_phi, g_scale = c["cos_phi"], c["sin_phi"], c["g_scale"]
        v1, v2, uz1, uz2 = c["v1"], c["v2"], c["uz1"], c["uz2"]
        n_slices = u.shape[1]
        ci = self._comp_idx
        eye = torch.eye(self._dim, dtype=self.cdtype, device=DEVICE)
        Uacc = eye
        Aops = [self._N_Q1, self._N_Q2]
        lam = [torch.zeros((n_slices, 4, self._dim), dtype=self.cdtype, device=DEVICE)
               for _ in Aops]
        for i in range(n_slices):
            H = self._H_DRIFT + \
                (u1_eff[i] * self.OMEGA_MAX) * self._X1 + \
                (u2_eff[i] * self.OMEGA_MAX) * self._X2
            if cos_phi is not None:
                gu3 = u3[i] * self.G_MAX
                if g_scale is not None:
                    gu3 = gu3 * g_scale[0, i]
                H = H + gu3 * cos_phi[0, i] * self._COUPLING_X
                H = H + gu3 * sin_phi[0, i] * self._COUPLING_Y
            else:
                H = H + (u3[i] * self.G_MAX) * self._COUPLING
            if v1 is not None:
                H = H + v1[0, i] * self._Y1 + v2[0, i] * self._Y2
            if uz1 is not None:
                H = H + (uz1[0, i] * self.STARK_MAX) * self._N_Q1 \
                      + (uz2[0, i] * self.STARK_MAX) * self._N_Q2
            Uhalf = torch.linalg.matrix_exp(-1j * H * (dt * 0.5))
            Ucenter = Uacc @ Uhalf                  # propagator 0 -> center of slice i
            for q, A in enumerate(Aops):
                Lq = Ucenter.conj().t() @ A @ Ucenter
                lam[q][i] = Lq[ci]                  # 4 comp rows, all columns (keep leakage)
            Uacc = Ucenter @ Uhalf                  # advance to end of slice i
        t_centers = (torch.arange(n_slices, dtype=self.rdtype, device=DEVICE) + 0.5) * dt
        return lam, t_centers

    def filter_function(self, u_stack, dt: float = 1.0, f_max_mhz: float = None,
                        n_freq: int = 300) -> dict:
        """First-order dephasing FILTER FUNCTION F(f) of the gate (per-qubit
        frequency noise), the cheap analytic sensitivity an arbitrary noise PSD
        multiplies -- no Monte-Carlo.

        For a frequency-noise source delta_q(t) on qubit q, the leading-order gate
        error generator is G = integral Lambda_q(t) delta_q(t) dt, and the
        entanglement infidelity averaged over the noise is
        ``1 - F = (1/2pi) integral S(omega) F(omega) domega`` with the (entanglement-
        fidelity-consistent) generalized filter function

            F(omega) = sum_q [ Tr(L~_q L~_q^dag)/d - |Tr L~_q|^2/d^2 ],
            L~_q(omega) = integral Lambda_q(t) e^{i omega t} dt,  d = 4.

        Returns {"freq_mhz", "omega", "F" (total), "F_per_qubit", "F0"} with F0 = F(0)
        the quasi-static value (so ``sigma^2 * F0`` is the static dephasing infidelity,
        validated against quasi_static_fidelity). Overlay your device's S(f) on F(f),
        or call filter_function_fidelity to integrate a 1/f^alpha PSD directly.
        """
        with torch.no_grad():
            lam, t = self._toggling_frame_ops(u_stack, dt)
            d = 4.0
            nyq_mhz = 0.5 / (dt * 1e-3)
            fmax = float(nyq_mhz if f_max_mhz is None else f_max_mhz)
            f_mhz = torch.linspace(0.0, fmax, int(n_freq), dtype=self.rdtype, device=DEVICE)
            omega = 2.0 * math.pi * (f_mhz / 1000.0)            # rad/ns
            ci = self._comp_idx
            rows = torch.arange(4, device=DEVICE)
            # L~_q(omega) = sum_i Lambda_q[i] e^{i omega t_i} dt  ([n_freq, 4, dim])
            phase = torch.exp(1j * omega.view(-1, 1) * t.view(1, -1)) * dt  # [n_freq, n_slices]
            F_total = torch.zeros(int(n_freq), dtype=self.rdtype, device=DEVICE)
            F_per = []
            for Lq in lam:
                Lt = torch.einsum('mn,nrk->mrk', phase.to(self.cdtype), Lq)  # [n_freq,4,dim]
                # term1 = (1/d) Tr(P_c L~ L~^dag P_c): Frobenius over the 4 comp rows
                # and ALL columns -> keeps the leakage (comp<->leakage) contribution.
                term1 = (Lt * Lt.conj()).sum(dim=(-2, -1)).real / d
                # term2 = (1/d^2)|Tr_comp(L~)|^2: the comp-diagonal (row r, column ci[r]).
                tr = Lt[:, rows, ci].sum(dim=-1)
                term2 = (tr.real ** 2 + tr.imag ** 2) / (d * d)
                Fq = (term1 - term2).clamp_min(0.0)
                F_per.append(Fq.cpu().numpy())
                F_total = F_total + Fq
        return {
            "freq_mhz": f_mhz.cpu().numpy(), "omega": omega.cpu().numpy(),
            "F": F_total.cpu().numpy(), "F_per_qubit": F_per,
            "F0": float(F_total[0].item()),
        }

    def filter_function_fidelity(self, u_stack, dt: float = 1.0, sigma_mhz: float = 0.3,
                                 alpha: float = 1.0, f_low_mhz: float = 1e-3,
                                 f_high_mhz: float = 5.0, n_freq: int = 400,
                                 quasi_static: bool = False) -> dict:
        """Analytic process-fidelity estimate under 1/f^alpha dephasing, via the
        filter function -- the cheap (no-Monte-Carlo) complement to
        colored_noise_fidelity / quasi_static_fidelity.

        Integrates the per-qubit filter function against a 1/f^alpha PSD of total
        per-qubit variance sigma^2 over [f_low, f_high]:
        ``1 - F = sigma_rad^2 * (integral f^-alpha F(2pi f) df) / (integral f^-alpha df)``
        -- a band-weighted average of F. ``quasi_static=True`` collapses this to the
        slow limit ``sigma_rad^2 * F(0)`` (the exact small-noise quasi_static_fidelity
        infidelity). First order in the noise: agrees with the colored-noise Monte
        Carlo for small sigma, and is what you'd sweep when designing for a measured
        noise spectrum. Returns {"F_proc","F_avg","infidelity","sigma_mhz","alpha",
        "f_low_mhz","f_high_mhz","F0"}.
        """
        sigma_rad = 2.0 * math.pi * (float(sigma_mhz) / 1000.0)
        d = 4.0
        if quasi_static:
            ff = self.filter_function(u_stack, dt, f_max_mhz=1.0, n_freq=2)
            F_used = ff["F0"]
        else:
            # Uses the SAME differentiable helper as the in-loop robust_filter
            # objective, so optimise-against and measure-against are identical.
            with torch.no_grad():
                lam, t = self._toggling_frame_ops(u_stack, dt)
                F_used = float(self._filter_band_F_used(
                    lam, t, dt, alpha, f_low_mhz, f_high_mhz, n_freq).item())
        infid = sigma_rad ** 2 * F_used
        f_proc = max(0.0, 1.0 - infid)
        return {
            "F_proc": f_proc, "F_avg": (d * f_proc + 1.0) / (d + 1.0),
            "infidelity": infid, "sigma_mhz": float(sigma_mhz), "alpha": float(alpha),
            "f_low_mhz": float(f_low_mhz), "f_high_mhz": float(f_high_mhz),
            "F0": float(self.filter_function(u_stack, dt, f_max_mhz=1.0, n_freq=2)["F0"]),
        }

    def _filter_band_F_used(self, lam, t, dt, alpha, f_low_mhz, f_high_mhz, n_freq):
        """1/f^alpha-weighted band average of the generalized dephasing filter
        function F over ``[f_low, f_high]`` MHz, from toggling-frame ops
        ``(lam, t)``. Returns a DIFFERENTIABLE torch scalar.

        Shared verbatim by ``filter_function_fidelity`` (the scorer, called under
        ``no_grad``) and ``_filter_dephasing_infidelity`` (the in-loop objective),
        so optimise-against and measure-against are the same estimator -- not two
        implementations that might drift. Same leakage-inclusive F as
        ``filter_function``: term1 keeps all columns (comp<->leakage), term2 is the
        comp-diagonal trace. Uses ``torch.logspace``/``torch.trapezoid`` (not the
        scorer's old NumPy pair) so gradients flow; the two agree to machine
        precision (asserted in tests/test_filter_in_loop.py)."""
        d = 4.0
        ci = self._comp_idx
        rows = torch.arange(4, device=DEVICE)
        f_mhz = torch.logspace(math.log10(f_low_mhz), math.log10(f_high_mhz),
                               int(n_freq), dtype=self.rdtype, device=DEVICE)
        omega = 2.0 * math.pi * (f_mhz / 1000.0)                  # rad/ns
        phase = torch.exp(1j * omega.view(-1, 1) * t.view(1, -1)) * dt
        F_curve = torch.zeros(int(n_freq), dtype=self.rdtype, device=DEVICE)
        for Lq in lam:
            Lt = torch.einsum('mn,nrk->mrk', phase.to(self.cdtype), Lq)
            t1 = (Lt * Lt.conj()).sum(dim=(-2, -1)).real / d
            tr = Lt[:, rows, ci].sum(dim=-1)
            F_curve = F_curve + (t1 - (tr.real ** 2 + tr.imag ** 2) / (d * d)).clamp_min(0.0)
        w = f_mhz ** (-float(alpha))
        return torch.trapezoid(w * F_curve, f_mhz) / torch.trapezoid(w, f_mhz)

    def _filter_dephasing_infidelity(self, x_clamped, dt, sigma_mhz, alpha,
                                     f_low_mhz, f_high_mhz, n_freq):
        """Per-seed first-order infidelity from slow 1/f^alpha frequency noise,
        ``sigma_rad^2 * <F>_band``, via the SAME filter function
        ``filter_function_fidelity`` scores with -- but differentiable and kept in
        the gradient. Putting the WHOLE noise band inside the objective is what
        ``robust_dephasing_sigma_mhz`` (the F(0) slow limit only) does not do:
        this hardens the gate across the mid-band too.

        ``x_clamped`` is ``[B, n_slices, C]``; returns ``[B]``. Loops over seeds
        because ``_toggling_frame_ops`` accumulates the control propagator
        sequentially per pulse (B is small in practice; batching it is a future
        perf option, not a correctness one)."""
        sigma_rad = 2.0 * math.pi * (float(sigma_mhz) / 1000.0)
        out = []
        for b in range(x_clamped.shape[0]):
            lam, t = self._toggling_frame_ops(x_clamped[b:b + 1], dt)
            F_used = self._filter_band_F_used(lam, t, dt, alpha, f_low_mhz,
                                              f_high_mhz, n_freq)
            out.append(sigma_rad ** 2 * F_used)
        return torch.stack(out)

    # ---- Spectator (always-on ZZ) crosstalk ------------------------------
    def spectator_fidelity(self, u_stack, dt: float = 1.0, zeta_mhz=0.1,
                           spectator_pop: float = 0.5) -> dict:
        """Gate-fidelity penalty from an always-on ZZ to an idle neighbour qubit.

        The dominant multi-qubit crosstalk this single-pair model otherwise omits.
        A spectator coupled to a gate qubit by a static (always-on) ZZ of rate
        ``zeta_mhz`` shifts that gate qubit's frequency by zeta when the neighbour
        is in |1>. For an OFF-RESONANT (detuned) neighbour the exchange is
        dispersively suppressed and this diagonal ZZ is the whole effect, so a
        neighbour frozen in state s is EXACTLY a static detuning zeta*s on the gate
        qubit -- modelled here through the detuning_offset primitive. This reduction
        is validated against a full 3-transmon (27-D) QuTiP simulation in
        tests/test_spectators.py. Resonant exchange / frequency collisions are the
        complementary regime (the spectator dynamically swaps population and cannot
        be frozen) -- see ``resonant_collision_fidelity``.

        zeta_mhz: ZZ rate (MHz). Scalar -> a neighbour of this strength on EACH
            gate qubit; (zeta1, zeta2) -> per gate qubit (set one to 0 for a single
            neighbour). Typical superconducting values are ~0.01-0.1 MHz.
        spectator_pop: P(neighbour in |1>). A *known, fixed* neighbour state is a
            deterministic single-qubit-Z detuning that virtual-Z re-tuning almost
            fully removes (verified on the shipped CZ: ~100% of the drop recovered).
            The cost that matters is an *unmeasured* neighbour whose state varies
            shot-to-shot: averaging the channel over its population makes it a
            dephasing channel. Re-tuning the virtual-Z for the neighbour's MEAN shift
            still removes part of that (~half at pop=0.5); the residual SPREAD across
            neighbour states is the genuinely irreducible part. The returned
            delta_r_spectator is therefore the conservative cost at the gate's NOMINAL
            frame -- an upper bound on the post-re-tuning residual. 0.5 = maximally-
            uncertain neighbour; pass its thermal n_th for an idle-cold one.

        Returns a dict:
          * f_proc_idle           neighbour in |0> (= the nominal gate).
          * f_proc_excited        neighbour frozen in |1>, RAW (un-recalibrated)
                                  conservative coherent penalty.
          * f_proc_spectator_avg  unmeasured neighbour, channel averaged over its
                                  state (at the gate's nominal frame).
          * f_avg_idle            nominal average gate fidelity (neighbour absent).
          * f_avg_spectator_avg   average gate fidelity with the unmeasured neighbour.
          * delta_r_spectator     f_avg_idle - f_avg_spectator_avg: the average-gate
                                  infidelity an unmeasured neighbour ADDS at the gate's
                                  nominal frame (marginal; excludes the gate's own
                                  error). Conservative -- virtual-Z re-tuning for the
                                  neighbour's mean shift removes part of it.
          * zz_phase_rad          conditional phase zeta*T_gate the neighbour
                                  imprints (intuition scalar), plus echoes of the
                                  inputs and the number of channel evaluations.
        """
        u = torch.as_tensor(u_stack, dtype=self.rdtype, device=DEVICE)
        if u.dim() == 2:
            u = u.unsqueeze(0)
        if isinstance(zeta_mhz, (tuple, list)):
            z1, z2 = float(zeta_mhz[0]), float(zeta_mhz[1])
        else:
            z1 = z2 = float(zeta_mhz)
        d = 4.0
        p = float(spectator_pop)

        def _rad(mhz):
            return 2.0 * math.pi * (mhz / 1000.0)               # MHz -> rad/ns

        # Per-axis neighbour states/weights; collapse an axis with zeta=0 (no
        # neighbour there) so it is not double-counted in the channel average.
        ax1 = [(0, 1.0 - p), (1, p)] if z1 != 0.0 else [(0, 1.0)]
        ax2 = [(0, 1.0 - p), (1, p)] if z2 != 0.0 else [(0, 1.0)]

        with torch.no_grad():
            # Evolve each distinct neighbour configuration's channel once.
            configs = {}
            for s1, _ in ax1:
                for s2, _ in ax2:
                    if (s1, s2) not in configs:
                        configs[(s1, s2)] = self.simulate_choi_batch(
                            u, dt=dt,
                            detuning_offset=(_rad(z1) * s1, _rad(z2) * s2))
            avg_choi = None
            for s1, w1 in ax1:
                for s2, w2 in ax2:
                    term = (w1 * w2) * configs[(s1, s2)]
                    avg_choi = term if avg_choi is None else avg_choi + term
            exc_key = (1 if z1 != 0.0 else 0, 1 if z2 != 0.0 else 0)
            f_idle = float(self._process_fidelity(configs[(0, 0)])[0])
            f_exc = float(self._process_fidelity(configs[exc_key])[0])
            f_avg_choi = float(self._process_fidelity(avg_choi)[0])

        f_avg_gate = (d * f_avg_choi + 1.0) / (d + 1.0)
        f_avg_idle = (d * f_idle + 1.0) / (d + 1.0)
        t_gate_ns = u.shape[1] * float(dt)
        zz_phase = max(abs(_rad(z1)), abs(_rad(z2))) * t_gate_ns
        return {
            "f_proc_idle": f_idle,
            "f_proc_excited": f_exc,
            "f_proc_spectator_avg": f_avg_choi,
            "f_avg_idle": f_avg_idle,
            "f_avg_spectator_avg": f_avg_gate,
            "delta_r_spectator": f_avg_idle - f_avg_gate,
            "zz_phase_rad": zz_phase,
            "zeta_mhz": (z1, z2),
            "spectator_pop": p,
            "n_evals": len(configs),
        }

    def multi_spectator_fidelity(self, u_stack, neighbours, dt: float = 1.0) -> dict:
        """Always-on-ZZ penalty from an ARBITRARY set of idle neighbours.

        The multi-spectator generalization of ``spectator_fidelity`` (which is the
        one-neighbour-per-gate-qubit special case). Each frozen, off-resonant
        neighbour adds a static diagonal ZZ -- i.e. a static detuning -- on the gate
        qubit it couples to. Diagonal shifts commute and ADD, so several neighbours
        on the same qubit are exactly one detuning equal to their sum, and an
        *unmeasured* ensemble averages the channel over every combination of neighbour
        states. This is the same exact reduction ``spectator_fidelity`` uses (validated
        to machine precision against a full 3-transmon QuTiP sim), applied to N
        neighbours; the additive-detuning step itself is cross-checked against an
        explicit multi-transmon QuTiP simulation in tests/test_spectators.py. The
        *resonant* / frequency-collision regime is handled by
        ``resonant_collision_fidelity`` (an evolving, exchange-coupled spectator).

        neighbours: list of ``(gate_qubit, zeta_mhz, pop)``:
            gate_qubit in {0, 1} (0 = first gate qubit q1, 1 = second q2),
            zeta_mhz   = that neighbour's always-on ZZ rate (MHz),
            pop        = P(neighbour in |1>) (0.5 = maximally uncertain; n_th if cold).
        Returns the spectator_fidelity keys (delta_r_spectator is the conservative
        nominal-frame infidelity the whole ensemble ADDS), plus n_neighbours/n_configs.
        """
        from itertools import product
        u = torch.as_tensor(u_stack, dtype=self.rdtype, device=DEVICE)
        if u.dim() == 2:
            u = u.unsqueeze(0)
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
        if avg_choi is None:          # no neighbours -> nominal gate
            avg_choi = idle

        f_idle = float(self._process_fidelity(idle)[0])
        f_avg_choi = float(self._process_fidelity(avg_choi)[0])
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

    # ---- Resonant / frequency-collision crosstalk ------------------------
    def resonant_collision_fidelity(self, u_stack, dt: float = 1.0,
                                    detuning_mhz=0.0, j_mhz: float = 8.0,
                                    couples_to: int = 2, diss_scale: float = 1.0):
        """Gate fidelity vs. a NEAR-RESONANT spectator that exchanges population.

        The complement of ``spectator_fidelity``. That method (and
        ``multi_spectator_fidelity``) FREEZE an off-resonant neighbour into a static
        diagonal ZZ -- exact only while the exchange is dispersively suppressed.
        When a spectator's frequency approaches a gate qubit's (a "frequency
        collision"), the exchange is RESONANT: population coherently SWAPS into the
        spectator during the gate and cannot be reduced to a detuning. This models
        that regime directly -- an explicitly EVOLVING third transmon coupled to a
        gate qubit by a transverse exchange J(a_g^dag a_s + a_g a_s^dag), propagated
        in the full (n_levels**3)-D open system (27-D at the default n_levels=3),
        the gate Hamiltonian taken from the SAME _smoothed_controls path the
        optimizer uses. Cross-checked against an independent QuTiP simulation in
        tests/test_collision.py (validate.collision_cross_check); the J=0 limit
        reproduces the bare-gate F_proc to machine precision (test guards the lift).

        detuning_mhz: spectator detuning FROM the coupled gate qubit (MHz). 0 = an
            exact frequency collision; large = far off-resonant (dispersive shift
            ~ J^2/detuning -> 0, recovering the bare gate). A scalar evaluates one
            point; an array/list evaluates the whole collision curve in ONE batched
            call (each detuning is a batch element).
        j_mhz: transverse exchange to the coupled gate qubit (MHz); a few-to-tens
            of MHz is a representative residual static coupling.
        couples_to: which gate qubit the spectator neighbours, 1 (q1) or 2 (q2).
        diss_scale: scales the gate-pair Lindblad rates (1.0 = nominal T1/T_phi;
            0.0 isolates the purely coherent collision error). The spectator is
            modelled as coherent (no added decoherence) -- the same conservative
            convention as the frozen-spectator cross-check.

        Returns a dict; the F/leakage entries are scalars for a scalar detuning and
        lists (one per detuning) for a sweep:
          * detuning_mhz        echo of the requested detuning(s)
          * f_proc / f_avg      gate-pair fidelity WITH the spectator present
          * f_proc_isolated     bare-gate F_proc (no spectator) -- the baseline,
            f_avg_isolated its average-gate form
          * delta_r_collision   f_avg_isolated - f_avg: average-gate infidelity the
            collision ADDS (>= 0; diverges as detuning -> 0)
          * spectator_leakage   population that swapped INTO the spectator
            (1 - P(spectator in |0>), averaged over the 4 computational inputs) --
            the smoking gun of a collision, ~0 far off-resonant
          * j_mhz, couples_to   echoes
        """
        u = torch.as_tensor(u_stack, dtype=self.rdtype, device=DEVICE)
        if u.dim() == 2:
            u = u.unsqueeze(0)
        u = u[:1]
        n_slices = u.shape[1]
        nl = self.n_levels
        d2 = self._dim                                   # nl**2 (gate pair)
        d3 = nl ** 3
        cdt = self.cdtype
        couples_to = int(couples_to)
        if couples_to not in (1, 2):
            raise ValueError("couples_to must be 1 (q1) or 2 (q2)")

        det_arr = np.atleast_1d(np.asarray(detuning_mhz, dtype=float)).ravel()
        B = det_arr.shape[0]
        det_rad = torch.as_tensor(2.0 * math.pi * det_arr / 1000.0,
                                  dtype=cdt, device=DEVICE)            # rad/ns, [B]

        # Single-transmon ladder, rebuilt from n_levels (matches _build_coupler_ops).
        _sub = torch.tensor([math.sqrt(k) for k in range(1, nl)],
                            dtype=cdt, device=DEVICE)
        a = torch.diag(_sub, 1)
        I = torch.eye(nl, dtype=cdt, device=DEVICE)
        I9 = torch.eye(d2, dtype=cdt, device=DEVICE)

        def lift(op9):                                   # gate-pair (d2) -> 3-body (d3)
            return torch.kron(op9.contiguous(), I)

        # Spectator (3rd tensor slot) + transverse exchange to the coupled qubit.
        a_s = torch.kron(I9, a)
        ad_s = a_s.conj().t()
        n_s = ad_s @ a_s
        anh_s = (self.profile.anharm_ghz_q2 if couples_to == 2
                 else self.profile.anharm_ghz_q1) * 2.0 * math.pi
        delta = (self.profile.freq_ghz_q2 - self.profile.freq_ghz_q1) * 2.0 * math.pi
        base = delta if couples_to == 2 else 0.0          # coupled-qubit freq, q1 frame
        a_g9 = torch.kron(I, a) if couples_to == 2 else torch.kron(a, I)
        a_g = lift(a_g9)
        j = 2.0 * math.pi * (float(j_mhz) / 1000.0)
        H_exch = j * (a_g.conj().t() @ a_s + a_g @ ad_s)
        H_s_const = base * n_s + 0.5 * anh_s * (ad_s @ ad_s @ a_s @ a_s)

        # Lifted gate-pair Lindblad operators (spectator stays coherent).
        Ls = [(lift(L), lift(L).conj().t().contiguous())
              for L in (self._L_T1_Q1, self._L_T1_Q2,
                        self._L_PHI_Q1, self._L_PHI_Q2)]
        for L in (self._L_TH_Q1, self._L_TH_Q2):
            if L is not None:
                Ls.append((lift(L), lift(L).conj().t().contiguous()))
        L_loss = lift(self._L_LOSS_SUM)

        def _Ldiss(r):
            jump = sum(L @ r @ Ld for (L, Ld) in Ls)
            anti = (L_loss @ r) + (r @ L_loss)
            return diss_scale * (jump - anti)

        def _diss_half(r):
            tau = 0.5 * dt
            Lr = _Ldiss(r)
            return r + tau * Lr + (0.5 * tau * tau) * _Ldiss(Lr)

        # Initial Choi stack |comp_i><comp_j| (x) |0><0|_spectator, batched / detuning.
        ci3 = (self._comp_idx * nl).tolist()              # comp index, spectator in |0>
        rho = torch.zeros((B, 16, d3, d3), dtype=cdt, device=DEVICE)
        for i in range(4):
            for jx in range(4):
                rho[:, i * 4 + jx, ci3[i], ci3[jx]] = 1.0

        ctrl = self._smoothed_controls(u, dt)
        u1_eff, u2_eff, u3 = ctrl["u1_eff"], ctrl["u2_eff"], ctrl["u3"]
        cos_phi, sin_phi, g_scale = ctrl["cos_phi"], ctrl["sin_phi"], ctrl["g_scale"]
        v1, v2, uz1, uz2 = ctrl["v1"], ctrl["v2"], ctrl["uz1"], ctrl["uz2"]

        with torch.no_grad():
            for i in range(n_slices):
                # Gate-pair H(t): identical assembly to simulate_gradient_batch.
                H9 = self._H_DRIFT \
                    + (u1_eff[0, i] * self.OMEGA_MAX) * self._X1 \
                    + (u2_eff[0, i] * self.OMEGA_MAX) * self._X2
                if cos_phi is not None:
                    gu3 = u3[0, i] * self.G_MAX
                    if g_scale is not None:
                        gu3 = gu3 * g_scale[0, i]
                    H9 = H9 + gu3 * cos_phi[0, i] * self._COUPLING_X
                    H9 = H9 + gu3 * sin_phi[0, i] * self._COUPLING_Y
                else:
                    H9 = H9 + (u3[0, i] * self.G_MAX) * self._COUPLING
                if v1 is not None:
                    H9 = H9 + v1[0, i] * self._Y1 + v2[0, i] * self._Y2
                if uz1 is not None:
                    H9 = H9 + (uz1[0, i] * self.STARK_MAX) * self._N_Q1
                    H9 = H9 + (uz2[0, i] * self.STARK_MAX) * self._N_Q2
                # Lift + spectator + exchange; only the spectator detuning is batched.
                H_common = lift(H9) + H_exch + H_s_const                  # [d3,d3]
                H_b = H_common.unsqueeze(0) + det_rad.view(B, 1, 1) * n_s  # [B,d3,d3]
                U = torch.linalg.matrix_exp(-1j * H_b * dt).unsqueeze(1)   # [B,1,d3,d3]
                Ud = U.conj().transpose(-2, -1)
                if self.step_order == 1:
                    rho = U @ rho @ Ud
                    rho = rho + dt * _Ldiss(rho)
                else:
                    rho = _diss_half(rho)
                    rho = U @ rho @ Ud
                    rho = _diss_half(rho)

            # Trace out the spectator -> 9-D Choi stack; reuse the validated metric.
            r6 = rho.reshape(B, 16, d2, nl, d2, nl)
            choi9 = torch.einsum('bmipjp->bmij', r6)       # partial trace over spectator
            f_proc = self._process_fidelity(choi9)         # [B]
            # Population that swapped INTO the spectator (1 - P(spectator |0>)),
            # averaged over the 4 computational inputs |i><i| (m in {0,5,10,15}).
            diag = rho.diagonal(dim1=-2, dim2=-1).real.reshape(B, 16, d2, nl)
            p_s0 = diag[..., 0].sum(dim=-1)                # [B,16] = P(spectator in |0>)
            spec_leak = (1.0 - p_s0[:, [0, 5, 10, 15]]).mean(dim=1).clamp(0.0, 1.0)
            # Bare-gate baseline (no spectator) via the standard 2-qubit path.
            f_iso = float(self._process_fidelity(
                self.simulate_choi_batch(u, dt=dt, diss_scale=diss_scale))[0])

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
            "couples_to": couples_to,
        }

    def tls_defect_fidelity(self, u_stack, dt: float = 1.0,
                            detuning_mhz=0.0, g_mhz: float = 2.0,
                            t1_tls_ns: float = 500.0, couples_to: int = 1,
                            diss_scale: float = 1.0):
        """Gate fidelity vs. an explicit, LOSSY two-level-system (TLS) defect.

        TLS defects in the amorphous oxides of a Josephson junction are the physics a
        CLASSICAL noise model (quasi-static / 1/f / colored_noise_fidelity, white
        Markovian T_phi) cannot capture: a TLS is a coherent QUANTUM bath mode that
        exchanges a real excitation with the qubit (vacuum-Rabi swap) AND carries its
        own short relaxation time -- so near resonance it both swaps population out of
        the computational space and drains it irreversibly. This evolves the gate pair
        plus ONE explicit two-level defect in the full (n_levels**2 * 2)-D open system
        (18-D at n_levels=3): a transverse exchange g (a_g^dag sigma_- + a_g sigma_+) to
        a gate qubit, the defect's own frequency, and its own T1 Lindblad jump. The gate
        Hamiltonian is taken from the SAME _smoothed_controls path the optimizer uses.

        It is the lossy cousin of ``resonant_collision_fidelity`` (a COHERENT evolving
        transmon): the new ingredient is the TLS's own dissipator, which is exactly what
        makes a real chip miss a decoherence-only sim. The g=0 limit reproduces the bare
        gate to machine precision (guards the lift); cross-checked against an independent
        QuTiP build in tests/test_tls.py.

        detuning_mhz: TLS frequency relative to the coupled gate qubit (MHz). 0 = an
            exact resonance (worst case). Scalar evaluates one point; an array/list
            sweeps the resonance curve in ONE batched call.
        g_mhz: transverse qubit-TLS coupling (MHz); ~0.1-a few MHz is representative.
        t1_tls_ns: the TLS's OWN energy-relaxation time (ns). TLS are often far less
            coherent than the qubit -- this is the irreversible-loss channel.
        couples_to: which gate qubit the defect neighbours, 1 (q1) or 2 (q2).
        diss_scale: scales ALL Lindblad rates (gate pair AND the TLS); 1.0 = nominal,
            0.0 isolates the purely COHERENT TLS-exchange error.

        Returns a dict (scalars for a scalar detuning, lists for a sweep):
          * detuning_mhz, g_mhz, t1_tls_ns, couples_to  echoes
          * f_proc / f_avg            gate-pair fidelity WITH the defect present
          * f_proc_isolated / f_avg_isolated  bare-gate baseline (no defect)
          * delta_r_tls               f_avg_isolated - f_avg: average-gate infidelity
            the defect ADDS (>= 0; peaks on resonance)
          * tls_excitation            P(TLS ends in |1>), averaged over the 4
            computational inputs -- population the defect pulled out of the qubits
        """
        u = torch.as_tensor(u_stack, dtype=self.rdtype, device=DEVICE)
        if u.dim() == 2:
            u = u.unsqueeze(0)
        u = u[:1]
        n_slices = u.shape[1]
        nl = self.n_levels
        d2 = self._dim                                   # nl**2 (gate pair)
        dT = d2 * 2                                       # gate pair (x) TLS (2-level)
        cdt = self.cdtype
        couples_to = int(couples_to)
        if couples_to not in (1, 2):
            raise ValueError("couples_to must be 1 (q1) or 2 (q2)")

        det_arr = np.atleast_1d(np.asarray(detuning_mhz, dtype=float)).ravel()
        B = det_arr.shape[0]
        det_rad = torch.as_tensor(2.0 * math.pi * det_arr / 1000.0,
                                  dtype=cdt, device=DEVICE)            # rad/ns, [B]

        # Single-transmon ladder + the gate-pair identity, matching _build_coupler_ops.
        _sub = torch.tensor([math.sqrt(k) for k in range(1, nl)], dtype=cdt, device=DEVICE)
        a = torch.diag(_sub, 1)
        I = torch.eye(nl, dtype=cdt, device=DEVICE)
        I9 = torch.eye(d2, dtype=cdt, device=DEVICE)
        I2 = torch.eye(2, dtype=cdt, device=DEVICE)
        sm = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=cdt, device=DEVICE)  # |1>-><0|

        def lift(op9):                                   # gate-pair (d2) -> 3-body (dT)
            return torch.kron(op9.contiguous(), I2)

        # TLS (2-level defect) operators in the joint space + transverse exchange.
        a_t = torch.kron(I9, sm)
        ad_t = a_t.conj().t()
        n_t = ad_t @ a_t                                 # projector onto TLS |1>
        a_g9 = torch.kron(I, a) if couples_to == 2 else torch.kron(a, I)
        a_g = lift(a_g9)
        g = 2.0 * math.pi * (float(g_mhz) / 1000.0)
        H_exch = g * (a_g.conj().t() @ a_t + a_g @ ad_t)
        # TLS frequency in the q1 frame: the coupled qubit's frame offset + detuning.
        delta = (self.profile.freq_ghz_q2 - self.profile.freq_ghz_q1) * 2.0 * math.pi
        base = delta if couples_to == 2 else 0.0
        H_t_const = base * n_t

        # Lindblad: lifted gate-pair jumps PLUS the TLS's own T1 jump (the new physics).
        Ls = [(lift(L), lift(L).conj().t().contiguous())
              for L in (self._L_T1_Q1, self._L_T1_Q2, self._L_PHI_Q1, self._L_PHI_Q2)]
        for L in (self._L_TH_Q1, self._L_TH_Q2):
            if L is not None:
                Ls.append((lift(L), lift(L).conj().t().contiguous()))
        L_tls = math.sqrt(1.0 / max(float(t1_tls_ns), 1e-9)) * a_t
        Ls.append((L_tls.contiguous(), L_tls.conj().t().contiguous()))
        L_loss = lift(self._L_LOSS_SUM) + 0.5 * (L_tls.conj().t() @ L_tls)

        def _Ldiss(r):
            jump = sum(L @ r @ Ld for (L, Ld) in Ls)
            anti = (L_loss @ r) + (r @ L_loss)
            return diss_scale * (jump - anti)

        def _diss_half(r):
            tau = 0.5 * dt
            Lr = _Ldiss(r)
            return r + tau * Lr + (0.5 * tau * tau) * _Ldiss(Lr)

        # Initial Choi stack |comp_i><comp_j| (x) |0><0|_TLS (defect starts cold).
        ciT = (self._comp_idx * 2).tolist()              # comp index, TLS in |0>
        rho = torch.zeros((B, 16, dT, dT), dtype=cdt, device=DEVICE)
        for i in range(4):
            for jx in range(4):
                rho[:, i * 4 + jx, ciT[i], ciT[jx]] = 1.0

        ctrl = self._smoothed_controls(u, dt)
        u1_eff, u2_eff, u3 = ctrl["u1_eff"], ctrl["u2_eff"], ctrl["u3"]
        cos_phi, sin_phi, g_scale = ctrl["cos_phi"], ctrl["sin_phi"], ctrl["g_scale"]
        v1, v2, uz1, uz2 = ctrl["v1"], ctrl["v2"], ctrl["uz1"], ctrl["uz2"]

        with torch.no_grad():
            for i in range(n_slices):
                H9 = self._H_DRIFT \
                    + (u1_eff[0, i] * self.OMEGA_MAX) * self._X1 \
                    + (u2_eff[0, i] * self.OMEGA_MAX) * self._X2
                if cos_phi is not None:
                    gu3 = u3[0, i] * self.G_MAX
                    if g_scale is not None:
                        gu3 = gu3 * g_scale[0, i]
                    H9 = H9 + gu3 * cos_phi[0, i] * self._COUPLING_X
                    H9 = H9 + gu3 * sin_phi[0, i] * self._COUPLING_Y
                else:
                    H9 = H9 + (u3[0, i] * self.G_MAX) * self._COUPLING
                if v1 is not None:
                    H9 = H9 + v1[0, i] * self._Y1 + v2[0, i] * self._Y2
                if uz1 is not None:
                    H9 = H9 + (uz1[0, i] * self.STARK_MAX) * self._N_Q1
                    H9 = H9 + (uz2[0, i] * self.STARK_MAX) * self._N_Q2
                H_common = lift(H9) + H_exch + H_t_const                   # [dT,dT]
                H_b = H_common.unsqueeze(0) + det_rad.view(B, 1, 1) * n_t   # [B,dT,dT]
                U = torch.linalg.matrix_exp(-1j * H_b * dt).unsqueeze(1)
                Ud = U.conj().transpose(-2, -1)
                if self.step_order == 1:
                    rho = U @ rho @ Ud
                    rho = rho + dt * _Ldiss(rho)
                else:
                    rho = _diss_half(rho)
                    rho = U @ rho @ Ud
                    rho = _diss_half(rho)

            # Trace out the TLS -> 9-D Choi stack; reuse the validated metric.
            r6 = rho.reshape(B, 16, d2, 2, d2, 2)
            choi9 = torch.einsum('bmipjp->bmij', r6)
            f_proc = self._process_fidelity(choi9)
            diag = rho.diagonal(dim1=-2, dim2=-1).real.reshape(B, 16, d2, 2)
            p_t1 = diag[..., 1].sum(dim=-1)                # [B,16] = P(TLS in |1>)
            tls_exc = p_t1[:, [0, 5, 10, 15]].mean(dim=1).clamp(0.0, 1.0)
            f_iso = float(self._process_fidelity(
                self.simulate_choi_batch(u, dt=dt, diss_scale=diss_scale))[0])

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
            "delta_r_tls": _s(f_avg_iso - f_avg_np),
            "tls_excitation": _s(tls_exc.detach().cpu().numpy()),
            "g_mhz": float(g_mhz),
            "t1_tls_ns": float(t1_tls_ns),
            "couples_to": couples_to,
        }

    # ---- Integrator convergence ------------------------------------------
    def dt_convergence(self, u_stack, dt: float = 1.0, refinements=(1, 2, 4),
                       metric: str = "process"):
        """Time-step convergence of the master-equation integrator.

        Holds the physical pulse fixed and shrinks the integration step dt,
        reporting the resulting gate fidelity for BOTH step orders plus a
        Richardson extrapolation to the dt→0 limit. This turns the old
        "first-order Euler, convergence not shown" caveat into a checkable
        claim: order 1 shows an O(dt) trend, order 2 an O(dt²) one that is
        already flat at dt = 1 ns.

        How the pulse is held fixed: each raw slice is repeated k times
        (zero-order hold) and dt → dt/k, so the total gate time T = n_slices·dt
        is preserved. The bandwidth smoother rebuilds its kernel for each dt to
        hold a FIXED physical cutoff (see _build_smoother_kernel), so the
        continuous-time control H(t) is, up to kernel discretization, the
        same at every k; only the integrator step changes. Because the unitary
        factor is exact at any dt, the integrator error being probed is purely
        the dissipator splitting. The cleanest single number is therefore
        splitting_err_at_dt = |order1 − order2| at the SAME k: identical pulse,
        identical dt, so it isolates the Trotter/Euler error from everything else.

        u_stack: [B, n_slices, n_channels] or [n_slices, n_channels] raw params
                 (the optimizer's parameter, e.g. an optimized pulse, NOT the
                 already-smoothed waveform). The 'best_raw_param' returned by
                 optimize_multi_seed is exactly this: feeding it here reproduces
                 the optimized pulse with no re-smoothing distortion.
        refinements: integer step-subdivision factors k; dt is divided by each.
                     The default (1, 2, 4) gives a clean factor-2 ladder.
        metric: 'process' (default, _process_fidelity) or 'state'
                (_avg_state_fidelity).

        Precision: with precision='single' (complex64) the integrator noise
        floor is ~1e-6, so the order-1 O(dt) trend stops halving cleanly once
        the refinement residual reaches that floor. Construct the optimizer with
        precision='double' to drop the floor by orders of magnitude and see the
        order-1 error halve cleanly to very fine dt (and order 2 quarter).

        Returns a dict: dt values, per-order fidelity lists, the Richardson
        dt→0 extrapolation per order, |F(dt)−F0| per order, and
        splitting_err_at_dt.
        """
        if u_stack.dim() == 2:
            u_stack = u_stack.unsqueeze(0)
        refs = sorted({int(k) for k in refinements})
        if refs[0] < 1:
            raise ValueError(f"refinements must all be >= 1, got {refinements}")
        if metric not in ("process", "state"):
            raise ValueError(f"metric must be 'process' or 'state', got {metric!r}")
        if metric == "process":
            fid_fn, sim_fn = self._process_fidelity, self.simulate_choi_batch
        else:
            fid_fn, sim_fn = self._avg_state_fidelity, self.simulate_gradient_batch

        saved_order = self.step_order
        out = {"dt": [dt / k for k in refs], "refinements": refs, "metric": metric}
        try:
            with torch.no_grad():
                for order in (1, 2):
                    self.step_order = order
                    fids = []
                    for k in refs:
                        # Zero-order hold: same physical waveform, finer grid.
                        u_ref = torch.repeat_interleave(u_stack, k, dim=1)
                        rho = sim_fn(u_ref, dt=dt / k)
                        fids.append(float(fid_fn(rho).mean().item()))
                    out[f"order{order}"] = fids
        finally:
            self.step_order = saved_order

        # Richardson extrapolation from the two finest steps, known order p, dt ratio r:
        #   F(h)=F0+C*h^p, F(h/r)=F0+C*(h/r)^p  =>  F0=(r^p*F(h/r)-F(h))/(r^p-1).
        for order, p in ((1, 1), (2, 2)):
            fids = out[f"order{order}"]
            if len(refs) >= 2:
                r = refs[-1] / refs[-2]            # dt(coarser)/dt(finer) of the pair
                rp = r ** p
                f0 = (rp * fids[-1] - fids[-2]) / (rp - 1.0)
            else:
                f0 = fids[-1]
            out[f"order{order}_extrap"] = f0
            out[f"order{order}_err_at_dt"] = abs(fids[0] - f0)
        # Pure integrator-splitting error at dt: same pulse & dt, order1 vs order2.
        out["splitting_err_at_dt"] = abs(out["order1"][0] - out["order2"][0])
        return out
