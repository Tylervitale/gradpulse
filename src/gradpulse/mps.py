"""gradpulse.mps -- evaluation-only matrix-product-state evaluator for the N-qubit
register, to score a pulse past the dense ~4-qubit wall in the WEAKLY-ENTANGLING regime.

Why this exists, and what it honestly is
----------------------------------------
The dense ``MultiQubitOptimizer`` is exact but exponential: the Hilbert space is
``n_levels**N`` and the open-system process fidelity evolves ``4**N`` Choi operators.
``process_fidelity_sparse`` pushes the *closed-system* score further by propagating the
``2**N`` computational state vectors, but it still carries ``2**N`` states and no
decoherence. This module compresses each evolved state into a matrix product state
(bounded bond dimension ``chi``), so a *single* low-entanglement input costs
``O(N * chi**2 * d)`` instead of ``O(d**N)`` -- reaching larger ``N`` when the gate
keeps entanglement low (the regime where MPS is the right tool, and the only regime
where it helps).

THE EXPONENTIAL DOES NOT GO AWAY -- IT MOVES. The exact process / entanglement fidelity
is a 2-design average over ``4**N`` basis operators; an MPS compresses *states*, not the
*number of basis operators*, so it CANNOT produce the exact ``F_proc`` cheaply. What it
produces is a **restricted-ensemble average gate fidelity witness**: the mean
input-output fidelity over a finite ensemble of low-entanglement (product) input states.
Because that ensemble is NOT a 2-design, the witness is NOT the entanglement fidelity and
does NOT map through ``F_avg = (d*F_proc+1)/(d+1)``; product-state ensembles typically
OVER-estimate (they under-sample the hard-to-preserve entangled inputs). It is reported
as what it is, with its bias measured against the dense value where both run -- never
collapsed onto the headline ``F_proc`` symbol.

Layers (each validated before the next is trusted)
--------------------------------------------------
* Layer 1 (this file, foundation): second-order Trotter (TEBD) evolution on the FULL
  state vector, built from the SAME local model as ``MultiQubitOptimizer``. Validated by
  convergence to ``process_fidelity_sparse`` (the exact closed-system F_proc) as the
  Trotter substep count grows -- this proves the local-operator re-derivation and the
  TEBD splitting before any bond truncation is introduced.
* Layer 2: MPS state with SVD truncation to ``chi`` (chi-convergence reported as a ship
  gate: witness vs chi + discarded weight, plateau at the N actually used).
* Layer 3: quantum-trajectory unraveling over pure MPS for the OPEN system (each
  trajectory is pure so positivity is automatic and chi(psi) << chi(rho); cost is a
  stated statistical error). Validated against the dense open-system value at small N.

Topology: requires a 1-D chain coupling graph (edges (i, i+1)) -- the native MPS
geometry. Non-chain graphs raise (honest scope, not a silent fallback).
"""
from __future__ import annotations

import math
from itertools import product

import numpy as np

from .liouville import _expm          # numpy-only, independently validated matrix exp

_2PI = 2.0 * math.pi


def _ladder(d: int) -> np.ndarray:
    """d-dimensional truncated annihilation operator."""
    return np.diag(np.sqrt(np.arange(1, d, dtype=float)), 1).astype(complex)


class ChainTEBD:
    """Second-order TEBD evolution of an N-transmon chain, re-derived in LOCAL form
    from a built ``MultiQubitOptimizer`` (so it uses identical model constants). Layer 1
    is exact (full state vector, no truncation): the correctness foundation that the MPS
    and trajectory layers build on. Evaluation-only -- no autograd through the SVD.
    """

    # ---- construction ----------------------------------------------------
    def __init__(self, opt):
        N = int(opt.N)
        if N < 2:
            raise ValueError("ChainTEBD needs N >= 2 qubits.")
        # Require a 1-D chain: every coupling edge must be a nearest-neighbour (i, i+1).
        edges = set()
        for (i, j) in list(opt.fixed_edges) + list(opt.tunable_edges):
            e = (min(i, j), max(i, j))
            if e[1] != e[0] + 1:
                raise ValueError(
                    f"ChainTEBD requires a 1-D chain coupling graph; edge {e} is not "
                    "nearest-neighbour. (Non-chain graphs are out of scope here.)")
            edges.add(e)
        self.N, self.d = N, int(opt.d)
        self.opt = opt
        self.use_drag = bool(opt.use_drag)
        # ---- local operators (d x d) ----
        a = _ladder(self.d)
        ad = a.conj().T
        n = ad @ a
        self._a, self._ad, self._n = a, ad, n
        self._X = a + ad                      # drive in-phase
        self._Y = 1j * (ad - a)               # drive quadrature (DRAG)
        self._num = n                         # frequency-control / dephasing operator
        self._I = np.eye(self.d, dtype=complex)
        # 2-site coupling operator ad_i a_j + a_i ad_j on the bond Hilbert space (d^2)
        self._C2 = np.kron(ad, a) + np.kron(a, ad)
        # ---- model constants pulled from the dense optimizer (identical model) ----
        prof = opt.profile
        f_ref = prof.f_ref_ghz
        self._delta = [_2PI * (prof.freqs_ghz[q] - f_ref) for q in range(N)]
        self._alpha = [_2PI * prof.anharm_mhz[q] / 1000.0 for q in range(N)]
        self.OMEGA_MAX = float(opt.OMEGA_MAX)
        self.G_MAX = float(opt.G_MAX)
        self.DELTA_MAX = float(getattr(opt, "DELTA_MAX", 0.0))
        self.drive_qubits = list(opt.drive_qubits)
        self.tunable_edges = [(min(i, j), max(i, j)) for (i, j) in opt.tunable_edges]
        self.freq_control_qubits = list(opt.freq_control_qubits)
        self._fixed_g = {}
        for (i, j) in opt.fixed_edges:
            e = (min(i, j), max(i, j))
            self._fixed_g[e] = _2PI * prof.couplings[(i, j)] / 1000.0
        # static on-site Hamiltonian per qubit (detuning + anharmonicity), d x d
        self._h_static = [self._delta[q] * n + 0.5 * self._alpha[q] * (ad @ ad @ a @ a)
                          for q in range(N)]
        # channel layout matches _hamiltonian_slice: drives, then tunable edges, then
        # frequency-control qubits.
        self._n_channels = (len(self.drive_qubits) + len(self.tunable_edges)
                            + len(self.freq_control_qubits))
        # Same rate convention as MultiQubitOptimizer._build_operators. Each jump is
        # (qubit, L, L^dag L); _Dloc[q] feeds the non-Hermitian H_eff = H - (i/2)*sum L^dag L.
        self._jumps = []
        self._Dloc = [np.zeros((self.d, self.d), dtype=complex) for _ in range(N)]
        for q in range(N):
            t1 = max(float(prof.t1_ns[q]), 1.0)
            t2 = max(float(prof.t2_ns[q]), 1.0)
            inv_tphi = max(1.0 / t2 - 1.0 / (2.0 * t1), 0.0)
            L1 = math.sqrt(1.0 / t1) * a
            self._jumps.append((q, L1, L1.conj().T @ L1))
            self._Dloc[q] = self._Dloc[q] + L1.conj().T @ L1
            if inv_tphi > 0.0:
                Lp = math.sqrt(2.0 * inv_tphi) * n
                self._jumps.append((q, Lp, Lp.conj().T @ Lp))
                self._Dloc[q] = self._Dloc[q] + Lp.conj().T @ Lp
        self._diss_on = False          # toggled True inside trajectory evolution

    @classmethod
    def from_optimizer(cls, opt):
        return cls(opt)

    # ---- per-slice control extraction ------------------------------------
    def _controls(self, waveform):
        """Smoothed [0,1] waveform -> signed u [Nt, n_channels], matching the dense
        ``2*x_smooth - 1`` convention (consumed verbatim, NOT re-smoothed)."""
        wf = np.asarray(waveform, dtype=float)
        if wf.ndim == 1:
            wf = wf[:, None]
        if wf.shape[1] != self._n_channels:
            raise ValueError(f"waveform has {wf.shape[1]} channels, expected "
                             f"{self._n_channels}")
        return np.clip(wf, 0.0, 1.0) * 2.0 - 1.0

    def _onsite_h(self, q, u, i, Nt, dt):
        """Time-dependent on-site Hamiltonian h_q(t) (d x d) at slice i: static drift
        plus this qubit's drive (+DRAG quadrature) and frequency control."""
        h = self._h_static[q].copy()
        # drive channels are first, in drive_qubits order
        if q in self.drive_qubits:
            ch = self.drive_qubits.index(q)
            h = h + (u[i, ch] * self.OMEGA_MAX) * self._X
            if self.use_drag:
                alpha = self._alpha[q]
                if 0 < i < Nt - 1:
                    du = (u[i + 1, ch] - u[i - 1, ch]) * self.OMEGA_MAX / (2.0 * dt)
                else:
                    du = 0.0
                h = h + (-du / alpha) * self._Y
        if q in self.freq_control_qubits:
            ch = (len(self.drive_qubits) + len(self.tunable_edges)
                  + self.freq_control_qubits.index(q))
            h = h + (u[i, ch] * self.DELTA_MAX) * self._num
        if self._diss_on:
            # non-Hermitian no-jump drift: H_eff = H - (i/2) sum_k L_k^dag L_k
            h = h - 0.5j * self._Dloc[q]
        return h

    def _bond_coupling_g(self, e, u, i):
        """Total coupling strength on bond e=(q,q+1) at slice i: fixed + tunable."""
        g = self._fixed_g.get(e, 0.0)
        if e in self.tunable_edges:
            ch = len(self.drive_qubits) + self.tunable_edges.index(e)
            g = g + u[i, ch] * self.G_MAX
        return g

    def _bond_hamiltonian(self, q, u, i, Nt, dt):
        """2-site Hamiltonian on bond (q, q+1): the coupling plus on-site terms folded
        in with weights that count each site exactly once across an open chain."""
        e = (q, q + 1)
        g = self._bond_coupling_g(e, u, i)
        Hb = g * self._C2
        # on-site distribution: end sites belong to one bond (weight 1), interior
        # sites are shared by two bonds (weight 1/2 each).
        w_left = 1.0 if q == 0 else 0.5
        w_right = 1.0 if (q + 1) == self.N - 1 else 0.5
        hq = self._onsite_h(q, u, i, Nt, dt)
        hq1 = self._onsite_h(q + 1, u, i, Nt, dt)
        Hb = Hb + w_left * np.kron(hq, self._I) + w_right * np.kron(self._I, hq1)
        return Hb

    def _bond_gate(self, q, u, i, Nt, dt, frac):
        """exp(-i H_bond(q) * dt * frac) as a [d, d, d, d] tensor
        (out_q, out_{q+1}, in_q, in_{q+1})."""
        Hb = self._bond_hamiltonian(q, u, i, Nt, dt)
        U = _expm(-1j * Hb * (dt * frac))
        d = self.d
        return U.reshape(d, d, d, d)

    # ---- full-statevector TEBD (Layer 1: exact, no truncation) -----------
    @staticmethod
    def _apply_two_site(psi, G4, q, N):
        """Apply a [d,d,d,d] 2-site gate on sites (q, q+1) to a batched state tensor
        ``psi`` of shape [M, d, d, ..., d] (N site axes after the batch axis)."""
        # contract G4 indices (in_q, in_{q+1}) = axes (2,3) with psi axes (q+1, q+2)
        # (the +1 offset is the batch axis). Result new axes (out_q, out_{q+1}) land at
        # the front; move them back into place.
        psi = np.tensordot(G4, psi, axes=([2, 3], [q + 1, q + 2]))
        # psi now: [out_q, out_{q+1}, M, <remaining site axes>]
        psi = np.moveaxis(psi, [0, 1], [q + 1, q + 2])
        # moveaxis with the batch axis: after tensordot the batch axis sits at position
        # 2; we want it back at 0.
        return psi

    def _tebd_step(self, psi, u, i, Nt, dt, substeps):
        """One time-slice of second-order symmetric-sweep TEBD on the full state tensor.
        Forward sweep of all bonds at dt/2 then backward at dt/2 -- the SAME bond order
        and step as ``evolve_mps``, so the untruncated MPS reproduces this to machine
        precision (isolating the MPS truncation as the only difference)."""
        N = self.N
        for _ in range(substeps):
            frac = 0.5 * (1.0 / substeps)
            for q in range(N - 1):                       # forward, half step
                psi = self._apply_two_site(psi, self._bond_gate(q, u, i, Nt, dt, frac), q, N)
            for q in range(N - 2, -1, -1):               # backward, half step
                psi = self._apply_two_site(psi, self._bond_gate(q, u, i, Nt, dt, frac), q, N)
        return psi

    def evolve_statevector(self, init_states, waveform, dt_ns=1.0, substeps=1):
        """Evolve a batch of full state vectors through the pulse with exact TEBD.

        ``init_states``: [M, D] complex (D = d**N), rows are input kets.
        Returns [M, D]. Exact up to the second-order Trotter error, which -> 0 as
        ``substeps`` grows (the Layer-1 convergence handle).
        """
        u = self._controls(waveform)
        Nt = u.shape[0]
        d, N = self.d, self.N
        M = init_states.shape[0]
        psi = np.asarray(init_states, dtype=complex).reshape((M,) + (d,) * N)
        for i in range(Nt):
            psi = self._tebd_step(psi, u, i, Nt, dt_ns, substeps)
        return psi.reshape(M, d ** N)

    # ---- MPS representation + truncated TEBD (Layer 2) -------------------
    # MPS = list of N tensors, each [Dl, d, Dr] (Dl=Dr=1 at open ends). The
    # orthogonality centre moves with each gate so truncation is optimal.

    def product_mps(self, local_kets):
        """MPS for a product state. ``local_kets``: [N, d] rows (one ket per site)."""
        return [np.asarray(k, dtype=complex).reshape(1, self.d, 1) for k in local_kets]

    @staticmethod
    def _apply_gate_mps(mps, q, G4, chi_max, sweep_dir, renorm=True):
        """Apply a [d,d,d,d] 2-site gate to MPS sites (q, q+1) and move the
        orthogonality centre in ``sweep_dir`` (+1 right / -1 left), truncating to
        ``chi_max``. Returns the discarded weight (sum of squared dropped Schmidt
        values, renormalized). Assumes the centre is at q (sweep right) or q+1 (left)."""
        A, Bt = mps[q], mps[q + 1]                     # [Dl,d,Dc], [Dc,d,Dr]
        Dl, d, Dc = A.shape
        Dr = Bt.shape[2]
        theta = np.tensordot(A, Bt, axes=([2], [0]))   # [Dl,d,d,Dr]
        theta = np.tensordot(G4, theta, axes=([2, 3], [1, 2]))  # [d,d,Dl,Dr]
        theta = np.transpose(theta, (2, 0, 1, 3))      # [Dl,d,d,Dr]
        mat = theta.reshape(Dl * d, d * Dr)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        total = float(np.sum(S ** 2))
        keep = min(chi_max, np.sum(S > 1e-14))
        keep = max(int(keep), 1)
        disc = float(np.sum(S[keep:] ** 2)) / max(total, 1e-300)
        U, S, Vh = U[:, :keep], S[:keep], Vh[:keep, :]
        if renorm:                                     # closed system: keep unit norm
            nrm = math.sqrt(float(np.sum(S ** 2)))
            S = S / max(nrm, 1e-300)
        # else (trajectory): keep the non-unitary norm decay -- it carries the
        # no-jump survival probability the MCWF sampler reads.
        if sweep_dir > 0:                              # centre -> q+1
            mps[q] = U.reshape(Dl, d, keep)
            mps[q + 1] = (np.diag(S) @ Vh).reshape(keep, d, Dr)
        else:                                          # centre -> q
            mps[q] = (U @ np.diag(S)).reshape(Dl, d, keep)
            mps[q + 1] = Vh.reshape(keep, d, Dr)
        return disc

    def evolve_mps(self, local_kets, waveform, dt_ns=1.0, substeps=1, chi_max=64):
        """Evolve a product-state MPS through the pulse with truncated symmetric-sweep
        TEBD. Returns (mps, max_discarded_weight). Second-order in dt (forward then
        backward half-step sweeps); exact as chi_max -> infinity."""
        u = self._controls(waveform)
        Nt, N = u.shape[0], self.N
        mps = self.product_mps(local_kets)
        max_disc = 0.0
        for i in range(Nt):
            for _ in range(substeps):
                sdt = (dt_ns / substeps)
                # forward sweep, half step (centre starts at 0, moves right)
                for q in range(N - 1):
                    G = self._bond_gate(q, u, i, Nt, dt_ns, 0.5 * sdt / dt_ns)
                    max_disc = max(max_disc, self._apply_gate_mps(mps, q, G, chi_max, +1))
                # backward sweep, half step (centre at N-1, moves left)
                for q in range(N - 2, -1, -1):
                    G = self._bond_gate(q, u, i, Nt, dt_ns, 0.5 * sdt / dt_ns)
                    max_disc = max(max_disc, self._apply_gate_mps(mps, q, G, chi_max, -1))
        return mps, max_disc

    def mps_to_vector(self, mps):
        """Contract an MPS to a full state vector [d**N] (small-N validation only)."""
        psi = mps[0]
        for q in range(1, self.N):
            psi = np.tensordot(psi, mps[q], axes=([-1], [0]))
        return psi.reshape(self.d ** self.N)

    # ---- exact closed-system process fidelity via TEBD (validation) ------
    def process_fidelity_tebd(self, waveform, dt_ns=1.0, substeps=1):
        """Closed-system F_proc over the full 2**N computational subspace, evolved with
        TEBD. Same metric as ``MultiQubitOptimizer.process_fidelity_sparse``; agrees with
        it as ``substeps`` -> infinity. This is the exact (exponential) quantity -- used
        to VALIDATE the TEBD machinery, not the cheap path. Evolves all 2**N inputs as
        full state vectors, so it is small-N only."""
        N, d = self.N, self.d
        comp = list(product((0, 1), repeat=N))
        D = d ** N
        # computational basis input kets as rows
        psi0 = np.zeros((len(comp), D), dtype=complex)
        comp_idx = []
        for r, bits in enumerate(comp):
            k = 0
            for b in bits:
                k = k * d + b
            psi0[r, k] = 1.0
            comp_idx.append(k)
        out = self.evolve_statevector(psi0, waveform, dt_ns, substeps)   # [dc, D]
        Uc = out[:, comp_idx].T                       # dc x dc comp-subspace evolution
        Ut = self.opt.u_target.detach().cpu().numpy()
        dc = len(comp)
        M = np.trace(Ut.conj().T @ Uc)
        return float((M.real ** 2 + M.imag ** 2) / (dc * dc))

    # ---- open-system trajectory unraveling (Layer 3) ---------------------
    def _right_canonicalize(self, mps):
        """Right-canonicalize sites 1..N-1 (each Vh-isometric) via SVD, leaving the
        orthogonality centre at site 0. Works on ANY MPS (no prior canonical form).
        Returns the state norm ||mps[0]||."""
        for q in range(self.N - 1, 0, -1):
            Dl, d, Dr = mps[q].shape
            U, S, Vh = np.linalg.svd(mps[q].reshape(Dl, d * Dr), full_matrices=False)
            mps[q] = Vh.reshape(-1, d, Dr)
            mps[q - 1] = np.tensordot(mps[q - 1], U * S, axes=([2], [0]))
        return math.sqrt(float(np.sum(np.abs(mps[0]) ** 2)))

    def _sweep_site_rdms(self, mps):
        """Single-site reduced density matrices for all sites. Assumes centre at 0;
        left-canonicalizes as it sweeps (centre ends at N-1). Returns [N] (d x d)."""
        rdms = []
        for q in range(self.N):
            M = mps[q]
            rho = np.einsum('asb,atb->st', M, M.conj())
            tr = float(np.trace(rho).real)
            rdms.append(rho / (tr if tr > 1e-300 else 1.0))
            if q < self.N - 1:
                Dl, d, Dr = M.shape
                U, S, Vh = np.linalg.svd(M.reshape(Dl * d, Dr), full_matrices=False)
                mps[q] = U.reshape(Dl, d, -1)
                mps[q + 1] = np.tensordot(S[:, None] * Vh, mps[q + 1], axes=([1], [0]))
        return rdms

    def evolve_trajectory(self, local_kets, waveform, dt_ns=1.0, substeps=1,
                          chi_max=64, rng=None):
        """One Monte-Carlo-wavefunction trajectory: pure-MPS evolution under the non-
        Hermitian H_eff with stochastic single-site jumps (norm-threshold sampling).
        Returns (mps, n_jumps); mps is normalized. ``rng``: np.random.Generator."""
        if rng is None:
            rng = np.random.default_rng()
        u = self._controls(waveform)
        Nt, N = u.shape[0], self.N
        self._diss_on = True
        try:
            mps = self.product_mps(local_kets)
            eps = float(rng.random())
            n_jumps = 0
            self._traj_max_disc = 0.0
            for i in range(Nt):
                for _ in range(substeps):
                    frac = 0.5 * (1.0 / substeps)
                    for q in range(N - 1):
                        dsc = self._apply_gate_mps(mps, q, self._bond_gate(q, u, i, Nt, dt_ns, frac),
                                                   chi_max, +1, renorm=False)
                        self._traj_max_disc = max(self._traj_max_disc, dsc)
                    for q in range(N - 2, -1, -1):
                        dsc = self._apply_gate_mps(mps, q, self._bond_gate(q, u, i, Nt, dt_ns, frac),
                                                   chi_max, -1, renorm=False)
                        self._traj_max_disc = max(self._traj_max_disc, dsc)
                norm2 = float(np.sum(np.abs(mps[0]) ** 2))       # centre at 0 after sweep
                if norm2 < eps:                                  # a jump occurred
                    mps[0] = mps[0] / math.sqrt(norm2)           # normalized pre-jump state
                    rdms = self._sweep_site_rdms(mps)
                    w = np.array([max(float(np.trace(LdL @ rdms[q]).real), 0.0)
                                  for (q, _L, LdL) in self._jumps])
                    tot = float(w.sum())
                    k = int(rng.choice(len(self._jumps), p=w / tot)) if tot > 0 else 0
                    q, L, _ = self._jumps[k]
                    mps[q] = np.tensordot(L, mps[q], axes=([1], [1])).transpose(1, 0, 2)
                    self._right_canonicalize(mps)                # centre -> 0
                    nrm = math.sqrt(float(np.sum(np.abs(mps[0]) ** 2)))
                    mps[0] = mps[0] / max(nrm, 1e-300)
                    n_jumps += 1
                    eps = float(rng.random())
            nrm = math.sqrt(float(np.sum(np.abs(mps[0]) ** 2)))
            mps[0] = mps[0] / max(nrm, 1e-300)
        finally:
            self._diss_on = False
        return mps, n_jumps

    def _embed_target_vector(self, local_kets):
        """Full D-vector U_target|psi_in> for a computational product input (rows of
        ``local_kets`` live in span{|0>,|1>} of each d-level site)."""
        psi = np.asarray(local_kets[0], dtype=complex)
        for q in range(1, self.N):
            psi = np.kron(psi, np.asarray(local_kets[q], dtype=complex))
        ci = self.opt._comp_idx.cpu().numpy()
        Ut = self.opt.u_target.detach().cpu().numpy()
        out = np.zeros_like(psi)
        out[ci] = Ut @ psi[ci]
        return out

    def witness_open(self, ensemble, waveform, dt_ns=1.0, substeps=1, chi_max=64,
                     n_traj=200, seed=0):
        """Restricted-ensemble average gate-fidelity WITNESS under the open system, via
        trajectory unraveling over pure MPS.

        This is NOT the entanglement/process fidelity: the product-state ensemble is not
        a 2-design, so the witness does not map through F_avg=(d*F_proc+1)/(d+1) and
        typically OVER-estimates (it under-samples hard-to-preserve entangled inputs). It
        is a fidelity witness, reported with its statistical (trajectory) error.

        ensemble: list of [N, d] computational product inputs. Returns
        {witness, sem, n_traj, mean_jumps}.
        """
        rng = np.random.default_rng(seed)
        per_input, sem_terms, total_jumps, max_disc = [], [], 0, 0.0
        for kets in ensemble:
            tgt = self._embed_target_vector(kets)
            fids = np.empty(n_traj)
            for t in range(n_traj):
                mps, nj = self.evolve_trajectory(kets, waveform, dt_ns, substeps, chi_max, rng)
                total_jumps += nj
                max_disc = max(max_disc, self._traj_max_disc)
                fids[t] = abs(np.vdot(tgt, self.mps_to_vector(mps))) ** 2
            per_input.append(float(fids.mean()))
            sem_terms.append(float(fids.var()) / n_traj)        # variance of this mean
        per_input = np.array(per_input)
        return {
            "witness": float(per_input.mean()),
            # SEM of the ensemble-averaged witness = sqrt(mean of per-input mean-variances)
            "sem": float(math.sqrt(sum(sem_terms)) / len(per_input)),
            "n_traj": n_traj,
            "mean_jumps": total_jumps / (len(ensemble) * n_traj),
            # chi-convergence ship gate: the worst Schmidt weight truncation discarded.
            # Small => chi_max captured the entanglement; large => raise chi_max (the
            # witness is not converged and must NOT be trusted).
            "max_discarded": float(max_disc),
        }
