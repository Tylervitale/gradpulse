"""Unitarity (purity) RB -- separate COHERENT (control) error from INCOHERENT (decoherence).

Standard interleaved RB gives only the TOTAL gate error r. It cannot tell whether that
error is coherent (a unitary over/under-rotation you could calibrate away) or incoherent
(T1/T2 decoherence you cannot). Unitarity RB (Wallman et al., NJP 2015) adds that split by
measuring how PURE the state stays, not just whether it survives:

  * survival RB:   apply m random Cliffords + recovery, measure P(|00>)  -> decay alpha -> r
  * unitarity RB:  apply m random Cliffords (NO recovery), measure the PURITY Tr(rho^2)
                   -> decay u (the "unitarity").  u=1 => purely coherent; u<1 => incoherent.

Decomposition (d=4):  r_incoherent = (d-1)/d * (1 - sqrt(u));  r_coherent = r - r_incoherent.

WHY IT COSTS ~3-9x A NORMAL RB RUN (the headline ask):
Survival reads ONE number from the computational basis -- P(|00>). Purity Tr(rho^2) is
Tr(rho^2) = (1/d) * sum_P <P>^2 over ALL d^2 = 16 two-qubit Paulis. The 15 non-identity
Pauli expectations are NOT all simultaneously measurable -- they need the 9 measurement
bases {X,Y,Z} (x) {X,Y,Z}, each a SEPARATE circuit (a basis-rotation appended before
readout). So every (sequence, length) point becomes 9 circuits instead of 1. That 9x in
circuits (modestly offset by fewer shots/basis) is the 3-9x cost. It is intrinsic to
measuring a quadratic functional of rho on a device that only reads the computational basis.

This file VERIFIES the protocol in simulation (exact purity from rho), then prints the
hardware circuit/cost multiplier. The hardware submission reuses the IRB path
(run_irb_on_braket.py) with the 9 basis-rotation circuits per point.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from gradpulse.rb import two_qubit_cliffords

D = 4
G = two_qubit_cliffords()
U = G.unitaries                                   # 4x4 computational-subspace Cliffords
RHO0 = np.zeros((D, D), complex); RHO0[0, 0] = 1.0
ZZ = np.diag(np.kron([1., -1.], [1., -1.]))       # diag(1,-1,-1,1)

def _noise(rho, kind, s):
    if kind in ("coherent", "mixed"):             # unitary over-rotation exp(i s ZZ)
        uc = np.diag(np.exp(1j * s * np.diag(ZZ)))
        rho = uc @ rho @ uc.conj().T
    if kind in ("depol", "mixed"):                # depolarizing toward maximally mixed
        rho = (1.0 - s) * rho + s * np.eye(D) / D
    return rho

def _grid_fit(m, y, floor):
    """y = A * base^k + floor (k = m for survival, m-1 for purity); LS over base."""
    m = np.asarray(m, float); yb = np.asarray(y, float) - floor
    best = None
    for base in np.linspace(0.5, 0.99999, 4000):
        bb = base ** m
        A = float(bb @ yb / (bb @ bb))
        err = float(np.sum((A * bb - yb) ** 2))
        if best is None or err < best[0]:
            best = (err, base)
    return best[1]

def run(kind, s, lengths, n_seq=60, seed=0):
    rng = np.random.default_rng(seed)
    surv, pur = [], []
    for m in lengths:
        sa = pa = 0.0
        for _ in range(n_seq):
            rho = RHO0.copy(); prod = np.eye(D, dtype=complex)
            for _ in range(m):
                idx = int(rng.integers(len(U)))
                rho = U[idx] @ rho @ U[idx].conj().T
                rho = _noise(rho, kind, s)
                prod = U[idx] @ prod
            pa += float(np.real(np.trace(rho @ rho)))          # PURITY (no recovery)
            rec = G.index_of(prod.conj().T)                    # recovery for SURVIVAL
            rs = _noise(U[rec] @ rho @ U[rec].conj().T, kind, s)
            sa += float(np.real(rs[0, 0]))
        surv.append(sa / n_seq); pur.append(pa / n_seq)
    surv, pur = np.array(surv), np.array(pur)
    alpha = _grid_fit(lengths, surv, 1.0 / D)
    u = _grid_fit(np.array(lengths) - 1, pur, 1.0 / D)
    r = (D - 1) / D * (1.0 - alpha)
    r_inc = (D - 1) / D * (1.0 - np.sqrt(max(u, 0.0)))
    r_coh = max(r - r_inc, 0.0)
    return dict(r=r, u=u, r_inc=r_inc, r_coh=r_coh,
                coh_frac=(r_coh / r if r > 0 else 0.0))

if __name__ == "__main__":
    L = [1, 2, 4, 8, 16, 32]
    print("=== VERIFY: does unitarity RB separate coherent vs incoherent? ===")
    print(f"{'noise model':22s}{'r(total)':>10s}{'u':>8s}{'r_coh':>9s}{'r_inc':>9s}{'coh %':>7s}")
    for label, kind, s in [("pure incoherent (depol)", "depol", 0.02),
                           ("pure coherent (ZZ rot)", "coherent", 0.05),
                           ("mixed (both)", "mixed", 0.02)]:
        d = run(kind, s, L)
        print(f"{label:22s}{d['r']*100:9.3f}%{d['u']:8.4f}{d['r_coh']*100:8.3f}%"
              f"{d['r_inc']*100:8.3f}%{d['coh_frac']*100:6.0f}%")
    print("\nExpect: depol -> ~0% coherent | ZZ rot -> ~100% coherent | mixed -> in between\n")

    # hardware cost multiplier vs a standard interleaved-RB run -- EXACT, via the
    # same cost function that prices the IRB run ($0.30/task + $0.000425/shot).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from gradpulse.braket_bridge import estimate_experiment_cost as E
    print("=== HARDWARE COST (exact, Rigetti-Cepheus pricing) ===")
    N_BASES = 9                                    # {X,Y,Z}x{X,Y,Z} to reconstruct purity
    POINTS = 6 * 5                                  # 6 lengths x 5 seeds (lean)
    lean = E(2 * POINTS, 300).total_usd            # the A run: ref+int survival IRB
    u_ref = E(POINTS * N_BASES, 300).total_usd      # ref-only purity: "noise is incoherent"
    u_int = E(2 * POINTS * N_BASES, 300).total_usd  # ref+int purity: split for the CZ itself
    print(f"lean survival IRB (the A run)      : {2*POINTS:4d} circ -> ${lean:7.2f}  (1.0x)")
    print(f"unitarity, ref-only  (x{N_BASES} bases)   : {POINTS*N_BASES:4d} circ -> "
          f"${u_ref:7.2f}  ({u_ref/lean:.1f}x)")
    print(f"unitarity, ref+int   (x{N_BASES} bases)   : {2*POINTS*N_BASES:4d} circ -> "
          f"${u_int:7.2f}  ({u_int/lean:.1f}x)")
    print("WHY: Braket bills per task (circuit). Purity needs 9 measurement bases, so 9x")
    print("circuits -> ~9x task fee, which is 70% of each shallow-circuit cost. The 9x is")
    print("intrinsic: a quadratic functional of rho (Tr rho^2) is unmeasurable in one basis.")
