"""Reproducible study: what budget/design resolves a GOOD (sub-0.5%) CZ on hardware.
Backs the resolution guidance in run_irb_on_braket.py and tests/test_irb_resolution.py.

Monte Carlo over the REAL 11520-Clifford group (sequence-to-sequence variance comes from
the actual per-Clifford gate counts). Models, for Cepheus pair (16,25):
  * per-CZ depolarizing at the device-reported r_CZ = 0.406%,
  * idle T2 decay over the circuit DURATION (the decisive effect a first pass missed) --
    each native gate carries exp(-t_gate/T2) of dephasing, so a sequence longer than
    ~T2/clifford-time floors at 1/d regardless of gate fidelity, and
  * the measured ASYMMETRIC readout (the length-1 canary showed errors piled into '10' ->
    one qubit reads 1 when it should read 0), via a per-qubit confusion matrix.

It is VALIDATED against the real canary (2026-06-20): length-1 survival ~0.74 and the
m=128 depth canary flooring at ~0.24 both fall out of the model. Conclusions (now matching
hardware):
  1. GO LONG (m~128) + FREE asymptote -- counterintuitively the BEST design. The long
     sequences floor at the asymptote, and BECAUSE real readout is asymmetric the asymptote
     is NOT 1/d=0.25 (it's ~0.20 here) and must be FIT; the floored long points are what pin
     it. A short ladder (m=32) leaves the asymptote underconstrained -> ~7x noisier
     (measured: m=128 free +-0.10% vs m=32 free +-0.75%, same budget).
  2. FIXED-1/d is the trap: tight but BIASED under asymmetric readout (m=128: +-0.08% but
     biased ~+0.09% past truth = confidently wrong). RB's decay rate -> r_CZ is SPAM-robust,
     so FREE recovers the truth; only the asymptote handling differs.
  (A single floored canary point looks "dead" -- that misread it as "m=128 too long". In a
   full ladder those points are the asymptote anchor. T2 sets WHERE the floor is, not a cap.)

Gate times (t_CZ, t_RX) are estimates; T2 is the measured 16.2/14.3 us. Run: python this_file.py
"""
import numpy as np

from gradpulse.rb import (COMP_IDX, _GENERATORS, _clifford_superop, _embed4_in_9,
                          _fit_single_exp, _rho0_vec, depolarizing_gate_superop,
                          native_superops, superop_from_unitary, two_qubit_cliffords)

# --- device numbers (Cepheus pair 16,25, live calibration 2026-06-20) ---
R_MEASURED = 4.06e-3                 # device interleaved-RB CZ error (the TOTAL, measured)
T2_US = 14.3                         # worse of the pair (q25); idle-dephasing time
T_CZ_NS, T_RX_NS = 60.0, 40.0        # gate durations (RZ is virtual = 0); estimates
# Split the measured r_CZ (which already includes in-gate coherence loss) into the
# depolarizing/coherent part + decoherence during the CZ's own duration, to avoid
# double-counting T2-during-CZ on top of the depolarizing parameter.
CZ_T2_R = 0.75 * (1.0 - np.exp(-(T_CZ_NS * 1e-3) / T2_US))
R_DEPOL = max(R_MEASURED - CZ_T2_R, 5e-4)        # coherent remainder (guard > 0)
P_DEPOL = 1.0 - R_DEPOL / 0.75                    # depolarizing param: 0.75*(1-p)=R_DEPOL
# Asymmetric readout calibrated to the length-1 canary counts {00:74,01:5,10:18,11:3}:
# first qubit reads 1 when true 0 with prob A1G0; |1> readout assumed good (0.98).
A1G0, B1G0, RO1 = 0.196, 0.063, 0.98


def _depol(alpha):
    """Depolarizing channel with parameter alpha (off-diagonals * alpha, drives toward
    maximally mixed) -- our effective idle-decoherence step over one gate."""
    return depolarizing_gate_superop(alpha, U4=np.eye(4, dtype=complex))


def decohered_native(cz_noisy):
    """Native-generator superops with idle T2 decay folded in per gate duration:
    H = one physical RX (-> exp(-t_RX/T2) dephasing), S = virtual RZ (no time), CZ ->
    exp(-t_CZ/T2). avg Clifford ~1.88 CZ, so this caps the coherent depth at ~T2."""
    a_rx = _depol(np.exp(-(T_RX_NS * 1e-3) / T2_US))
    a_cz = _depol(np.exp(-(T_CZ_NS * 1e-3) / T2_US))
    ideal = native_superops(superop_from_unitary(_embed4_in_9(_GENERATORS["CZ"])))
    return {
        "H1": a_rx @ ideal["H1"], "H2": a_rx @ ideal["H2"],   # 1 RX each
        "S1": ideal["S1"], "S2": ideal["S2"],                 # virtual, no decoherence
        "CZ": a_cz @ cz_noisy,                                # gate error + T2
    }, (a_cz @ cz_noisy)


def _confusion():
    """4x4 readout confusion C[measured, true] = C_A (x) C_B from the canary asymmetry."""
    ca = np.array([[1 - A1G0, 1 - RO1], [A1G0, RO1]])
    cb = np.array([[1 - B1G0, 1 - RO1], [B1G0, RO1]])
    return np.kron(ca, cb)


CONF = _confusion()


def seq_pops(group, native, m, n_seq, rng, cache, cz=None):
    """Per-sequence final computational populations [P00,P01,P10,P11] (no readout yet)."""
    n_cliff = len(group)
    cz_ideal = _embed4_in_9(_GENERATORS["CZ"]) if cz is not None else None
    rho0 = _rho0_vec()
    out = np.empty((n_seq, 4))
    for j in range(n_seq):
        vec = rho0.copy()
        ideal = np.eye(9, dtype=complex)
        for _ in range(m):
            idx = int(rng.integers(n_cliff))
            vec = _clifford_superop(idx, group, native, cache) @ vec
            ideal = _embed4_in_9(group.unitaries[idx]) @ ideal
            if cz is not None:
                vec = cz @ vec
                ideal = cz_ideal @ ideal
        rec = group.index_of((ideal.conj().T)[np.ix_(COMP_IDX, COMP_IDX)])
        vec = _clifford_superop(rec, group, native, cache) @ vec
        rho = vec.reshape(9, 9)
        out[j] = [float(np.real(rho[i, i])) for i in COMP_IDX]
    return out


def _measured_survival(pops):
    """Apply the asymmetric readout confusion -> probability of measuring '00'."""
    return (CONF @ pops.T)[0]          # row 0 = P(measure 00) per sequence


def fit_alpha(m, y, b=None):
    """Single-exp fit. b=None -> free asymptote (3-param); else fixed asymptote b."""
    m = np.asarray(m, float)
    if b is None:
        a, _, _ = _fit_single_exp(m, y)
        return a
    yb = np.asarray(y, float) - b
    best = None
    for a in np.linspace(0.50, 0.99999, 4000):
        basis = a ** m
        A = float(np.dot(basis, yb) / np.dot(basis, basis))
        err = float(np.sum((A * basis - yb) ** 2))
        if best is None or err < best[0]:
            best = (err, a)
    return best[1]


def experiment(group, native, cz, lengths, n_seq, n_shots, rng, cache, fixed_b):
    ref = np.empty(len(lengths)); intl = np.empty(len(lengths))
    for li, m in enumerate(lengths):
        pr = _measured_survival(seq_pops(group, native, m, n_seq, rng, cache))
        pi = _measured_survival(seq_pops(group, native, m, n_seq, rng, cache, cz=cz))
        ref[li] = (rng.binomial(n_shots, np.clip(pr, 0, 1)) / n_shots).mean()
        intl[li] = (rng.binomial(n_shots, np.clip(pi, 0, 1)) / n_shots).mean()
    b = 0.25 if fixed_b else None
    return 0.75 * (1.0 - fit_alpha(lengths, intl, b) / fit_alpha(lengths, ref, b))


def cost(n_lengths, n_seq, n_shots):
    n_circ = n_lengths * n_seq * 2
    return 0.30 * n_circ + 0.000425 * n_circ * n_shots


def run(label, group, native, cz, cache, lengths, n_seq, n_shots, fixed_b, T=100):
    rng = np.random.default_rng(2024)
    rs = np.array([experiment(group, native, cz, lengths, n_seq, n_shots, rng, cache,
                              fixed_b) for _ in range(T)]) * 100
    mean, std = rs.mean(), rs.std()
    lo95, hi95 = np.percentile(rs, [2.5, 97.5])
    pred = R_MEASURED * 100
    print(f"\n  {label}: max len {max(lengths)}, "
          f"{'FIXED 1/d' if fixed_b else 'free'} asymptote, {n_seq}seq x {n_shots}sh "
          f"(${cost(len(lengths), n_seq, n_shots):.0f})")
    print(f"     r_CZ = {mean:.3f}% +/- {std:.3f}%  (bias {mean-pred:+.3f}%)  "
          f"95% CI [{lo95:.3f},{hi95:.3f}]%  excl 2x? {hi95<0.812}  incl 0? {lo95<=0}")


def main():
    group = two_qubit_cliffords()
    cz_noisy = depolarizing_gate_superop(P_DEPOL)
    native, cz_deco = decohered_native(cz_noisy)
    cache: dict = {}
    print(f"Cepheus (16,25): r_CZ={R_MEASURED*100:.3f}%, T2={T2_US}us, "
          f"t_CZ={T_CZ_NS:.0f}ns t_RX={T_RX_NS:.0f}ns, asymmetric readout\n" + "=" * 70)

    # --- VALIDATE the model against the real canary (len-1 ~0.74, m=128 ~0.24) ---
    rng = np.random.default_rng(0)
    s1 = _measured_survival(seq_pops(group, native, 1, 400, rng, cache)).mean()
    s128 = _measured_survival(seq_pops(group, native, 128, 200, rng, cache, cz=cz_deco)).mean()
    print("VALIDATION vs real canary (2026-06-20):")
    print(f"  length-1 survival: model {s1:.3f}  vs measured 0.740")
    print(f"  m=128 survival:    model {s128:.3f}  vs measured 0.240 (floor ~0.20-0.25)")
    # measured asymptote under asymmetric readout (true state -> maximally mixed)
    asym = float((CONF @ np.full(4, 0.25))[0])
    print(f"  readout-induced asymptote (NOT 1/d=0.25): {asym:.3f} -> fixed-1/d fit is biased")

    LONG = (1, 2, 4, 8, 16, 32, 64, 128)
    print("\n" + "=" * 70 + "\nDESIGN (free asymptote): GO LONG -- the floored long points PIN the\n"
          "asymmetric-readout asymptote; a short ladder leaves it underconstrained:")
    run("(A) long  m=128", group, native, cz_deco, cache, LONG, 15, 400, False)
    run("(B) mid   m=44 ", group, native, cz_deco, cache, (1, 2, 4, 8, 16, 32, 44), 15, 400, False)
    run("(C) short m=32 ", group, native, cz_deco, cache, (1, 2, 4, 8, 16, 24, 32), 15, 400, False)

    print("\n" + "=" * 70 + "\nFIT at the long ladder: FREE asymptote (unbiased) vs FIXED 1/d (biased\n"
          "by the asymmetric readout, asymptote is really ~0.20 not 0.25):")
    run("(D) m=128 FREE asymptote ", group, native, cz_deco, cache, LONG, 15, 400, False)
    run("(E) m=128 FIXED 1/d", group, native, cz_deco, cache, LONG, 15, 400, True)

    print("\n" + "=" * 70 + "\nBUDGET at the winning design (m=128, free asymptote):")
    run("(F) cheap  7seq x 500sh ", group, native, cz_deco, cache, LONG, 7, 500, False)
    run("(G) mid   15seq x 400sh ", group, native, cz_deco, cache, LONG, 15, 400, False)
    run("(H) tight 25seq x 800sh ", group, native, cz_deco, cache, LONG, 25, 800, False)


if __name__ == "__main__":
    main()
