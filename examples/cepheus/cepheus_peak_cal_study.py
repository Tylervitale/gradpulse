"""Offline proof that the Level-B peak calibration RESOLVES the optimum before any QPU spend.

Why this exists: the first $33 on-device peak sweep was wasted because it selected the peak by
``min(fit r_cz)``. On short-ladder noise the interleaved-RB fit clamps r_cz->0, so a DESTROYED
gate (whose fit happened to clamp) was crowned "best". This script reproduces that failure and
proves the fix -- selecting by MAX interleaved survival -- using the REAL 11520-element Clifford
group (gradpulse.rb) under realistic shot + finite-sequence noise.

Model: the Cliffords run the device's NATIVE CZ (~0.45% err); the INTERLEAVED gate is the
gradpulse pulse at scale ``s`` -> conditional phase pi*s (ideal CZ at s=1) plus a leakage floor.
The cal must find the min of the resulting U-shaped error curve under noise.

Result (the numbers backing run_irb_on_braket.select_best_peak / fit_resolved): across four
budgets including the exact failed $33 config, max-survival recovers the optimum 100% of the
time vs ~38-54% for fit->argmin. The estimator was the bug, not the budget.

Run:  python examples/cepheus_peak_cal_study.py
"""
import importlib.util
import pathlib

import numpy as np

import gradpulse.rb as rb

# import fit_irb from the sibling experiment script
_EX = pathlib.Path(__file__).resolve().parent / "run_irb_on_braket.py"
spec = importlib.util.spec_from_file_location("irb_ex", _EX)
irb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(irb)

GROUP = rb.two_qubit_cliffords()
NAT_P = 0.9955                      # device native CZ used inside the Cliffords (~0.45% err)
NATIVE = rb.native_superops(rb.depolarizing_gate_superop(NAT_P, leak=0.0))


def pulse_superop(s, p=0.992, leak=0.006):
    """gradpulse pulse at scale s: conditional phase pi*s (ideal CZ at s=1) + floor."""
    U4 = np.diag([1.0, 1.0, 1.0, np.exp(1j * np.pi * s)]).astype(complex)
    return rb.depolarizing_gate_superop(p, U4=U4, leak=leak)


def true_r(s, lengths=(1, 2, 4, 8, 16, 32)):
    """Noise-free per-scale CZ error, to define the optimum and check estimators against it."""
    rng = np.random.default_rng(0)
    intl, _ = rb._run_rb(GROUP, rb.native_superops(pulse_superop(s)), list(lengths), 30, rng,
                         interleave=pulse_superop(s))
    ref, _ = rb._run_rb(GROUP, NATIVE, list(lengths), 30, rng, interleave=None)
    a_i = rb._fit_single_exp(np.array(lengths), intl)[0]
    a_r = rb._fit_single_exp(np.array(lengths), ref)[0]
    return rb._r_from_alpha(a_i / a_r if a_r > 0 else 1.0)


SCALES = np.linspace(0.6, 1.4, 9)        # scan through the optimum at s=1


def measure(s, lengths, n_seq, shots, rng):
    """One noisy cal measurement at scale s: real RB survivals + binomial shot noise."""
    intl, _ = rb._run_rb(GROUP, rb.native_superops(pulse_superop(s)), lengths, n_seq, rng,
                         interleave=pulse_superop(s))
    tot = n_seq * shots
    return rng.binomial(tot, np.clip(intl, 0, 1)) / tot


def ref_meas(lengths, n_seq, shots, rng):
    ref, _ = rb._run_rb(GROUP, NATIVE, lengths, n_seq, rng, interleave=None)
    tot = n_seq * shots
    return rng.binomial(tot, np.clip(ref, 0, 1)) / tot


def mc(lengths, n_seq, shots, s_opt, n_mc=24, tol=0.12):
    """Fraction of MC trials each estimator's chosen scale is within tol of the true optimum."""
    okA = okB = 0
    Lmax = max(lengths)
    for t in range(n_mc):
        rng = np.random.default_rng(1000 + t)
        refy = ref_meas(lengths, n_seq, shots, rng)
        rA, sB = [], []
        for s in SCALES:
            iy = measure(s, lengths, n_seq, shots, rng)
            refseq = [{"length": m, "survival": float(refy[i]), "interleaved": False}
                      for i, m in enumerate(lengths)]
            intseq = [{"length": m, "survival": float(iy[i]), "interleaved": True}
                      for i, m in enumerate(lengths)]
            rA.append(irb.fit_irb(refseq, intseq, lengths, n_boot=1)["r_cz"])   # estimator A: fit
            sB.append(iy[lengths.index(Lmax)])                                  # estimator B: surv
        okA += abs(SCALES[int(np.argmin(rA))] - s_opt) <= tol
        okB += abs(SCALES[int(np.argmax(sB))] - s_opt) <= tol
    return okA / n_mc, okB / n_mc


def main():
    print("true CZ error vs scale (noise-free, the thing the cal must find the min of):")
    tr = [true_r(s) for s in SCALES]
    for s, r in zip(SCALES, tr):
        print(f"  s={s:.2f}: r_cz={r:.4f}")
    s_opt = SCALES[int(np.argmin(tr))]
    print(f"  -> true optimum scale = {s_opt:.2f}\n")

    print("MC resolution accuracy (fraction within 0.12 of true optimum):")
    for tag, L, ns, sh in [
            ("FAILED $33 cfg  (len1-8, 2seed, 500sh)", [1, 2, 4, 8], 2, 500),
            ("B short          (len1-8, 4seed, 300sh)", [1, 2, 4, 8], 4, 300),
            ("B deep           (len1-16,4seed,300sh)", [1, 2, 4, 8, 16], 4, 300),
            ("B deeper         (len1-32,4seed,300sh)", [1, 2, 4, 8, 16, 32], 4, 300)]:
        a, b = mc(L, ns, sh, s_opt)
        print(f"  {tag}:  fit(A)={a:.0%}   max-survival(B)={b:.0%}")


if __name__ == "__main__":
    main()
