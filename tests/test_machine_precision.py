"""Machine-precision gate: the independent solvers don't merely agree to ~10^-7,
they CONVERGE to machine precision -- and the operating-point gap is provably pure
Trotter splitting error, not a model disagreement.

Why this test exists (the epistemic point)
-------------------------------------------
``tests/test_reproducibility.py`` locks the *reported* numbers via three-solver
consensus at the 1 ns operating point, where the genuinely independent legs sit
~2.3x10^-7 apart. That gap is the headline cross-check -- but on its own a reader
could ask the sharp question: *is 2.3x10^-7 a controlled discretization error, or a
real disagreement between two models that just happens to be small?* This file
answers it, and in doing so separates the two things the word "triple-solver" is
doing:

  * **Implementation check (machine precision, least probative).** The PyTorch
    optimizer and QuTiP's matched piecewise-constant build share the *same*
    Lie-Trotter split, so they agree to ~10^-14. That confirms the operator build
    and the fidelity contraction were transcribed correctly in two independent
    codebases -- but, sharing the split, it says NOTHING about the split itself.
    Demonstrated here by: optimizer(K=1) == QuTiP to <10^-11, while BOTH sit
    ~2.3x10^-7 from the exact-generator Liouvillian (different scheme).

  * **Physics check (the load-bearing number).** The NumPy Liouvillian takes the
    exact exponential of the *full* generator per slice -- which, because the
    controls are piecewise-constant, is the EXACT solution of the discretized
    master equation. The optimizer's split approximates that same exact propagator;
    the 2.3x10^-7 is purely the split's O(dt) error. Refining the integrator's
    sub-stepping (holding the physical pulse fixed) drives the optimizer's F_proc to
    the Liouvillian value to machine precision -- Richardson-extrapolated to dt->0,
    they agree to ~10^-13. THIS is what earns "independently cross-checked to machine
    precision (in the converged limit)": two solvers sharing no operator-build, no
    matrix-exponential, and no stepping scheme, meeting at the float64 floor.

All of this is evaluation-only and never enters the optimization loop: 2.3x10^-7 is
~4 orders of magnitude below the ~10^-3 decoherence floor, so it is physically
irrelevant to what the optimizer does. The value here is purely epistemic -- it
upgrades the claim from "the independent solvers agree to 10^-7" to "the independent
solvers converge to machine precision, and the operating-point gap is provably pure
first-order splitting error."

Tolerances are DERIVED from the measured convergence on the committed pulse (see the
constants below), not tuned to pass. Double precision is mandatory throughout: the
single-precision (~10^-6) integrator floor would bury the entire 10^-7 signal.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradpulse import (ParametricCouplerProfile, ParametricCZOptimizer,
                       liouville_f_proc)
from gradpulse.parametric import DEVICE

FIXTURE = Path(__file__).parent / "fixtures" / "reference_cz_pulse.json"

# --- Tolerances, derived from the measured convergence (double precision) -------
# Matched pair (optimizer K=1 vs QuTiP): measured ~1x10^-14 -> gate 3 orders up.
TOL_MATCHED = 1e-11
# Richardson(dt->0) vs the exact Liouvillian: measured 1.4x10^-13 (order 1),
# 5.9x10^-14 (order 2) -> gate ~70x above, still flags any real divergence.
TOL_MACHINE = 1e-11
# Two exact, independent methods (Liouvillian vs adaptive mesolve): measured
# 8.4x10^-8 (mesolve is adaptive-tolerance-limited, NOT model-limited) -> gate at
# 1x10^-6, ~12x margin, still far below any physical scale.
TOL_EXACT_PAIR = 1e-6
# The documented operating-point split-bound: measured 2.26x10^-7. Pin it to the
# decade [1x10^-7, 1x10^-6] so the headline "~2x10^-7" stays honest.
GAP_LO, GAP_HI = 1e-7, 1e-6


def _meta():
    return json.loads(FIXTURE.read_text())


def _envelope(meta):
    return np.load(FIXTURE.parent / Path(meta["pulse_npy"]).name)


def _eval_opt(meta, step_order=1):
    """Double-precision evaluation optimizer that consumes the saved envelope as the
    literal piecewise-constant pulse (bandwidth off, clamp) -- the same convention as
    the QuTiP and Liouvillian validators."""
    prof = ParametricCouplerProfile(**meta["profile"])
    return ParametricCZOptimizer(
        prof, bandwidth_mhz=0.0, use_drag=False, n_channels=int(meta["n_channels"]),
        activation="clamp", precision="double", step_order=step_order)


def _trotter_fproc(meta, step_order, K):
    """F_proc from the optimizer's split integrator with K sub-steps per control
    slice. ``repeat_interleave`` holds each slice value across K finer steps, so the
    PHYSICAL pulse is unchanged (still piecewise-constant) and only the integrator
    resolution (dt -> 1/K) refines. As K grows the split product approaches the exact
    per-slice propagator the Liouvillian computes in one shot."""
    opt = _eval_opt(meta, step_order=step_order)
    env = torch.as_tensor(_envelope(meta), dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    env = env.repeat_interleave(K, dim=1)
    choi = opt.simulate_choi_batch(env, dt=1.0 / K)
    return float(opt._process_fidelity(choi).mean())


def _richardson(vals_by_K, p0):
    """Romberg/Richardson extrapolation to dt->0. ``vals_by_K`` maps K (=1,2,4,...,
    geometric halving) to F. Successively eliminates error orders dt^p0, dt^(p0+1),
    ... -- p0=1 for Lie-Trotter (O(dt)), p0=2 for the symmetric Strang split (O(dt^2))."""
    col = [vals_by_K[k] for k in sorted(vals_by_K)]
    p = p0
    while len(col) > 1:
        f = 2.0 ** p
        col = [(f * col[i + 1] - col[i]) / (f - 1.0) for i in range(len(col) - 1)]
        p += 1
    return col[0]


def _liouville():
    meta = _meta()
    return liouville_f_proc(ParametricCouplerProfile(**meta["profile"]),
                            _envelope(meta), "cz", 1.0)


# --- The implementation check: machine precision, but blind to the split --------
def test_matched_pair_is_machine_precision_but_shares_the_split():
    """The optimizer and QuTiP share the Lie-Trotter scheme, so they agree to
    machine precision (validating the independent operator build + fidelity
    contraction) -- yet BOTH sit ~2.3x10^-7 from the exact-generator Liouvillian.
    This is the whole point: the machine-precision leg confirms the implementation,
    not the physics."""
    pytest.importorskip("qutip")
    from gradpulse.validate import qutip_f_proc
    meta = _meta()
    f_opt = _trotter_fproc(meta, step_order=1, K=1)
    f_qutip = qutip_f_proc(ParametricCouplerProfile(**meta["profile"]),
                           _envelope(meta), "cz", 1.0)
    f_liou = _liouville()
    # Same scheme, independent code -> machine precision.
    assert abs(f_opt - f_qutip) < TOL_MATCHED, (
        f"matched pair drifted: optimizer {f_opt:.14f} vs QuTiP {f_qutip:.14f} "
        f"(|d|={abs(f_opt - f_qutip):.2e}); these share the split and must agree.")
    # ... but the SAME-scheme QuTiP value is ~2.3e-7 from the different-scheme
    # Liouvillian: the machine-precision agreement cannot see the splitting error.
    assert GAP_LO < abs(f_qutip - f_liou) < GAP_HI, (
        f"the matched-pair-vs-Liouville gap left the documented decade: "
        f"{abs(f_qutip - f_liou):.2e} not in [{GAP_LO:.0e}, {GAP_HI:.0e}].")


# --- The operating-point split-bound is a controlled O(dt) error ----------------
def test_operating_point_gap_is_the_documented_split_bound():
    """At the 1 ns operating point the optimizer's split sits in the documented
    ~2x10^-7 band below the exact Liouvillian -- the number the README/paper quote
    as the load-bearing independent cross-check."""
    meta = _meta()
    gap = abs(_trotter_fproc(meta, step_order=1, K=1) - _liouville())
    assert GAP_LO < gap < GAP_HI, (
        f"operating-point split-bound {gap:.2e} left [{GAP_LO:.0e}, {GAP_HI:.0e}]; "
        "the headline ~2x10^-7 cross-check number has moved -- investigate the scheme.")


def test_trotter_converges_first_order_in_dt():
    """Refining the integrator (K=1,2,4,8,16; physical pulse fixed) drives the
    split's distance to the exact Liouvillian down monotonically, and the halving
    ratio -> 2 -- the signature of O(dt) Lie-Trotter error. This proves the
    2.3x10^-7 is a controlled discretization, not a fixed model disagreement."""
    meta = _meta()
    f_liou = _liouville()
    errs = {K: abs(_trotter_fproc(meta, step_order=1, K=K) - f_liou)
            for K in (1, 2, 4, 8, 16)}
    seq = [errs[K] for K in (1, 2, 4, 8, 16)]
    # Monotone decreasing error as dt shrinks.
    assert all(b < a for a, b in zip(seq, seq[1:])), f"error not monotone in dt: {seq}"
    # Finest halving ratio approaches 2 (first order). Measured ~1.95.
    ratio = errs[8] / errs[16]
    assert ratio > 1.8, (
        f"finest dt-halving ratio {ratio:.2f} < 1.8 -- convergence is not the "
        "expected first order; the stepping scheme may have changed.")


# --- The physics check: machine precision between genuinely independent solvers --
def test_independent_solvers_converge_to_machine_precision():
    """HEADLINE. Richardson-extrapolate the optimizer's split to dt->0 and compare to
    the exact full-generator Liouvillian. They share NO operator build, NO matrix
    exponential (torch matrix_exp vs a self-contained NumPy Pade), and NO stepping
    scheme -- yet meet at the float64 floor (~10^-13). NumPy-only, so it runs without
    the [validate] extra: the strongest cross-check is always available."""
    meta = _meta()
    F1 = {K: _trotter_fproc(meta, step_order=1, K=K) for K in (1, 2, 4, 8, 16)}
    R1 = _richardson(F1, p0=1)
    f_liou = _liouville()
    assert abs(R1 - f_liou) < TOL_MACHINE, (
        f"independent solvers FAILED to converge: Richardson(dt->0) {R1:.15f} vs "
        f"exact Liouvillian {f_liou:.15f} differ by {abs(R1 - f_liou):.2e} "
        f"(> {TOL_MACHINE:.0e}). A real divergence between independently written "
        "solvers -- not a stale constant.")


def test_strang_split_also_converges_to_machine_precision():
    """Corroboration with a DIFFERENT integrator: the 2nd-order symmetric Strang
    split (step_order=2, O(dt^2)) Richardson-extrapolates to the same exact
    Liouvillian value to machine precision. Two different splitting schemes both
    landing on the exact answer rules out a scheme-specific coincidence."""
    meta = _meta()
    F2 = {K: _trotter_fproc(meta, step_order=2, K=K) for K in (1, 2, 4, 8)}
    R2 = _richardson(F2, p0=2)
    f_liou = _liouville()
    assert abs(R2 - f_liou) < TOL_MACHINE, (
        f"Strang Richardson(dt->0) {R2:.15f} vs Liouvillian {f_liou:.15f} differ by "
        f"{abs(R2 - f_liou):.2e} (> {TOL_MACHINE:.0e}).")


def test_two_exact_methods_agree_independently():
    """A second exact reference by a DIFFERENT numerical method: QuTiP's adaptive
    ODE solver (mesolve on the zero-order-hold staircase, tight tolerances) agrees
    with the NumPy Liouvillian well below any physical scale. The residual is the
    adaptive solver's tolerance floor (~10^-7 here), not a model difference -- so
    this leg confirms the answer is solver-independent, while the Richardson test
    above is the one that reaches machine precision."""
    pytest.importorskip("qutip")
    from gradpulse.validate import mesolve_zoh_fproc
    meta = _meta()
    f_meso = mesolve_zoh_fproc(ParametricCouplerProfile(**meta["profile"]),
                               _envelope(meta), "cz", 1.0)
    assert abs(f_meso - _liouville()) < TOL_EXACT_PAIR, (
        f"two exact methods disagree: mesolve {f_meso:.12f} vs Liouvillian "
        f"{_liouville():.12f} (|d|={abs(f_meso - _liouville()):.2e} > {TOL_EXACT_PAIR:.0e}).")
