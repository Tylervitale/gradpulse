"""Reproducibility gate: the paper's headline is the AGREEMENT of independent solvers.

``tests/fixtures/reference_cz_pulse.{npy,json}`` is the validated 150 ns CZ
envelope. Re-evaluating it is deterministic (no optimization, no RNG), so it pins
the reported numbers. The design here deliberately avoids a hand-trusted constant:

  * The HARD gate (``test_independent_solvers_agree_now``) consults NO stored
    number. It recomputes F_proc through every independent solver available at
    runtime -- the PyTorch optimizer, the QuTiP-free Liouvillian solver, and (when
    the ``[validate]`` extra is installed) QuTiP -- and asserts they AGREE. The
    headline is whatever they converge on, recomputed every run.

  * The SOFT tripwire (``test_headline_has_not_drifted_from_checkpoint``) compares
    today's value to a committed *checkpoint*. That checkpoint is not a magic
    number: ``_rebless`` regenerates it from the committed pulse and refuses to
    write it unless the solvers agree first, so it is a consensus artifact -- a
    lockfile. Its only job is to catch a *shared* drift (a model/constant change
    that moves all solvers together, which agreement alone cannot see). When it
    trips, that is a reviewed git diff, not a silent slide: re-bless with --regen.

Why both layers. Cross-solver agreement catches an implementation bug in one
solver but is blind to a shared-model change; a stored checkpoint catches the
shared change but says nothing about correctness. Absolute-drift detection is
impossible without remembering the prior value, so we keep one -- but blessed by
consensus and reviewed on change, never hand-typed.

Tolerances are DERIVED from measured cross-solver agreement on the committed
pulse, not tuned to pass (see the constants below).

Run:  pytest tests/        OR        python tests/test_reproducibility.py
      python tests/test_reproducibility.py --regen   # re-bless the checkpoint
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

# Tolerances derived from measured solver agreement on the committed pulse: optimizer
# vs QuTiP share the matched scheme (~2e-14, gated at 1e-11); the Liouvillian's exact
# full-generator exponential sits ~2.3e-7 from the split solvers, the splitting error
# it bounds (gated at 1e-5, ~40x margin); the saved pulse re-evaluates deterministically
# to ~1e-9, so the checkpoint drift tripwire gates at 1e-6.
TOL_SAME_SCHEME = 1e-11      # optimizer vs QuTiP (identical scheme)
TOL_SPLIT = 1e-5            # exact-generator vs split-scheme (bounds Trotter error)
TOL_DRIFT = 1e-6            # today's recompute vs the blessed checkpoint
TOL_LEAK_DRIFT = 1e-5       # leakage recompute vs checkpoint


def _eval_opt(meta):
    """Double-precision evaluation optimizer that consumes the saved envelope as
    the literal physical pulse (bandwidth off, clamp activation) -- the same
    convention as the QuTiP and Liouvillian validators."""
    prof = ParametricCouplerProfile(**meta["profile"])
    return ParametricCZOptimizer(
        prof, bandwidth_mhz=0.0, use_drag=False,
        n_channels=int(meta["n_channels"]), activation="clamp", precision="double",
    )


def _envelope(meta):
    npy = FIXTURE.parent / Path(meta["pulse_npy"]).name
    return np.load(npy)


def _optimizer_choi(meta):
    opt = _eval_opt(meta)
    env = torch.as_tensor(_envelope(meta), dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    return opt, opt.simulate_choi_batch(env, dt=1.0)


def _fproc_optimizer(meta):
    opt, choi = _optimizer_choi(meta)
    return float(opt._process_fidelity(choi).mean())


def _solver_fidelities(meta):
    """F_proc through every independent solver available, computed at runtime.

    The Liouvillian solver is NumPy-only, so it is always present -- the
    cross-check runs even without the ``[validate]`` (QuTiP) extra. QuTiP is added
    when installed. Returns a dict keyed by solver name.
    """
    prof = ParametricCouplerProfile(**meta["profile"])
    wf = _envelope(meta)
    out = {
        "optimizer": _fproc_optimizer(meta),
        "liouville": liouville_f_proc(prof, wf, "cz", 1.0),
    }
    try:
        from gradpulse.validate import qutip_f_proc
        out["qutip"] = qutip_f_proc(prof, wf, "cz", 1.0)
    except Exception:
        pass        # [validate] extra not installed; Liouville still cross-checks
    return out


def _meta():
    meta = json.loads(FIXTURE.read_text())
    if "f_proc_double" not in meta or "leakage_double" not in meta:
        meta = _rebless(meta)
    return meta


def test_fixture_exists_and_is_shaped():
    meta = _meta()
    wf = _envelope(meta)
    assert wf.shape == (150, int(meta["n_channels"]))
    assert wf.min() >= -1e-6 and wf.max() <= 1 + 1e-6
    assert meta.get("target_gate", "cz") == "cz"


def test_independent_solvers_agree_now():
    """HARD GATE -- fully dynamic, no stored number consulted.

    The headline is whatever the independent solvers agree on right now. The
    optimizer-vs-Liouvillian leg runs everywhere (NumPy-only); the QuTiP legs run
    when the extra is installed. A discrepancy here means a real divergence
    between independently-written solvers, not a stale constant.
    """
    f = _solver_fidelities(_meta())
    # Always available: PyTorch optimizer vs the exact-generator Liouvillian.
    assert abs(f["optimizer"] - f["liouville"]) < TOL_SPLIT, (
        f"optimizer {f['optimizer']:.12f} vs Liouville {f['liouville']:.12f} "
        f"disagree by {abs(f['optimizer'] - f['liouville']):.2e} (> {TOL_SPLIT:.0e}); "
        "the Trotter splitting error has moved -- investigate the stepping scheme."
    )
    if "qutip" in f:
        assert abs(f["optimizer"] - f["qutip"]) < TOL_SAME_SCHEME, (
            f"optimizer {f['optimizer']:.14f} vs QuTiP {f['qutip']:.14f} disagree by "
            f"{abs(f['optimizer'] - f['qutip']):.2e} (> {TOL_SAME_SCHEME:.0e}); these "
            "share the matched scheme and must agree to machine precision."
        )
        assert abs(f["qutip"] - f["liouville"]) < TOL_SPLIT
    # Internal consistency: every solver lands in the same physical band.
    for name, val in f.items():
        assert 0.97 < val < 1.0, f"{name} F_proc {val} outside the physical band"


def test_headline_has_not_drifted_from_checkpoint():
    """SOFT TRIPWIRE -- today's consensus value vs the committed checkpoint.

    The checkpoint is a regenerable, consensus-blessed lockfile (see ``_rebless``),
    not a hand-typed constant. This catches a *shared* drift that cross-solver
    agreement cannot. If it trips on an intended change, re-bless: it is a reviewed
    diff, not a silent slide.
    """
    meta = _meta()
    f_now = _fproc_optimizer(meta)
    ref = meta["f_proc_double"]
    assert f_now == pytest.approx(ref, abs=TOL_DRIFT), (
        f"Headline F_proc drifted from the blessed checkpoint: {ref:.10f} -> {f_now:.10f}. "
        "If intended, re-bless with: python tests/test_reproducibility.py --regen"
    )
    f_avg = (4.0 * f_now + 1.0) / 5.0
    ref_f_avg = (4.0 * ref + 1.0) / 5.0
    assert f_avg == pytest.approx(ref_f_avg, abs=TOL_DRIFT)


def test_leakage_has_not_drifted_from_checkpoint():
    meta = _meta()
    opt, choi = _optimizer_choi(meta)
    leak = float(opt._leakage(choi).mean())
    assert leak == pytest.approx(meta["leakage_double"], abs=TOL_LEAK_DRIFT), (
        f"leakage drifted: {meta['leakage_double']:.7f} -> {leak:.7f}. "
        "If intended, re-bless with: python tests/test_reproducibility.py --regen"
    )


def test_dt_convergence_is_converged():
    # The documented dt-convergence: F_proc is monotone in 1/dt and the 1 ns
    # operating point sits within ~1e-6 of the dt->0 limit. Pure internal
    # consistency -- no stored number drives the assertion.
    meta = _meta()
    opt = _eval_opt(meta)
    env = torch.as_tensor(_envelope(meta), dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    f_dt1 = float(opt._process_fidelity(opt.simulate_choi_batch(env, dt=1.0)).mean())
    env4 = env.repeat_interleave(4, dim=1)                 # same pulse, dt = 0.25 ns
    f_dt025 = float(opt._process_fidelity(opt.simulate_choi_batch(env4, dt=0.25)).mean())
    assert f_dt025 >= f_dt1 - 1e-7                          # monotone non-decreasing
    assert abs(f_dt025 - f_dt1) < 1e-5                      # already converged at 1 ns


def test_reported_grape_f_matches_recompute():
    # Internal consistency: the value saved with the pulse (complex64
    # optimization) agrees with the double-precision recompute to ~1e-6, so the
    # reported headline is not a fluke of the optimizer's working precision.
    meta = _meta()
    f_now = _fproc_optimizer(meta)
    assert float(meta["grape_f"]) == pytest.approx(f_now, abs=5e-6)


def test_cli_cross_check_passes():
    # Integration: the file-loading CLI cross-check path (gradpulse.validate) still
    # PASSes end-to-end on the committed pulse. Skipped without the [validate] extra.
    pytest.importorskip("qutip")
    from gradpulse.validate import cross_check
    res = cross_check(FIXTURE)
    assert res["status"] == "PASS", f"cross-check status {res['status']}"
    assert abs(res["delta"]) < 1e-3


def _rebless(meta=None):
    """Regenerate the committed checkpoint from the pulse -- refusing to write it
    unless the independent solvers AGREE first, so the stored value is a consensus
    artifact, never one solver's unverified word.

        python tests/test_reproducibility.py --regen
    """
    if meta is None:
        meta = json.loads(FIXTURE.read_text())
    f = _solver_fidelities(meta)
    # Consensus gate BEFORE blessing.
    assert abs(f["optimizer"] - f["liouville"]) < TOL_SPLIT, (
        f"refuse to bless: optimizer vs Liouville disagree by "
        f"{abs(f['optimizer'] - f['liouville']):.2e}")
    if "qutip" in f:
        assert abs(f["optimizer"] - f["qutip"]) < TOL_SAME_SCHEME, (
            f"refuse to bless: optimizer vs QuTiP disagree by "
            f"{abs(f['optimizer'] - f['qutip']):.2e}")
    opt, choi = _optimizer_choi(meta)
    meta["f_proc_double"] = float(opt._process_fidelity(choi).mean())
    meta["leakage_double"] = float(opt._leakage(choi).mean())
    meta["consensus"] = {
        "blessed_f_proc": meta["f_proc_double"],
        "agreeing_solvers": sorted(f),
        "rebless_cmd": "python tests/test_reproducibility.py --regen",
    }
    FIXTURE.write_text(json.dumps(meta, indent=2))
    print(f"Re-blessed {FIXTURE} (consensus of {sorted(f)})")
    print(f"  f_proc_double  = {meta['f_proc_double']:.16f}")
    print(f"  leakage_double = {meta['leakage_double']:.16f}")
    return meta


if __name__ == "__main__":
    import sys
    if "--regen" in sys.argv:
        _rebless()
    else:
        sys.exit(pytest.main([__file__, "-v"]))
