"""Mutation tests: the triple-solver cross-check is non-vacuous -- it CATCHES bugs.

The honest worry about "three solvers agree to ~1e-7 / 1e-14" is that the agreement
could be *vacuous*: maybe the solvers share a transcription or modelling bug and would
agree even if the physics were wrong. ``test_machine_precision.py`` proves the agreement
is real and reaches the float64 floor in the converged limit; this file proves it has
TEETH -- a real bug in the master equation breaks the agreement by orders of magnitude,
so passing the cross-check is a genuine constraint, not a coincidence.

Method. The PyTorch optimizer and the NumPy Liouvillian share no operator build, no
matrix exponential, and no integration scheme. We inject one classic master-equation
bug at a time and confirm the Liouvillian's F_proc then DIVERGES from the pristine
optimizer value -- the consensus gate in ``test_reproducibility.py`` would trip on
every one. Two fault kinds, handled differently *on purpose*:

  * **Parameter faults -- injected through the PUBLIC profile API, no patching.** A
    bug-sized error in a physical input (collapse rate, drive amplitude, a dropped
    dephasing channel) is just a wrong profile field, so we feed the Liouvillian a
    corrupted profile and the optimizer the pristine one. No coupling to internals.

  * **Structural fault -- the SOLE monkeypatch.** Dropping the -1/2{L^dag L, rho}
    anticommutator (the textbook non-CPTP Lindblad bug) has no parameter equivalent;
    it exists only as code. To exercise it through the *real* ``liouville_f_proc``
    pipeline (not a separate toy reimplementation, which would prove far less) we
    monkeypatch the one builder. ``monkeypatch`` auto-restores after the test and
    raises loudly if the internal is renamed, so it fails fast rather than rotting.

Measured shifts (pristine cross-check gap ~2.3e-7):
    q1 T1 input wrong by 2x                         9.9e-4
    parametric drive amplitude 30% wrong            3.8e-4   (smallest -> ~1700x the gap)
    a whole pure-dephasing channel omitted          1.7e-3
    dropped anticommutator (non-CPTP)               9.7e-3   (F overshoots past 1.0)
Thresholds below are DERIVED from these with margin, not tuned to pass. This is what
upgrades "the implementation is wired right (trust us)" into "the implementation is
wired right, and here is proof the cross-check would catch it if it weren't."

Note on completeness: this demonstrates teeth on representative bug *classes*; it is not
a systematic mutation score over every operator. The canonical tool for that is a
mutation-testing framework (mutmut / cosmic-ray) run in CI -- a heavier, separate scope.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import gradpulse.liouville as Lmod
from gradpulse import (ParametricCouplerProfile, ParametricCZOptimizer,
                       liouville_f_proc)
from gradpulse.parametric import DEVICE

# The reference fixture uses representative-default device parameters by design (it is a
# code-consistency fixture, not a hardware claim); silence that one advisory warning.
pytestmark = pytest.mark.filterwarnings(
    "ignore:ParametricCouplerProfile is using representative published-typical")

FIXTURE = Path(__file__).parent / "fixtures" / "reference_cz_pulse.json"

# Thresholds DERIVED from the measured shifts (smallest 3.8e-4 over a 2.3e-7 baseline).
# A real bug must move F by at least MIN_SHIFT *and* by at least MIN_RATIO times the
# pristine gap -- both comfortably below every measured shift.
MIN_SHIFT = 1e-4
MIN_RATIO = 100.0
PRISTINE_TOL = 1e-6   # the cross-check holds: |opt - liou| sits at the ~2.3e-7 split-bound


def _meta():
    return json.loads(FIXTURE.read_text())


def _envelope(meta):
    return np.load(FIXTURE.parent / Path(meta["pulse_npy"]).name)


def _f_opt(meta):
    """Independent reference: optimizer F_proc on the fixture pulse (PyTorch, with a
    different operator build, matrix_exp, and Trotter scheme from the Liouvillian)."""
    prof = ParametricCouplerProfile(**meta["profile"])
    opt = ParametricCZOptimizer(
        prof, bandwidth_mhz=0.0, use_drag=False, n_channels=int(meta["n_channels"]),
        activation="clamp", precision="double", step_order=1)
    env = torch.as_tensor(_envelope(meta), dtype=opt.rdtype, device=DEVICE).unsqueeze(0)
    return float(opt._process_fidelity(opt.simulate_choi_batch(env, dt=1.0)).mean())


def _f_liou(meta, **profile_overrides):
    """Liouvillian F_proc, optionally with bug-sized profile-field overrides."""
    prof = ParametricCouplerProfile(**{**meta["profile"], **profile_overrides})
    return liouville_f_proc(prof, _envelope(meta), "cz", 1.0)


# Parameter-class faults, each a bug-sized wrong value in one PUBLIC profile field. The
# closure receives the pristine profile dict P and returns the override(s).
PARAM_FAULTS = {
    # q1 T1 input wrong by 2x. In the T2 parameterization this also shifts pure
    # dephasing, exactly as a real input error would; net divergence measured 9.9e-4.
    "t1_off_2x":         lambda P: {"t1_ns_q1": P["t1_ns_q1"] / 2},
    # parametric drive amplitude 30% wrong -> the optimized pulse misfires; 3.8e-4.
    "drive_off_30pct":   lambda P: {"omega_max_mhz": P["omega_max_mhz"] * 1.3},
    # a whole pure-dephasing channel omitted: T2 = 2*T1 is the no-dephasing limit; 1.7e-3.
    "dephasing_omitted": lambda P: {"t2_ns_q1": 2 * P["t1_ns_q1"]},
}


def test_pristine_cross_check_holds():
    """Precondition: the independent optimizer and Liouvillian agree at the documented
    ~2.3e-7 split-bound. The mutation margins below are relative to this gap."""
    meta = _meta()
    gap = abs(_f_opt(meta) - _f_liou(meta))
    assert gap < PRISTINE_TOL, (
        f"pristine cross-check gap {gap:.2e} > {PRISTINE_TOL:.0e}; fix the agreement "
        "itself before trusting the mutation margins.")


@pytest.mark.parametrize("name", list(PARAM_FAULTS))
def test_cross_check_catches_parameter_fault(name):
    """A bug-sized error in one solver's physical inputs -- injected through the PUBLIC
    profile API, no patching of internals -- drives the Liouvillian away from the
    independent optimizer by >> the agreement tolerance. The cross-check catches it."""
    meta = _meta()
    f_opt = _f_opt(meta)
    base_gap = abs(f_opt - _f_liou(meta))
    shift = abs(_f_liou(meta, **PARAM_FAULTS[name](meta["profile"])) - f_opt)
    assert shift > MIN_SHIFT and shift > MIN_RATIO * base_gap, (
        f"parameter fault '{name}' moved F by only {shift:.2e} (baseline gap "
        f"{base_gap:.2e}); the cross-check might not catch this bug class.")


def test_cross_check_catches_structural_fault(monkeypatch):
    """The one fault with no parameter equivalent: dropping the -1/2{L^dag L, rho}
    anticommutator (the textbook non-CPTP Lindblad bug). It exists only as code, so this
    is the SOLE mutation that needs monkeypatch -- and it must corrupt the real
    liouville_f_proc pipeline to be meaningful. The non-trace-preserving generator sends
    the 'fidelity' past 1.0 (measured 9.7e-3 from the optimizer)."""
    meta = _meta()
    f_opt = _f_opt(meta)
    base_gap = abs(f_opt - _f_liou(meta))

    def buggy_lindbladian(H, L_ops, dim):
        Id = np.eye(dim, dtype=complex)
        S = -1j * (np.kron(Id, H) - np.kron(H.T, Id))
        for L in L_ops:
            S = S + np.kron(L.conj(), L)   # drops -1/2{L^dag L, rho}: non-CPTP
        return S
    monkeypatch.setattr(Lmod, "_lindbladian", buggy_lindbladian)

    shift = abs(liouville_f_proc(ParametricCouplerProfile(**meta["profile"]),
                                 _envelope(meta), "cz", 1.0) - f_opt)
    assert shift > MIN_SHIFT and shift > MIN_RATIO * base_gap, (
        f"structural fault (dropped anticommutator) moved F by only {shift:.2e}; the "
        "cross-check should catch a non-CPTP generator by a wide margin.")
