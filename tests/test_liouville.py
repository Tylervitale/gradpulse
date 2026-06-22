"""Unit tests for gradpulse.liouville -- the third, QuTiP-free independent solver.

These exercise the solver in isolation: its self-contained matrix exponential,
its independence (NumPy only -- no QuTiP/PyTorch/SciPy import), and its agreement
with the QuTiP path across code paths the headline cross-check does not reach
(static detuning, the iSWAP-family target). The headline agreement itself lives
in test_reproducibility.py::test_independent_solvers_agree_now.
"""
import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from gradpulse import ParametricCouplerProfile, liouville_f_proc
from gradpulse.liouville import _expm

FIXTURE = Path(__file__).parent / "fixtures" / "reference_cz_pulse.json"


def _meta():
    return json.loads(FIXTURE.read_text())


def _ref():
    meta = _meta()
    wf = np.load(FIXTURE.parent / Path(meta["pulse_npy"]).name)
    return ParametricCouplerProfile(**meta["profile"]), wf


def test_expm_matches_analytic_cases():
    # Diagonal: expm(diag(x)) = diag(exp(x)).
    D = np.diag([0.3, -1.2, 2.0]).astype(complex)
    assert np.allclose(_expm(D), np.diag(np.exp([0.3, -1.2, 2.0])))
    # Skew-symmetric generator -> rotation: expm([[0,-t],[t,0]]) = [[cos,-sin],[sin,cos]].
    t = 0.7
    G = np.array([[0.0, -t], [t, 0.0]], dtype=complex)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    assert np.allclose(_expm(G), R)
    # Nilpotent: expm([[0,1],[0,0]]) = [[1,1],[0,1]] (exact, terminating series).
    N = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    assert np.allclose(_expm(N), np.array([[1.0, 1.0], [0.0, 1.0]]))


def test_expm_large_norm_scaling():
    # A large-norm matrix exercises the scaling-and-squaring path (s > 0). Compare
    # the eigendecomposition exp for a diagonalizable normal matrix.
    rng = np.random.default_rng(0)
    H = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    H = H + H.conj().T                                    # Hermitian -> normal
    M = 40.0j * H                                         # large norm, skew-Hermitian
    w, V = np.linalg.eigh(H)
    ref = V @ np.diag(np.exp(40.0j * w)) @ V.conj().T
    assert np.allclose(_expm(M), ref, atol=1e-10)


def test_liouville_is_pure_numpy():
    # The third solver's independence rests on sharing no code with the solvers it
    # checks: it must import neither QuTiP, PyTorch, nor SciPy.
    src = inspect.getsource(inspect.getmodule(liouville_f_proc))
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update(n.name.split(".")[0] for n in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "numpy" in imported
    assert {"qutip", "torch", "scipy"} & imported == set(), (
        f"liouville must stay NumPy-only; found {imported}")


def test_liouville_reference_in_physical_band():
    # Runs with no QuTiP installed: a sanity floor on the headline pulse.
    prof, wf = _ref()
    f = liouville_f_proc(prof, wf, "cz", 1.0)
    assert 0.98 < f < 1.0


def test_liouville_matches_qutip_with_detuning():
    # A static detuning offset (the robustness / quasi-static / spectator-ZZ paths)
    # must agree between QuTiP and the Liouvillian solver -- a code path the
    # headline cross-check never exercises. Both compute whatever fidelity the CZ
    # pulse yields under the offset; the point is that they AGREE.
    pytest.importorskip("qutip")
    from gradpulse.validate import qutip_f_proc
    prof, wf = _ref()
    for det in (0.01, (0.0, -0.02)):
        f_q = qutip_f_proc(prof, wf, "cz", 1.0, detuning_offset=det)
        f_l = liouville_f_proc(prof, wf, "cz", 1.0, detuning_offset=det)
        assert abs(f_q - f_l) < 1e-5, f"detuning={det}: qutip {f_q} vs liouville {f_l}"


def test_liouville_matches_qutip_iswap_target():
    # The iSWAP-family target path agrees too (evaluated against the CZ pulse, so
    # the fidelity is low -- but both solvers must report the SAME low number).
    pytest.importorskip("qutip")
    from gradpulse.validate import qutip_f_proc
    prof, wf = _ref()
    f_q = qutip_f_proc(prof, wf, "iswap", 1.0)
    f_l = liouville_f_proc(prof, wf, "iswap", 1.0)
    assert abs(f_q - f_l) < 1e-5


def test_liouville_accepts_dict_profile():
    # Profiles may be a dataclass or a plain dict (mirrors qutip_f_proc).
    meta = _meta()
    wf = np.load(FIXTURE.parent / Path(meta["pulse_npy"]).name)
    f_dataclass = liouville_f_proc(ParametricCouplerProfile(**meta["profile"]), wf, "cz", 1.0)
    f_dict = liouville_f_proc(meta["profile"], wf, "cz", 1.0)
    assert abs(f_dataclass - f_dict) < 1e-12


# --------------------------------------------------------------------------- #
#  Cross-resonance: the library-independent third solver for the SECOND       #
#  architecture. Until this, CR's only independent referee was QuTiP, so its  #
#  cross-check was a *double*-solver one whose two non-optimizer legs shared a #
#  library. liouville_cr_f_proc closes that: a NumPy-only exact-generator      #
#  solver that must reproduce the QuTiP referee on the SAME pulse.            #
# --------------------------------------------------------------------------- #

def _cr_synthetic_pulse(use_target_cancel):
    """A deterministic, smooth, physically realizable CR drive.

    Built through the optimizer's own bandwidth smoother (no optimization needed:
    the independence claim -- two no-shared-code solvers agree on the SAME pulse --
    holds for any pulse, optimized or not), so the test is fast and deterministic.
    Returns (optimizer_factory_kwargs, smoothed_signed_waveform).
    """
    import torch
    from gradpulse.crossresonance import (CrossResonanceProfile,
                                          CrossResonanceZXOptimizer, DEVICE)
    prof = CrossResonanceProfile()
    n_ch = 2 if use_target_cancel else 1
    t = np.linspace(0.0, 1.0, 200)
    raw = np.zeros((200, n_ch))
    raw[:, 0] = 0.8 * np.sin(np.pi * t)                   # control CR drive
    if n_ch == 2:
        raw[:, 1] = 0.3 * np.sin(2.0 * np.pi * t)         # target cancellation tone
    opt = CrossResonanceZXOptimizer(prof, use_drag=True,
                                    use_target_cancel=use_target_cancel,
                                    precision="double")
    wf = opt.smoothed_waveform(
        torch.tensor(raw, dtype=opt.rdtype, device=DEVICE)).detach().cpu().numpy()
    return prof, wf


@pytest.mark.parametrize("use_target_cancel", [False, True])
@pytest.mark.parametrize("echo", [False, True])
def test_liouville_cr_matches_qutip(use_target_cancel, echo):
    # The CR analogue of the headline parametric cross-check: the NumPy-only
    # Liouvillian must reproduce the independent QuTiP referee on the SAME pulse,
    # across both channel layouts and the echoed/non-echoed sequence. They share
    # no library, so agreement rules out a QuTiP-specific artifact in CR's
    # cross-check -- the gap this solver was added to close. The residual is the
    # Trotter splitting error (QuTiP splits; Liouville is exact-generator), which
    # sits far below the 1e-3 ship gate.
    pytest.importorskip("qutip")
    import torch
    from gradpulse.crossresonance import (CrossResonanceZXOptimizer, DEVICE)
    from gradpulse.validate import cr_cross_check
    from gradpulse.liouville import liouville_cr_f_proc

    prof, wf = _cr_synthetic_pulse(use_target_cancel)
    vz = [0.3, -0.4]
    opt = CrossResonanceZXOptimizer(prof, use_drag=True,
                                    use_target_cancel=use_target_cancel,
                                    echo=echo, precision="double")
    f_q = cr_cross_check(opt, wf, vz=vz, echo=echo)
    f_l = liouville_cr_f_proc(prof, wf, vz=vz, echo=echo, use_drag=True)
    assert abs(f_q - f_l) < 1e-5, (
        f"target_cancel={use_target_cancel} echo={echo}: "
        f"qutip {f_q:.10f} vs liouville {f_l:.10f} (d={abs(f_q - f_l):.2e})")


def test_liouville_cr_accepts_dict_profile():
    # As for the parametric solver: dataclass and plain-dict profiles agree.
    from dataclasses import asdict
    from gradpulse.liouville import liouville_cr_f_proc
    prof, wf = _cr_synthetic_pulse(use_target_cancel=True)
    f_dataclass = liouville_cr_f_proc(prof, wf, vz=(0.1, 0.2), use_drag=True)
    f_dict = liouville_cr_f_proc(asdict(prof), wf, vz=(0.1, 0.2), use_drag=True)
    assert abs(f_dataclass - f_dict) < 1e-12


# --------------------------------------------------------------------------- #
#  General N-qubit register: the library-independent leg for the THIRD        #
#  architecture's closed-system path. The novel content -- a gate on a subset #
#  with identity on the rest, and a spectator coupled in the drift so its      #
#  crosstalk is inside the propagation -- is reconstructed in pure NumPy and   #
#  must reproduce the optimizer. (No Trotter split in the closed system, so    #
#  the two exact propagators agree to machine precision in double precision.)  #
# --------------------------------------------------------------------------- #

def _nqubit_synthetic(use_drag, freq_control_qubits):
    """A 3-qubit chain (0-1-2), CZ on (0,1) with spectator 2 coupled to 1 -- the
    in-loop-crosstalk case -- plus a deterministic smooth control through the
    optimizer's own smoother. Returns (optimizer, smoothed_waveform)."""
    import torch
    from gradpulse.multiqubit import (MultiQubitProfile, MultiQubitOptimizer,
                                      DEVICE)
    prof = MultiQubitProfile()                          # 0-1-2 chain, 12 MHz edges
    opt = MultiQubitOptimizer(prof, target_gate="cz", target_qubits=(0, 1),
                              open_system=False, use_drag=use_drag,
                              freq_control_qubits=freq_control_qubits,
                              precision="double", verbose=False)
    n = opt.n_channels
    t = np.linspace(0.0, 1.0, 160)
    raw = np.stack([0.5 + 0.4 * np.sin((c + 1) * np.pi * t) for c in range(n)], axis=1)
    xs = torch.tensor(raw, dtype=opt.rdtype, device=DEVICE).unsqueeze(0).clamp(0.0, 1.0)
    sm = opt._smooth(xs, opt._smoother(160, 1.0))[0].detach().cpu().numpy()
    return opt, sm


@pytest.mark.parametrize("use_drag,freq_control", [
    (False, None), (True, None), (False, [0, 1])])
def test_liouville_nqubit_closed_matches_optimizer(use_drag, freq_control):
    # The independent NumPy unitary propagator + subset-target reconstruction must
    # reproduce MultiQubitOptimizer's closed-system F_proc, with the spectator's
    # always-on coupling carried in the drift. No shared operator-build or expm
    # code; double precision, so they meet at machine precision.
    from gradpulse.liouville import liouville_nqubit_closed_f_proc
    opt, sm = _nqubit_synthetic(use_drag, freq_control)
    f_opt = opt.process_fidelity(sm, dt_ns=1.0)
    f_liou = liouville_nqubit_closed_f_proc(opt, sm, dt_ns=1.0)
    assert abs(f_opt - f_liou) < 1e-9, (
        f"drag={use_drag} freq_control={freq_control}: "
        f"optimizer {f_opt:.12f} vs NumPy {f_liou:.12f} (d={abs(f_opt - f_liou):.2e})")
