"""Gradient checkpointing of the open-system slice loop.

Checkpointing trades compute for memory: it must return the SAME forward value and
the SAME gradients as the plain loop -- only the peak autograd memory differs. These
tests pin that equivalence to ~1e-6 for both the parametric (9-D) and the multiqubit
(4**N Choi) propagators, and that the default path (checkpoint_segments=0) is
untouched.
"""
import numpy as np
import torch

import gradpulse as gp
from gradpulse.multiqubit import MultiQubitProfile, MultiQubitOptimizer
from gradpulse.parametric import DEVICE


def test_parametric_checkpoint_matches_value_and_grad():
    opt = gp.ParametricCZOptimizer(n_channels=4, activation="sigmoid")
    x0 = torch.randn(1, 60, 4, device=DEVICE, dtype=opt.rdtype)

    def fid(seg):
        x = x0.clone().requires_grad_(True)
        rho = opt.simulate_choi_batch(x, dt=1.0, checkpoint_segments=seg)
        f = opt._process_fidelity(rho).mean()
        f.backward()
        return float(f.detach()), x.grad.detach().clone()

    f_plain, g_plain = fid(0)
    f_ckpt, g_ckpt = fid(6)
    assert abs(f_plain - f_ckpt) < 1e-6, f"forward differs: {f_plain} vs {f_ckpt}"
    assert torch.max(torch.abs(g_plain - g_ckpt)).item() < 1e-5, "gradients differ"


def test_parametric_checkpoint_default_is_unchanged():
    """checkpoint_segments=0 must be bit-identical to not passing it at all."""
    opt = gp.ParametricCZOptimizer(n_channels=3, activation="sigmoid")
    x = torch.randn(1, 50, 3, device=DEVICE, dtype=opt.rdtype)
    with torch.no_grad():
        a = opt.simulate_choi_batch(x, dt=1.0)
        b = opt.simulate_choi_batch(x, dt=1.0, checkpoint_segments=0)
    assert torch.equal(a, b)


def test_multiqubit_checkpoint_matches_value_and_grad():
    p = MultiQubitProfile(n_qubits=3, n_levels=2,
                          couplings={(0, 1): 12.0, (1, 2): 12.0})
    opt = MultiQubitOptimizer(p, target_gate="cz", target_qubits=(0, 1),
                              open_system=True, verbose=False)
    kernel = opt._smoother(40, 1.0)
    raw0 = 0.1 * torch.randn(1, 40, opt.n_channels, device=DEVICE, dtype=opt.rdtype)

    def fid(seg):
        raw = raw0.clone().requires_grad_(True)
        xs = opt._smooth(torch.sigmoid(raw), kernel)
        rho = opt._propagate_choi(xs, 1.0, checkpoint_segments=seg)
        f = opt._process_fidelity_choi(rho)[0]
        f.backward()
        return float(f.detach()), raw.grad.detach().clone()

    f_plain, g_plain = fid(0)
    f_ckpt, g_ckpt = fid(5)
    assert abs(f_plain - f_ckpt) < 1e-6
    assert torch.max(torch.abs(g_plain - g_ckpt)).item() < 1e-5


def test_multiqubit_optimize_with_checkpointing_runs():
    p = MultiQubitProfile(n_qubits=3, n_levels=2,
                          couplings={(0, 1): 12.0, (1, 2): 12.0})
    opt = MultiQubitOptimizer(p, target_gate="cz", target_qubits=(0, 1),
                              open_system=True, verbose=False)
    res = opt.optimize(n_slices=30, dt_ns=1.0, iterations=15, n_seeds=1,
                       checkpoint_segments=4, verbose=False)
    assert np.isfinite(res["best_fidelity"])


def test_auto_checkpointing_resolves_and_is_exact():
    """checkpoint_segments='auto' resolves to round(sqrt(n_slices)) for long pulses
    (and 0 for short ones). Since checkpointing is exact, 'auto' must give the same
    optimisation result as the explicit split it resolves to -- it only changes peak
    memory, never the math."""
    import math
    opt = gp.ParametricCZOptimizer(n_channels=3, activation="sigmoid", precision="double")
    N = 100
    cfg = dict(n_seeds=1, iterations=8, n_slices=N, lbfgs_polish=False, lr=0.02)
    f_auto = opt.optimize_multi_seed(
        checkpoint_segments="auto",
        rng=torch.Generator(device=DEVICE).manual_seed(0), **cfg)["best_fidelity"]
    f_expl = opt.optimize_multi_seed(
        checkpoint_segments=round(math.sqrt(N)),
        rng=torch.Generator(device=DEVICE).manual_seed(0), **cfg)["best_fidelity"]
    assert abs(f_auto - f_expl) < 1e-9
    # A short pulse keeps the plain (fastest) path: 'auto' -> 0, still runs.
    f_short = opt.optimize_multi_seed(
        checkpoint_segments="auto", n_slices=40, n_seeds=1, iterations=5,
        lbfgs_polish=False, rng=torch.Generator(device=DEVICE).manual_seed(0))["best_fidelity"]
    assert np.isfinite(f_short)
