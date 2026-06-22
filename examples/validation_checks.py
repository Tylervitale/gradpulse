"""Supplementary validation checks for the gradpulse paper.

Produces the four numbers the paper's Validation section quotes beyond the
QuTiP cross-check:

  1. Autodiff vs finite-difference gradient   (correctness of the torch.autograd path)
  2. dt-convergence of F_proc                 (first-order Lindblad step error)
  3. Leakage out of the computational subspace
  4. GPU-vs-CPU timing                        (why batching on a consumer GPU helps)

All numbers print to stdout. Run on GPU:

    python examples/validation_checks.py

and again with the GPU hidden to get the CPU timing row:

    CUDA_VISIBLE_DEVICES=-1 python examples/validation_checks.py --timing-only

Checks 1-3 are device-independent; only the timing differs by device.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch
import torch.utils.benchmark as benchmark

from gradpulse import ParametricCouplerProfile, ParametricCZOptimizer
from gradpulse.parametric import DEVICE

PROFILE = ParametricCouplerProfile(
    freq_ghz_q1=4.85, freq_ghz_q2=5.05,
    anharm_ghz_q1=-0.20, anharm_ghz_q2=-0.20,
    t1_ns_q1=30_000, t2_ns_q1=25_000,
    t1_ns_q2=30_000, t2_ns_q2=25_000,
    g_max_mhz=12.0, omega_max_mhz=50.0,
)


def _opt(**kw):
    base = dict(bandwidth_mhz=80.0, use_drag=False, n_channels=3, activation="sigmoid")
    base.update(kw)
    return ParametricCZOptimizer(PROFILE, **base)


def check_finite_difference(float64=False):
    """Central finite differences vs autograd on a small toy pulse.

    Done in raw (pre-sigmoid) parameter space -- exactly the space the optimizer's
    Adam/L-BFGS update. In the default complex64 path the float32 finite-difference
    floor (~1e-2 in L2 magnitude) dominates, so we also offer a float64/complex128
    path (``float64=True``, bandwidth off so the whole pipeline is double precision)
    for a crisp magnitude check. Cosine similarity isolates direction agreement and
    is precision-robust either way.
    """
    if float64:
        xdt, eps = torch.float64, 1e-5
        opt = _opt(bandwidth_mhz=0.0, activation="sigmoid", precision="double")
    else:
        xdt, eps = torch.float32, 2e-3
        opt = _opt()
    n_slices = 6
    torch.manual_seed(0)
    x0 = torch.randn(1, n_slices, 3, dtype=xdt, device=DEVICE)

    def loss_of(x):
        rho = opt.simulate_choi_batch(x, dt=1.0)
        return (1.0 - opt._process_fidelity(rho).mean())

    x = x0.clone().requires_grad_(True)
    loss_of(x).backward()
    g_auto = x.grad.detach().reshape(-1).clone()

    flat = x0.reshape(-1)
    g_fd = torch.zeros_like(g_auto)
    for i in range(flat.numel()):
        xp = flat.clone(); xp[i] += eps
        xm = flat.clone(); xm[i] -= eps
        lp = loss_of(xp.view_as(x0)).item()
        lm = loss_of(xm.view_as(x0)).item()
        g_fd[i] = (lp - lm) / (2 * eps)

    rel = float((g_auto - g_fd).norm() / g_fd.norm().clamp_min(1e-12))
    cos = float(torch.nn.functional.cosine_similarity(g_auto.double(), g_fd.double(), dim=0))
    tag = "float64/complex128" if float64 else "float32/complex64"
    print("\n[1] FINITE-DIFFERENCE GRADIENT CHECK")
    print(f"    toy pulse: {n_slices} slices x 3 ch ({flat.numel()} params), {tag}, eps={eps}")
    print(f"    relative L2 error (autograd vs central FD): {rel:.2e}")
    print(f"    cosine similarity:                          {cos:.8f}")
    return {"rel_l2": rel, "cosine": cos, "n_params": int(flat.numel()), "eps": eps}


def _reference_pulse():
    """The 150 ns CZ envelope checks 2-3 run on.

    Prefer the *shipped* pulse (``cz_pulse.npy`` from examples/optimize_cz.py) so
    every supplementary number traces to the one artifact the QuTiP cross-check
    also validates; fall back to optimizing a fresh reference if it is absent.
    """
    p = Path("cz_pulse.npy")
    if p.exists():
        env = np.load(p)
        meta = json.loads(Path("cz_pulse.json").read_text()) if Path("cz_pulse.json").exists() else {}
        f = float(meta.get("grape_f", float("nan")))
        print(f"      using shipped cz_pulse.npy (reported F_proc = {f:.7f})")
        return env, f
    print("      cz_pulse.npy not found; optimizing a fresh reference ...")
    opt = _opt()
    res = opt.optimize_multi_seed(
        n_seeds=4, iterations=200, n_slices=150, dt_ns=1.0,
        warm_start_mode="parametric_cz", use_process_fidelity=True, lbfgs_polish=True,
    )
    return res["best_waveform"], float(res["best_fidelity"])


def check_dt_convergence(envelope, precision="double", ks=(1, 2, 4, 8, 16, 32)):
    """F_proc vs integration step. The smoothed envelope is the fixed physical
    pulse; we subdivide each 1 ns piecewise-constant slice into k equal sub-steps
    (same amplitude) and integrate at dt=1/k. Since matrix_exp is exact within a
    constant slice, the unitary evolution is k-independent and the only
    dt-dependence is the first-order Lindblad split -- so this isolates the
    dissipator's discretization error.

    Run in precision='double' by default: in single (complex64) the accumulated
    float32 round-off over hundreds-to-thousands of steps competes with the
    discretization error and corrupts the convergence curve. A bandwidth=0,
    clamp-activation optimizer feeds the envelope as the literal physical pulse
    (no re-sigmoid/re-smoothing) -- the same convention as the QuTiP validator.
    """
    evalopt = _opt(bandwidth_mhz=0.0, activation="clamp", precision=precision)
    tag = "complex128" if precision == "double" else "complex64"
    env = torch.as_tensor(envelope, dtype=evalopt.rdtype, device=DEVICE)  # [150,3] in [0,1]
    rows = {}
    print(f"\n[2] dt-CONVERGENCE OF F_proc  ({tag}, first-order Lindblad step)")
    for k in ks:
        up = env.repeat_interleave(k, dim=0).unsqueeze(0)   # [1, 150k, 3], same pulse
        rho = evalopt.simulate_choi_batch(up, dt=1.0 / k)
        fproc = float(evalopt._process_fidelity(rho).mean())
        rows[1.0 / k] = fproc
        print(f"    dt = {1.0/k:>7.4f} ns ({up.shape[1]:>5} steps):  F_proc = {fproc:.7f}")
    return rows


def check_leakage(envelope, precision="double"):
    """Population leaked out of the 2-qubit computational subspace for the
    validated (no-DRAG) pulse (double precision)."""
    evalopt = _opt(bandwidth_mhz=0.0, activation="clamp", precision=precision)
    env = torch.as_tensor(envelope, dtype=evalopt.rdtype, device=DEVICE).unsqueeze(0)
    rho = evalopt.simulate_choi_batch(env, dt=1.0)
    leak = float(evalopt._leakage(rho).mean())
    fproc = float(evalopt._process_fidelity(rho).mean())
    tag = "complex128" if precision == "double" else "complex64"
    print(f"\n[3] LEAKAGE (validated no-DRAG pulse, {tag})")
    print(f"    F_proc (eval path):                 {fproc:.7f}")
    print(f"    leakage out of comp. subspace:      {leak:.2e}")
    return {"leakage": leak, "fproc_eval": fproc}


def check_error_budget(envelope, precision="double"):
    """Decompose the validated pulse's infidelity and report channel unitarity.

    Two independent views of the same coherent-vs-incoherent split:
      * ablation -- rerun with the Lindblad dissipator switched off
        (diss_scale=0): the residual error is coherent control + leakage; the
        rest of the total is the T1/T_phi decoherence floor.
      * unitarity u of the full noisy channel (Wallman et al. 2015), whose
        implied incoherent floor (d-1)/d (1 - sqrt(u)) is compared against the
        ablation. The two agreeing is a self-consistency check on the budget.
    """
    evalopt = _opt(bandwidth_mhz=0.0, activation="clamp", precision=precision)
    eb = evalopt.error_budget(envelope, dt=1.0)
    tag = "complex128" if precision == "double" else "complex64"
    rt = eb["r_total"]
    pct = (lambda x: 100.0 * x / rt if rt > 0 else float("nan"))
    print(f"\n[4] ERROR BUDGET (validated pulse, {tag})")
    print(f"    F_avg (avg gate fidelity):          {eb['F_avg']:.7f}")
    print(f"    total infidelity r_total:           {rt:.2e}")
    print(f"      control + leakage (ablation):     {eb['r_control_leakage']:.2e}"
          f"  ({pct(eb['r_control_leakage']):.0f}% of total)")
    print(f"      decoherence floor  (ablation):    {eb['r_decoherence']:.2e}"
          f"  ({pct(eb['r_decoherence']):.0f}% of total)")
    print(f"    channel unitarity u:                {eb['unitarity']:.5f}")
    print(f"      incoherent floor (from u):        {eb['r_incoherent_unitarity']:.2e}")
    print(f"      coherent excess  (from u):        {eb['coherent_excess']:.2e}")
    return eb


def _env_info():
    """Capture the hardware/software environment so the timing table is
    interpretable (and reproducible-by-construction) rather than a bare number."""
    cpu = platform.processor() or "unknown CPU"
    try:                                    # /proc/cpuinfo gives a real model name
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "logical_cores": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "python": platform.python_version(),
        "os": platform.platform(),
    }


def check_timing(batches=(1, 16, 64, 256), min_run_time=3.0):
    """Wall-clock per forward (fidelity eval) and per forward+backward (gradient
    step), swept over batch size, measured with ``torch.utils.benchmark.Timer``
    (blocked_autorange: adaptive iteration counts, CUDA-synced, robust median/IQR).

    Absolute milliseconds are not portable across machines; what IS reproducible is
    (a) the *shape* -- GPU wall-clock ~flat across batch (launch-bound) vs CPU
    scaling with batch -- and (b) the crossover batch. We report median [p25-p75] so
    a re-run landing inside the band is consistent rather than a contradiction."""
    opt = _opt()
    env = _env_info()
    where = env["gpu"] if DEVICE.type == "cuda" else env["cpu"]
    print(f"\n[5] TIMING on {DEVICE} ({where})")
    print(f"    env: torch {env['torch']} / CUDA {env['cuda_build']} / python "
          f"{env['python']} / {env['logical_cores']} cores "
          f"({env['torch_threads']} torch-threads) / {env['os']}")
    print(f"    method: torch.utils.benchmark blocked_autorange "
          f"(min_run_time={min_run_time}s); value = median [p25-p75]")

    def _meas(callable_):
        m = benchmark.Timer(stmt="f()", globals={"f": callable_}
                            ).blocked_autorange(min_run_time=min_run_time)
        ts = 1e3 * np.asarray(m.times)
        return (1e3 * m.median,
                float(np.percentile(ts, 25)), float(np.percentile(ts, 75)), len(ts))

    out = {}
    for B in batches:
        x = torch.randn(B, 150, 3, dtype=torch.float32, device=DEVICE, requires_grad=True)

        def fwd():
            rho = opt.simulate_choi_batch(x, dt=1.0)
            return (1.0 - opt._process_fidelity(rho).mean())

        def fwd_nograd():
            with torch.no_grad():
                fwd()

        def step():
            x.grad = None
            fwd().backward()

        fwd_m, fwd_lo, fwd_hi, nf = _meas(fwd_nograd)
        step_m, step_lo, step_hi, ns = _meas(step)
        out[B] = {"fwd_ms": fwd_m, "fwd_iqr": [fwd_lo, fwd_hi],
                  "step_ms": step_m, "step_iqr": [step_lo, step_hi],
                  "n_samples": [nf, ns]}
        print(f"    batch={B:>3}:  fwd {fwd_m:7.1f} [{fwd_lo:5.1f}-{fwd_hi:5.1f}] ms   "
              f"step {step_m:7.1f} [{step_lo:5.1f}-{step_hi:5.1f}] ms   "
              f"(per seed fwd {fwd_m/B:.2f}, step {step_m/B:.2f}; n={nf}/{ns})")
    return {"env": env, "device": str(DEVICE), "batches": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing-only", action="store_true",
                    help="skip optimization + checks 1-3, only run the device timing")
    ap.add_argument("--fd64", action="store_true",
                    help="only run the finite-difference gradient check in float64")
    args = ap.parse_args()

    print("=" * 64)
    print(f"  gradpulse supplementary validation  (device={DEVICE})")
    print("=" * 64)

    if args.fd64:
        check_finite_difference(float64=True)
        return
    if args.timing_only:
        check_timing()
        return

    check_finite_difference(float64=True)
    print("\n[opt] reference 150 ns CZ for checks 2-3 ...")
    envelope, fproc = _reference_pulse()
    print(f"      reference F_proc = {fproc:.7f}")
    check_dt_convergence(envelope, precision="single")    # shows float32 round-off confound
    check_dt_convergence(envelope, precision="double")    # clean discretization curve
    check_leakage(envelope, precision="double")
    check_error_budget(envelope, precision="double")
    check_timing()
    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
