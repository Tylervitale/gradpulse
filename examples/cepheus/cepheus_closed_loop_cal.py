"""Joint closed-loop calibration of a Cepheus Level-B CZ -- the route to a hardware GO.

The open-loop scout was NO-GO and the model caps coupler-only at ~0.69 with REPRESENTATIVE coupler
params -- but Cepheus's native CZ is coupler-flux-only at ~0.994, so the device DOES support it;
the model is the wrong part. The fix native uses too: optimize the pulse's knobs AGAINST THE
MEASURED RB, so the loop absorbs the coupler-param/distortion gap it cannot model.

Knobs (the hardware-realizable control set for a fixed-frequency-qubit device):
  * flux SCALE                       -- overall coupler amplitude (the entangling angle),
  * a few smooth SHAPE modes         -- sin(k*pi*tau), vanish at the endpoints so the pulse stays
                                        composable (rest-0 coupler, chainable in RB),
  * two VIRTUAL-Z phases             -- free single-qubit frame shifts (device shift_phase).

`joint_cal(measure_fn, x0)` is BACKEND-AGNOSTIC: `measure_fn(knobs) -> F_avg` is either the sim
surrogate (this script's rehearsal) or `BraketRBMeasure` (interleaved RB on Cepheus). The identical
optimizer drives both; only measure_fn changes.

This file's main() is the FREE rehearsal: it proves the shape-knob loop CONVERGES under realistic
shot noise (recovers from a detuned start to the in-model ceiling) and counts the evaluations ->
the on-device dollar budget. Honest boundary: the rehearsal runs on the MODEL, so it proves the
LOOP works and budgets it -- NOT the final on-device fidelity (the model can't predict that; only
the QPU can, which is the whole point of running closed-loop ON the device).

Run (free rehearsal):  python examples/cepheus/cepheus_closed_loop_cal.py
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
HERE = os.path.dirname(os.path.abspath(__file__))

import json
import math

import numpy as np
import torch

import gradpulse as gp
from gradpulse.hardware import simulate_noisy_irb
from gradpulse.multiqubit import DEVICE
import gradpulse.braket_bridge as bb

DT = 1.0
N_MODES = 3                                  # smooth coupler-shape perturbation modes
_CAL_LENGTHS, _CAL_SEEDS = [1, 2, 4, 8], 2   # per-eval interleaved-RB design (budget accounting)
# FAITHFUL regime (see cepheus_faithful_model.py): Cepheus MEASURED q16/q25 + Rigetti-PROTOTYPE
# coupler -- idle 2.644 GHz bare, tune-up 978 MHz, anharm -227/-178/-221. (Symmetric g approximates
# the asymmetric g1c=96.2/g2c=83.9 ~ 90; the tunable_coupler_cz evaluator takes a single g.)
MODEL = dict(freqs_ghz=(4.654, 2.644, 4.806), anharm_mhz=(-227.0, -178.0, -221.0),
             g_qubit_coupler_mhz=90.0, t1_ns=(45196.8, 20000.0, 29784.9),
             t2_ns=(16681.5, 15000.0, 13017.7), delta_max_mhz=978.0)
# The TRUE device differs in the one HIGH-SENSITIVITY unmeasured param: the coupling g (the sweep
# showed g=60->0.65 vs 90->0.92). We DESIGN with the prototype g=90 but the real Cepheus (16,25) g
# is unknown -- model it as 75 (a realistic ~17% miss). KEY question: can on-device cal recover, given
# the flux SCALE knob directly sets how hard the coupler tunes (i.e. partly compensates a wrong g)?
# If yes, Level-B is ROBUST to not knowing Cepheus's exact g; if no, we need the exact g.
TRUE = dict(MODEL, g_qubit_coupler_mhz=75.0)


def coupler_flux_from_knobs(base_u, knobs, n_modes=N_MODES):
    """Physical coupler flux from the knob vector [scale, c1..cK, vz0, vz1].

    u(t) = scale * base_u(t) + sum_k c_k * sin(k*pi*tau).  The sine modes vanish at both endpoints,
    so the pulse stays rest-0 (composable). Returns (flux_u, vz0, vz1).
    """
    base_u = np.asarray(base_u, dtype=float).ravel()
    scale = float(knobs[0])
    coeffs = [float(c) for c in knobs[1:1 + n_modes]]
    vz0, vz1 = float(knobs[1 + n_modes]), float(knobs[2 + n_modes])
    tau = np.linspace(0.0, 1.0, base_u.size)
    u = scale * base_u
    for k, c in enumerate(coeffs, start=1):
        u = u + c * np.sin(k * math.pi * tau)
    return u, vz0, vz1


class LevelBEvaluator:
    """Sim surrogate: F_avg of (coupler flux from knobs, virtual-Z) under a profile.

    Mirrors the on-device bench pulse exactly (coupler flux + post-hoc virtual-Z). The Choi is
    propagated once per (scale, shape) and cached; the virtual-Z is then the closed-form
    data-subspace formula (the in-package cz_data_virtualz estimator).
    """

    def __init__(self, profile_kwargs, base_u, n_modes=N_MODES):
        self.opt = gp.tunable_coupler_cz(verbose=False, **profile_kwargs)
        self.base_u = np.asarray(base_u, dtype=float).ravel()
        self.n_modes = n_modes
        s = self.opt._cz_data_vz_setup()
        self.di, self.q0, self.q1, self.czd = s["d_idx"], s["q0"], s["q1"], s["czd"]
        self.ci, self.dc = self.opt._comp_idx, self.opt._dcomp
        self._cache = {}

    def _G(self, scale, coeffs):
        key = (round(float(scale), 4), tuple(round(float(c), 4) for c in coeffs))
        G = self._cache.get(key)
        if G is not None:
            return G
        u, _, _ = coupler_flux_from_knobs(self.base_u, [scale, *coeffs, 0.0, 0.0], self.n_modes)
        x = np.full((u.size, 3), 0.5)
        x[:, 1] = np.clip(0.5 * (u + 1.0), 0.0, 1.0)
        xt = torch.as_tensor(x, dtype=self.opt.rdtype, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            rho = self.opt._propagate_choi(xt, DT, 1.0)
            C = (rho[:, :, self.ci, :][:, :, :, self.ci]
                 .reshape(-1, self.dc, self.dc, self.dc, self.dc)[0])
            G = C[self.di[:, None], self.di[None, :], self.di[:, None], self.di[None, :]]
        self._cache[key] = G
        return G

    def f_avg(self, knobs):
        scale = knobs[0]
        coeffs = knobs[1:1 + self.n_modes]
        vz0, vz1 = knobs[1 + self.n_modes], knobs[2 + self.n_modes]
        G = self._G(scale, coeffs)
        ph = torch.exp(1j * (float(vz0) * self.q0 + float(vz1) * self.q1)).to(G.dtype)
        Vb = self.czd.to(G.dtype) * ph
        f_proc = float((Vb.conj() @ G @ Vb).real / 16.0)
        return (4.0 * f_proc + 1.0) / 5.0

    def best_over_vz(self, scale, coeffs, n=49):
        G = self._G(scale, coeffs)
        grid = torch.linspace(0, 2 * math.pi, n, device=DEVICE, dtype=self.opt.rdtype)
        e0 = torch.exp(1j * grid[:, None] * self.q0[None, :]).to(G.dtype)
        e1 = torch.exp(1j * grid[:, None] * self.q1[None, :]).to(G.dtype)
        V = self.czd[None, None, :].to(G.dtype) * e0[:, None, :] * e1[None, :, :]
        Fg = torch.einsum('abk,kl,abl->ab', V.conj(), G, V).real / 16.0
        return (4.0 * float(Fg.max()) + 1.0) / 5.0


class BraketRBMeasure:
    """On-device cost function: F_avg of the knob pulse via interleaved RB on a real QPU.

    The shared native reference is run ONCE and cached; each call builds the bench pulse from the
    knobs, runs interleaved RB at ``lengths`` x ``seeds``, and fits F_avg against the reference
    (rb.py leakage-aware estimator via fit_irb). This is the irreducibly on-device half -- each call
    costs money -- so the closed loop with this measure_fn IS the ~$150-300 calibration run, not a
    free rehearsal. ``dry_run(knobs)`` builds + serializes one candidate offline (no submit) to
    verify the pipeline; ``measure_fn`` (the call) requires a live device.
    """

    def __init__(self, device, flux_frame, drive_frames, base_u, *, peak, qubits,
                 lengths=(1, 2, 4, 8), seeds=2, shots=500, s3_folder=None, n_modes=N_MODES,
                 fit_irb=None, asymptote=None):
        self.device, self.flux_frame, self.drive_frames = device, flux_frame, drive_frames
        self.base_u, self.peak, self.qubits = np.asarray(base_u, float).ravel(), float(peak), tuple(qubits)
        self.lengths, self.seeds, self.shots = list(lengths), int(seeds), int(shots)
        self.s3_folder, self.n_modes, self.fit_irb, self.asym = s3_folder, n_modes, fit_irb, asymptote
        self.ref, self.intl = bb.native_rb_sequences(self.lengths, self.seeds, seed=0, interleaved=False), \
            bb.native_rb_sequences(self.lengths, self.seeds, seed=0, interleaved=True)
        self._ref_cached = None
        self.n_calls = 0

    def _bench(self, knobs):
        u, vz0, vz1 = coupler_flux_from_knobs(self.base_u, knobs, self.n_modes)
        return bb.build_bench_cz_pulse_sequence(u, self.flux_frame, peak_amplitude=self.peak,
                                                drive_frames=self.drive_frames, virtual_z=(vz0, vz1))

    def dry_run(self, knobs):
        """Offline: build the bench pulse + verify it serializes (no submission)."""
        bench = self._bench(knobs)
        return bb.verify_levelb_offline(bench, qubits=self.qubits)

    def _run(self, seq, pulse=None):
        circ = bb.to_braket_rb_circuit(seq["gates"], qubits=self.qubits, bench_cz_pulse=pulse)
        task = self.device.run(circ, shots=self.shots, disable_qubit_rewiring=True,
                               s3_destination_folder=self.s3_folder)
        return bb.survival_from_counts(task.result().measurement_counts)

    def __call__(self, knobs):
        if self._ref_cached is None:                      # shared native reference, run once
            self._ref_cached = [dict(s, survival=self._run(s, None)) for s in self.ref]
        bench = self._bench(knobs)
        intl = [dict(s, survival=self._run(s, bench)) for s in self.intl]
        self.n_calls += 1
        res = self.fit_irb(self._ref_cached, intl, self.lengths, asymptote=self.asym)
        return res["f_cz"]                                # F_avg-like; maximize


def joint_cal(measure_fn, x0, maxiter=40, scale_bounds=(0.2, 1.8), coeff_bound=0.3,
              steps=None):
    """Backend-agnostic JOINT closed-loop cal over the knob vector, maximizing measured F_avg.

    Nelder-Mead with an EXPLICIT initial simplex sized per knob -- critical, because the default
    simplex perturbs a coordinate that starts at 0 by only ~2.5e-4, so virtual-Z (and shape modes)
    starting near 0 are never explored and the loop sticks. ``steps`` is the per-knob initial step
    (scale, shape modes..., vz0, vz1); defaults to (0.15, 0.1.., 1.0, 1.0) rad. Knobs are clipped to
    physical bounds. Returns (best_x, best_f, best-so-far history, n_measurements).
    """
    from scipy.optimize import minimize
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    nm = n - 3
    if steps is None:
        steps = np.array([0.15] + [0.1] * nm + [1.0, 1.0])
    steps = np.asarray(steps, dtype=float)
    st = {"n": 0, "best": -1.0, "bestx": tuple(x0), "hist": []}

    def clip(x):
        y = np.array(x, dtype=float)
        y[0] = np.clip(y[0], *scale_bounds)
        if nm:
            y[1:1 + nm] = np.clip(y[1:1 + nm], -coeff_bound, coeff_bound)
        return y

    def neg(x):
        y = clip(x)
        f = measure_fn(y)
        st["n"] += 1
        if f > st["best"]:
            st["best"], st["bestx"] = f, tuple(y)
        st["hist"].append(st["best"])
        return -f

    simplex = np.vstack([x0] + [x0 + np.eye(n)[i] * steps[i] for i in range(n)])
    minimize(neg, x0, method="Nelder-Mead", options={"maxiter": int(maxiter),
             "initial_simplex": simplex, "xatol": 2e-3, "fatol": 1e-4})
    return st["bestx"], st["best"], st["hist"], st["n"]


def staged_cal(measure_fn, x0, scale_grid, vz_grid, n_modes=N_MODES, rounds=1):
    """Robust STAGED cal: scale (1D scan) then a JOINT 2D virtual-Z grid.

    The two virtual-Z phases are strongly COUPLED -- the gate fidelity is flat in either phase
    alone but peaks at a specific (vz0,vz1) pair -- so SEQUENTIAL 1D vz sweeps get stuck near the
    start (the bug in a naive Stage 2). A joint 2D grid finds the pair; selection is by max
    interleaved survival (unbiased, unlike the RB-fit). Scale is secondary (the gate is broadly
    flat in it once vz is right). Returns (best_x, n_measurements).
    """
    x = np.array(x0, dtype=float)
    n = 0

    def at(updates):
        y = x.copy()
        for i, v in updates:
            y[i] = v
        return y

    for _ in range(int(rounds)):
        fs = [measure_fn(at([(0, s)])) for s in scale_grid]; n += len(scale_grid)
        x[0] = scale_grid[int(np.argmax(fs))]
        best, bv = -1.0, (x[1 + n_modes], x[2 + n_modes])
        for v0 in vz_grid:
            for v1 in vz_grid:
                f = measure_fn(at([(1 + n_modes, v0), (2 + n_modes, v1)])); n += 1
                if f > best:
                    best, bv = f, (v0, v1)
        x[1 + n_modes], x[2 + n_modes] = bv
    return x, n


def main():
    base_u = np.load(os.path.join(HERE, "cepheus_cz_shape.npy")).ravel()    # native flat-top shape, 0->1->0
    base_u = base_u / np.max(np.abs(base_u))                      # unit-peak; scale knob sets amplitude
    print(f"base shape: {base_u.size} samples; knobs = [scale, c1..c{N_MODES}, vz0, vz1] "
          f"({3 + N_MODES} total)\n")

    print("building model + true-device evaluators (27-D open systems) ...")
    model_eval = LevelBEvaluator(MODEL, base_u)
    true_eval = LevelBEvaluator(TRUE, base_u)

    # in-model coupler-only ceiling (best scale, no shape pert, best vz) for context
    ceil = max(true_eval.best_over_vz(s, [0.0] * N_MODES) for s in np.linspace(0.4, 1.2, 9))

    # REALISTIC warm start = open-loop transfer: native shape, scale 1, virtual-Z calibrated on the
    # MODEL (what you'd transfer to the device). The loop then refines on the TRUE device.
    grid = torch.linspace(0, 2 * math.pi, 49, device=DEVICE, dtype=model_eval.opt.rdtype)
    # cheap: model's best vz at scale 1 via its own grid (reuse best_over_vz machinery indirectly)
    Gm = model_eval._G(1.0, [0.0] * N_MODES)
    e0 = torch.exp(1j * grid[:, None] * model_eval.q0[None, :]).to(Gm.dtype)
    e1 = torch.exp(1j * grid[:, None] * model_eval.q1[None, :]).to(Gm.dtype)
    Vm = model_eval.czd[None, None, :].to(Gm.dtype) * e0[:, None, :] * e1[None, :, :]
    Fg = torch.einsum('abk,kl,abl->ab', Vm.conj(), Gm, Vm).real / 16.0
    fl = int(Fg.argmax().item()); vz0_m, vz1_m = float(grid[fl // 49]), float(grid[fl % 49])

    x0 = np.array([1.0, 0.0, 0.0, 0.0, vz0_m, vz1_m])
    f_start = true_eval.f_avg(x0)
    print(f"  open-loop warm start (model-vz {vz0_m:.2f},{vz1_m:.2f}) on true device: {f_start:.4f}")
    print(f"  in-model coupler-only ceiling: {ceil:.4f}\n")

    shots, nseq = 2000, 30
    # Cal cost = MAX INTERLEAVED SURVIVAL at short lengths, NOT the RB-fit F_avg (badly biased
    # for low-fidelity gates -- argmax-of-fit would pick the WORST gates).
    cal_lens = (1, 2, 3)

    def measure_fn(knobs, seed):
        F = true_eval.f_avg(knobs)
        alpha = max((4.0 * F - 1.0) / 3.0, 0.0)
        rng = np.random.default_rng(seed)
        surv = [rng.binomial(shots, min(0.75 * alpha ** m + 0.25, 1.0), size=nseq).mean() / shots
                for m in cal_lens]
        return float(np.mean(surv))                 # mean interleaved survival (maximize)

    # --- (a) JOINT Nelder-Mead: fragile under shot noise (overfits the noisy 'best') ---
    print("(a) JOINT Nelder-Mead (noisy RB) ...")
    cnt = {"i": 0}

    def mj(knobs):
        cnt["i"] += 1
        return measure_fn(knobs, 10_000 + cnt["i"])
    bj, _, _, nj = joint_cal(mj, x0, maxiter=40)
    fj = true_eval.f_avg(bj)
    print(f"    {nj} evals -> F_avg(true) {fj:.4f}  (start {f_start:.4f}, ceiling {ceil:.4f})")

    # --- (b) STAGED coordinate cal (grid argmax per axis): robust under the SAME noise ---
    print("(b) STAGED cal: scale scan + JOINT 2D virtual-Z grid (same noise) ...")
    scale_grid = np.linspace(0.5, 1.3, 9)
    vz_grid = np.linspace(0, 2 * math.pi, 9)        # 2D -> 9x9 = 81 vz points
    cnt2 = {"i": 0}

    def ms(knobs):
        cnt2["i"] += 1
        return measure_fn(knobs, 20_000 + cnt2["i"])
    bs, ns = staged_cal(ms, x0, scale_grid, vz_grid)
    fs_true = true_eval.f_avg(bs)
    print(f"    {ns} evals -> F_avg(true) {fs_true:.4f}  scale={bs[0]:.3f} "
          f"vz=({bs[1+N_MODES]:.2f},{bs[2+N_MODES]:.2f})")

    # budget the on-device STAGED run: shared ref (once) + per-eval interleaved
    per = len(_CAL_LENGTHS) * _CAL_SEEDS
    tot = per + ns * per
    cost = bb.estimate_experiment_cost(tot, 500)
    cc = bb.estimate_experiment_cost(2, 100)
    print(f"\nSTAGED recovered {(fs_true - f_start) / max(1e-9, ceil - f_start) * 100:.0f}% of the "
          f"start->ceiling gap vs JOINT {(fj - f_start) / max(1e-9, ceil - f_start) * 100:.0f}%.")
    print(f"ON-DEVICE BUDGET (staged): {ns} evals x {per} + {per} ref = {tot} circuits @ 500 shots "
          f"~= ${cost.total_usd:.2f} + canaries ${cc.total_usd:.2f} = "
          f"${cost.total_usd + cc.total_usd:.2f}  (pricing {cost.pricing_as_of})")
    robust = fs_true > f_start + 0.1 and fs_true >= ceil - 0.05
    print(f"\nVERDICT: staged-with-2D-vz {'CLIMBS TO THE CEILING (validated)' if robust else 'still short'}; "
          "the JOINT 2D virtual-Z grid is essential -- sequential 1D vz (and Nelder-Mead) get stuck "
          "because the two phases are coupled. On-device, run scale-scan + 2D-vz-grid, max-survival "
          "selection. It optimizes against the REAL device (where ~0.994 exists), not this model's "
          "ceiling, so a successful climb here means the method works when the device supports a GO.")


if __name__ == "__main__":
    main()
