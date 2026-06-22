"""Decoherence in the loop vs optimise-coherent-then-multiply: a measured head-to-head.

This package argues that folding decoherence in *after* optimisation -- the common
``F ~= F_coherent * exp(-t_g/T)`` budget -- misleads, and that putting decoherence
*inside* the GRAPE gradient does better. ``run_head_to_head`` runs both recipes on
the same device across a sweep of gate durations and returns the measured gap, so
the claim is demonstrated, not asserted. Nothing here is hard-coded: every operating
point falls out of the sweep.

Both recipes reuse one primitive -- ``ParametricCZOptimizer.optimize_multi_seed``
with ``diss_scale`` set to ``1.0`` (true open-system objective) or ``0.0`` (coherent
objective). At each duration ``t_g`` the module records:

* ``A``  in-loop      -- optimise the true open-system ``F_proc`` (``diss_scale=1``).
* ``B``  coherent     -- optimise the coherent ``F_proc`` (``diss_scale=0``), then
    - ``B_predicted`` -- MULTIPLY by the analytic coherence budget the recipe trusts,
      ``F_coh * (1 - EPG_dec(t_g))`` with ``EPG_dec`` from
      :func:`gradpulse.literature.analytic_coherence_limit_epg`;
    - ``B_delivered`` -- SCORE that coherent-optimal pulse under the true open system
      (what it actually delivers on the device).

Each method then chooses its own best duration -- in-loop by its true fidelity, the
multiply recipe by its prediction -- and the summary reports the delivered gap and
its decomposition into a pulse-shaping part (in-loop's gradient finds a better pulse
family at fixed duration) and a duration-selection part (the analytic budget points
the recipe at the wrong duration).
"""
from __future__ import annotations

from .literature import analytic_coherence_limit_epg, f_avg as _f_avg


def resample_pulse(wf, n_new):
    """Linearly resample a ``[n_old, C]`` pulse envelope onto ``n_new`` slices
    (time axis = dim 0, channels preserved). Used to warm-start one sweep point
    from the previous point's solution when the slice count changes -- a longer
    gate is seeded by the stretched shorter-gate pulse instead of from scratch."""
    import torch
    t = torch.as_tensor(wf, dtype=torch.float64)
    if t.shape[0] == int(n_new):
        return t.cpu().numpy()
    x = t.t().unsqueeze(0)                                   # [1, C, n_old]
    y = torch.nn.functional.interpolate(x, size=int(n_new), mode="linear",
                                        align_corners=True)
    return y.squeeze(0).t().contiguous().cpu().numpy()      # [n_new, C]


def run_head_to_head(profile, durations_ns, *, dt_ns: float = 1.0, n_seeds: int = 2,
                     iterations: int = 150, lbfgs_iters: int = 40, lr: float = 0.02,
                     precision: str = "double", seed: int = 0, verbose: bool = True,
                     warm_start_chain: bool = False):
    """Run the in-loop vs optimise-coherent-then-multiply head-to-head.

    profile:      a ``ParametricCouplerProfile`` (use a regime where decoherence and
                  leakage genuinely compete -- e.g. a few-microsecond T1/T2 -- or the
                  two recipes converge and there is nothing to see).
    durations_ns: gate durations to sweep (``n_slices = round(t_g / dt_ns)``).
    warm_start_chain: if True, warm-start each duration from the previous one's
                  solution (resampled to the new slice count), which converges far
                  faster on a fine sweep. A chains from A and B from B, so the A/B
                  fairness (identical but for ``diss_scale``) is preserved. Default
                  False keeps every point an independent from-scratch optimisation
                  (fully reproducible, no cross-point coupling). Sweep in monotonic
                  duration order when using it so consecutive pulses are similar.

    Returns ``{"rows": [...per-duration...], "summary": {...}}``; see the module
    docstring. A and B at a given duration share seeds and settings, so the only
    difference between them is whether decoherence is inside the gradient.
    """
    import torch
    from .parametric import ParametricCZOptimizer, DEVICE

    durations_ns = [float(t) for t in durations_ns]
    if not durations_ns:
        raise ValueError("durations_ns must be non-empty")
    opt = ParametricCZOptimizer(profile, precision=precision)

    rows = []
    prev_a = prev_b = None          # previous-duration solutions for warm-start chaining
    for t_g in durations_ns:
        n_slices = int(round(t_g / dt_ns))
        common = dict(n_seeds=n_seeds, iterations=iterations, n_slices=n_slices,
                      dt_ns=dt_ns, lr=lr, lbfgs_iters=lbfgs_iters)
        # Warm-start each chain from its OWN previous solution (resampled), never
        # the other's -- so A and B still differ only in diss_scale.
        ws_a = resample_pulse(prev_a, n_slices) if (warm_start_chain and prev_a is not None) else None
        ws_b = resample_pulse(prev_b, n_slices) if (warm_start_chain and prev_b is not None) else None
        # Same seed for A and B so the ONLY difference is diss_scale.
        rng_a = torch.Generator(device=DEVICE).manual_seed(seed)
        rng_b = torch.Generator(device=DEVICE).manual_seed(seed)
        res_a = opt.optimize_multi_seed(label=f"inloop_{n_slices}", diss_scale=1.0,
                                        rng=rng_a, warm_start_pulse=ws_a, **common)
        eb_a = opt.error_budget(res_a["best_raw_param"])
        res_b = opt.optimize_multi_seed(label=f"coherent_{n_slices}", diss_scale=0.0,
                                        rng=rng_b, warm_start_pulse=ws_b, **common)
        eb_b = opt.error_budget(res_b["best_raw_param"])
        if warm_start_chain:
            prev_a, prev_b = res_a["best_waveform"], res_b["best_waveform"]

        epg_dec = analytic_coherence_limit_epg(profile, t_g)
        f_coh = _f_avg(eb_b["F_proc_closed"])      # coherent gate quality (closed)
        f_pred = f_coh * (1.0 - epg_dec)           # multiply-after PREDICTION
        f_deliver = eb_b["F_avg"]                  # what the coherent pulse delivers
        rows.append({
            "t_g_ns": t_g, "n_slices": n_slices,
            "f_inloop": eb_a["F_avg"],
            "f_coherent": f_coh,
            "f_predicted": f_pred,
            "f_delivered": f_deliver,
            "epg_dec_analytic": epg_dec,
        })
        if verbose:
            print(f"  t_g={t_g:6.1f} ns | in-loop={eb_a['F_avg']:.5f}  "
                  f"coherent={f_coh:.5f}  predicted={f_pred:.5f}  "
                  f"delivered={f_deliver:.5f}")

    # Each method picks its own optimal duration.
    a_star = max(rows, key=lambda r: r["f_inloop"])           # in-loop, by true fidelity
    b_pred = max(rows, key=lambda r: r["f_predicted"])        # multiply recipe's choice
    b_open = max(rows, key=lambda r: r["f_delivered"])        # coherent family, open-scored
    summary = {
        "inloop_best_duration_ns": a_star["t_g_ns"],
        "inloop_best_f_avg": a_star["f_inloop"],
        "multiply_chosen_duration_ns": b_pred["t_g_ns"],
        "multiply_predicted_f_avg": b_pred["f_predicted"],
        "multiply_delivered_f_avg": b_pred["f_delivered"],
        "coherent_openscored_best_duration_ns": b_open["t_g_ns"],
        "coherent_openscored_best_f_avg": b_open["f_delivered"],
        # Total edge of in-loop over the multiply recipe (each at its own optimum).
        "delivered_fidelity_gain_vs_multiply": a_star["f_inloop"] - b_pred["f_delivered"],
        # Decomposition of that edge:
        "pulse_shaping_gain": a_star["f_inloop"] - b_open["f_delivered"],
        "duration_selection_loss_of_multiply": b_open["f_delivered"] - b_pred["f_delivered"],
        # How optimistic the analytic budget was at the duration it chose.
        "multiplicative_overprediction": b_pred["f_predicted"] - b_pred["f_delivered"],
        "picks_shorter_gate": a_star["t_g_ns"] < b_pred["t_g_ns"],
    }
    return {"rows": rows, "summary": summary}
