"""gradpulse.viz -- quick matplotlib views of optimization results.

Every result dict and analysis dict gradpulse returns is already plottable; these
helpers just save the boilerplate for the four things a scientist looks at after a
run: the pulse, the convergence curve, the error budget, and the robustness sweep.

Needs matplotlib (the ``[viz]`` extra: ``pip install gradpulse[viz]``); it is
imported lazily so the gradpulse core keeps no hard dependency on it. Each function
takes an optional ``ax`` and returns the matplotlib ``Axes`` (or ``Figure`` for the
multi-panel ones) so you can restyle, compose, or ``savefig`` the result.

    import gradpulse as gp
    from gradpulse import viz
    r = gp.optimize_cz()
    viz.plot_pulse(r)
    viz.plot_convergence(r)
    viz.plot_error_budget(r["optimizer"].error_budget(r["best_raw_param"]))
"""
from __future__ import annotations

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover - only without the extra
        raise ImportError(
            "gradpulse.viz needs matplotlib. Install it with "
            "`pip install gradpulse[viz]` (or `pip install matplotlib`)."
        ) from e


# Default per-channel labels for the parametric-coupler layouts (3/4/6 channels);
# anything else falls back to "ch N". Override with channel_labels=[...].
_PARAMETRIC_CHANNELS = ["q1 drive", "q2 drive", "coupler",
                        "coupler phase", "stark q1", "stark q2"]


def _as_waveform(result):
    """Accept a result dict or a raw [n_slices, n_channels] array; return (wf, dt)."""
    if isinstance(result, dict):
        wf = np.asarray(result["best_waveform"])
        dt = float(result.get("dt_ns", 1.0))
    else:
        wf = np.asarray(result)
        dt = 1.0
    if wf.ndim == 1:
        wf = wf[:, None]
    return wf, dt


def plot_pulse(result, dt_ns=None, ax=None, channel_labels=None):
    """Plot each control channel's envelope vs time.

    ``result`` is an optimizer result dict (uses ``best_waveform`` and, if present,
    ``dt_ns``) or a raw ``[n_slices, n_channels]`` array. ``dt_ns`` overrides the
    time step. Returns the Axes.
    """
    plt = _plt()
    wf, dt = _as_waveform(result)
    if dt_ns is not None:
        dt = float(dt_ns)
    n_slices, n_ch = wf.shape
    t = np.arange(n_slices) * dt
    if channel_labels is None:
        channel_labels = [_PARAMETRIC_CHANNELS[c] if c < len(_PARAMETRIC_CHANNELS)
                          else f"ch {c}" for c in range(n_ch)]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    for c in range(n_ch):
        ax.plot(t, wf[:, c], label=channel_labels[c], lw=1.8)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("control amplitude (a.u.)")
    ax.set_title(f"Optimized pulse ({n_slices} slices x {dt:g} ns = {n_slices*dt:g} ns)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_convergence(result, ax=None, infidelity=True):
    """Plot the optimization convergence curve from ``result["history"]``.

    ``infidelity=True`` (default) plots ``1 - F`` on a log axis (the usual way to
    read how many nines you reached); ``False`` plots fidelity linearly. Returns
    the Axes.
    """
    plt = _plt()
    hist = np.asarray(result["history"] if isinstance(result, dict) else result,
                      dtype=float)
    it = np.arange(len(hist))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    if infidelity:
        ax.semilogy(it, np.clip(1.0 - hist, 1e-12, None), lw=1.8)
        ax.set_ylabel("infidelity  1 - F")
    else:
        ax.plot(it, hist, lw=1.8)
        ax.set_ylabel("fidelity F")
    ax.set_xlabel("iteration")
    title = "Convergence"
    if isinstance(result, dict) and "converged" in result:
        title += f"  (converged={result['converged']}, final F={hist[-1]:.5f})"
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    return ax


def plot_error_budget(budget, ax=None):
    """Bar chart of the infidelity decomposition from ``error_budget()``.

    Splits total infidelity into the control/leakage part and the decoherence
    floor, with the unitarity-based coherent excess alongside. Returns the Axes.
    """
    plt = _plt()
    labels = ["total", "control+leakage", "decoherence floor"]
    keys = ["r_total", "r_control_leakage", "r_decoherence"]
    vals = [float(budget[k]) for k in keys]
    if "coherent_excess" in budget:
        labels.append("coherent excess")
        vals.append(float(budget["coherent_excess"]))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    colors = ["#444", "#1f77b4", "#d62728", "#ff7f0e"][:len(vals)]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("infidelity contribution")
    fp = budget.get("F_proc")
    ax.set_title("Error budget" + (f"  (F_proc={fp:.5f})" if fp is not None else ""))
    ax.bar_label(bars, fmt="%.2e", fontsize=8, padding=2)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(alpha=0.3, axis="y")
    return ax


def plot_robustness(sweep, axes=None):
    """Plot fidelity vs miscalibration for each axis in a ``robustness_sweep()``.

    ``sweep`` is ``{axis_name: {"x", "unit", "F_proc", "F_avg"}}``. One subplot per
    axis. Returns the Figure.
    """
    plt = _plt()
    names = list(sweep.keys())
    if axes is None:
        fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 3.5),
                                 squeeze=False)
        axes = axes[0]
    else:
        fig = axes[0].figure
    for ax, name in zip(axes, names):
        s = sweep[name]
        ax.plot(s["x"], s["F_proc"], "o-", lw=1.8, ms=4)
        ax.set_xlabel(s.get("unit", name))
        ax.set_ylabel("F_proc")
        ax.set_title(f"Robustness: {name}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
