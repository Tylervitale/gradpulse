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

def plot_state_heatmap(density_matrix, ax=None, cmap="magma"):
    """Plot the populations and coherences of a density matrix over time as a heatmap.

    ``density_matrix`` should be a 3D array of shape [time_steps, N, N] representing the
    density matrix at each time step.
    Returns the Axes.
    """
    plt = _plt()
    dm = np.asarray(density_matrix)
    if dm.ndim != 3 or dm.shape[1] != dm.shape[2]:
        raise ValueError("density_matrix must be a 3D array of shape [time_steps, N, N]")

    n_steps, N, _ = dm.shape

    # We'll plot the absolute value of the density matrix elements
    # Reshape it to 2D for imshow: [N*N, time_steps] where we flatten the matrix for each step
    dm_flat = np.abs(dm).reshape(n_steps, N * N).T

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(dm_flat, aspect='auto', cmap=cmap, origin='lower',
                   extent=[0, n_steps - 1, -0.5, N * N - 0.5])

    # Set y-ticks to correspond to matrix elements
    yticks = np.arange(N * N)
    yticklabels = [f"|{i}><{j}|" for i in range(N) for j in range(N)]

    # If N is large, maybe don't show all ticks, but for small N (e.g. 3 or 4) it's fine.
    if N * N <= 25:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
    else:
        # Just show populations
        pop_indices = [i * N + i for i in range(N)]
        pop_labels = [f"|{i}><{i}|" for i in range(N)]
        ax.set_yticks(pop_indices)
        ax.set_yticklabels(pop_labels)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Density matrix element")
    ax.set_title("State Evolution Heatmap")
    plt.colorbar(im, ax=ax, label="|ρ_ij|")

    return ax

def plot_bloch_trajectory(states, ax=None, cmap="viridis"):
    """Plot a dynamic Bloch sphere trajectory with color gradients representing time evolution.

    ``states`` should be an array of shape [time_steps, 3] representing the Bloch vector
    at each time step.
    Returns the Axes.
    """
    plt = _plt()

    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        pass

    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("states must be a 2D array of shape [time_steps, 3]")

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='lightgray', alpha=0.1, rstride=5, cstride=5)

    # Draw axes
    ax.plot([-1, 1], [0, 0], [0, 0], color='k', linestyle='--', linewidth=0.5)
    ax.plot([0, 0], [-1, 1], [0, 0], color='k', linestyle='--', linewidth=0.5)
    ax.plot([0, 0], [0, 0], [-1, 1], color='k', linestyle='--', linewidth=0.5)

    # Plot trajectory with color gradient
    time_steps = len(states)
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, time_steps))

    for i in range(time_steps - 1):
        ax.plot(states[i:i+2, 0], states[i:i+2, 1], states[i:i+2, 2], color=colors[i], linewidth=2)

    # Mark start and end
    ax.scatter(*states[0], color='green', marker='o', s=50, label='Start')
    ax.scatter(*states[-1], color='red', marker='x', s=50, label='End')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Bloch Sphere Trajectory')
    ax.legend()

    # Add a colorbar for time
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=time_steps-1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Time step", pad=0.1, shrink=0.7)

    return ax

def plot_spectrogram(result, dt_ns=None, ax=None, channel_idx=0, cmap="viridis", NFFT=256, noverlap=128):
    """Plot a spectrogram of the synthesized pulse for visual debugging of frequency content.

    ``result`` is an optimizer result dict or raw array.
    ``channel_idx`` specifies which control channel to plot (default: 0).
    Returns the Axes.
    """
    plt = _plt()
    wf, dt = _as_waveform(result)

    if dt_ns is not None:
        dt = float(dt_ns)

    if channel_idx >= wf.shape[1]:
        raise ValueError(f"channel_idx {channel_idx} out of bounds for {wf.shape[1]} channels")

    signal = wf[:, channel_idx]

    # The time step is dt (in ns). Sampling frequency is 1 / dt (in GHz).
    Fs = 1.0 / dt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    # If NFFT is larger than the signal, adjust it
    nfft_actual = min(NFFT, len(signal))
    noverlap_actual = min(noverlap, nfft_actual - 1)

    # We only plot positive frequencies if the signal is real, but let's assume it could be complex
    # or just use default specgram which handles both. For envelope pulses, they might be complex
    # if represented as I + jQ. But the waveform might be split into real channels.
    # We'll just pass the signal directly.
    Pxx, freqs, bins, im = ax.specgram(signal, NFFT=nfft_actual, Fs=Fs, noverlap=noverlap_actual, cmap=cmap)

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Frequency (GHz)")
    ax.set_title(f"Spectrogram (Channel {channel_idx})")

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Power / Frequency (dB/Hz)")

    return ax
