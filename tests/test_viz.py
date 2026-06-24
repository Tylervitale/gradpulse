"""Smoke tests for the gradpulse.viz plotting helpers (needs matplotlib)."""
import pytest

pytest.importorskip("matplotlib", reason="needs the [viz] extra")
import matplotlib
matplotlib.use("Agg")   # headless

import numpy as np

from gradpulse import viz


def test_plot_pulse_from_array_and_dict():
    wf = np.random.RandomState(0).rand(40, 3)
    ax = viz.plot_pulse(wf, dt_ns=1.0)
    assert ax.lines and len(ax.lines) == 3
    ax2 = viz.plot_pulse({"best_waveform": wf, "dt_ns": 2.0})
    assert ax2.lines


def test_plot_convergence():
    hist = list(np.linspace(0.5, 0.999, 50))
    ax = viz.plot_convergence({"history": hist, "converged": True}, infidelity=True)
    assert ax.lines
    ax2 = viz.plot_convergence(hist, infidelity=False)
    assert ax2.lines


def test_plot_error_budget():
    budget = {"F_proc": 0.99, "r_total": 0.01, "r_control_leakage": 0.002,
              "r_decoherence": 0.008, "coherent_excess": 0.001}
    ax = viz.plot_error_budget(budget)
    assert len(ax.patches) >= 3   # bars


def test_plot_robustness():
    sweep = {"amplitude": {"x": [-0.1, 0, 0.1], "unit": "amp err",
                           "F_proc": [0.97, 0.99, 0.97], "F_avg": [0.98, 0.99, 0.98]},
             "frequency": {"x": [-1, 0, 1], "unit": "MHz",
                           "F_proc": [0.98, 0.99, 0.98], "F_avg": [0.98, 0.99, 0.98]}}
    fig = viz.plot_robustness(sweep)
    assert len(fig.axes) == 2



def test_plot_state_heatmap():
    # 3D array: 10 time steps, 2x2 density matrix
    dm = np.zeros((10, 2, 2), dtype=complex)
    dm[:, 0, 0] = np.linspace(1, 0, 10)
    dm[:, 1, 1] = np.linspace(0, 1, 10)
    ax = viz.plot_state_heatmap(dm)
    assert ax.images

def test_plot_bloch_trajectory():
    # 2D array: 20 time steps, 3 Bloch vector components
    t = np.linspace(0, 2*np.pi, 20)
    states = np.column_stack((np.cos(t), np.sin(t), np.zeros_like(t)))
    ax = viz.plot_bloch_trajectory(states)
    assert ax.collections or ax.lines

def test_plot_spectrogram():
    wf = np.random.RandomState(0).rand(1000, 2)
    ax = viz.plot_spectrogram(wf, dt_ns=1.0, channel_idx=0)
    assert ax.images
def test_plot_state_heatmap_error():
    with pytest.raises(ValueError, match="density_matrix must be a 3D array"):
        viz.plot_state_heatmap(np.zeros((10, 2)))

def test_plot_bloch_trajectory_error():
    with pytest.raises(ValueError, match="states must be a 2D array of shape"):
        viz.plot_bloch_trajectory(np.zeros((10, 2)))

def test_plot_spectrogram_error():
    wf = np.random.RandomState(0).rand(100, 2)
    with pytest.raises(ValueError, match="channel_idx 2 out of bounds"):
        viz.plot_spectrogram(wf, channel_idx=2)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
