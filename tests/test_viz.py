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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
