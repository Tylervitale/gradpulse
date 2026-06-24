import numpy as np
import pytest
from src.gradpulse.mitigation import (
    stretch_pulse,
    fold_pulse,
    fold_sequence,
    fit_linear,
    fit_polynomial,
    fit_exponential,
    fit_richardson,
    ZNE
)

def test_stretch_pulse_1d():
    wf = np.array([1.0, 1.0, 1.0])
    stretched = stretch_pulse(wf, 2.0)
    assert stretched.shape[0] == 6
    assert np.allclose(stretched, 0.5)

def test_stretch_pulse_2d():
    wf = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    stretched = stretch_pulse(wf, 2.0)
    assert stretched.shape == (6, 2)
    assert np.allclose(stretched[:, 0], 0.5)
    assert np.allclose(stretched[:, 1], 1.0)

def test_fold_pulse():
    wf = np.array([1.0, 2.0, 3.0])
    folded = fold_pulse(wf, 3)
    assert folded.shape[0] == 9
    assert np.allclose(folded[:3], wf)
    assert np.allclose(folded[3:6], -np.flip(wf, axis=0))
    assert np.allclose(folded[6:], wf)

def test_fold_sequence():
    wf1 = np.array([1.0, 2.0])
    wf2 = np.array([3.0, 4.0])
    seq = [wf1, wf2]
    folded = fold_sequence(seq, 3)
    assert len(folded) == 6
    assert np.allclose(folded[0], wf1)
    assert np.allclose(folded[1], wf2)
    assert np.allclose(folded[2], -np.flip(wf2, axis=0))
    assert np.allclose(folded[3], -np.flip(wf1, axis=0))
    assert np.allclose(folded[4], wf1)
    assert np.allclose(folded[5], wf2)

def test_extrapolation_linear():
    scales = [1, 2, 3]
    values = [0.9, 0.8, 0.7]
    val = fit_linear(scales, values)
    assert np.isclose(val, 1.0)

def test_extrapolation_richardson():
    scales = [1, 2, 3]
    values = [0.9, 0.8, 0.7]
    val = fit_richardson(scales, values)
    assert np.isclose(val, 1.0)

def test_zne_class():
    def mock_expectation(wf):
        # A mock expectation where stretching linearly decreases expectation
        # Let's say expectation = 1.0 - 0.1 * len(wf) / original_len
        # So scale_factor = len(wf) / original_len
        return 1.0 - 0.1 * (wf.shape[0] / 3)

    wf = np.array([1.0, 2.0, 3.0])
    zne = ZNE(mock_expectation, scaling_method='stretch', extrapolation_method='linear')
    scales = [1, 2, 3]
    # expected values: 0.9, 0.8, 0.7
    # extrapolated to 0
    val = zne(wf, scales)
    assert np.isclose(val, 1.0)

def test_zne_fold():
    def mock_expectation(wf):
        # Let's say expectation = 1.0 - 0.1 * (len(wf)/3)
        # where scale_factor = 1, 3, 5 -> len(wf)/3 = 1, 3, 5
        return 1.0 - 0.1 * (wf.shape[0] / 3)

    wf = np.array([1.0, 2.0, 3.0])
    zne = ZNE(mock_expectation, scaling_method='fold', extrapolation_method='linear')
    scales = [1, 3, 5]
    # values: 0.9, 0.7, 0.5
    # slope: (0.5 - 0.9) / 4 = -0.1
    # intercept: 0.9 - (-0.1 * 1) = 1.0
    val = zne(wf, scales)
    assert np.isclose(val, 1.0)
