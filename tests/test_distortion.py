import pytest
import torch
import numpy as np
from gradpulse.distortion import Predistorter

def test_forward_simulate():
    line_response = torch.tensor([1.0, 0.5, 0.25])
    predistorter = Predistorter(line_response=line_response)

    pulse = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])

    # Forward pass of impulse should give line_response
    output = predistorter.forward_simulate(pulse)

    assert torch.allclose(output[:3], line_response)
    assert torch.allclose(output[3:], torch.tensor([0.0, 0.0]))

def test_invert_tikhonov():
    line_response = torch.tensor([1.0, 0.5])
    predistorter = Predistorter(line_response=line_response)

    ideal_pulse = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])

    # Without noise, small lambda_reg should be a good inversion
    predistorted = predistorter.invert_tikhonov(ideal_pulse, lambda_reg=1e-6)

    # Forward simulate it
    recovered = predistorter.forward_simulate(predistorted)

    # It should be close to the ideal pulse (using larger atol because of FFT aliasing at ends)
    assert torch.allclose(recovered, ideal_pulse, atol=2e-2)

def test_predistort_optimization():
    line_response = torch.tensor([1.0, 0.8, 0.4])
    predistorter = Predistorter(line_response=line_response)

    ideal_pulse = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 0.0])

    # Run predistortion loop
    predistorted = predistorter.predistort(ideal_pulse, iterations=200, lr=1e-2, use_tikhonov_init=True)

    # Simulating the predistorted pulse should yield something very close to ideal_pulse
    received_pulse = predistorter.forward_simulate(predistorted)

    loss = torch.nn.functional.mse_loss(received_pulse, ideal_pulse).item()

    assert loss < 2e-3, f"Predistortion failed to converge, final MSE: {loss}"
