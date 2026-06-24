"""Cable distortion hack module."""
import torch
import numpy as np

class Predistorter:
    """
    Actively invert the measured transfer functions of cryogenic wiring and control lines
    to pre-distort ideal pulses.
    """
    def __init__(self, line_response: torch.Tensor, dt_ns: float = 1.0, device: str = "cpu"):
        """
        Args:
            line_response: A 1D time-domain impulse response kernel.
            dt_ns: Time step in nanoseconds.
            device: 'cpu' or 'cuda'.
        """
        self.device = torch.device(device)
        self.line_response = line_response.to(self.device)
        self.dt_ns = dt_ns

    def forward_simulate(self, pulse: torch.Tensor) -> torch.Tensor:
        """
        Forward-propagate the pulse through the line response using frequency-domain convolution.
        pulse and line_response are assumed to be 1D tensors.
        """
        pulse = pulse.to(self.device)
        if pulse.ndim != 1:
            raise ValueError("pulse must be a 1D tensor.")

        N = len(pulse)
        M = len(self.line_response)
        L = N + M - 1

        # Next power of 2 for fast FFT
        n_fft = 2 ** int(np.ceil(np.log2(L)))

        pulse_fft = torch.fft.rfft(pulse, n=n_fft)
        kernel_fft = torch.fft.rfft(self.line_response, n=n_fft)

        convolved_fft = pulse_fft * kernel_fft
        convolved = torch.fft.irfft(convolved_fft, n=n_fft)

        # To match the length of the original pulse without phase delay padding
        # we can just take the first N samples or use 'same' padding.
        # For line response, standard convolution spreads it out. We will take
        # the first N samples assuming the impulse response is causal and peaked near t=0.
        # But if the peak is shifted, the output will be shifted.
        # It's safest to let the caller handle padding or just truncate to N.
        # Here we truncate to N.
        return convolved[:N]

    def invert_tikhonov(self, ideal_pulse: torch.Tensor, lambda_reg: float = 1e-3) -> torch.Tensor:
        """Iterate Tikhonov regularization.

        Iterative regularization in the frequency domain.
        H_inv = H* / (|H|^2 + lambda_reg)
        """
        ideal_pulse = ideal_pulse.to(self.device)
        N = len(ideal_pulse)
        M = len(self.line_response)
        L = N + M - 1
        n_fft = 2 ** int(np.ceil(np.log2(L)))

        pulse_fft = torch.fft.rfft(ideal_pulse, n=n_fft)
        kernel_fft = torch.fft.rfft(self.line_response, n=n_fft)

        kernel_mag_sq = torch.abs(kernel_fft)**2
        # Tikhonov inverse filter
        inv_filter = torch.conj(kernel_fft) / (kernel_mag_sq + lambda_reg)

        predistorted_fft = pulse_fft * inv_filter
        predistorted = torch.fft.irfft(predistorted_fft, n=n_fft)

        # We truncated to N in forward_simulate, here we just return N
        return predistorted[:N]

    def predistort(self, ideal_pulse: torch.Tensor, iterations: int = 100, lr: float = 1e-2,
                   use_tikhonov_init: bool = True, lambda_reg: float = 1e-3) -> torch.Tensor:
        """
        Feedback loop that simulates the forward-propagated distorted pulse
        to minimize the residual error between the target shape and the received shape.
        """
        ideal_pulse = ideal_pulse.to(self.device)

        if use_tikhonov_init:
            predistorted_pulse = self.invert_tikhonov(ideal_pulse, lambda_reg=lambda_reg).detach()
        else:
            predistorted_pulse = ideal_pulse.clone().detach()

        predistorted_pulse.requires_grad_(True)

        optimizer = torch.optim.Adam([predistorted_pulse], lr=lr)

        for i in range(iterations):
            optimizer.zero_grad()

            received_pulse = self.forward_simulate(predistorted_pulse)

            # Truncate received pulse to match ideal_pulse length if necessary,
            # though forward_simulate already truncates to N.
            # Compare up to the length of ideal_pulse
            loss = torch.nn.functional.mse_loss(received_pulse, ideal_pulse)

            loss.backward()
            optimizer.step()

        return predistorted_pulse.detach()
