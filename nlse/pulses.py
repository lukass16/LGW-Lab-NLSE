"""Common pulse envelopes used as solver inputs."""

import numpy as np
import torch


def gaussian(t, tau, amplitude):
    """Generate a Gaussian pulse."""
    return amplitude * torch.exp(-(t ** 2) / (2 * tau ** 2))


def strong_soliton(t, beta2_k, gamma_k, tau):
    """Fundamental soliton envelope for the strong pulse A_k."""
    a_2_squared = -beta2_k / (gamma_k * tau ** 2)
    a_2 = np.sqrt(a_2_squared)
    return a_2 * torch.cosh(t / tau) ** (-1)
