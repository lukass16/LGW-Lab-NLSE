"""Split-step Fourier solvers for the NLSE and coupled XPM."""

import torch


def dispersion_operator(A, beta2, Nt, dt, dz):
    """Apply the dispersion operator for the linear step in Fourier space using PyTorch."""
    omega = 2 * torch.pi * torch.fft.fftfreq(Nt, dt, device=A.device)
    linear_op = torch.exp(0.5j * beta2 * omega**2 * dz)
    A_ft = torch.fft.fft(A)
    A_ft *= linear_op
    return torch.fft.ifft(A_ft)


def drift_dispersion_operator(A, beta2, delta_beta, Nt, dt, dz):
    """Apply the dispersion operator with a first-order walk-off term."""
    omega = 2 * torch.pi * torch.fft.fftfreq(Nt, dt, device=A.device)
    linear_op = torch.exp(1.0j * (0.5 * beta2 * omega**2 - delta_beta * omega) * dz)
    A_ft = torch.fft.fft(A)
    A_ft *= linear_op
    return torch.fft.ifft(A_ft)


def nonlinear_operator(gamma, A, dz):
    """Apply the nonlinear operator for the nonlinear step in real space using PyTorch."""
    return A * torch.exp(1j * gamma * torch.abs(A)**2 * dz)


def nonlinear_operator_xpm(gamma_j, A_j, A_k, dz):
    """Apply the XPM operator for the nonlinear step in real space using PyTorch."""
    return A_j * torch.exp(1j * gamma_j * (torch.abs(A_j)**2 + 2 * torch.abs(A_k)**2) * dz)


def split_step_fourier(A0, dz, Nz, beta2, gamma, Lt):
    """Implement the Split-Step Fourier Method using PyTorch."""
    A = A0.clone()
    Nt = len(A)
    dt = Lt / Nt
    A_evolution = torch.zeros((Nt, Nz + 1), dtype=torch.complex64, device=A0.device)
    A_evolution[:, 0] = A0

    for i in range(Nz):
        A = nonlinear_operator(gamma, A, dz)
        A = dispersion_operator(A, beta2, Nt, dt, dz)
        A_evolution[:, i + 1] = A

    return A_evolution


def split_step_fourier_xpm(
    A0_j, A0_k, dz, Nz, beta2_j, beta2_k, gamma_j, gamma_k, Lt, strangsplitting=True
):
    """Split-step Fourier with XPM. A_j is the weak field, A_k the strong field."""
    A_j = A0_j.clone()
    A_k = A0_k.clone()
    Nt = len(A_j)
    dt = Lt / Nt
    A_j_evolution = torch.zeros((Nt, Nz + 1), dtype=torch.complex64, device=A0_j.device)
    A_k_evolution = torch.zeros((Nt, Nz + 1), dtype=torch.complex64, device=A0_k.device)
    A_j_evolution[:, 0] = A0_j
    A_k_evolution[:, 0] = A0_k

    for i in range(Nz):
        _A_j = A_j.clone()
        _A_k = A_k.clone()

        if strangsplitting:
            A_j = nonlinear_operator_xpm(gamma_j, _A_j, _A_k, dz / 2)
            A_j = dispersion_operator(A_j, beta2_j, Nt, dt, dz)
            A_j = nonlinear_operator_xpm(gamma_j, A_j, _A_k, dz / 2)

            A_k = nonlinear_operator_xpm(gamma_k, _A_k, _A_j, dz / 2)
            A_k = dispersion_operator(A_k, beta2_k, Nt, dt, dz)
            A_k = nonlinear_operator_xpm(gamma_k, A_k, _A_j, dz / 2)

            A_j_evolution[:, i + 1] = A_j
            A_k_evolution[:, i + 1] = A_k
        else:
            A_j = nonlinear_operator_xpm(gamma_j, _A_j, _A_k, dz)
            A_j = dispersion_operator(A_j, beta2_j, Nt, dt, dz)
            A_j_evolution[:, i + 1] = A_j

            A_k = nonlinear_operator_xpm(gamma_k, _A_k, _A_j, dz)
            A_k = dispersion_operator(A_k, beta2_k, Nt, dt, dz)
            A_k_evolution[:, i + 1] = A_k

    return A_j_evolution, A_k_evolution


def drift_split_step_fourier_xpm(
    A0_j, A0_k, dz, Nz, beta2_j, beta2_k, delta_beta, gamma_j, gamma_k, Lt,
    strangsplitting=True,
):
    """XPM split-step Fourier with a walk-off term on the strong (and unsplit) field."""
    A_j = A0_j.clone()
    A_k = A0_k.clone()
    Nt = len(A_j)
    dt = Lt / Nt
    A_j_evolution = torch.zeros((Nt, Nz + 1), dtype=torch.complex64, device=A0_j.device)
    A_k_evolution = torch.zeros((Nt, Nz + 1), dtype=torch.complex64, device=A0_k.device)
    A_j_evolution[:, 0] = A0_j
    A_k_evolution[:, 0] = A0_k

    for i in range(Nz):
        _A_j = A_j.clone()
        _A_k = A_k.clone()

        if strangsplitting:
            A_j = nonlinear_operator_xpm(gamma_j, _A_j, _A_k, dz / 2)
            A_j = dispersion_operator(A_j, beta2_j, Nt, dt, dz)
            A_j = nonlinear_operator_xpm(gamma_j, A_j, _A_k, dz / 2)

            A_k = nonlinear_operator_xpm(gamma_k, _A_k, _A_j, dz / 2)
            A_k = drift_dispersion_operator(A_k, beta2_k, delta_beta, Nt, dt, dz)
            A_k = nonlinear_operator_xpm(gamma_k, A_k, _A_j, dz / 2)

            A_j_evolution[:, i + 1] = A_j
            A_k_evolution[:, i + 1] = A_k
        else:
            A_j = nonlinear_operator_xpm(gamma_j, _A_j, _A_k, dz)
            A_j = drift_dispersion_operator(A_j, beta2_j, delta_beta, Nt, dt, dz)
            A_j_evolution[:, i + 1] = A_j

            A_k = nonlinear_operator_xpm(gamma_k, _A_k, _A_j, dz)
            A_k = drift_dispersion_operator(A_k, beta2_k, delta_beta, Nt, dt, dz)
            A_k_evolution[:, i + 1] = A_k

    return A_j_evolution, A_k_evolution


def split_step_fourier_xpm_batch(
    A0_j, A0_k, dz, Nz, beta2_j, beta2_k, gamma_j, gamma_k, Lt, strangsplitting=True
):
    """Batched XPM split-step Fourier.

    Args:
        A0_j: Initial weak wave, tensor of shape (B, Nt)
        A0_k: Initial strong wave, tensor of shape (Nt,) — same for all batches
        dz: Propagation step size (scalar)
        Nz: Number of propagation steps (integer)
        beta2_j / beta2_k: Dispersion, scalar or tensor of shape (B,)
        gamma_j / gamma_k: Nonlinearity, scalar or tensor of shape (B,)
        Lt: Time window (scalar)
        strangsplitting: Whether to use Strang splitting (default: True)

    Returns:
        A_j_evolution: (B, Nt, Nz+1)
        A_k_evolution: (B, Nt, Nz+1)
    """
    A_j = A0_j.clone()
    B, Nt = A_j.shape
    A_k = A0_k.unsqueeze(0).expand(B, -1).clone()
    dt = Lt / Nt

    A_j_evolution = torch.zeros((B, Nt, Nz + 1), dtype=torch.complex64, device=A0_j.device)
    A_k_evolution = torch.zeros((B, Nt, Nz + 1), dtype=torch.complex64, device=A0_j.device)
    A_j_evolution[:, :, 0] = A0_j
    A_k_evolution[:, :, 0] = A_k

    for i in range(Nz):
        _A_j = A_j.clone()
        _A_k = A_k.clone()

        if strangsplitting:
            A_j = nonlinear_operator_xpm(gamma_j, _A_j, _A_k, dz / 2)
            A_j = dispersion_operator(A_j, beta2_j, Nt, dt, dz)
            A_j = nonlinear_operator_xpm(gamma_j, A_j, _A_k, dz / 2)

            A_k = nonlinear_operator_xpm(gamma_k, _A_k, _A_j, dz / 2)
            A_k = dispersion_operator(A_k, beta2_k, Nt, dt, dz)
            A_k = nonlinear_operator_xpm(gamma_k, A_k, _A_j, dz / 2)

            A_j_evolution[:, :, i + 1] = A_j
            A_k_evolution[:, :, i + 1] = A_k
        else:
            A_j = nonlinear_operator_xpm(gamma_j, _A_j, _A_k, dz)
            A_j = dispersion_operator(A_j, beta2_j, Nt, dt, dz)
            A_j_evolution[:, :, i + 1] = A_j

            A_k = nonlinear_operator_xpm(gamma_k, _A_k, _A_j, dz)
            A_k = dispersion_operator(A_k, beta2_k, Nt, dt, dz)
            A_k_evolution[:, :, i + 1] = A_k

    return A_j_evolution, A_k_evolution


def time_derivative_fft(A, Nt, Lt):
    """Compute the first time derivative of the pulse using FFT."""
    dt = Lt / Nt
    N = A.shape[-1]
    fft_A = torch.fft.fft(A.detach())
    frequencies = torch.fft.fftfreq(N, d=dt, device=A.device)
    omega = 2 * torch.pi * frequencies
    fft_dA_dt = 1j * omega * fft_A
    return torch.fft.ifft(fft_dA_dt).real


def get_freqs(Nt, Lt, device=None):
    dt = Lt / Nt
    if device is None:
        device = torch.device('cpu')
    return torch.fft.fftshift(torch.fft.fftfreq(Nt, dt, device=device))


def fft(Ain):
    return torch.fft.fftshift(torch.fft.fft(Ain))


def ifft(Ain):
    return torch.fft.ifft(torch.fft.ifftshift(Ain))


def get_energy(A, dt):
    return torch.trapz(torch.abs(A)**2, dx=torch.tensor(dt, device=A.device))
