"""Hermite–Gauss basis and time ↔ coefficient conversions."""

import math

import torch


def hermite_gauss_stable(n, t, t0=1.0, offset=0.0):
    """Stable computation of ψ_n(t; t0) using orthonormal recursion."""
    x = (t + offset) / t0
    psi_prev = torch.zeros_like(x, device=t.device)
    psi_curr = (1 / (math.pi**0.25 * math.sqrt(t0))) * torch.exp(-0.5 * x**2)

    if n == 0:
        return psi_curr
    for k in range(1, n + 1):
        psi_next = math.sqrt(2 / k) * x * psi_curr - math.sqrt((k - 1) / k) * psi_prev
        psi_prev, psi_curr = psi_curr, psi_next
    return psi_curr


def get_hg_basis(N_modes, t, t0=1.0, offset=0.0):
    """Precompute HG basis functions as a (N_modes, len(t)) tensor on ``t``'s device."""
    hg_basis = torch.zeros(N_modes, len(t), dtype=torch.float32, device=t.device)
    for n in range(N_modes):
        hg_basis[n] = hermite_gauss_stable(n, t, t0, offset)
    print(f"Precomputed {N_modes} HG basis functions on grid of {len(t)} points on device {t.device}")
    return hg_basis


def time_to_hg(A, hg_basis, dt):
    integrand = hg_basis * A[None, :]
    return torch.trapz(integrand, dx=dt, dim=1)


def hg_to_time(coefficients, hg_basis):
    return torch.sum(coefficients[:, None] * hg_basis, dim=0)


def time_to_trunc_hg(A, hg_basis, dt, num_modes):
    """Project onto the first ``num_modes`` HG functions.

    Accepts an unbatched pulse ``(Nt,)`` or a batch ``(B, Nt)``.
    """
    if A.ndim == 1:
        return time_to_hg(A, hg_basis[:num_modes, :], dt)
    B = A.shape[0]
    return torch.stack([time_to_hg(A[i], hg_basis[:num_modes, :], dt) for i in range(B)])
