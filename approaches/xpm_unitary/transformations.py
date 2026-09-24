"""Unitary target matrices for XPM inverse design."""

import numpy as np
import torch

VALID_TRANSFORMATIONS = ["identity", "permutation", "rotation", "arbitrary", "beamsplitter"]


def generate_transformation(transformation_name, num_modes, device, seed=27):
    """Generate a unitary transformation matrix.

    Args:
        transformation_name: One of ``identity``, ``permutation``, ``rotation``,
            ``arbitrary``, ``beamsplitter``.
        num_modes: Size of the transformation matrix.
        device: Torch device.
        seed: Random seed for reproducibility (default: 27).

    Returns:
        Unitary matrix of shape ``(num_modes, num_modes)``, complex dtype.
    """
    torch.manual_seed(seed)

    if transformation_name == "identity":
        U = torch.eye(num_modes, dtype=torch.cfloat, device=device)

    elif transformation_name == "permutation":
        idx = torch.randperm(num_modes, device=device)
        U = torch.eye(num_modes, dtype=torch.cfloat, device=device)[:, idx]

    elif transformation_name == "rotation":
        theta = torch.rand(1, device=device) * 2 * torch.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        rotation_2x2 = torch.zeros((2, 2), dtype=torch.float32, device=device)
        rotation_2x2[0, 0] = cos_theta
        rotation_2x2[0, 1] = -sin_theta
        rotation_2x2[1, 0] = sin_theta
        rotation_2x2[1, 1] = cos_theta

        U = torch.zeros((num_modes, num_modes), dtype=torch.cfloat, device=device)
        num_blocks = num_modes // 2
        for block_idx in range(num_blocks):
            start = block_idx * 2
            U[start:start + 2, start:start + 2] = rotation_2x2.to(torch.cfloat)
        if num_modes % 2 == 1:
            U[-1, -1] = 1.0

    elif transformation_name == "beamsplitter":
        bs = torch.tensor([[1.0, 1j], [1j, 1.0]], dtype=torch.cfloat, device=device) / np.sqrt(2)
        U = torch.zeros((num_modes, num_modes), dtype=torch.cfloat, device=device)
        num_blocks = num_modes // 2
        for block_idx in range(num_blocks):
            start = block_idx * 2
            U[start:start + 2, start:start + 2] = bs
        if num_modes % 2 == 1:
            U[-1, -1] = 1.0

    elif transformation_name == "arbitrary":
        H = torch.zeros((num_modes, num_modes), dtype=torch.cfloat, device=device)
        diag_real = torch.randn(num_modes, device=device).to(torch.cfloat)
        H[torch.arange(num_modes), torch.arange(num_modes)] = diag_real
        for i in range(num_modes):
            for j in range(i + 1, num_modes):
                re = torch.randn(1, device=device)
                im = torch.randn(1, device=device)
                val = re + 1j * im
                H[i, j] = val
                H[j, i] = val.conj()
        U = torch.matrix_exp(1j * H)

    else:
        raise ValueError(
            f"Unknown transformation: {transformation_name}. "
            f"Valid options: {VALID_TRANSFORMATIONS}"
        )

    return U


def is_unitary(matrix, atol=1e-5):
    """Check if a matrix is unitary within tolerance."""
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    if torch.is_complex(matrix):
        prod = matrix @ matrix.conj().T
    else:
        prod = matrix @ matrix.T
    return torch.allclose(prod, identity, atol=atol)
