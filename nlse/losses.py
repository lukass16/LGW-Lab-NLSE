"""Losses and metrics that compare a realized transformation V with a target U.

V and U are (B, B) Hermite–Gauss coefficient matrices: row i is the output
for input mode i, and <V_i, U_i> = sum_j conj(V_ij) U_ij. Losses are ordered
from weakest to strictest; the fidelity-type ones are negated so that every
loss is minimized.

    name           definition                      sees                          blind to
    hg             1/B² Σ_ij (|V_ij|² - |U_ij|²)²  power in every element        all phases
    normalized_hg  hg on row-normalized |V|, |U|   power distribution per row    phases, row norms
    fid_energy     -1/B Σ_i (Σ_j |V_ij||U_ij|)²    magnitude overlap per row     all phases
    fid_phase      -1/B Σ_i |<V_i, U_i>|²          relative phases within a row  one phase per row
    trace          -|Tr(V† U) / B|²                relative phases across rows   one global phase
    hg_phase       1/B² ||V - U||_F²               every element incl. phase     nothing
"""

import torch
import torch.nn.functional as F


def intensity_loss(output, target):
    """Time-domain MSE on intensities ``|A|²`` (the ``basic`` loss)."""
    return F.mse_loss(torch.abs(output)**2, torch.abs(target)**2)


def hg_loss(V, U):
    return F.mse_loss(torch.abs(V)**2, torch.abs(U)**2)


def normalized_hg_loss(V, U, scale=1.0, eps=1e-10):
    V_n = V / (torch.norm(V, dim=1, keepdim=True) + eps) * scale
    U_n = U / (torch.norm(U, dim=1, keepdim=True) + eps) * scale
    return F.mse_loss(torch.abs(V_n), torch.abs(U_n))


def fid_energy_loss(V, U):
    overlaps = torch.sum(torch.abs(V) * torch.abs(U), dim=1)
    return -torch.mean(overlaps**2)


def fid_phase_loss(V, U):
    overlaps = torch.sum(V.conj() * U, dim=1)
    return -torch.mean(torch.abs(overlaps)**2)


def trace_loss(V, U):
    B = V.shape[0]
    overlaps = torch.sum(V.conj() * U, dim=1)
    return -torch.abs(torch.sum(overlaps) / B)**2


def hg_phase_loss(V, U):
    return torch.mean(torch.abs(V - U)**2)


HG_LOSSES = {
    'hg': hg_loss,
    'normalized_hg': normalized_hg_loss,
    'fid_energy': fid_energy_loss,
    'fid_phase': fid_phase_loss,
    'trace': trace_loss,
    'hg_phase': hg_phase_loss,
}
LOSS_NAMES = ['basic', *HG_LOSSES]


def eval_frobenius_norm(V, U):
    """Mean ``|V - U|²``. 0 is perfect."""
    return torch.mean(torch.abs(V - U)**2).item()


def eval_trace_fidelity(V, U):
    """``trace`` fidelity normalized by the target's own, in percent."""
    B = V.shape[0]
    actual = torch.abs(torch.sum(torch.sum(V.conj() * U, dim=1)) / B)**2
    ideal = torch.abs(torch.sum(torch.sum(U.conj() * U, dim=1)) / B)**2
    return (actual / (ideal + 1e-10) * 100).item()


def eval_hg_fidelity(V, U):
    """Mean per-row overlap ``|<V_i, U_i>|`` normalized by the target's own, in percent."""
    actual = torch.mean(torch.abs(torch.sum(V.conj() * U, dim=1))).item()
    ideal = torch.mean(torch.abs(torch.sum(U.conj() * U, dim=1))).item()
    return actual / (ideal + 1e-10) * 100
