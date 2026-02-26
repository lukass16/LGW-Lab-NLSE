"""
Ring Resonator Training: Arbitrary Strong Pulses
==================================================
Standalone script for inverse design of per-loop strong pulses in a ring
resonator via XPM. Each resonator step has its own independent strong pulse
parameterized in the HG basis (N_resonator x N_modes complex parameters).

Usage:
    python ring_resonator_arbitrary.py                          # defaults
    python ring_resonator_arbitrary.py --N_train 50 --lr 5e-4   # custom
    python ring_resonator_arbitrary.py --output_dir results/     # save plots
"""

import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F

from nlse import (
    dispersion_operator,
    nonlinear_operator_xpm,
    get_hg_basis,
    time_to_hg,
    hg_to_time,
    get_energy,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def gaussian(t, tau, amplitude):
    """Generate a Gaussian pulse."""
    return amplitude * torch.exp(-(t ** 2) / (2 * tau ** 2))


def strong_soliton(t, beta2_k, gamma_k, tau):
    """Create the strong soliton pulse A_k."""
    a_2_squared = -beta2_k / (gamma_k * tau ** 2)
    a_2 = np.sqrt(a_2_squared)
    return a_2 * torch.cosh(t / tau) ** (-1)


def propagate_L(A0_j, A0_k, dz, Nz, beta2_j, beta2_k, gamma_j, gamma_k, Lt,
                strangsplitting=True):
    """Propagate weak (j) and strong (k) pulses through distance L."""
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


def beam_splitter(a, d, r, t_coeff):
    """Beam splitter: a = waveguide input, d = resonator pulse."""
    b = t_coeff * a + r * d   # into resonator
    c = r * a + t_coeff * d   # out of resonator
    return b, c


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Ring Resonator Arbitrary Strong-Pulse Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Simulation parameters
    sim = p.add_argument_group("Simulation parameters")
    sim.add_argument("--Lz", type=float, default=1e-3, help="Propagation distance (km)")
    sim.add_argument("--Nz", type=int, default=100, help="Number of z steps")
    sim.add_argument("--Lt", type=float, default=3.0, help="Temporal window size")
    sim.add_argument("--Nt", type=int, default=2048, help="Number of temporal points")
    sim.add_argument("--N_modes", type=int, default=100, help="Number of HG modes (strong pulse basis)")
    sim.add_argument("--B", type=int, default=16, help="Truncated HG basis dimension (weak pulse / loss)")

    # Medium parameters
    med = p.add_argument_group("Medium parameters")
    med.add_argument("--beta2_j", type=float, default=-10.0, help="GVD for weak wave")
    med.add_argument("--beta2_k", type=float, default=-10.0, help="GVD for strong wave")
    med.add_argument("--gamma_j", type=float, default=1.0, help="Nonlinearity for weak wave")
    med.add_argument("--gamma_k", type=float, default=1.0, help="Nonlinearity for strong wave")

    # Pulse / resonator parameters
    pulse = p.add_argument_group("Pulse / resonator parameters")
    pulse.add_argument("--tau", type=float, default=0.1, help="Pulse temporal width")
    pulse.add_argument("--R", type=float, default=0.4, help="Ring resonator reflectivity")
    pulse.add_argument("--amplitude", type=float, default=0.32, help="Weak input amplitude")
    pulse.add_argument("--N_resonator", type=int, default=10, help="Number of resonator loops")

    # Training parameters
    train = p.add_argument_group("Training parameters")
    train.add_argument("--mode", type=int, default=6, help="HG mode index for weak pulse")
    train.add_argument("--N_train", type=int, default=10, help="Number of training iterations")
    train.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # Output
    out = p.add_argument_group("Output")
    out.add_argument("--output_dir", type=str, default=".", help="Directory for saved plots")
    out.add_argument("--show", action="store_true", help="Show plots interactively (off by default for cluster use)")
    out.add_argument("--dpi", type=int, default=200, help="DPI for saved figures")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not args.show:
        matplotlib.use("Agg")

    os.makedirs(args.output_dir, exist_ok=True)

    # Derived simulation quantities
    dz = args.Lz / args.Nz
    dt = args.Lt / args.Nt
    T_coeff = 1 - args.R
    r_coeff = np.sqrt(args.R)
    t_coeff = np.sqrt(T_coeff)

    # Temporal grid and HG basis
    t = torch.linspace(-args.Lt / 2, args.Lt / 2, args.Nt)
    hg_basis = get_hg_basis(args.N_modes, t, args.tau)
    hg_basis_B = hg_basis[:args.B]

    # Input pulses
    weak_input = gaussian(t, args.tau, args.amplitude)
    strong_input = strong_soliton(t, args.beta2_k, args.gamma_k, args.tau)

    # Weak pulse from chosen HG mode
    A_j_input = hg_basis[args.mode]

    # Target HG coefficient vector (in truncated basis)
    y_hg = torch.zeros(args.B)
    y_hg[args.mode] = 1

    # -------------------------------------------------------------------
    # Loss & forward
    # -------------------------------------------------------------------
    def loss_function(final_j):
        """Fidelity loss on HG coefficients (phase-aware, truncated basis)."""
        final_j_hg = time_to_hg(final_j, hg_basis_B, dt)
        dot_products = torch.sum(final_j_hg.conj() * y_hg)
        fidelity_avg = torch.abs(dot_products) ** 2
        return -fidelity_avg

    def forward(A_k_hg_stack_param):
        A_k_stack = torch.stack(
            [hg_to_time(A_k_hg_stack_param[i], hg_basis) for i in range(args.N_resonator)]
        )
        A_j = A_j_input * t_coeff
        for i in range(args.N_resonator):
            A_j_ev, _ = propagate_L(
                A_j, A_k_stack[i], dz, args.Nz,
                args.beta2_j, args.beta2_k, args.gamma_j, args.gamma_k, args.Lt,
            )
            A_j = A_j_ev[:, -1]
            A_j, _ = beam_splitter(weak_input, A_j, r_coeff, t_coeff)
        return A_j * t_coeff

    # -------------------------------------------------------------------
    # Trainable parameters & optimizer
    # -------------------------------------------------------------------
    strong_input_coeffs = time_to_hg(strong_input, hg_basis, dt)
    init_stack = torch.stack([strong_input_coeffs for _ in range(args.N_resonator)])
    A_k_hg_stack = torch.nn.Parameter(init_stack.clone().detach().requires_grad_(True))
    optimizer = torch.optim.Adam([A_k_hg_stack], lr=args.lr)

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    losses = []
    print(f"Starting training for {args.N_train} iterations ...")
    for i in tqdm(range(args.N_train), desc="Training"):
        optimizer.zero_grad()
        A_j_out = forward(A_k_hg_stack)
        loss = loss_function(A_j_out)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"Final loss: {losses[-1]:.6f}")

    # Reconstruct A_k_stack with optimized parameters for plotting
    with torch.no_grad():
        A_k_stack = torch.stack(
            [hg_to_time(A_k_hg_stack[i].detach(), hg_basis) for i in range(args.N_resonator)]
        )

    # -------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------

    # 1. Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, label="Training Loss", color="tab:blue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "loss_curve.png"), dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)

    # 2. Optimized HG coefficients per resonator step
    fig, ax = plt.subplots(figsize=(10, 4))
    init_coeffs = time_to_hg(strong_input, hg_basis, dt).numpy()
    mode_indices = np.arange(args.N_modes)
    ax.stem(mode_indices, np.abs(init_coeffs) ** 2, linefmt="r-", markerfmt="ro",
            basefmt=" ", label="Initial (soliton)")
    cmap = plt.cm.viridis
    for loop_idx in range(args.N_resonator):
        opt_coeffs = A_k_hg_stack[loop_idx].detach().cpu().numpy()
        color = cmap(loop_idx / args.N_resonator)
        ax.stem(mode_indices, np.abs(opt_coeffs) ** 2, linefmt="-", markerfmt="o",
                basefmt=" ", label=f"Loop {loop_idx}")
        ax.get_children()[-3].set_color(color)
        ax.get_children()[-4].set_color(color)
    ax.set_xlabel("HG Mode Index")
    ax.set_ylabel("Coefficient Intensity |c_n|²")
    ax.set_title("Strong Pulse HG Coefficients: Initial vs Optimized (per loop)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "hg_coefficients.png"), dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)

    # 3. Strong pulse evolution
    fig = plt.figure(figsize=(12, 7), dpi=args.dpi)
    cmap = plt.cm.plasma
    n_pulses = len(A_k_stack)
    colors = [cmap(i / n_pulses) for i in range(n_pulses)]
    t_np = t.cpu().numpy()
    for i, pulse in enumerate(A_k_stack):
        pulse_np = pulse.detach().cpu().numpy()
        intensity = np.abs(pulse_np) ** 2
        plt.plot(t_np, intensity, color=colors[i], alpha=0.7, linewidth=1.5)
        max_idx = np.argmax(intensity)
        plt.text(
            t_np[max_idx], intensity[max_idx], f"Loop {i}",
            fontsize=8, ha="left", va="bottom", color=colors[i],
        )
    plt.xlabel("Time")
    plt.ylabel("Intensity |A|²")
    plt.title("Strong Pulse Evolution in Ring Resonator")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "strong_pulse_evolution.png"), dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)

    print(f"Plots saved to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
