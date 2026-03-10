"""
Ring Resonator Training: Scaled Amplitude (Batched)
====================================================
Batch-aware inverse design of strong pulse amplitudes in a ring resonator via
XPM. Optimizes a shared strong pulse shape (HG coefficients) and per-loop
scaling factors (alphas) to preserve multiple HG modes simultaneously.

The weak-pulse inputs are the first `num_modes` HG basis functions (modes
0, 1, ..., num_modes-1).  The loss encourages each input mode to emerge as
itself at the output (identity transformation across all modes).

Usage:
    python ring_resonator_scaling_batch.py --config configs/config_ring_scaling_fidelity.yaml
    python ring_resonator_scaling_batch.py --num_modes 4 --N_train 50  # CLI only
    python ring_resonator_scaling_batch.py --config configs/config_ring_scaling_trace.yaml --lr 1e-4  # config + CLI override
"""

import argparse
import json
import os

import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")

from nlse import (
    split_step_fourier_xpm_batch,
    get_hg_basis,
    time_to_hg,
    hg_to_time,
    plot_mode_hg_coeffs,
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


def beam_splitter(a, d, r, t_coeff):
    """Beam splitter: a = waveguide input, d = resonator pulse.

    Works for both unbatched (Nt,) and batched (B, Nt) tensors thanks to
    broadcasting (a may be (Nt,) while d is (B, Nt)).
    """
    b = t_coeff * a + r * d
    c = r * a + t_coeff * d
    return b, c


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _load_config(config_path):
    """Load a YAML config. Returns (flat_dict, raw_config)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    flat = {}
    for section in cfg.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat, cfg


def parse_args():
    # Pre-parse to grab --config before building the full parser
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None,
                     help="Path to YAML config file (values override defaults; "
                          "explicit CLI flags override config)")
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(
        parents=[pre],
        description="Ring Resonator Scaled-Amplitude Training (Batched)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sim = p.add_argument_group("Simulation parameters")
    sim.add_argument("--Lz", type=float, default=1e-3, help="Propagation distance (km)")
    sim.add_argument("--Nz", type=int, default=100, help="Number of z steps")
    sim.add_argument("--Lt", type=float, default=3.0, help="Temporal window size")
    sim.add_argument("--Nt", type=int, default=2048, help="Number of temporal points")
    sim.add_argument("--N_modes", type=int, default=100,
                     help="Number of HG modes for the strong-pulse basis")
    sim.add_argument("--B", type=int, default=16,
                     help="Truncated HG basis dimension (weak pulse / loss)")

    med = p.add_argument_group("Medium parameters")
    med.add_argument("--beta2_j", type=float, default=-10.0, help="GVD for weak wave")
    med.add_argument("--beta2_k", type=float, default=-10.0, help="GVD for strong wave")
    med.add_argument("--gamma_j", type=float, default=1.0, help="Nonlinearity for weak wave")
    med.add_argument("--gamma_k", type=float, default=1.0, help="Nonlinearity for strong wave")

    pulse = p.add_argument_group("Pulse / resonator parameters")
    pulse.add_argument("--tau", type=float, default=0.1, help="Pulse temporal width")
    pulse.add_argument("--R", type=float, default=0.4, help="Ring resonator reflectivity")
    pulse.add_argument("--amplitude", type=float, default=0.32, help="Weak input amplitude")
    pulse.add_argument("--N_resonator", type=int, default=10, help="Number of resonator loops")

    train = p.add_argument_group("Training parameters")
    train.add_argument("--num_modes", type=int, default=8,
                       help="Number of HG modes to train on (batch size)")
    train.add_argument("--N_train", type=int, default=10, help="Number of training iterations")
    train.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train.add_argument("--loss_fn", type=str, default="fidelity",
                       choices=["fidelity", "trace"],
                       help="Loss function: fidelity (per-mode avg) or trace")

    out = p.add_argument_group("Output")
    out.add_argument("--output_dir", type=str, default=".", help="Directory for saved plots")
    out.add_argument("--show", action="store_true",
                     help="Show plots interactively (off by default for cluster use)")
    out.add_argument("--dpi", type=int, default=200, help="DPI for saved figures")
    out.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    # Override argparse defaults with config values (CLI flags still win)
    raw_config = {}
    if pre_args.config:
        cfg_defaults, raw_config = _load_config(pre_args.config)
        p.set_defaults(**cfg_defaults)
        print(f"Loaded config from {pre_args.config}")

    args = p.parse_args()
    args._raw_config = raw_config
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not args.show:
        matplotlib.use("Agg")

    os.makedirs(args.output_dir, exist_ok=True)

    num_modes = args.num_modes
    assert num_modes <= args.B, (
        f"num_modes ({num_modes}) must be <= B ({args.B}), "
        "since the truncated basis is used for the loss."
    )

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

    # Input pulses -----------------------------------------------------------
    weak_input = gaussian(t, args.tau, args.amplitude)          # (Nt,) beam-splitter input
    strong_input = strong_soliton(t, args.beta2_k, args.gamma_k, args.tau)  # (Nt,)

    # Batch of weak-pulse inputs: modes 0 .. num_modes-1, shape (num_modes, Nt)
    A_j_inputs = hg_basis[:num_modes]

    # Target HG coefficient matrix: identity in the truncated basis
    # y_hg[i, j] = 1 if i == j, else 0  → mode i should emerge as mode i
    y_hg = torch.zeros(num_modes, args.B)
    for i in range(num_modes):
        y_hg[i, i] = 1.0

    # -------------------------------------------------------------------
    # Loss functions & forward (batched)
    # -------------------------------------------------------------------
    def _final_hg(final_j):
        """Helper: convert batch of time-domain outputs to HG coefficients."""
        return torch.stack(
            [time_to_hg(final_j[i], hg_basis_B, dt) for i in range(num_modes)]
        )  # (num_modes, B)

    def fidelity_loss(final_j):
        """Per-mode fidelity loss averaged over all modes (phase-aware)."""
        final_j_hg = _final_hg(final_j)
        dot_products = torch.sum(final_j_hg.conj() * y_hg, dim=1)  # (num_modes,)
        fidelity_avg = torch.mean(torch.abs(dot_products) ** 2)
        return -fidelity_avg

    def trace_loss(final_j):
        """Trace-based loss: sums complex overlaps across all modes before
        taking the magnitude, so relative phases between mode pairs matter.
        Analogous to |Tr(U_actual^dag U_target)|^2 / num_modes^2.
        """
        final_j_hg = _final_hg(final_j)
        dot_products = torch.sum(final_j_hg.conj() * y_hg, dim=1)  # (num_modes,)
        trace_fidelity = torch.abs(torch.sum(dot_products) / num_modes) ** 2
        return -trace_fidelity

    LOSS_FNS = {"fidelity": fidelity_loss, "trace": trace_loss}
    loss_function = LOSS_FNS[args.loss_fn]
    print(f"Using loss function: {args.loss_fn}")

    def forward(A_k_coeffs_param, alphas_param):
        """Ring-resonator forward pass for a batch of weak-pulse inputs.

        Returns:
            final weak pulses, shape (num_modes, Nt)
        """
        A_k_time = hg_to_time(A_k_coeffs_param, hg_basis)  # (Nt,) shared strong pulse

        # A_j starts as the batch of HG modes scaled by t_coeff
        A_j = A_j_inputs * t_coeff  # (num_modes, Nt)

        for i in range(args.N_resonator):
            A_k_scaled = A_k_time * alphas_param[i]  # (Nt,) per-loop scaling

            # Batched propagation: A_j (num_modes, Nt), A_k_scaled (Nt,)
            A_j_ev, _ = split_step_fourier_xpm_batch(
                A_j, A_k_scaled, dz, args.Nz,
                args.beta2_j, args.beta2_k,
                args.gamma_j, args.gamma_k, args.Lt,
            )
            A_j = A_j_ev[:, :, -1]  # (num_modes, Nt) — take final z-step

            # Beam splitter: weak_input (Nt,) broadcasts with A_j (num_modes, Nt)
            A_j, _ = beam_splitter(weak_input, A_j, r_coeff, t_coeff)

        return A_j * t_coeff  # (num_modes, Nt)

    # -------------------------------------------------------------------
    # Trainable parameters & optimizer
    # -------------------------------------------------------------------
    strong_input_coeffs = time_to_hg(strong_input, hg_basis, dt)
    A_k_coeffs = torch.nn.Parameter(strong_input_coeffs.clone().detach().requires_grad_(True))
    alphas = torch.nn.Parameter(torch.ones(args.N_resonator).clone().detach().requires_grad_(True))
    optimizer = torch.optim.Adam([A_k_coeffs, alphas], lr=args.lr)

    # -------------------------------------------------------------------
    # Wandb initialisation
    # -------------------------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb_cfg = args._raw_config.get("wandb", {})
        wandb.init(
            project=wandb_cfg.get("project", "ring-resonator"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("name"),
            tags=wandb_cfg.get("tags", []),
            notes=wandb_cfg.get("notes", ""),
            config=vars(args),
            dir=args.output_dir,
        )
    elif use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Continuing without wandb.")
        use_wandb = False

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    losses = []
    best_loss = float("inf")
    print(f"Training on modes 0..{num_modes - 1} (batch size = {num_modes})")
    print(f"Starting training for {args.N_train} iterations ...")
    for i in tqdm(range(args.N_train), desc="Training"):
        optimizer.zero_grad()
        A_j_out = forward(A_k_coeffs, alphas)          # (num_modes, Nt)
        loss = loss_function(A_j_out)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        if loss_val < best_loss:
            best_loss = loss_val

        # Periodic eval + wandb logging
        with torch.no_grad():
            out_hg = _final_hg(A_j_out)
            dots = torch.sum(out_hg.conj() * y_hg, dim=1)
            avg_fidelity = torch.mean(torch.abs(dots) ** 2).item()
            trace_fid = (torch.abs(torch.sum(dots) / num_modes) ** 2).item()

        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "iteration": i,
                "loss": loss_val,
                "best_loss": best_loss,
                "avg_fidelity": avg_fidelity,
                "trace_fidelity": trace_fid,
            })

        if (i + 1) % max(1, args.N_train // 10) == 0:
            print(f"  Iter {i+1}/{args.N_train}: loss={loss_val:.6f}, "
                  f"avg_fid={avg_fidelity:.6f}, trace_fid={trace_fid:.6f}")

    print(f"Final loss: {losses[-1]:.6f}")

    # Reconstruct A_k_stack with optimized parameters for plotting
    with torch.no_grad():
        A_k_optimized = hg_to_time(A_k_coeffs.detach(), hg_basis)
        A_k_stack = torch.stack(
            [A_k_optimized * alphas[i].detach() for i in range(args.N_resonator)]
        )

    # -------------------------------------------------------------------
    # Per-mode fidelity evaluation
    # -------------------------------------------------------------------
    with torch.no_grad():
        A_j_final = forward(A_k_coeffs, alphas)  # (num_modes, Nt)
        final_hg = torch.stack(
            [time_to_hg(A_j_final[i], hg_basis_B, dt) for i in range(num_modes)]
        )
        per_mode_fid = torch.abs(torch.sum(final_hg.conj() * y_hg, dim=1)) ** 2

    print("\nPer-mode fidelity:")
    for m in range(num_modes):
        print(f"  Mode {m}: {per_mode_fid[m].item():.6f}")
    print(f"  Average : {per_mode_fid.mean().item():.6f}")

    # Save losses to JSON
    losses_path = os.path.join(args.output_dir, "losses.json")
    with open(losses_path, "w") as f:
        json.dump({"losses": losses, "final_loss": losses[-1],
                   "best_loss": best_loss,
                   "per_mode_fidelity": per_mode_fid.cpu().tolist()}, f, indent=2)

    # Save checkpoint for later analysis
    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pt")
    torch.save({
        "scenario": "scaling",
        "A_k_coeffs": A_k_coeffs.detach().cpu(),
        "alphas": alphas.detach().cpu(),
        "args": {k: v for k, v in vars(args).items() if k != "_raw_config"},
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

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

    # 2. Optimized alphas
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(alphas.detach().cpu().numpy(), marker="o", linestyle="-", color="tab:green")
    ax.set_xlabel("Resonator Step")
    ax.set_ylabel("Alpha (Strong Pulse Scale)")
    ax.set_title(r"Optimized $\alpha$ (Strong Pulse Amplitudes)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "optimized_alphas.png"), dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)

    # 3. Optimized HG coefficients of the strong pulse
    fig, ax = plt.subplots(figsize=(10, 4))
    init_coeffs = time_to_hg(strong_input, hg_basis, dt).numpy()
    opt_coeffs = A_k_coeffs.detach().cpu().numpy()
    mode_indices = np.arange(args.N_modes)
    ax.stem(mode_indices, np.abs(init_coeffs) ** 2, linefmt="r-", markerfmt="ro",
            basefmt=" ", label="Initial (soliton)")
    ax.stem(mode_indices, np.abs(opt_coeffs) ** 2, linefmt="b-", markerfmt="bo",
            basefmt=" ", label="Optimized")
    ax.set_xlabel("HG Mode Index")
    ax.set_ylabel("Coefficient Intensity |c_n|²")
    ax.set_title("Strong Pulse HG Coefficients: Initial vs Optimized")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "hg_coefficients.png"), dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)

    # 4. Strong pulse evolution across resonator loops
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

    # 5. HG coefficient comparison (magnitude + phase) via plot_mode_hg_coeffs
    y_time = A_j_inputs.detach()  # target: input modes should emerge as-is
    A_j_ev_dummy = A_j_final.detach().unsqueeze(-1)  # (num_modes, Nt, 1)
    plot_mode_hg_coeffs(
        y_time, A_j_ev_dummy, hg_basis, dt,
        num_modes=num_modes,
        transformation_name="Ring Resonator (identity)",
    )
    plt.savefig(os.path.join(args.output_dir, "mode_hg_coeffs.png"),
                dpi=args.dpi, bbox_inches="tight")
    plt.close("all")

    # Log final metrics and finish wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "final_loss": losses[-1],
            "best_loss": best_loss,
            "final_avg_fidelity": per_mode_fid.mean().item(),
        })
        wandb.finish()

    print(f"Plots saved to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
