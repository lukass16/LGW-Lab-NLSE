"""
Analyze XPM Identity Operator Training Results

This script loads the trained model from a run directory and creates a plot
showing how well the strong wave produces each of the 16 target modes.

Usage:
    python analyze_xpm_results.py --run-dir runs/run_YYYYMMDD_HHMMSS [--show]
    
    --show: Display plot interactively instead of saving to plots folder
"""

import argparse
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch

from nlse import *


def load_trained_model(run_dir, device):
    """Load config and trained theta from run directory."""
    run_path = Path(run_dir)
    
    # Load config
    config_path = run_path / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load theta
    theta_path = run_path / "parameters" / "theta.npy"
    theta = torch.tensor(np.load(theta_path), dtype=torch.float32, device=device)
    
    return config, theta


def reconstruct_setup(config, device):
    """Reconstruct the training setup from config."""
    sim = config['simulation']
    med = config['medium']
    pulse = config['pulse']
    train = config['training']
    
    # Extract parameters
    Lz = float(sim['Lz'])
    Nz = int(sim['Nz'])
    Lt = float(sim['Lt'])
    Nt = int(sim['Nt'])
    N_modes = int(sim['N_modes'])
    beta2_j = float(med['beta2_j'])
    beta2_k = float(med['beta2_k'])
    gamma_j = float(med['gamma_j'])
    gamma_k = float(med['gamma_k'])
    tau = float(pulse['tau'])
    batch_size = int(train['batch_size'])
    amplitude_downscale = float(train['amplitude_downscale'])
    
    # Calculate derived parameters
    dz = Lz / Nz
    dt = Lt / Nt
    
    # Create time grid
    t = torch.linspace(-Lt/2, Lt/2, Nt, dtype=torch.float32, device=device)
    
    # Define HG basis
    hg_basis = get_hg_basis(N_modes, t, tau)
    
    # Get batch of pulses (same as training)
    B = batch_size
    x = get_hg_basis(B, t, tau) * amplitude_downscale  # inputs
    y = x.clone().detach()  # targets
    
    return {
        't': t,
        'hg_basis': hg_basis,
        'x': x,
        'y': y,
        'dz': dz,
        'Nz': Nz,
        'beta2_j': beta2_j,
        'beta2_k': beta2_k,
        'gamma_j': gamma_j,
        'gamma_k': gamma_k,
        'Lt': Lt
    }


def forward_pass(theta, setup):
    """Run forward pass with trained theta."""
    Ain_k = hg_to_time(theta, setup['hg_basis'])
    A_j_evolution, A_k_evolution = split_step_fourier_xpm_batch(
        setup['x'], Ain_k, setup['dz'], setup['Nz'],
        setup['beta2_j'], setup['beta2_k'],
        setup['gamma_j'], setup['gamma_k'],
        setup['Lt']
    )
    return A_j_evolution, A_k_evolution


def plot_mode_comparison(setup, A_j_evolution, run_dir, show_plot=False):
    """Create a 4x4 grid plot showing all 16 modes: target vs final output."""
    y = setup['y']
    t = setup['t']
    B = y.shape[0]
    
    # Calculate time window for plotting (center 30%)
    plot_percent = 0.3
    total_points = len(t)
    center_points = int(total_points * plot_percent)
    start_idx = (total_points - center_points) // 2
    end_idx = start_idx + center_points
    t_plot = t[start_idx:end_idx].cpu().numpy()
    
    # Get final outputs
    final_outputs = A_j_evolution[:, :, -1].detach()
    
    # Create 4x4 subplot grid
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for mode_idx in range(B):
        ax = axes[mode_idx]
        
        # Extract data for this mode
        target_intensity = np.abs(y[mode_idx].detach().cpu().numpy())[start_idx:end_idx]**2
        output_intensity = np.abs(final_outputs[mode_idx].detach().cpu().numpy())[start_idx:end_idx]**2
        
        # Plot target and output
        ax.plot(t_plot, target_intensity, 'r--', linewidth=2, label='Target', alpha=0.7)
        ax.plot(t_plot, output_intensity, 'b-', linewidth=2, label='Output', alpha=0.8)
        
        # Calculate and display MSE
        mse = np.mean((target_intensity - output_intensity)**2)
        ax.set_title(f'Mode {mode_idx}\nMSE: {mse:.2e}', fontsize=10)
        ax.set_xlabel('Time', fontsize=9)
        ax.set_ylabel('Intensity |A|²', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Target vs Final Output for All 16 Modes', fontsize=16, y=0.995)
    plt.tight_layout()
    
    if show_plot:
        # Display plot interactively
        plt.show()
    else:
        # Save plot to plots folder in run directory
        plots_dir = Path(run_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)
        output_path = plots_dir / "mode_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze XPM Identity Operator Results')
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Path to run directory containing trained model')
    parser.add_argument('--show', action='store_true',
                        help='Display plot interactively instead of saving to file')
    
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load trained model
    print(f"Loading model from {args.run_dir}...")
    config, theta = load_trained_model(args.run_dir, device)
    print(f"Loaded theta with shape: {theta.shape}")
    
    # Reconstruct setup
    print("Reconstructing simulation setup...")
    setup = reconstruct_setup(config, device)
    print(f"Batch size: {setup['x'].shape[0]}")
    
    # Run forward pass
    print("Running forward pass...")
    A_j_evolution, A_k_evolution = forward_pass(theta, setup)
    
    # Create comparison plot
    print("Generating mode comparison plot...")
    plot_mode_comparison(setup, A_j_evolution, args.run_dir, show_plot=args.show)
    
    # Print summary statistics
    final_outputs = A_j_evolution[:, :, -1].detach()
    targets = setup['y']
    mse_per_mode = torch.mean((torch.abs(final_outputs)**2 - torch.abs(targets)**2)**2, dim=1)
    print(f"\nSummary Statistics:")
    print(f"  Mean MSE per mode: {mse_per_mode.mean().item():.2e}")
    print(f"  Std MSE per mode: {mse_per_mode.std().item():.2e}")
    print(f"  Min MSE: {mse_per_mode.min().item():.2e}")
    print(f"  Max MSE: {mse_per_mode.max().item():.2e}")
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

