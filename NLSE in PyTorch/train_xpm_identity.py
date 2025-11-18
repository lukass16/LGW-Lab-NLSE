"""
XPM Identity Operator Training Script

This script trains an XPM identity operator using the Hermite-Gauss basis.
It can be run as a batch job and includes wandb logging, parameter saving,
and plot generation.

Usage:
    python train_xpm_identity.py [--config config.yaml] [--run-dir runs/run_001]
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import yaml
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch jobs
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Try to import wandb, make it optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")

# Import NLSE module
from nlse import *



"""--------------------------------- Helper Functions ---------------------------------"""

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config):
    """Setup and return the device to use."""
    use_cuda = config['device']['use_cuda'] and torch.cuda.is_available()
    if use_cuda:
        device_id = config['device']['device_id']
        device = torch.device(f'cuda:{device_id}')
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(device_id)}")
        print(f"Memory Available: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
    return device


def create_run_directory(base_dir='runs', name=None):
    """Create a unique run directory with given name or timestamp."""
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    if name:
        run_dir = base_path / name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_path / f"run_{timestamp}"
    
    run_dir.mkdir(exist_ok=True)
    
    return run_dir


def save_config_to_run_dir(config, run_dir):
    """Save the configuration to the run directory."""
    config_path = run_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return config_path


"""--------------------------------- Training Setup Functions ---------------------------------"""

def strong_soliton(t, beta2_k, gamma_k, tau):
    """Create the strong pulse (A_k) based on the soliton parameters."""
    # calculate secondary parameters
    a_2_squared = -beta2_k / (gamma_k * tau**2)
    a_2 = np.sqrt(a_2_squared)
    
    # Define strong pulse waveform
    Ain_k = a_2 * torch.cosh(t / tau)**(-1)
    return Ain_k


def setup_training(config, device):
    """Setup training data, model, and optimizer."""
    # Extract parameters from dictionaries
    sim = config['simulation']
    med = config['medium']
    pulse = config['pulse']
    train = config['training']
    
    # Extract simulation parameters (convert to appropriate types)
    Lz = float(sim['Lz'])
    Nz = int(sim['Nz'])
    Lt = float(sim['Lt'])
    Nt = int(sim['Nt'])
    N_modes = int(sim['N_modes'])
    
    # Extract medium parameters (convert to appropriate types)
    beta2_j = float(med['beta2_j'])
    beta2_k = float(med['beta2_k'])
    gamma_j = float(med['gamma_j'])
    gamma_k = float(med['gamma_k'])
    
    # Extract pulse parameters (convert to appropriate types)
    tau = float(pulse['tau'])
    
    # Extract training parameters (convert to appropriate types)
    batch_size = int(train['batch_size'])
    amplitude_downscale = float(train['amplitude_downscale'])
    mask_percent = float(train['mask_percent'])
    m = int(train['m'])
    w_mse = float(train['w_mse'])
    w_pen = float(train['w_pen'])
    lr = float(train['lr'])
    
    # Calculate derived parameters
    dz = Lz / Nz
    dt = Lt / Nt
    
    # Create time grid
    t = torch.linspace(-Lt/2, Lt/2, Nt, dtype=torch.float32, device=device)
    
    # Define HG basis
    hg_basis = get_hg_basis(N_modes, t, tau)
    
    # Get batch of pulses
    B = batch_size
    x = get_hg_basis(B, t, tau) * amplitude_downscale  # inputs
    y = x.clone().detach()  # labels
    
    # Define input strong pulse
    A_strong = strong_soliton(t, beta2_k, gamma_k, tau)
    A_strong_hg = time_to_hg(A_strong, hg_basis, dt)
    theta = torch.nn.Parameter(A_strong_hg.clone().detach().requires_grad_(True))
    
    # Define penalty mask
    penalty = torch.zeros_like(t, device=device)
    penalty[:int(Nt/2 - Nt*mask_percent/2)] = 1
    penalty[int(Nt/2 + Nt*mask_percent/2):] = 1
    mask = 1 - penalty
    iters = np.linspace(0, Nz-1, m).astype(int)
    
    # Define loss function
    def loss_function(A_j_evolution, A_k_evolution):
        # MSE loss
        mse_loss = F.mse_loss(torch.abs(A_j_evolution[:, :, -1])**2, torch.abs(y)**2)
        
        pen_loss = 0.0
        zeros = torch.zeros(B, Nt, dtype=torch.float32, device=device)
        # Penalization loss - to enforce none of the waves to stray outside the simulation bounds
        for i in iters:
            pen_loss = pen_loss + F.mse_loss(torch.abs(A_k_evolution[:, :, i]*penalty)**2, zeros)
        
        return w_mse * mse_loss, w_pen * pen_loss
    
    #! TEST: Alternative loss function - HG coefficients
    # define y_hg - the target HG coefficients (simply identity tensor of size B x B)
    y_hg = torch.eye(B, dtype=torch.float32, device=device)
    # Create a smaller HG basis with only B modes for the loss computation
    hg_basis_B = get_hg_basis(B, t, tau)
    
    def hg_loss_function(A_j_evolution, A_k_evolution):
        final_j = A_j_evolution[:, :, -1] # shape: (B, Nt)
        # Compute HG coefficients for each signal in the batch
        final_j_hg = torch.stack([time_to_hg(final_j[i], hg_basis_B, dt) for i in range(B)]) # shape: (B, B)
        # calculate the MSE loss between the final HG coefficients and the target HG coefficients
        mse_loss = F.mse_loss(final_j_hg, y_hg)
        
        
        pen_loss = 0.0
        zeros = torch.zeros(B, Nt, dtype=torch.float32, device=device)
        # Penalization loss - to enforce none of the waves to stray outside the simulation bounds
        for i in iters:
            pen_loss = pen_loss + F.mse_loss(torch.abs(A_k_evolution[:, :, i]*penalty)**2, zeros)
            
        return w_mse * mse_loss, w_pen * pen_loss
    
    #! TEST: Alternative loss function - HG coefficients normalized
    def normalized_hg_loss_function(A_j_evolution, A_k_evolution):
        final_j = A_j_evolution[:, :, -1] # shape: (B, Nt)
        # Compute HG coefficients for each signal in the batch
        final_j_hg = torch.stack([time_to_hg(final_j[i], hg_basis_B, dt) for i in range(B)]) # shape: (B, B)
        # Add epsilon for numerical stability during normalization
        eps = 1e-10
        final_j_hg_normalized = final_j_hg / (torch.norm(final_j_hg, dim=1, keepdim=True) + eps)
        # calculate the MSE loss between the final HG coefficients and the target HG coefficients
        mse_loss = F.mse_loss(final_j_hg_normalized, y_hg)
    
        pen_loss = 0.0
        zeros = torch.zeros(B, Nt, dtype=torch.float32, device=device)
        # Penalization loss - to enforce none of the waves to stray outside the simulation bounds
        for i in iters:
            pen_loss = pen_loss + F.mse_loss(torch.abs(A_k_evolution[:, :, i]*penalty)**2, zeros)
            
        return w_mse * mse_loss, w_pen * pen_loss
    
    # Define forward pass
    def forward(theta, hg_basis):
        Ain_k = hg_to_time(theta, hg_basis)
        return split_step_fourier_xpm_batch(
            x, Ain_k, dz, Nz, 
            beta2_j, beta2_k, 
            gamma_j, gamma_k, 
            Lt
        )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam([theta], lr=lr)
    
    print(f"Optimizing HG basis coefficient theta of size {theta.shape[0]}")
    
    return {
        't': t,
        'hg_basis': hg_basis,
        'x': x,
        'y': y,
        'theta': theta,
        'penalty': penalty,
        'mask': mask,
        'iters': iters,
        'loss_function': loss_function,
        'hg_loss_function': hg_loss_function,
        'normalized_hg_loss_function': normalized_hg_loss_function,
        'forward': forward,
        'optimizer': optimizer,
        'dz': dz,
        'dt': dt
    }


def train_loop(config, training_setup, device, run_dir, use_wandb=True, loss_fn_name='basic'):
    """Main training loop."""
    train = config['training']
    sim = config['simulation']
    plot = config['plotting']
    
    # Extract training setup
    theta = training_setup['theta']
    hg_basis = training_setup['hg_basis']
    
    # Select the loss function based on the loss_fn_name argument
    if loss_fn_name == 'hg':
        loss_function = training_setup['hg_loss_function']
        print(f"Using HG coefficient loss function")
    elif loss_fn_name == 'normalized_hg':
        loss_function = training_setup['normalized_hg_loss_function']
        print(f"Using normalized HG coefficient loss function")
    else:  # 'basic'
        loss_function = training_setup['loss_function']
        print(f"Using basic loss function")
    
    forward = training_setup['forward']
    optimizer = training_setup['optimizer']
    
    # Initialize wandb if requested and available
    if use_wandb and WANDB_AVAILABLE:
        wandb_config = config.get('wandb', {})
        wandb.init(
            project=wandb_config.get('project', 'xpm-identity-operator'),
            entity=wandb_config.get('entity'),
            name=wandb_config.get('name'),
            tags=wandb_config.get('tags', []),
            notes=wandb_config.get('notes', ''),
            config=config,
            dir=str(run_dir)
        )
    elif use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Continuing without wandb logging.")
        use_wandb = False
    
    # Training loop
    losses_mse = []
    losses_pen = []
    losses = []
    
    # Track best model for checkpoint saving
    best_loss = float('inf')
    best_theta = None
    best_iteration = -1
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"\nStarting training for {train['N_train']} iterations...")
    for i in tqdm(range(train['N_train']), desc="Training"):
        optimizer.zero_grad()
        A_j_evolution, A_k_evolution = forward(theta, hg_basis)
        loss_mse, loss_pen = loss_function(A_j_evolution, A_k_evolution)
        loss = loss_mse + loss_pen
        loss.backward()
        optimizer.step()
        
        # Save losses
        loss_mse_val = loss_mse.item()
        loss_pen_val = loss_pen.item()
        loss_val = loss.item()
        
        losses_mse.append(loss_mse_val)
        losses_pen.append(loss_pen_val)
        losses.append(loss_val)
        
        # Check if this is the best model so far
        if loss_val < best_loss:
            best_loss = loss_val
            best_iteration = i
            # Save a deep copy of the best theta
            best_theta = theta.data.clone().detach()
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
            torch.save({
                'iteration': i,
                'theta': best_theta,
                'loss': loss_val,
                'loss_mse': loss_mse_val,
                'loss_pen': loss_pen_val,
            }, checkpoint_path)
        
        # Log to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'iteration': i,
                'loss': loss_val,
                'loss_mse': loss_mse_val,
                'loss_pen': loss_pen_val,
                'best_loss': best_loss,
            })
        
        # Print progress every few iterations
        if (i + 1) % max(1, train['N_train'] // 10) == 0:
            print(f"Iteration {i+1}/{train['N_train']}: Loss={loss_val:.6f}, MSE={loss_mse_val:.6f}, Pen={loss_pen_val:.6f}, Best={best_loss:.6f} (iter {best_iteration+1})")
    
    # Restore best model before final evaluation
    if best_theta is not None:
        print(f"\nRestoring best model from iteration {best_iteration+1} (loss={best_loss:.6f})")
        theta.data.copy_(best_theta)
    
    # Final forward pass for evaluation (using best model)
    with torch.no_grad():
        A_j_evolution, A_k_evolution = forward(theta, hg_basis)
    
    # Save final losses
    final_losses = {
        'losses': losses,
        'losses_mse': losses_mse,
        'losses_pen': losses_pen,
        'final_loss': losses[-1],
        'final_loss_mse': losses_mse[-1],
        'final_loss_pen': losses_pen[-1],
        'best_loss': best_loss,
        'best_iteration': best_iteration,
        'best_loss_mse': losses_mse[best_iteration] if best_iteration >= 0 else None,
        'best_loss_pen': losses_pen[best_iteration] if best_iteration >= 0 else None,
    }
    
    # Save losses to file
    losses_path = run_dir / "losses.json"
    with open(losses_path, 'w') as f:
        json.dump(final_losses, f, indent=2)
    
    # Log final metrics to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            'final_loss': losses[-1],
            'final_loss_mse': losses_mse[-1],
            'final_loss_pen': losses_pen[-1],
            'best_loss': best_loss,
            'best_iteration': best_iteration,
        })
        wandb.finish()
    
    return A_j_evolution, A_k_evolution, final_losses, best_iteration, best_loss


def save_plots(config, training_setup, A_j_evolution, A_k_evolution, losses, run_dir):
    """Generate and save all plots."""
    sim = config['simulation']
    plot = config['plotting']
    train = config['training']
    
    # Convert simulation parameters to appropriate types
    Lz = float(sim['Lz'])
    Nz = int(sim['Nz'])
    
    t = training_setup['t']
    x = training_setup['x']
    y = training_setup['y']
    
    # Calculate plot indices
    plot_percent = plot['plot_percent']
    total_points = len(t)
    center_points = int(total_points * plot_percent)
    start_idx = (total_points - center_points) // 2
    end_idx = start_idx + center_points
    t_center = t[start_idx:end_idx].cpu().numpy()
    
    mode_nr = plot['mode_nr']
    
    # Create plots directory
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(losses['losses'], color='m', marker="o", label='Total Loss', linewidth=2, markersize=4)
    plt.plot(losses['losses_mse'], color='b', marker="s", label='MSE Loss', linewidth=2, markersize=4)
    plt.plot(losses['losses_pen'], color='r', marker="^", label='Penalty Loss', linewidth=2, markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "training_loss.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Intensity evolution plots
    # Note: plot_intensity_evolution calls plt.show(), but with Agg backend it won't display
    # We save the figure after the function call
    plot_intensity_evolution(
        A_j_evolution[mode_nr, :, :], 
        t, 
        Lz, 
        Nz, 
        wave_name='Wave j'
    )
    # Get the current figure and save it
    fig = plt.gcf()
    fig.savefig(plots_dir / "intensity_evolution_wave_j.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    plot_intensity_evolution(
        A_k_evolution[mode_nr, :, :], 
        t, 
        Lz, 
        Nz, 
        wave_name='Wave k'
    )
    fig = plt.gcf()
    fig.savefig(plots_dir / "intensity_evolution_wave_k.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 4. Co-wave evolution
    plot_cowave_evolution(
        A_j_evolution[mode_nr, :, :].detach().clone(),
        A_k_evolution[mode_nr, :, :].detach().clone(),
        t,
        Lz,
        Nz
    )
    fig = plt.gcf()
    fig.savefig(plots_dir / "cowave_evolution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 5. Target vs Final wave comparison
    plot_percent_comparison = 0.3
    center_points_comp = int(total_points * plot_percent_comparison)
    start_idx_comp = (total_points - center_points_comp) // 2
    end_idx_comp = start_idx_comp + center_points_comp
    t_center_comp = t[start_idx_comp:end_idx_comp].cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    ax1.plot(t_center_comp, np.abs(x[mode_nr].detach().cpu().numpy())[start_idx_comp:end_idx_comp]**2, 
            'b-', label='Input Wave', linewidth=2)
    ax1.plot(t_center_comp, np.abs(y[mode_nr].detach().cpu().numpy())[start_idx_comp:end_idx_comp]**2, 
            'r--', label='Target Wave', linewidth=2)
    ax1.plot(t_center_comp, np.abs(A_j_evolution[mode_nr, :, -1].detach().cpu().numpy())[start_idx_comp:end_idx_comp]**2, 
            'g-', label='Final Wave', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Intensity |A|²')
    ax1.set_title('Input vs Target vs Final Wave (j)')
    ax1.legend()
    ax1.grid()
    
    ax2.plot(t_center_comp, np.abs(A_k_evolution[mode_nr, :, 0].detach().cpu().numpy())[start_idx_comp:end_idx_comp]**2, 
            'b-', label='Strong Input Pulse', linewidth=2)
    ax2.plot(t_center_comp, np.abs(A_k_evolution[mode_nr, :, -1].detach().cpu().numpy())[start_idx_comp:end_idx_comp]**2, 
            'g-', label='Strong Output Pulse', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Intensity |A|²')
    ax2.set_title('Wave k: Input vs Output')
    ax2.legend()
    ax2.grid()
    
    plt.tight_layout()
    plt.savefig(plots_dir / "wave_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {plots_dir}")


def save_model_parameters(training_setup, run_dir, best_iteration=None, best_loss=None):
    # *somewhat redundant, since the best model is already saved in the checkpoint, but it saves a clean additional numpy copy of the best theta
    """Save the trained model parameters (best model from training)."""
    theta = training_setup['theta']
    params_dir = run_dir / "parameters"
    params_dir.mkdir(exist_ok=True)
    
    # Save theta as numpy array
    theta_path = params_dir / "theta.npy"
    np.save(theta_path, theta.detach().cpu().numpy())
    
    # Save as PyTorch state dict
    state_dict_path = params_dir / "theta_state_dict.pt"
    save_dict = {'theta': theta}
    if best_iteration is not None:
        save_dict['best_iteration'] = best_iteration
    if best_loss is not None:
        save_dict['best_loss'] = best_loss
    torch.save(save_dict, state_dict_path)
    
    if best_iteration is not None and best_loss is not None:
        print(f"Best model parameters (iteration {best_iteration+1}, loss={best_loss:.6f}) saved to {params_dir}")
    else:
        print(f"Model parameters saved to {params_dir}")


"""--------------------------------- Main Function ---------------------------------"""

def main():
    parser = argparse.ArgumentParser(description='Train XPM Identity Operator')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Custom run directory (default: auto-generated)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--loss-fn', type=str, default=None,
                        choices=['basic', 'hg', 'normalized_hg'],
                        help='Loss function to use: basic, hg, or normalized_hg (overrides config file)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Determine loss function: command-line arg overrides config file
    if args.loss_fn is not None:
        loss_fn = args.loss_fn
        print(f"Loss function set via command-line: {loss_fn}")
    else:
        # Get from config file, default to 'basic' if not specified
        loss_fn = config.get('training', {}).get('loss_fn', 'basic')
        if loss_fn not in ['basic', 'hg', 'normalized_hg']:
            print(f"Warning: Invalid loss_fn '{loss_fn}' in config file. Using 'basic' instead.")
            loss_fn = 'basic'
        print(f"Loss function from config file: {loss_fn}")
    
    # Setup device
    device = setup_device(config)
    
    # Create run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Use wandb name if available, otherwise use timestamp
        wandb_name = config.get('wandb', {}).get('name')
        run_dir = create_run_directory(name=wandb_name)
    
    print(f"\nRun directory: {run_dir}")
    
    # Update config with the actual loss function being used
    if 'training' not in config:
        config['training'] = {}
    config['training']['loss_fn'] = loss_fn
    
    # Save configuration to run directory
    config_path = save_config_to_run_dir(config, run_dir)
    print(f"Configuration saved to {config_path}")
    
    # Setup training
    print("\nSetting up training...")
    training_setup = setup_training(config, device)
    
    # Run training
    print("\nStarting training...")
    A_j_evolution, A_k_evolution, losses, best_iteration, best_loss = train_loop(
        config, training_setup, device, run_dir, 
        use_wandb=not args.no_wandb,
        loss_fn_name=loss_fn
    )
    
    # Save plots
    print("\nGenerating and saving plots...")
    save_plots(config, training_setup, A_j_evolution, A_k_evolution, losses, run_dir)
    
    # Save model parameters (best model)
    print("\nSaving model parameters...")
    save_model_parameters(training_setup, run_dir, best_iteration, best_loss)
    
    print(f"\n{'='*80}")
    print(f"Training completed successfully!")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

