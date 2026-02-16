"""
XPM Unitary Operator Training Script

This script trains an XPM operator to implement arbitrary unitary transformations
using the Hermite-Gauss basis. It can be run as a batch job and includes wandb 
logging, parameter saving, and plot generation.

Supported transformations:
    - identity: Identity transformation (y = x)
    - permutation: Random permutation of modes
    - rotation: Block-diagonal 2x2 rotations
    - arbitrary: Arbitrary unitary from random Hermitian matrix

Usage:
    python train_xpm_identity.py [--config config.yaml] [--run-dir runs/run_001] [--transformation identity]
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


VALID_TRANSFORMATIONS = ['identity', 'permutation', 'rotation', 'arbitrary']


def generate_transformation(transformation_name, num_modes, device, seed=27):
    """
    Generate a unitary transformation matrix.
    
    Args:
        transformation_name: One of 'identity', 'permutation', 'rotation', 'arbitrary'
        num_modes: Number of modes (size of the transformation matrix)
        device: Torch device
        seed: Random seed for reproducibility (default: 27)
    
    Returns:
        U: Unitary transformation matrix of shape (num_modes, num_modes), complex dtype
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    if transformation_name == 'identity':
        U = torch.eye(num_modes, dtype=torch.cfloat, device=device)
    
    elif transformation_name == 'permutation':
        # Generate a random permutation of the columns of an identity matrix
        idx = torch.randperm(num_modes, device=device)
        U = torch.eye(num_modes, dtype=torch.cfloat, device=device)[:, idx]
    
    elif transformation_name == 'rotation':
        # Block-diagonal 2x2 rotations
        theta = torch.rand(1, device=device) * 2 * torch.pi  # random angle in [0, 2pi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        rotation_2x2 = torch.zeros((2, 2), dtype=torch.float32, device=device)
        rotation_2x2[0, 0] = cos_theta
        rotation_2x2[0, 1] = -sin_theta
        rotation_2x2[1, 0] = sin_theta
        rotation_2x2[1, 1] = cos_theta
        
        # Create block-diagonal matrix with 2x2 rotation repeated along diagonal
        U = torch.zeros((num_modes, num_modes), dtype=torch.cfloat, device=device)
        num_blocks = num_modes // 2
        for block_idx in range(num_blocks):
            start = block_idx * 2
            U[start:start+2, start:start+2] = rotation_2x2.to(torch.cfloat)
        # Handle odd number of modes - last mode maps to itself
        if num_modes % 2 == 1:
            U[-1, -1] = 1.0
    
    elif transformation_name == 'arbitrary':
        # Arbitrary unitary from exponentiated Hermitian matrix
        H = torch.zeros((num_modes, num_modes), dtype=torch.cfloat, device=device)
        
        # Diagonal entries must be real
        diag_real = torch.randn(num_modes, device=device).to(torch.cfloat)
        H[torch.arange(num_modes), torch.arange(num_modes)] = diag_real
        
        # Off-diagonal entries: h_ij = conj(h_ji)
        for i in range(num_modes):
            for j in range(i+1, num_modes):
                re = torch.randn(1, device=device)
                im = torch.randn(1, device=device)
                val = re + 1j * im
                H[i, j] = val
                H[j, i] = val.conj()
        
        # Exponentiate to get unitary
        U = torch.matrix_exp(1j * H)
    
    else:
        raise ValueError(f"Unknown transformation: {transformation_name}. "
                        f"Valid options: {VALID_TRANSFORMATIONS}")
    
    return U


def is_unitary(matrix, atol=1e-5):
    """Check if a matrix is unitary within tolerance."""
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    if torch.is_complex(matrix):
        prod = matrix @ matrix.conj().T
    else:
        prod = matrix @ matrix.T
    return torch.allclose(prod, identity, atol=atol)


"""--------------------------------- Training Setup Functions ---------------------------------"""

def strong_soliton(t, beta2_k, gamma_k, tau):
    """Create the strong pulse (A_k) based on the soliton parameters."""
    # calculate secondary parameters
    a_2_squared = -beta2_k / (gamma_k * tau**2)
    a_2 = np.sqrt(a_2_squared)
    
    # Define strong pulse waveform
    Ain_k = a_2 * torch.cosh(t / tau)**(-1)
    return Ain_k


def setup_training(config, device, transformation_name='identity'):
    """Setup training data, model, and optimizer.
    
    Args:
        config: Configuration dictionary
        device: Torch device
        transformation_name: Name of the transformation to train ('identity', 'permutation', 'rotation', 'arbitrary')
    """
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
    
    # Get random seed for transformation (optional, defaults to 27)
    transformation_seed = int(train.get('transformation_seed', 27))
    
    # Calculate derived parameters
    dz = Lz / Nz
    dt = Lt / Nt
    
    # Create time grid
    t = torch.linspace(-Lt/2, Lt/2, Nt, dtype=torch.float32, device=device)
    
    # Define HG basis
    hg_basis = get_hg_basis(N_modes, t, tau)
    
    # Get batch of pulses (inputs are HG modes)
    B = batch_size
    x = hg_basis[:B, :] * amplitude_downscale  # inputs: shape (B, Nt)
    
    # Generate the target unitary transformation
    U = generate_transformation(transformation_name, B, device, seed=transformation_seed)
    print(f"Transformation '{transformation_name}' is unitary: {is_unitary(U)}")
    
    # Generate labels by applying transformation in HG coefficient space
    x_hg = torch.eye(B, dtype=torch.cfloat, device=device) * amplitude_downscale  # shape: (B, B)
    y_hg = U @ x_hg  # shape: (B, B) - target HG coefficients
    
    # Convert target HG coefficients back to time domain
    hg_basis_B = hg_basis[:B, :]  # truncated basis for batch size
    y = torch.stack([hg_to_time(y_hg[i], hg_basis_B) for i in range(B)])  # shape: (B, Nt)
    
    # Define input strong pulse
    A_strong = strong_soliton(t, beta2_k, gamma_k, tau)
    A_strong_hg = time_to_hg(A_strong, hg_basis, dt).to(torch.cfloat)
    theta = torch.nn.Parameter(A_strong_hg.clone().detach().requires_grad_(True))
    
    # Define penalty mask
    penalty = torch.zeros_like(t, device=device)
    penalty[:int(Nt/2 - Nt*mask_percent/2)] = 1
    penalty[int(Nt/2 + Nt*mask_percent/2):] = 1
    mask = 1 - penalty
    iters = np.linspace(0, Nz-1, m).astype(int)
    
    # Define loss function (compares intensity profiles)
    def loss_function(A_j_evolution, A_k_evolution):
        # MSE loss on intensity (works for complex y)
        mse_loss = F.mse_loss(torch.abs(A_j_evolution[:, :, -1])**2, torch.abs(y)**2)
        
        pen_loss = 0.0
        zeros = torch.zeros(B, Nt, dtype=torch.float32, device=device)
        # Penalization loss - to enforce none of the waves to stray outside the simulation bounds
        for i in iters:
            pen_loss = pen_loss + F.mse_loss(torch.abs(A_k_evolution[:, :, i]*penalty)**2, zeros)
        
        return w_mse * mse_loss, w_pen * pen_loss
    
    # Alternative loss function - HG coefficients
    def hg_loss_function(A_j_evolution, A_k_evolution):
        final_j = A_j_evolution[:, :, -1]  # shape: (B, Nt)
        # Compute HG coefficients for each signal in the batch
        final_j_hg = torch.stack([time_to_hg(final_j[i], hg_basis_B, dt) for i in range(B)])  # shape: (B, B)
        # MSE loss between HG coefficients (use abs for complex coefficients)
        mse_loss = F.mse_loss(torch.abs(final_j_hg)**2, torch.abs(y_hg)**2)
        
        pen_loss = 0.0
        zeros = torch.zeros(B, Nt, dtype=torch.float32, device=device)
        # Penalization loss - to enforce none of the waves to stray outside the simulation bounds
        for i in iters:
            pen_loss = pen_loss + F.mse_loss(torch.abs(A_k_evolution[:, :, i]*penalty)**2, zeros)
            
        return w_mse * mse_loss, w_pen * pen_loss
    
    # Alternative loss function - HG coefficients normalized
    def normalized_hg_loss_function(A_j_evolution, A_k_evolution):
        final_j = A_j_evolution[:, :, -1]  # shape: (B, Nt)
        # Compute HG coefficients for each signal in the batch
        final_j_hg = torch.stack([time_to_hg(final_j[i], hg_basis_B, dt) for i in range(B)])  # shape: (B, B)
        # Add epsilon for numerical stability during normalization
        eps = 1e-10
        final_j_hg_normalized = final_j_hg / (torch.norm(final_j_hg, dim=1, keepdim=True) + eps) * amplitude_downscale
        # Normalize target as well for fair comparison
        y_hg_normalized = y_hg / (torch.norm(y_hg, dim=1, keepdim=True) + eps) * amplitude_downscale
        # MSE loss (use abs for complex coefficients)
        mse_loss = F.mse_loss(torch.abs(final_j_hg_normalized), torch.abs(y_hg_normalized))
    
        pen_loss = 0.0
        zeros = torch.zeros(B, Nt, dtype=torch.float32, device=device)
        # Penalization loss - to enforce none of the waves to stray outside the simulation bounds
        for i in iters:
            pen_loss = pen_loss + F.mse_loss(torch.abs(A_k_evolution[:, :, i]*penalty)**2, zeros)
            
        return w_mse * mse_loss, w_pen * pen_loss
    
    # Fidelity-based loss function (phase-aware)
    def fid_phase_loss_function(A_j_evolution, A_k_evolution):
        """Fidelity loss on HG coefficients (phase-aware, no normalization)"""
        final_j = A_j_evolution[:, :, -1]  # shape: (B, Nt)
        # Compute HG coefficients for each signal in the batch
        final_j_hg = torch.stack([time_to_hg(final_j[i], hg_basis_B, dt) for i in range(B)])  # shape: (B, B)
        
        # Penalization loss
        pen_loss = 0.0
        zeros = torch.zeros(B, Nt, dtype=torch.float32, device=device)
        for i in iters:
            pen_loss = pen_loss + F.mse_loss(torch.abs(A_k_evolution[:, :, i]*penalty)**2, zeros)
        
        # Fidelity loss (phase-aware)
        dot_products = torch.sum(final_j_hg.conj() * y_hg, dim=1)  # sum along rows
        fidelity_avg = torch.mean(torch.abs(dot_products)**2)
        fid_loss = -fidelity_avg  # negative because we want to maximize fidelity
        
        return w_mse * fid_loss, w_pen * pen_loss
    
    # Fidelity-based loss function (energy-only)
    def fid_energy_loss_function(A_j_evolution, A_k_evolution):
        """Fidelity loss on HG coefficient energies (phase-agnostic, no normalization)"""
        final_j = A_j_evolution[:, :, -1]  # shape: (B, Nt)
        # Compute HG coefficients for each signal in the batch
        final_j_hg = torch.stack([time_to_hg(final_j[i], hg_basis_B, dt) for i in range(B)])  # shape: (B, B)
        
        # Penalization loss
        pen_loss = 0.0
        zeros = torch.zeros(B, Nt, dtype=torch.float32, device=device)
        for i in iters:
            pen_loss = pen_loss + F.mse_loss(torch.abs(A_k_evolution[:, :, i]*penalty)**2, zeros)
        
        # Convert to energy (magnitude) only - no conjugate needed as we're taking abs
        final_j_hg_energy = torch.abs(final_j_hg)
        y_hg_energy = torch.abs(y_hg)
        
        # Fidelity loss (energy-only)
        dot_products = torch.sum(final_j_hg_energy * y_hg_energy, dim=1)  # sum along rows
        fidelity_avg = torch.mean(dot_products**2)
        fid_loss = -fidelity_avg  # negative because we want to maximize fidelity
        
        return w_mse * fid_loss, w_pen * pen_loss
    
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
    print(f"Training for transformation: {transformation_name}")
    
    return {
        't': t,
        'hg_basis': hg_basis,
        'hg_basis_B': hg_basis_B,
        'x': x,
        'y': y,
        'y_hg': y_hg,
        'U': U,
        'transformation_name': transformation_name,
        'theta': theta,
        'penalty': penalty,
        'mask': mask,
        'iters': iters,
        'loss_function': loss_function,
        'hg_loss_function': hg_loss_function,
        'normalized_hg_loss_function': normalized_hg_loss_function,
        'fid_phase_loss_function': fid_phase_loss_function,
        'fid_energy_loss_function': fid_energy_loss_function,
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
    elif loss_fn_name == 'fid_phase':
        loss_function = training_setup['fid_phase_loss_function']
        print(f"Using fidelity (phase-aware) loss function")
    elif loss_fn_name == 'fid_energy':
        loss_function = training_setup['fid_energy_loss_function']
        print(f"Using fidelity (energy-only) loss function")
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
        
        # Universal evaluation metric - compute phase-aware fidelity for comparison across runs
        with torch.no_grad():
            final_j = A_j_evolution[:, :, -1]
            final_j_hg = torch.stack([time_to_hg(final_j[j], training_setup['hg_basis_B'], training_setup['dt']) 
                                     for j in range(final_j.shape[0])])
            # Fidelity calculation (phase-aware)
            # Compute target fidelity (perfect case: target vs target)
            target_dot_products = torch.sum(training_setup['y_hg'].conj() * training_setup['y_hg'], dim=1)
            target_fidelity_avg = torch.mean(torch.abs(target_dot_products)).item()
            # Compute actual fidelity (output vs target)
            dot_products = torch.sum(final_j_hg.conj() * training_setup['y_hg'], dim=1)
            actual_fidelity_avg = torch.mean(torch.abs(dot_products)).item()
            # Normalize to percentage (0-100, can exceed 100 if output amplitude > target)
            eval_hg_fidelity = (actual_fidelity_avg / (target_fidelity_avg + 1e-10)) * 100
        
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
                'eval_hg_fidelity': eval_hg_fidelity,
            }, checkpoint_path)
        
        # Log to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'iteration': i,
                'loss': loss_val,
                'loss_mse': loss_mse_val,
                'loss_pen': loss_pen_val,
                'best_loss': best_loss,
                'eval_hg_fidelity': eval_hg_fidelity,  # Universal evaluation metric (phase-aware fidelity)
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
        # Compute final eval_hg_fidelity (phase-aware)
        final_j = A_j_evolution[:, :, -1]
        final_j_hg = torch.stack([time_to_hg(final_j[j], training_setup['hg_basis_B'], training_setup['dt']) 
                                 for j in range(final_j.shape[0])])
        # Fidelity calculation (phase-aware)
        # Compute target fidelity (perfect case: target vs target)
        target_dot_products = torch.sum(training_setup['y_hg'].conj() * training_setup['y_hg'], dim=1)
        target_fidelity_avg = torch.mean(torch.abs(target_dot_products)).item()
        # Compute actual fidelity (output vs target)
        dot_products = torch.sum(final_j_hg.conj() * training_setup['y_hg'], dim=1)
        actual_fidelity_avg = torch.mean(torch.abs(dot_products)).item()
        # Normalize to percentage (0-100, can exceed 100 if output amplitude > target)
        final_eval_hg_fidelity = (actual_fidelity_avg / (target_fidelity_avg + 1e-10)) * 100
    
    # Save final losses
    final_losses = {
        'losses': losses,
        'losses_mse': losses_mse,
        'losses_pen': losses_pen,
        'final_loss': losses[-1],
        'final_loss_mse': losses_mse[-1],
        'final_loss_pen': losses_pen[-1],
        'final_eval_hg_fidelity': final_eval_hg_fidelity,
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
            'final_eval_hg_fidelity': final_eval_hg_fidelity,
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
    U = training_setup['U']
    transformation_name = training_setup['transformation_name']
    
    params_dir = run_dir / "parameters"
    params_dir.mkdir(exist_ok=True)
    
    # Save theta as numpy array
    theta_path = params_dir / "theta.npy"
    np.save(theta_path, theta.detach().cpu().numpy())
    
    # Save transformation matrix as numpy array
    U_path = params_dir / "transformation_matrix.npy"
    np.save(U_path, U.detach().cpu().numpy())
    
    # Save as PyTorch state dict
    state_dict_path = params_dir / "theta_state_dict.pt"
    save_dict = {
        'theta': theta,
        'transformation_matrix': U,
        'transformation_name': transformation_name,
    }
    if best_iteration is not None:
        save_dict['best_iteration'] = best_iteration
    if best_loss is not None:
        save_dict['best_loss'] = best_loss
    torch.save(save_dict, state_dict_path)
    
    if best_iteration is not None and best_loss is not None:
        print(f"Best model parameters (iteration {best_iteration+1}, loss={best_loss:.6f}) saved to {params_dir}")
        print(f"Transformation '{transformation_name}' matrix saved to {U_path}")
    else:
        print(f"Model parameters saved to {params_dir}")


def visualize_unitary_comparison(config, training_setup, run_dir):
    """
    Visualize the target vs actual unitary transformation matrices.
    
    This function performs a simulation run with the trained parameters and reconstructs
    the actual unitary transformation by comparing input and output HG coefficients.
    Creates a side-by-side comparison plot of target and actual unitary matrices.
    
    Args:
        config: Configuration dictionary
        training_setup: Dictionary containing training setup (theta, forward function, etc.)
        run_dir: Path to run directory for saving the plot
    """
    print("\nGenerating unitary transformation comparison...")
    
    # Extract training setup components
    theta = training_setup['theta']
    forward = training_setup['forward']
    hg_basis = training_setup['hg_basis']
    hg_basis_B = training_setup['hg_basis_B']
    dt = training_setup['dt']
    U_target = training_setup['U']
    transformation_name = training_setup['transformation_name']
    
    # Get batch size from x
    x = training_setup['x']
    batch_size = x.shape[0]
    
    # Extract amplitude scaling factor
    amplitude_downscale = float(config['training']['amplitude_downscale'])
    
    # Run simulation with best parameters
    with torch.no_grad():
        A_j_evolution, A_k_evolution = forward(theta, hg_basis)
        
        # Extract final outputs for each mode
        final_outputs = A_j_evolution[:, :, -1]  # shape: (B, Nt)
        
        # Convert outputs to HG coefficients for each batch element
        # Each batch element corresponds to a different input mode
        output_hg_coeffs = torch.stack([
            time_to_hg(final_outputs[i], hg_basis_B, dt) 
            for i in range(batch_size)
        ])  # shape: (B, B)
        
        # The input HG coefficients are essentially identity matrix scaled
        # input_hg_coeffs[i, j] = amplitude_downscale if i==j, else 0
        # So the actual transformation matrix is obtained by dividing by the scaling
        U_actual = output_hg_coeffs / amplitude_downscale
    
    # Convert to numpy for plotting
    U_target_np = U_target.detach().cpu().numpy()
    U_actual_np = U_actual.detach().cpu().numpy()
    
    # Calculate magnitude arrays
    U_target_mag = np.abs(U_target_np)
    U_actual_mag = np.abs(U_actual_np)
    
    # Determine common scale for both plots
    vmin = min(U_target_mag.min(), U_actual_mag.min())
    vmax = max(U_target_mag.max(), U_actual_mag.max())
    
    # Create visualization with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Target unitary transformation (magnitude)
    im1 = ax1.imshow(
        U_target_mag, 
        cmap='viridis', 
        aspect='auto',
        interpolation='nearest',
        vmin=vmin,
        vmax=vmax
    )
    ax1.set_title(f'Target Unitary: {transformation_name}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Input Mode', fontsize=12)
    ax1.set_ylabel('Output Mode', fontsize=12)
    ax1.set_xticks(range(batch_size))
    ax1.set_yticks(range(batch_size))
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('|U|', fontsize=12)
    
    # Plot 2: Actual unitary transformation from simulation (magnitude)
    im2 = ax2.imshow(
        U_actual_mag, 
        cmap='viridis', 
        aspect='auto',
        interpolation='nearest',
        vmin=vmin,
        vmax=vmax
    )
    ax2.set_title('Actual Unitary from Simulation', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Input Mode', fontsize=12)
    ax2.set_ylabel('Output Mode', fontsize=12)
    ax2.set_xticks(range(batch_size))
    ax2.set_yticks(range(batch_size))
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('|U|', fontsize=12)
    
    # Calculate and display fidelity metric
    fidelity = torch.mean(torch.abs(U_actual.conj() * U_target)).item()
    mse = torch.mean(torch.abs(U_actual - U_target)**2).item()
    
    plt.suptitle(
        f'Unitary Transformation Comparison\nFidelity: {fidelity:.4f} | MSE: {mse:.6f}',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    
    # Save the figure
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    save_path = plots_dir / "unitary_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Unitary comparison plot saved to {save_path}")
    print(f"  Fidelity: {fidelity:.4f}")
    print(f"  MSE: {mse:.6f}")
    
    # Also save the actual unitary matrix as numpy array
    params_dir = run_dir / "parameters"
    params_dir.mkdir(exist_ok=True)
    U_actual_path = params_dir / "actual_unitary_matrix.npy"
    np.save(U_actual_path, U_actual_np)
    print(f"Actual unitary matrix saved to {U_actual_path}")


"""--------------------------------- Main Function ---------------------------------"""

def main():
    parser = argparse.ArgumentParser(description='Train XPM Unitary Operator')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Custom run directory (default: auto-generated)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--loss-fn', type=str, default=None,
                        choices=['basic', 'hg', 'normalized_hg', 'fid_phase', 'fid_energy'],
                        help='Loss function to use: basic, hg, normalized_hg, fid_phase, or fid_energy (overrides config file)')
    parser.add_argument('--transformation', type=str, default=None,
                        choices=VALID_TRANSFORMATIONS,
                        help=f'Transformation to train: {", ".join(VALID_TRANSFORMATIONS)} (overrides config file)')
    
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
        if loss_fn not in ['basic', 'hg', 'normalized_hg', 'fid_phase', 'fid_energy']:
            print(f"Warning: Invalid loss_fn '{loss_fn}' in config file. Using 'basic' instead.")
            loss_fn = 'basic'
        print(f"Loss function from config file: {loss_fn}")
    
    # Determine transformation: command-line arg overrides config file
    if args.transformation is not None:
        transformation = args.transformation
        print(f"Transformation set via command-line: {transformation}")
    else:
        # Get from config file, default to 'identity' if not specified
        transformation = config.get('training', {}).get('transformation', 'identity')
        if transformation not in VALID_TRANSFORMATIONS:
            print(f"Warning: Invalid transformation '{transformation}' in config file. Using 'identity' instead.")
            transformation = 'identity'
        print(f"Transformation from config file: {transformation}")
    
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
    
    # Update config with the actual loss function and transformation being used
    if 'training' not in config:
        config['training'] = {}
    config['training']['loss_fn'] = loss_fn
    config['training']['transformation'] = transformation
    
    # Save configuration to run directory
    config_path = save_config_to_run_dir(config, run_dir)
    print(f"Configuration saved to {config_path}")
    
    # Setup training
    print("\nSetting up training...")
    training_setup = setup_training(config, device, transformation_name=transformation)
    
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
    
    # Generate unitary transformation comparison visualization
    visualize_unitary_comparison(config, training_setup, run_dir)
    
    print(f"\n{'='*80}")
    print(f"Training completed successfully!")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

