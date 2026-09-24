"""Train an XPM fiber to implement a target unitary in the Hermite–Gauss basis.

    python -m approaches.xpm_unitary.train --config approaches/xpm_unitary/configs/fiber/config.yaml

How a run works (see ``main``):

1. ``setup_training`` builds the time grid and HG basis, the weak inputs x (the
   first B HG modes), the target coefficients y_hg = U @ x_hg, and the trainable
   strong pulse theta (its HG coefficients, seeded with a soliton).
2. ``train_loop`` propagates all B inputs through the fiber with the same strong
   pulse, projects each output back onto the first B HG modes to get the realized
   matrix V, and minimizes  w_mse * loss(V, U) + w_pen * boundary_penalty.
   The loss definitions live in ``nlse/losses.py``.
3. The best theta, the realized V and a comparison plot are written to ``runs/<name>/``.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch jobs
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")

from nlse import get_hg_basis, hg_to_time, time_to_hg, split_step_fourier_xpm_batch, strong_soliton
from nlse.losses import (
    HG_LOSSES,
    LOSS_NAMES,
    intensity_loss,
    eval_frobenius_norm,
    eval_trace_fidelity,
    eval_hg_fidelity,
)
from approaches.xpm_unitary.transformations import VALID_TRANSFORMATIONS, generate_transformation, is_unitary


DEFAULT_CONFIG = "approaches/xpm_unitary/configs/fiber/config.yaml"


# =============================================================================
# 1. Problem setup
# =============================================================================

def setup_training(config, device, transformation_name='identity'):
    """Build inputs, targets, the trainable strong pulse, the forward model and the optimizer."""
    sim = config['simulation']
    med = config['medium']
    pulse = config['pulse']
    train = config['training']

    Lz, Nz = float(sim['Lz']), int(sim['Nz'])
    Lt, Nt = float(sim['Lt']), int(sim['Nt'])
    N_modes = int(sim['N_modes'])
    beta2_j, beta2_k = float(med['beta2_j']), float(med['beta2_k'])
    gamma_j, gamma_k = float(med['gamma_j']), float(med['gamma_k'])
    tau = float(pulse['tau'])

    B = int(train['batch_size'])  # number of input HG modes, i.e. the size of U
    amplitude_downscale = float(train['amplitude_downscale'])
    mask_percent = float(train['mask_percent'])
    m = int(train['m'])
    w_mse = float(train['w_mse'])
    w_pen = float(train['w_pen'])
    transformation_seed = int(train.get('transformation_seed', 27))

    # When set, the strong pulse propagates with beta2_k = gamma_k = 0 (no SPM, no
    # dispersion) but is still seeded from the configured soliton; the weak pulse
    # still sees XPM from |A_k|^2 via gamma_j.
    test_spm_dispersion_off = bool(train.get('test_spm_dispersion_off', False))

    dz = Lz / Nz
    dt = Lt / Nt
    t = torch.linspace(-Lt/2, Lt/2, Nt, dtype=torch.float32, device=device)
    hg_basis = get_hg_basis(N_modes, t, tau)
    hg_basis_B = hg_basis[:B, :]

    # Inputs: the first B HG modes. Targets: U applied in HG-coefficient space.
    x = hg_basis_B * amplitude_downscale                                      # (B, Nt)
    U = generate_transformation(transformation_name, B, device, seed=transformation_seed)
    print(f"Transformation '{transformation_name}' is unitary: {is_unitary(U)}")
    x_hg = torch.eye(B, dtype=torch.cfloat, device=device) * amplitude_downscale
    y_hg = U @ x_hg                                                           # (B, B)
    y = torch.stack([hg_to_time(y_hg[i], hg_basis_B) for i in range(B)])      # (B, Nt)

    # Trainable parameters: HG coefficients of the strong pulse.
    A_strong = strong_soliton(t, beta2_k, gamma_k, tau)
    theta = torch.nn.Parameter(time_to_hg(A_strong, hg_basis, dt).to(torch.cfloat).clone().detach())

    if test_spm_dispersion_off:
        beta2_k_prop, gamma_k_prop = 0.0, 0.0
        print("[TEST] test_spm_dispersion_off=True: strong pulse propagated with "
              f"beta2_k=0, gamma_k=0 (initial pulse still defined with beta2_k={beta2_k}, gamma_k={gamma_k}).")
    else:
        beta2_k_prop, gamma_k_prop = beta2_k, gamma_k

    def forward(theta, hg_basis):
        Ain_k = hg_to_time(theta, hg_basis)
        return split_step_fourier_xpm_batch(x, Ain_k, dz, Nz, beta2_j, beta2_k_prop, gamma_j, gamma_k_prop, Lt)

    def output_hg(A_j_evolution):
        """Realized matrix V: row i = HG coefficients of the output for input mode i."""
        final_j = A_j_evolution[:, :, -1]
        return torch.stack([time_to_hg(final_j[i], hg_basis_B, dt) for i in range(B)])

    # Penalize strong-pulse intensity outside the central mask_percent of the window,
    # sampled at m positions along z.
    penalty = torch.zeros_like(t, device=device)
    penalty[:int(Nt/2 - Nt*mask_percent/2)] = 1
    penalty[int(Nt/2 + Nt*mask_percent/2):] = 1
    iters = np.linspace(0, Nz-1, m).astype(int)
    zeros = torch.zeros(B, Nt, dtype=torch.float32, device=device)

    def boundary_penalty(A_k_evolution):
        pen = 0.0
        for i in iters:
            pen = pen + F.mse_loss(torch.abs(A_k_evolution[:, :, i] * penalty)**2, zeros)
        return pen

    def make_loss(name):
        """Return loss(A_j_evolution, A_k_evolution) -> (weighted main loss, weighted penalty)."""
        if name == 'basic':
            main = lambda A_j_evolution: intensity_loss(A_j_evolution[:, :, -1], y)
        elif name == 'normalized_hg':
            main = lambda A_j_evolution: HG_LOSSES[name](output_hg(A_j_evolution), y_hg, amplitude_downscale)
        else:
            main = lambda A_j_evolution: HG_LOSSES[name](output_hg(A_j_evolution), y_hg)

        def loss(A_j_evolution, A_k_evolution):
            return w_mse * main(A_j_evolution), w_pen * boundary_penalty(A_k_evolution)
        return loss

    optimizer = make_optimizer(train.get('optimizer', 'adam'), theta, train)
    scheduler = make_scheduler(optimizer, train)

    print(f"Optimizing HG basis coefficient theta of size {theta.shape[0]}")
    print(f"Training for transformation: {transformation_name}")
    print(f"Using optimizer: {train.get('optimizer', 'adam').lower()}")

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
        'forward': forward,
        'output_hg': output_hg,
        'make_loss': make_loss,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'dz': dz,
        'dt': dt,
    }


def make_optimizer(name, theta, train):
    lr = float(train['lr'])
    name = name.lower()
    if name == 'adam':
        return torch.optim.Adam([theta], lr=lr)
    if name == 'sgd':
        return torch.optim.SGD([theta], lr=lr)
    if name == 'lbfgs':
        # 'strong_wolfe' makes lr an upper bound on the step length and backtracks
        # whenever the loss would increase, so one overshoot cannot blow theta up.
        line_search = train.get('lbfgs_line_search', 'strong_wolfe')
        if isinstance(line_search, str) and line_search.lower() == 'none':
            line_search = None
        return torch.optim.LBFGS(
            [theta], lr=lr,
            max_iter=int(train.get('lbfgs_max_iter', 10)),
            history_size=int(train.get('lbfgs_history_size', 10)),
            line_search_fn=line_search,
        )
    raise ValueError(f"Unknown optimizer: '{name}'. Valid options: adam, sgd, lbfgs")


def make_scheduler(optimizer, train):
    name = train.get('learning_schedule', 'none').lower()
    if name == 'none':
        return None
    if name == 'cosine_warm_restarts':
        T_0 = int(train.get('schedule_T0', 50))
        T_mult = int(train.get('schedule_T_mult', 2))
        eta_min = float(train.get('schedule_eta_min', 0))
        print(f"Using learning schedule: {name} (T_0={T_0}, T_mult={T_mult}, eta_min={eta_min})")
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
    raise ValueError(f"Unknown learning_schedule: '{name}'. Valid options: none, cosine_warm_restarts")


# =============================================================================
# 2. Training loop
# =============================================================================

def evaluate(V, U):
    """Loss-independent metrics, so runs with different losses can be compared."""
    return {
        'eval_hg_fidelity': eval_hg_fidelity(V, U),
        'eval_frobenius_norm': eval_frobenius_norm(V, U),
        'eval_trace_fidelity': eval_trace_fidelity(V, U),
    }


def optimizer_step(optimizer, forward, loss_fn, theta, hg_basis):
    """One optimization step. Returns (A_j_evolution, A_k_evolution, loss_mse, loss_pen)."""
    if not isinstance(optimizer, torch.optim.LBFGS):
        optimizer.zero_grad()
        A_j_evo, A_k_evo = forward(theta, hg_basis)
        loss_mse, loss_pen = loss_fn(A_j_evo, A_k_evo)
        (loss_mse + loss_pen).backward()
        optimizer.step()
        return A_j_evo, A_k_evo, loss_mse, loss_pen

    last = {}

    def closure():
        optimizer.zero_grad()
        A_j_evo, A_k_evo = forward(theta, hg_basis)
        l_mse, l_pen = loss_fn(A_j_evo, A_k_evo)
        l = l_mse + l_pen
        last.update(A_j=A_j_evo, A_k=A_k_evo, mse=l_mse, pen=l_pen)
        # A non-finite trial step returns +inf with zero gradient so the line search
        # backtracks instead of committing it and corrupting theta for the rest of the run.
        if not torch.isfinite(l):
            if theta.grad is None:
                theta.grad = torch.zeros_like(theta)
            else:
                theta.grad.detach_()
                theta.grad.zero_()
            return torch.full_like(l.detach(), float('inf'))
        l.backward()
        return l

    optimizer.step(closure)
    return last['A_j'], last['A_k'], last['mse'], last['pen']


def train_loop(config, setup, run_dir, use_wandb=True, loss_fn_name='basic'):
    train = config['training']
    N_train = int(train['N_train'])
    theta = setup['theta']
    hg_basis = setup['hg_basis']
    forward = setup['forward']
    output_hg = setup['output_hg']
    y_hg = setup['y_hg']
    optimizer = setup['optimizer']
    scheduler = setup['scheduler']
    transformation_name = setup['transformation_name']

    # Curriculum: trace loss first, then hg_phase (optionally with another optimizer).
    curriculum = bool(train.get('curriculum_learning', False))
    if curriculum:
        switch_iter = int(N_train * float(train.get('curriculum_switch', 0.8)))
        loss_fn = setup['make_loss']('trace')
        print(f"Curriculum learning enabled: trace loss for iters 0-{switch_iter}, "
              f"hg_phase loss for iters {switch_iter}-{N_train}")
    else:
        loss_fn = setup['make_loss'](loss_fn_name)
        print(f"Using loss function: {loss_fn_name}")

    if use_wandb and WANDB_AVAILABLE:
        wandb_config = config.get('wandb', {})
        wandb.init(
            project=wandb_config.get('project', 'xpm-identity-operator'),
            entity=wandb_config.get('entity'),
            name=wandb_config.get('name'),
            tags=wandb_config.get('tags', []),
            notes=wandb_config.get('notes', ''),
            config=config,
            dir=str(run_dir),
        )
    elif use_wandb:
        print("Warning: wandb requested but not available. Continuing without wandb logging.")
        use_wandb = False

    losses, losses_mse, losses_pen = [], [], []
    best_loss, best_theta, best_iteration = float('inf'), None, -1
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    # Flush losses.json + summary.json ~20 times so a SLURM kill still leaves artifacts.
    partial_save_interval = max(1, N_train // 20)
    status = 'completed'

    def save(status, final_eval=None):
        return save_run_artifacts(
            run_dir, config, loss_fn_name, transformation_name,
            losses, losses_mse, losses_pen,
            best_iteration, best_loss if best_iteration >= 0 else None,
            final_eval=final_eval, status=status,
        )

    print(f"\nStarting training for {N_train} iterations...")
    for i in tqdm(range(N_train), desc="Training"):
        if curriculum and i == switch_iter:
            loss_fn = setup['make_loss']('hg_phase')
            next_opt = train.get('curriculum_optimizer', train.get('optimizer', 'adam')).lower()
            current_opt = 'lbfgs' if isinstance(optimizer, torch.optim.LBFGS) else type(optimizer).__name__.lower()
            if next_opt != current_opt:
                optimizer = make_optimizer(next_opt, theta, train)
                tqdm.write(f"[Curriculum] Switching to hg_phase loss + {next_opt} optimizer at iteration {i}")
            else:
                optimizer.state.clear()
                tqdm.write(f"[Curriculum] Switching to hg_phase loss at iteration {i} (optimizer state reset)")

        A_j_evolution, A_k_evolution, loss_mse, loss_pen = optimizer_step(
            optimizer, forward, loss_fn, theta, hg_basis
        )
        if scheduler is not None:
            scheduler.step()

        loss_mse_val = loss_mse.item()
        loss_pen_val = loss_pen.item()
        loss_val = (loss_mse + loss_pen).item()
        losses_mse.append(loss_mse_val)
        losses_pen.append(loss_pen_val)
        losses.append(loss_val)

        if not all(math.isfinite(v) for v in (loss_val, loss_mse_val, loss_pen_val)):
            if best_iteration >= 0:
                msg = (f"\n[NaN guard] Non-finite loss at iteration {i+1} (loss={loss_val}). Stopping early. "
                       f"Best so far: iter {best_iteration+1}, loss={best_loss:.6f}.")
            else:
                msg = (f"\n[NaN guard] Non-finite loss at iteration {i+1} (loss={loss_val}). "
                       f"No finite iteration was recorded; the optimizer likely overshot at step 0 -- "
                       f"try a smaller lr / lbfgs_max_iter.")
            tqdm.write(msg)
            status = 'nan'
            save(status)
            break

        with torch.no_grad():
            metrics = evaluate(output_hg(A_j_evolution), y_hg)

        if loss_val < best_loss:
            best_loss, best_iteration = loss_val, i
            best_theta = theta.data.clone().detach()
            torch.save({
                'iteration': i,
                'theta': best_theta,
                'loss': loss_val,
                'loss_mse': loss_mse_val,
                'loss_pen': loss_pen_val,
                **metrics,
            }, checkpoint_dir / "best_checkpoint.pt")

        if use_wandb:
            wandb.log({
                'iteration': i,
                'loss': loss_val,
                'loss_mse': loss_mse_val,
                'loss_pen': loss_pen_val,
                'best_loss': best_loss,
                **metrics,
                'learning_rate': optimizer.param_groups[0]['lr'],
            })

        if (i + 1) % max(1, N_train // 10) == 0:
            print(f"Iteration {i+1}/{N_train}: Loss={loss_val:.6f}, MSE={loss_mse_val:.6f}, "
                  f"Pen={loss_pen_val:.6f}, Best={best_loss:.6f} (iter {best_iteration+1})")

        if (i + 1) % partial_save_interval == 0:
            save('in_progress')

    # Re-evaluate the best model (skipped if no finite iteration was recorded).
    final_eval = None
    A_j_evolution = A_k_evolution = None
    if best_theta is not None:
        print(f"\nRestoring best model from iteration {best_iteration+1} (loss={best_loss:.6f})")
        theta.data.copy_(best_theta)
        with torch.no_grad():
            A_j_evolution, A_k_evolution = forward(theta, hg_basis)
            final_eval = evaluate(output_hg(A_j_evolution), y_hg)
    else:
        print("\nNo finite best model recorded -- skipping final evaluation forward pass.")

    final_losses, _ = save(status, final_eval)

    if use_wandb:
        final_eval = final_eval or {}
        wandb.log({
            'final_loss': losses[-1] if losses else None,
            'final_loss_mse': losses_mse[-1] if losses_mse else None,
            'final_loss_pen': losses_pen[-1] if losses_pen else None,
            'final_eval_hg_fidelity': final_eval.get('eval_hg_fidelity'),
            'final_eval_frobenius_norm': final_eval.get('eval_frobenius_norm'),
            'final_eval_trace_fidelity': final_eval.get('eval_trace_fidelity'),
            'best_loss': best_loss if best_iteration >= 0 else None,
            'best_iteration': best_iteration if best_iteration >= 0 else None,
            'run_status': status,
        })
        wandb.finish()

    return A_j_evolution, A_k_evolution, final_losses, best_iteration, best_loss


# =============================================================================
# 3. Saving results
# =============================================================================

def save_run_artifacts(
    run_dir, config, loss_fn_name, transformation_name,
    losses, losses_mse, losses_pen,
    best_iteration, best_loss,
    final_eval=None,
    status='in_progress',
):
    """Write losses.json and summary.json atomically. Safe to call repeatedly."""
    has_best = best_iteration is not None and best_iteration >= 0
    final_eval = final_eval or {}
    final_losses = {
        'losses': losses,
        'losses_mse': losses_mse,
        'losses_pen': losses_pen,
        'final_loss': losses[-1] if losses else None,
        'final_loss_mse': losses_mse[-1] if losses_mse else None,
        'final_loss_pen': losses_pen[-1] if losses_pen else None,
        'final_eval_hg_fidelity': final_eval.get('eval_hg_fidelity'),
        'final_eval_frobenius_norm': final_eval.get('eval_frobenius_norm'),
        'final_eval_trace_fidelity': final_eval.get('eval_trace_fidelity'),
        'best_loss': best_loss if has_best else None,
        'best_iteration': best_iteration if has_best else None,
        'best_loss_mse': losses_mse[best_iteration] if has_best else None,
        'best_loss_pen': losses_pen[best_iteration] if has_best else None,
        'iterations_completed': len(losses),
        'status': status,
    }
    atomic_write_json(run_dir / 'losses.json', final_losses)

    summary = {
        'batch_size': int(config['training']['batch_size']),
        'loss_fn': loss_fn_name,
        'transformation': transformation_name,
        'optimizer': config['training'].get('optimizer', 'adam'),
        'lr': float(config['training']['lr']),
        'N_train': int(config['training']['N_train']),
        'N_modes': int(config['simulation']['N_modes']),
        'iterations_completed': len(losses),
        'status': status,
        'best_iteration': best_iteration if has_best else None,
        'best_loss': best_loss if has_best else None,
        'best_eval_trace_fidelity': final_eval.get('eval_trace_fidelity'),
        'best_eval_frobenius_norm': final_eval.get('eval_frobenius_norm'),
        'best_eval_hg_fidelity': final_eval.get('eval_hg_fidelity'),
        'run_dir': str(run_dir),
    }
    atomic_write_json(run_dir / 'summary.json', summary)
    return final_losses, summary


def save_model_parameters(setup, run_dir, best_iteration=None, best_loss=None):
    """Save the best theta and the target U as .npy plus a torch dict."""
    theta = setup['theta']
    U = setup['U']
    transformation_name = setup['transformation_name']
    params_dir = run_dir / "parameters"
    params_dir.mkdir(exist_ok=True)

    np.save(params_dir / "theta.npy", theta.detach().cpu().numpy())
    np.save(params_dir / "transformation_matrix.npy", U.detach().cpu().numpy())
    save_dict = {'theta': theta, 'transformation_matrix': U, 'transformation_name': transformation_name}
    if best_iteration is not None:
        save_dict['best_iteration'] = best_iteration
    if best_loss is not None:
        save_dict['best_loss'] = best_loss
    torch.save(save_dict, params_dir / "theta_state_dict.pt")

    if best_iteration is not None and best_loss is not None:
        print(f"Best model parameters (iteration {best_iteration+1}, loss={best_loss:.6f}) saved to {params_dir}")
    else:
        print(f"Model parameters saved to {params_dir}")


def visualize_unitary_comparison(config, setup, run_dir):
    """Plot |U| next to the realized |V| and save V to parameters/actual_unitary_matrix.npy."""
    theta = setup['theta']
    U_target = setup['U']
    transformation_name = setup['transformation_name']
    batch_size = setup['x'].shape[0]
    amplitude_downscale = float(config['training']['amplitude_downscale'])

    with torch.no_grad():
        A_j_evolution, _ = setup['forward'](theta, setup['hg_basis'])
        # Inputs are amplitude_downscale * identity in HG space, so dividing recovers V.
        U_actual = setup['output_hg'](A_j_evolution) / amplitude_downscale

    U_target_np = U_target.detach().cpu().numpy()
    U_actual_np = U_actual.detach().cpu().numpy()
    U_target_mag = np.abs(U_target_np)
    U_actual_mag = np.abs(U_actual_np)
    vmin = min(U_target_mag.min(), U_actual_mag.min())
    vmax = max(U_target_mag.max(), U_actual_mag.max())

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, mag, title in [
        (axes[0], U_target_mag, f'Target Unitary: {transformation_name}'),
        (axes[1], U_actual_mag, 'Actual Unitary from Simulation'),
    ]:
        im = ax.imshow(mag, cmap='viridis', aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Input Mode', fontsize=12)
        ax.set_ylabel('Output Mode', fontsize=12)
        ax.set_xticks(range(batch_size))
        ax.set_yticks(range(batch_size))
        plt.colorbar(im, ax=ax).set_label('|U|', fontsize=12)

    fidelity = torch.mean(torch.abs(U_actual.conj() * U_target)).item()
    mse = torch.mean(torch.abs(U_actual - U_target)**2).item()
    plt.suptitle(f'Unitary Transformation Comparison\nFidelity: {fidelity:.4f} | MSE: {mse:.6f}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    save_path = plots_dir / "unitary_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Unitary comparison plot saved to {save_path} (fidelity {fidelity:.4f}, MSE {mse:.6f})")

    params_dir = run_dir / "parameters"
    params_dir.mkdir(exist_ok=True)
    np.save(params_dir / "actual_unitary_matrix.npy", U_actual_np)


# =============================================================================
# Run bookkeeping
# =============================================================================

class Tee:
    """Duplicates writes to both a stream and a log file."""

    def __init__(self, stream, filepath):
        self.stream = stream
        self.file = open(filepath, 'w', encoding='utf-8', errors='replace', buffering=1)

    def write(self, data):
        try:
            self.stream.write(data)
        except UnicodeEncodeError:
            encoding = self.stream.encoding or 'utf-8'
            self.stream.write(data.encode(encoding, errors='replace').decode(encoding))
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def setup_device(config):
    use_cuda = config['device']['use_cuda'] and torch.cuda.is_available()
    if not use_cuda:
        print("Using device: cpu")
        return torch.device('cpu')
    device_id = config['device']['device_id']
    device = torch.device(f'cuda:{device_id}')
    print(f"Using device: {device} ({torch.cuda.get_device_name(device_id)}, "
          f"{torch.cuda.get_device_properties(device_id).total_memory / 1e9:.2f} GB)")
    return device


def make_run_dir(run_dir=None, name=None, base_dir='runs'):
    if run_dir:
        path = Path(run_dir)
    else:
        path = Path(base_dir) / (name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_json(path, obj):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train an XPM fiber to implement a target unitary')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Path to configuration YAML file')
    parser.add_argument('--run-dir', default=None, help='Run directory (default: runs/<wandb name or timestamp>)')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--loss-fn', default=None, choices=LOSS_NAMES, help='Overrides training.loss_fn')
    parser.add_argument('--transformation', default=None, choices=VALID_TRANSFORMATIONS,
                        help='Overrides training.transformation')
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config.setdefault('training', {})

    loss_fn = args.loss_fn or config['training'].get('loss_fn', 'basic')
    if loss_fn not in LOSS_NAMES:
        print(f"Warning: Invalid loss_fn '{loss_fn}' in config file. Using 'basic' instead.")
        loss_fn = 'basic'
    transformation = args.transformation or config['training'].get('transformation', 'identity')
    if transformation not in VALID_TRANSFORMATIONS:
        print(f"Warning: Invalid transformation '{transformation}' in config file. Using 'identity' instead.")
        transformation = 'identity'
    config['training']['loss_fn'] = loss_fn
    config['training']['transformation'] = transformation

    device = setup_device(config)
    run_dir = make_run_dir(args.run_dir, name=config.get('wandb', {}).get('name'))
    print(f"\nRun directory: {run_dir}")

    log_path = run_dir / "train.log"
    tee_out, tee_err = Tee(sys.stdout, log_path), Tee(sys.stderr, log_path)
    sys.stdout, sys.stderr = tee_out, tee_err

    with open(run_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    setup = setup_training(config, device, transformation_name=transformation)
    _, _, _, best_iteration, best_loss = train_loop(
        config, setup, run_dir, use_wandb=not args.no_wandb, loss_fn_name=loss_fn
    )

    # Skip parameter/plot saving if no finite best model was found (they would be NaN).
    if best_iteration is not None and best_iteration >= 0:
        save_model_parameters(setup, run_dir, best_iteration, best_loss)
        visualize_unitary_comparison(config, setup, run_dir)
    else:
        print("\nSkipping model-parameter save and unitary visualisation: no finite best model was recorded.")

    summary_path = run_dir / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    summary['config_path'] = str(args.config)
    atomic_write_json(summary_path, summary)

    status = summary.get('status', 'completed')
    print(f"\n{'='*80}")
    print("Training completed successfully!" if status == 'completed'
          else f"Training ended early (status='{status}'). Partial artifacts saved.")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*80}")

    sys.stdout, sys.stderr = tee_out.stream, tee_err.stream
    tee_out.close()
    tee_err.close()


if __name__ == '__main__':
    main()
