"""Optional matplotlib helpers. Importing this module does not configure a backend."""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from nlse.modes import time_to_hg, hg_to_time, time_to_trunc_hg
from nlse.solvers import split_step_fourier, time_derivative_fft


def plot_inputs_and_target(Ain_j, Ain_k, A_target, t):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.plot(t.detach().cpu().numpy(), np.abs(Ain_j.detach().cpu().numpy())**2, 'b-', linewidth=2, label='|A_j|²')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Intensity')
    ax1.set_title('(Weak) Wave j')
    ax1.grid(True)
    ax2.plot(t.detach().cpu().numpy(), np.abs(Ain_k.detach().cpu().numpy())**2, 'r-', linewidth=2, label='|A_k|²')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Intensity')
    ax2.set_title('(Strong) Wave k')
    ax2.grid(True)
    ax3.plot(t.detach().cpu().numpy(), np.abs(Ain_j.detach().cpu().numpy())**2, 'b--', linewidth=2, label='|A_j|²')
    ax3.plot(t.detach().cpu().numpy(), np.abs(A_target.detach().cpu().numpy())**2, 'g-', linewidth=2, label='|A_target|²')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Intensity')
    ax3.set_title('Target vs Input Wave j')
    ax3.legend()
    ax3.grid(True)
    plt.tight_layout()
    plt.show()

def plot_intensity_evolution(A_evolution, t, Lz, Nz, wave_name='Wave'):
    """Plot the intensity evolution in 3D using PyTorch."""
    intensity_evolution = torch.abs(A_evolution)**2
    intensity_evolution = intensity_evolution.detach().cpu().numpy().T
    
    t = t.detach().cpu().numpy()
    z = torch.linspace(0, Lz, Nz+1).detach().cpu().numpy()

    T, Z = np.meshgrid(t, z)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(T, Z, intensity_evolution, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('z (km)')
    ax.set_zlabel('Intensity |A|^2')
    ax.set_title('Intensity Evolution in NLSE for ' + wave_name)
    plt.show()
    
def plot_intensity_comparison(input, output, t):
    """Plot the intensity of the input and output pulses for comparison."""
    plt.figure(figsize=(10, 5))
    plt.plot(t.cpu() if hasattr(t, 'cpu') else t, torch.abs(input).detach().cpu().numpy()**2, label='Input')
    plt.plot(t.cpu() if hasattr(t, 'cpu') else t, torch.abs(output).detach().cpu().numpy()**2, label='Output')
    plt.title('Intensity Comparison')
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid()
    plt.show()


def plot_spectrum(A0, Nt, Lt):
    """Plot the spectrum of the initial pulse."""
    dt = Lt / Nt
    freq = torch.fft.fftfreq(Nt, dt, device=A0.device)
    freq_shifted = torch.fft.fftshift(freq)
    A0_ft = torch.fft.fft(A0)
    A0_ft_shifted = torch.fft.fftshift(A0_ft)
    plt.figure(figsize=(10, 5))
    plt.plot(freq_shifted.detach().cpu().numpy(), torch.abs(A0_ft_shifted.detach()).cpu().numpy())
    plt.title('Spectrum of the Pulse')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Intensity')
    plt.show()


def plot_spectrum_comparison(input, output, Nt, Lt):
    """Plot the spectrum of the initial pulse and the output pulse for comparison."""
    dt = Lt / Nt
    freq = torch.fft.fftfreq(Nt, dt, device=input.device)
    freq_shifted = torch.fft.fftshift(freq)
    input_ft = torch.fft.fft(input.detach())
    input_ft_shifted = torch.fft.fftshift(input_ft)
    output_ft = torch.fft.fft(output.detach())
    output_ft_shifted = torch.fft.fftshift(output_ft)
    plt.figure(figsize=(10, 5))
    plt.plot(freq_shifted.cpu().numpy(), torch.abs(input_ft_shifted).cpu().numpy(), label='Input')
    plt.plot(freq_shifted.cpu().numpy(), torch.abs(output_ft_shifted).cpu().numpy(), label='Output')
    plt.title('Spectrum of the Pulse')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()


def simulate_and_plot(A0, dz, Nz, beta2, gamma, Lt):
    """Simulate the NLSE using the Split-Step Fourier Method and plot the intensity evolution."""
    A_evolution = split_step_fourier(A0, dz, Nz, beta2, gamma, Lt)
    t = torch.linspace(0, Lt, len(A0), device=A0.device)
    plot_intensity_evolution(torch.abs(A_evolution)**2, t, Nz*dz, Nz)


def plot_time_derivative(A0, Nt, Lt):
    """Plot the time derivative of the pulse."""
    t = torch.linspace(0, Lt, Nt, device=A0.device)
    dA_dt = time_derivative_fft(A0, Nt, Lt)
    plt.figure(figsize=(10, 5))
    plt.plot(t.cpu().numpy(), dA_dt.cpu().numpy(), color="m")
    plt.title('Time Derivative of the Pulse')
    plt.xlabel('Time')
    plt.ylabel('dA/dt')
    plt.grid()
    plt.show()


def plot_temporal_waveform(z_plot_list, A_evolution, t, Lz, Nz):
    """Plot the intensity waveform at specific z values."""
    plt.figure(figsize=(10, 5))
    for z_plot in z_plot_list:
        index = int(z_plot / Lz * Nz)
        plt.plot(t.cpu().numpy(), torch.abs(A_evolution[:, index]).cpu().numpy()**2, label='z= %.f' % z_plot)
    plt.title('Intensity Waveform')
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()
    
def plot_cowave_evolution(A_j_evolution, A_k_evolution, t, Lz, Nz):
    # Create 2D intensity evolution plots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Convert time and distance arrays for plotting
    t_plot = t.cpu().numpy()
    z_plot = np.linspace(0, Lz, Nz+1)  # Nz+1 points to match the evolution array

    # Create meshgrids for 2D plotting
    T, Z = np.meshgrid(t_plot, z_plot)

    # Plot Wave j (weak pulse) intensity evolution
    intensity_j = torch.abs(A_j_evolution)**2
    im1 = axes[0].pcolormesh(T, Z, intensity_j.cpu().numpy().T, shading='auto', cmap='hot')
    axes[0].set_title('Channel 1 (Weak pulse) |A_j|²', fontsize=14)
    axes[0].set_xlabel('Retarded Time T [ps]')
    axes[0].set_ylabel('Propagation Distance z [km]')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Intensity')

    # Plot Wave k (strong trap) intensity evolution
    intensity_k = torch.abs(A_k_evolution)**2
    im2 = axes[1].pcolormesh(T, Z, intensity_k.cpu().numpy().T, shading='auto', cmap='plasma')
    axes[1].set_title('Channel 2 (Strong Trap) |A_k|²', fontsize=14)
    axes[1].set_xlabel('Retarded Time T [ps]')
    axes[1].set_ylabel('Propagation Distance z [km]')
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Intensity')

    plt.tight_layout()
    plt.show()
    
def plot_transformation_matrix(U, transformation_name):
    """
    Visualize the transformation matrix U in terms of its magnitude, real, and imaginary parts.

    Args:
        U: torch.Tensor (complex) transformation matrix.
        transformation_name: str, name of the transformation to display in titles.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Magnitude
    im0 = axes[0].imshow(torch.abs(U).cpu().numpy(), cmap='viridis', aspect='equal')
    axes[0].set_title(f'|U| (Magnitude)\n{transformation_name}', fontsize=12)
    axes[0].set_xlabel('Input Mode')
    axes[0].set_ylabel('Output Mode')
    plt.colorbar(im0, ax=axes[0])

    # Real part
    im1 = axes[1].imshow(torch.real(U).cpu().numpy(), cmap='RdBu', aspect='equal', vmin=-1, vmax=1)
    axes[1].set_title(f'Re(U) (Real Part)\n{transformation_name}', fontsize=12)
    axes[1].set_xlabel('Input Mode')
    axes[1].set_ylabel('Output Mode')
    plt.colorbar(im1, ax=axes[1])

    # Imaginary part
    im2 = axes[2].imshow(torch.imag(U).cpu().numpy(), cmap='RdBu', aspect='equal', vmin=-1, vmax=1)
    axes[2].set_title(f'Im(U) (Imaginary Part)\n{transformation_name}', fontsize=12)
    axes[2].set_xlabel('Input Mode')
    axes[2].set_ylabel('Output Mode')
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle(f'Transformation Matrix: {transformation_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_mode_comparison(y, t, A_j_evolution, plot_percent=0.3, transformation_name=""):
    """Create a grid plot showing all modes: target vs final output."""
    B = y.shape[0]

    # Calculate grid dimensions
    n_cols = 4
    n_rows = (B + n_cols - 1) // n_cols

    # Calculate time window for plotting
    total_points = len(t)
    center_points = int(total_points * plot_percent)
    start_idx = (total_points - center_points) // 2
    end_idx = start_idx + center_points
    t_plot = t[start_idx:end_idx].cpu().numpy()

    # Get final outputs
    final_outputs = A_j_evolution[:, :, -1].detach()

    # Create subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for mode_idx in range(B):
        ax = axes[mode_idx]

        # Extract data for this mode (use abs for complex values)
        target_intensity = np.abs(y[mode_idx].detach().cpu().numpy())[start_idx:end_idx] ** 2
        output_intensity = np.abs(final_outputs[mode_idx].detach().cpu().numpy())[start_idx:end_idx] ** 2

        # Plot target and output
        ax.plot(t_plot, target_intensity, 'r--', linewidth=2, label='Target', alpha=0.7)
        ax.plot(t_plot, output_intensity, 'b-', linewidth=2, label='Output', alpha=0.8)

        # Calculate and display MSE
        mse = np.mean((target_intensity - output_intensity) ** 2)
        ax.set_title(f'Mode {mode_idx}\nMSE: {mse:.2e}', fontsize=10)
        ax.set_xlabel('Time', fontsize=9)
        ax.set_ylabel('Intensity |A|²', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(B, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Target vs Final Output for All {B} Modes\nTransformation: {transformation_name}',
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.show()


def plot_mode_comparison_hg(y, A_j_evolution, hg_basis, dt, num_modes=None, transformation_name=""):
    """
    Show a mode-by-mode comparison of the TARGET and OUTPUT in the HG basis (as |coeff|^2),
    using a grid plot (4x4) where each subplot shows a single mode: target (red) vs output (blue).
    """
    # y: (B, Nt), A_j_evolution: (B, Nt, Nz+1)
    B = y.shape[0]
    if num_modes is None:
        num_modes = B

    # Calculate grid dimensions
    n_cols = 4
    n_rows = (B + n_cols - 1) // n_cols

    # Compute coefficients
    coeff_target = []
    coeff_output = []
    for mode_idx in range(B):
        c_target = time_to_trunc_hg(y[mode_idx].detach(), hg_basis, dt, num_modes).cpu().numpy()
        c_output = time_to_trunc_hg(A_j_evolution[mode_idx, :, -1].detach(), hg_basis, dt, num_modes).cpu().numpy()
        coeff_target.append(c_target)
        coeff_output.append(c_output)
    coeff_target = np.stack(coeff_target)  # (B, num_modes)
    coeff_output = np.stack(coeff_output)  # (B, num_modes)

    # Prepare subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    # Plot for each mode its HG coefficient stem plots (target vs output)
    mode_indices = np.arange(num_modes)
    for mode_idx in range(B):
        ax = axes[mode_idx]
        ct = np.abs(coeff_target[mode_idx])**2
        co = np.abs(coeff_output[mode_idx])**2
        ax.stem(mode_indices, ct, basefmt=' ', linefmt='r-', markerfmt='ro', label='Target')
        ax.stem(mode_indices, co, basefmt=' ', linefmt='b-', markerfmt='bo', label='Output')
        # Compute MSE for this mode in HG coef
        mse_hg = np.mean(co - ct)
        ax.set_title(f"Mode {mode_idx}\nHG Intensity MSE: {mse_hg:.2e}", fontsize=10)
        ax.set_xlabel("Truncated HG Mode n", fontsize=9)
        ax.set_ylabel("|cₙ|²", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(B, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"Target vs Output HG Coefficient Intensities\nfor All {B} Modes (Transformation: {transformation_name})", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.show()
    
    
def plot_mode_hg_coeffs(y, A_j_evolution, hg_basis, dt, num_modes=None, threshold_percent = 0.5, transformation_name=""):
    # y: (B, Nt), A_j_evolution: (B, Nt, Nz+1)
    B = y.shape[0]
    if num_modes is None:
        num_modes = B

    coeff_target = []
    coeff_output = []
    for mode_idx in range(B):
        c_target = time_to_trunc_hg(y[mode_idx].detach(), hg_basis, dt, num_modes).cpu().numpy()
        c_output = time_to_trunc_hg(A_j_evolution[mode_idx, :, -1].detach(), hg_basis, dt, num_modes).cpu().numpy()
        coeff_target.append(c_target)
        coeff_output.append(c_output)
    coeff_target = np.stack(coeff_target)  # (B, num_modes) — row i = output coefficients for input mode i
    coeff_output = np.stack(coeff_output)  # (B, num_modes)

    coeff_indices = np.arange(num_modes)
    colors = plt.cm.tab20(np.arange(B))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Magnitude plot ---
    ax1 = axes[0]
    for i in range(B):
        row_max = np.max(np.abs(coeff_target[i, :]))
        if row_max == 0:
            continue
        significant = np.abs(coeff_target[i, :]) > threshold_percent * row_max #* note: filtering is based on target coefficients
        js = coeff_indices[significant]

        ax1.scatter(js, np.abs(coeff_target[i, js]), s=60, c=[colors[i]], marker='o', alpha=0.7)
        ax1.scatter(js, np.abs(coeff_output[i, js]), s=60, c=[colors[i]], marker='x', alpha=0.7)

    ax1.set_xlabel('HG Coefficient Index j')
    ax1.set_ylabel('Coefficient Magnitude')
    ax1.set_title('Magnitude (color = input mode i, x-pos = coeff j)')
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(coeff_indices)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Target', markersize=8, alpha=0.7),
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='gray', markeredgecolor='gray', label='Output', markersize=8, alpha=0.7),
    ] + [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], label=f'mode {i}', markersize=8)
        for i in range(B)
    ]
    ax1.legend(handles=legend_elements, fontsize=8, ncol=2)

    # --- Phase plot ---
    ax2 = axes[1]
    target_phase = np.angle(coeff_target)
    output_phase = np.angle(coeff_output)

    for i in range(B):
        row_max = np.max(np.abs(coeff_target[i, :]))
        if row_max == 0:
            continue
        significant = np.abs(coeff_target[i, :]) > 0.5 * row_max
        js = coeff_indices[significant]

        ax2.scatter(js, target_phase[i, js], s=60, c=[colors[i]], marker='o', alpha=0.7)
        ax2.scatter(js, output_phase[i, js], s=60, c=[colors[i]], marker='x', alpha=0.7)

    ax2.set_xlabel('HG Coefficient Index j')
    ax2.set_ylabel('Phase (radians)')
    ax2.set_title('Phase (color = input mode i, x-pos = coeff j)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(coeff_indices)

    legend_elements_phase = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Target', markersize=8, alpha=0.7),
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='gray', markeredgecolor='gray', label='Output', markersize=8, alpha=0.7),
    ] + [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], label=f'mode {i}', markersize=8)
        for i in range(B)
    ]
    ax2.legend(handles=legend_elements_phase, fontsize=8, ncol=2)

    plt.suptitle(f'HG Coefficient Comparison - {transformation_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    coeff_error = np.abs(coeff_target - coeff_output)
    print(f"\nHG Coefficient Error Metrics:")
    print(f"  Mean coefficient error: {coeff_error.mean().item():.2e}")
    print(f"  Max coefficient error: {coeff_error.max().item():.2e}")


def analyze_pulse_in_hg_basis(pulse, hg_basis, t, pulse_name="Pulse"):
    
    dt = t[1]-t[0]
    hg_coefficients = time_to_hg(pulse, hg_basis, dt)

    # Reconstruct the pulse from the truncated HG basis
    reconstructed_pulse = hg_to_time(hg_coefficients, hg_basis)
    
    # Plot the HG coefficients
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: HG coefficients
    mode_indices = torch.arange(hg_basis.shape[0])
    ax1.stem(mode_indices.cpu().numpy(), hg_coefficients.cpu().numpy(), basefmt=' ')
    ax1.set_xlabel('HG Mode Index n')
    ax1.set_ylabel('Coefficient Amplitude')
    ax1.set_title(f'Hermite-Gauss Coefficients of {pulse_name}')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Original vs reconstructed pulse
    ax2.plot(t.cpu().numpy(), pulse.cpu().numpy(), 'b-', linewidth=2, label=f'Original {pulse_name}')
    ax2.plot(t.cpu().numpy(), reconstructed_pulse.cpu().numpy(), 'r--', linewidth=2, label=f'Reconstructed (N={hg_basis.shape[0]} modes)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'Original vs Reconstructed {pulse_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Reconstruction error
    error = pulse - reconstructed_pulse
    ax3.plot(t.cpu().numpy(), error.cpu().numpy(), 'g-', linewidth=1)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Error')
    ax3.set_title('Reconstruction Error')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"Original pulse energy: {torch.trapz(pulse**2, dx=dt):.6f}")
    print(f"Reconstructed pulse energy: {torch.trapz(reconstructed_pulse**2, dx=dt):.6f}")
    print(f"RMS error: {torch.sqrt(torch.mean(error**2)):.6f}")
    print(f"Max coefficient magnitude: {torch.max(torch.abs(hg_coefficients)):.6f}")
    print(f"Number of significant coefficients (>1% of max): {torch.sum(torch.abs(hg_coefficients) > 0.01 * torch.max(torch.abs(hg_coefficients)))}")
    
    return error


def analyze(output, target, hg_basis, dt, num_modes, Lt, Nt, label1="Target", label2="Output"): # analyze the output and target pulse in the HG and Time domains
    # takes an unbatched output [Nt] and a target [Nt]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # 1. Intensity plot (plot only the middle 0.2 percent)
    intensity_output = output.abs()**2
    intensity_target = target.abs()**2
    t_plot = np.linspace(-Lt/2, Lt/2, Nt)
    mse_time = F.mse_loss(intensity_output, intensity_target).item()
    percent = 0.2  # 0.2 percent as fraction
    Nt_middle = int(Nt * percent)
    left_idx = (Nt - Nt_middle) // 2
    right_idx = left_idx + Nt_middle
    axs[0].plot(t_plot[left_idx:right_idx], intensity_target[left_idx:right_idx], label=f'{label1} |A|^2', color = 'r')
    axs[0].plot(t_plot[left_idx:right_idx], intensity_output[left_idx:right_idx], label=f'{label2} |A|^2', linestyle='--', color = 'b')
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Intensity")
    axs[0].set_title(f"Pulse Intensity (MSE={mse_time:.3e})")
    axs[0].legend()
    axs[0].grid(True)

    # 2. HG Coefficient plot (now as a stem plot)
    coeff_output = time_to_trunc_hg(output, hg_basis, dt, num_modes).numpy()
    coeff_target = time_to_trunc_hg(target, hg_basis, dt, num_modes).numpy()
    coeff_output_int = np.abs(coeff_output)**2
    coeff_target_int = np.abs(coeff_target)**2
    mse_hg = np.mean(coeff_output_int - coeff_target_int)
    mode_indices = np.arange(num_modes)
    axs[1].stem(mode_indices, coeff_target_int, label=f'{label1} HG Coeff.', basefmt=' ', linefmt='r-', markerfmt='ro')
    axs[1].stem(mode_indices, coeff_output_int, label=f'{label2} HG Coeff.', basefmt=' ', linefmt='b-', markerfmt='bo')
    axs[1].set_xlabel("Truncated HG Mode n")
    axs[1].set_ylabel("Coeff Amplitude")
    axs[1].set_title(f"Truncated HG Coefficient \"Intensity\" (MSE={mse_hg:.3e})")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
