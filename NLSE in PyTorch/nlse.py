# nlse.py

# General imports
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Torch imports
import torch


# Update the matplotlib configuration parameters
plt.rcParams.update({'font.size': 14})

# Fix OpenMP Environment Error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


"""" Define Operators and Functions """
def dispersion_operator(A, beta2, Nt, dt, dz):
    """Apply the dispersion operator for the linear step in Fourier space using PyTorch."""
    omega = 2 * torch.pi * torch.fft.fftfreq(Nt, dt)
    linear_op = torch.exp(0.5j * beta2 * omega**2 * dz)
    A_ft = torch.fft.fft(A)  # Transform to Fourier space
    A_ft *= linear_op  # Apply linear operator
    return torch.fft.ifft(A_ft)  # Transform back to real space
    #! note - function changed on 10/12/2025 could lead to errors in other files, if implemented directly

def nonlinear_operator(gamma, A, dz):
    """Apply the nonlinear operator for the nonlinear step in real space using PyTorch."""
    return A * torch.exp(1j * gamma * torch.abs(A)**2 * dz)

def nonlinear_operator_xpm(gamma_j, A_j, A_k, dz):
    """Apply the XPM operator for the nonlinear step in real space using PyTorch."""
    return A_j * torch.exp(1j * gamma_j * (torch.abs(A_j)**2 + 2 * torch.abs(A_k)**2) * dz)

def split_step_fourier(A0, dz, Nz, beta2, gamma, Lt):
    """Implement the Split-Step Fourier Method using PyTorch."""
    A = A0.clone()
    Nt = len(A)
    dt = Lt / Nt
    A_evolution = torch.zeros((Nt, Nz+1), dtype=torch.complex64)
    A_evolution[:, 0] = A0  # Set first column of A_evolution to be the initial pulse

    for i in range(Nz):
        A = nonlinear_operator(gamma, A, dz)  # Apply nonlinear operator
        A = dispersion_operator(A, beta2, Nt, dt, dz)  # Apply dispersion operator

        A_evolution[:, i+1] = A

    return A_evolution

def split_step_fourier_xpm(A0_j, A0_k, dz, Nz, beta2_j, beta2_k, gamma_j, gamma_k, Lt, strangsplitting = True): # A_j = weak, A_k = strong
    """Implement the Split-Step Fourier Method with XPM using PyTorch."""
    A_j = A0_j.clone()
    A_k = A0_k.clone()
    Nt = len(A_j)
    dt = Lt / Nt
    A_j_evolution = torch.zeros((Nt, Nz+1), dtype=torch.complex64)
    A_k_evolution = torch.zeros((Nt, Nz+1), dtype=torch.complex64)
    A_j_evolution[:, 0] = A0_j  # Set first column of A_j_evolution to be the initial pulse
    A_k_evolution[:, 0] = A0_k  # Set first column of A_k_evolution to be the initial pulse

    for i in range(Nz):
        # we want to preserve symmetry in both of the waves - always use the wave at i-th step to go to i+1-th step ? is this that important?
        _A_j = A_j.clone() 
        _A_k = A_k.clone()
        
        if strangsplitting:
            A_j = nonlinear_operator_xpm(gamma_j, _A_j, _A_k, dz/2)  # Apply XPM nonlinear operator
            A_j = dispersion_operator(A_j, beta2_j, Nt, dt, dz)  # Apply dispersion operator
            A_j = nonlinear_operator_xpm(gamma_j, A_j, _A_k, dz/2)  # Apply XPM nonlinear operator
            
            A_k = nonlinear_operator_xpm(gamma_k, _A_k, _A_j, dz/2)  # Apply XPM nonlinear operator
            A_k = dispersion_operator(A_k, beta2_k, Nt, dt, dz)  # Apply dispersion operator
            A_k = nonlinear_operator_xpm(gamma_k, A_k, _A_j, dz/2)  # Apply XPM nonlinear operator
            
            A_j_evolution[:, i+1] = A_j
            A_k_evolution[:, i+1] = A_k
            
        else:
            # for weak wave j
            A_j = nonlinear_operator_xpm(gamma_j, _A_j, _A_k, dz)  # Apply XPM nonlinear operator
            A_j = dispersion_operator(A_j, beta2_j, Nt, dt, dz)  # Apply dispersion operator
            A_j_evolution[:, i+1] = A_j
            
            # for strong wave k
            A_k = nonlinear_operator_xpm(gamma_k, _A_k, _A_j, dz)  # Apply XPM nonlinear operator
            A_k = dispersion_operator(A_k, beta2_k, Nt, dt, dz)  # Apply dispersion operator
            A_k_evolution[:, i+1] = A_k

    return A_j_evolution, A_k_evolution

def time_derivative_fft(A, Nt, Lt):
    """Compute the first time derivative of the pulse using FFT."""
    dt = Lt / Nt
    N = A.shape[-1]  # Number of time points
    fft_A = torch.fft.fft(A.detach())
    frequencies = torch.fft.fftfreq(N, d=dt)  # Frequency bins
    omega = 2 * torch.pi * frequencies  # Angular frequency
    fft_dA_dt = 1j * omega * fft_A  # Compute derivative in frequency domain
    dA_dt = torch.fft.ifft(fft_dA_dt).real  # Transform back to time domain
    return dA_dt

# short forms for FFT operations
def get_freqs(Nt, Lt):
    dt = Lt / Nt
    freqs = torch.fft.fftshift(torch.fft.fftfreq(Nt, dt))
    return freqs

def fft(Ain):
    return torch.fft.fftshift(torch.fft.fft(Ain))

def ifft(Ain):
    return torch.fft.ifft(torch.fft.ifftshift(Ain))
    #return torch.fft.ifft(Ain)
    
def get_energy(A, dt):
    return torch.trapz(torch.abs(A)**2, dx=torch.tensor(dt))

"""" Plotting functions """
def plot_inputs_and_target(Ain_j, Ain_k, A_target, t):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.plot(t.detach().numpy(), np.abs(Ain_j.detach().numpy())**2, 'b-', linewidth=2, label='|A_j|²')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Intensity')
    ax1.set_title('(Weak) Wave j')
    ax1.grid(True)
    ax2.plot(t.detach().numpy(), np.abs(Ain_k.detach().numpy())**2, 'r-', linewidth=2, label='|A_k|²')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Intensity')
    ax2.set_title('(Strong) Wave k')
    ax2.grid(True)
    ax3.plot(t.detach().numpy(), np.abs(Ain_j.detach().numpy())**2, 'b--', linewidth=2, label='|A_j|²')
    ax3.plot(t.detach().numpy(), np.abs(A_target.detach().numpy())**2, 'g-', linewidth=2, label='|A_target|²')
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
    intensity_evolution = intensity_evolution.detach().numpy().T
    
    t = t.detach().numpy()
    z = torch.linspace(0, Lz, Nz+1).detach().numpy()

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
    plt.plot(t, torch.abs(input).detach().numpy()**2, label='Input')
    plt.plot(t, torch.abs(output).detach().numpy()**2, label='Output')
    plt.title('Intensity Comparison')
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid()
    plt.show()


def plot_spectrum(A0, Nt, Lt):
    """Plot the spectrum of the initial pulse."""
    dt = Lt / Nt
    freq = torch.fft.fftfreq(Nt, dt)
    freq_shifted = torch.fft.fftshift(freq)
    A0_ft = torch.fft.fft(A0)
    A0_ft_shifted = torch.fft.fftshift(A0_ft)
    plt.figure(figsize=(10, 5))
    plt.plot(freq_shifted.detach().numpy(), torch.abs(A0_ft_shifted.detach()).numpy())
    plt.title('Spectrum of the Pulse')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Intensity')
    plt.show()


def plot_spectrum_comparison(input, output, Nt, Lt):
    """Plot the spectrum of the initial pulse and the output pulse for comparison."""
    dt = Lt / Nt
    freq = torch.fft.fftfreq(Nt, dt)
    freq_shifted = torch.fft.fftshift(freq)
    input_ft = torch.fft.fft(input.detach())
    input_ft_shifted = torch.fft.fftshift(input_ft)
    output_ft = torch.fft.fft(output.detach())
    output_ft_shifted = torch.fft.fftshift(output_ft)
    plt.figure(figsize=(10, 5))
    plt.plot(freq_shifted, torch.abs(input_ft_shifted).numpy(), label='Input')
    plt.plot(freq_shifted, torch.abs(output_ft_shifted).numpy(), label='Output')
    plt.title('Spectrum of the Pulse')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()


def simulate_and_plot(A0, dz, Nz, beta2, gamma, Lt):
    """Simulate the NLSE using the Split-Step Fourier Method and plot the intensity evolution."""
    A_evolution = split_step_fourier(A0, dz, Nz, beta2, gamma, Lt)
    t = torch.linspace(0, Lt, len(A0))
    plot_intensity_evolution(torch.abs(A_evolution)**2, t, Nz*dz, Nz)


def plot_time_derivative(A0, Nt, Lt):
    """Plot the time derivative of the pulse."""
    t = torch.linspace(0, Lt, Nt)
    dA_dt = time_derivative_fft(A0, Nt, Lt)
    plt.figure(figsize=(10, 5))
    plt.plot(t, dA_dt, color="m")
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
        plt.plot(t.numpy(), torch.abs(A_evolution[:, index])**2, label='z= %.f' % z_plot)
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
    

""" Hermite-Gauss HG Basis Operations"""

def hermite_gauss_stable(n, t, t0=1.0):
    """
    Stable computation of ψ_n(t; t0) using orthonormal recursion.
    """
    x = t / t0
    psi_prev = torch.zeros_like(x)
    psi_curr = (1 / (math.pi**0.25 * math.sqrt(t0))) * torch.exp(-0.5 * x**2)
    
    if n == 0:
        return psi_curr
    for k in range(1, n + 1):
        psi_next = math.sqrt(2 / k) * x * psi_curr - math.sqrt((k - 1) / k) * psi_prev
        psi_prev, psi_curr = psi_curr, psi_next
    return psi_curr

def get_hg_basis(N_modes, t, t0=1.0):
    # Precompute all HG basis functions as a 2D tensor
    # Shape: (N_modes, len(t)) where each row is ψ_n(t)
    hg_basis = torch.zeros(N_modes, len(t), dtype=torch.float64)

    for n in range(N_modes):
        hg_basis[n] = hermite_gauss_stable(n, t, t0)
        
    print(f"Precomputed {N_modes} HG basis functions on grid of {len(t)} points")
    
    return hg_basis

def time_to_hg(A, hg_basis, dt):
    # Matrix multiplication: each row of hg_basis dotted with A
    # Shape: (N_modes,) = (N_modes, len(t)) @ (len(t),)
    integrand = hg_basis * A[None, :]  # Broadcast A to match hg_basis shape
    coefficients = torch.trapz(integrand, dx=dt, dim=1)
    return coefficients

def hg_to_time(coefficients, hg_basis):
    # Matrix-vector multiplication: coefficients^T @ hg_basis
    # Shape: (len(t),) = (N_modes,) @ (N_modes, len(t))
    A = torch.sum(coefficients[:, None] * hg_basis, dim=0)
    return A

def analyze_pulse_in_hg_basis(pulse, hg_basis, t, pulse_name="Pulse"):
    
    dt = t[1]-t[0]
    hg_coefficients = time_to_hg(pulse, hg_basis, dt)

    # Reconstruct the pulse from the truncated HG basis
    reconstructed_pulse = hg_to_time(hg_coefficients, hg_basis)
    
    # Plot the HG coefficients
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: HG coefficients
    mode_indices = torch.arange(hg_basis.shape[0])
    ax1.stem(mode_indices.numpy(), hg_coefficients.numpy(), basefmt=' ')
    ax1.set_xlabel('HG Mode Index n')
    ax1.set_ylabel('Coefficient Amplitude')
    ax1.set_title(f'Hermite-Gauss Coefficients of {pulse_name}')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Original vs reconstructed pulse
    ax2.plot(t.numpy(), pulse.numpy(), 'b-', linewidth=2, label=f'Original {pulse_name}')
    ax2.plot(t.numpy(), reconstructed_pulse.numpy(), 'r--', linewidth=2, label=f'Reconstructed (N={hg_basis.shape[0]} modes)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'Original vs Reconstructed {pulse_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Reconstruction error
    error = pulse - reconstructed_pulse
    ax3.plot(t.numpy(), error.numpy(), 'g-', linewidth=1)
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
