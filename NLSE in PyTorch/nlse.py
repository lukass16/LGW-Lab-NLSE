# nlse.py

# General imports
import numpy as np
import matplotlib.pyplot as plt
import os

# Torch imports
import torch


# Update the matplotlib configuration parameters
plt.rcParams.update({'font.size': 14})

# Fix OpenMP Environment Error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


"""" Define Operators and Functions """

def dispersion_operator(beta2, dt, Nt, dz):
    """Calculate the dispersion operator for the linear step in Fourier space using PyTorch."""
    omega = 2 * torch.pi * torch.fft.fftfreq(Nt, dt)
    return torch.exp(0.5j * beta2 * omega**2 * dz)


def nonlinear_operator(gamma, A, dz):
    """Apply the nonlinear operator for the nonlinear step in real space using PyTorch."""
    return A * torch.exp(1j * gamma * torch.abs(A)**2 * dz)


def split_step_fourier(A0, dz, Nz, beta2, gamma, Lt):
    """Implement the Split-Step Fourier Method using PyTorch."""
    A = A0.clone()
    Nt = len(A)
    dt = Lt / Nt
    linear_op = dispersion_operator(beta2, dt, Nt, dz)
    A_evolution = torch.zeros((Nt, Nz+1), dtype=torch.complex64)
    A_evolution[:, 0] = A0  # Set first column of A_evolution to be the initial pulse

    for i in range(Nz):
        A = nonlinear_operator(gamma, A, dz)  # Apply nonlinear operator
        A_ft = torch.fft.fft(A)  # Transform to Fourier space
        A_ft *= linear_op  # Apply linear operator
        A = torch.fft.ifft(A_ft)  # Transform back to real space

        A_evolution[:, i+1] = A

    return A_evolution

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

"""" Plotting functions """

def plot_intensity_evolution(A_evolution, t, Lz, Nz):
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
    ax.set_title('Intensity Evolution in NLSE')
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
    
    

