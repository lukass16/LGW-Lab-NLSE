"""
Test script to verify GPU compatibility of nlse.py
Run this to ensure all functions work correctly on both CPU and GPU
"""

import torch
import numpy as np
from nlse import *

def test_gpu_compatibility():
    """Test that all major functions work on GPU (if available)"""
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    print("-" * 60)
    
    # Test parameters
    Lz = 1e-3
    Nz = 50
    dz = Lz / Nz
    Lt = 10
    Nt = 1024
    dt = Lt / Nt
    beta2 = -10
    gamma = 1
    tau = 0.1
    N_modes = 20
    
    # Test 1: Create time grid on device
    print("\n1. Testing time grid creation...")
    t = torch.linspace(-Lt/2, Lt/2, Nt, dtype=torch.float32, device=device)
    print(f"   [OK] Time grid created on {t.device}")
    
    # Test 2: Create HG basis
    print("\n2. Testing HG basis creation...")
    hg_basis = get_hg_basis(N_modes, t, tau)
    print(f"   [OK] HG basis created on {hg_basis.device}")
    
    # Test 3: Create initial pulse
    print("\n3. Testing initial pulse creation...")
    A0 = torch.exp(-t**2 / tau**2).to(torch.complex64)
    print(f"   [OK] Initial pulse created on {A0.device}")
    
    # Test 4: Test dispersion operator
    print("\n4. Testing dispersion operator...")
    A_disp = dispersion_operator(A0, beta2, Nt, dt, dz)
    print(f"   [OK] Dispersion operator output on {A_disp.device}")
    
    # Test 5: Test nonlinear operator
    print("\n5. Testing nonlinear operator...")
    A_nonlin = nonlinear_operator(gamma, A0, dz)
    print(f"   [OK] Nonlinear operator output on {A_nonlin.device}")
    
    # Test 6: Test split-step Fourier method
    print("\n6. Testing split-step Fourier method...")
    A_evolution = split_step_fourier(A0, dz, Nz, beta2, gamma, Lt)
    print(f"   [OK] Evolution computed, shape: {A_evolution.shape}, device: {A_evolution.device}")
    
    # Test 7: Test XPM operators
    print("\n7. Testing XPM operators...")
    A0_j = A0 * 0.1  # Weak pulse
    A0_k = A0 * 1.0  # Strong pulse
    A_j_evolution, A_k_evolution = split_step_fourier_xpm(
        A0_j, A0_k, dz, Nz, beta2, beta2, gamma, gamma, Lt
    )
    print(f"   [OK] XPM evolution computed")
    print(f"     - A_j on {A_j_evolution.device}, shape: {A_j_evolution.shape}")
    print(f"     - A_k on {A_k_evolution.device}, shape: {A_k_evolution.shape}")
    
    # Test 8: Test batched XPM
    print("\n8. Testing batched XPM...")
    B = 8
    x = get_hg_basis(B, t, tau) * 0.1
    A_j_batch, A_k_batch = split_step_fourier_xpm_batch(
        x, A0_k, dz, Nz, beta2, beta2, gamma, gamma, Lt
    )
    print(f"   [OK] Batched XPM computed")
    print(f"     - A_j_batch on {A_j_batch.device}, shape: {A_j_batch.shape}")
    print(f"     - A_k_batch on {A_k_batch.device}, shape: {A_k_batch.shape}")
    
    # Test 9: Test HG transformations
    print("\n9. Testing HG transformations...")
    pulse = torch.exp(-t**2 / tau**2)
    hg_coeffs = time_to_hg(pulse, hg_basis, dt)
    pulse_reconstructed = hg_to_time(hg_coeffs, hg_basis)
    print(f"   [OK] HG transform: coefficients on {hg_coeffs.device}")
    print(f"   [OK] Inverse HG transform: pulse on {pulse_reconstructed.device}")
    
    # Test 10: Test energy calculation
    print("\n10. Testing energy calculation...")
    energy = get_energy(A0, dt)
    print(f"    [OK] Energy calculated: {energy.item():.6f}")
    
    # Test 11: Test FFT utilities
    print("\n11. Testing FFT utilities...")
    freqs = get_freqs(Nt, Lt, device=device)
    A_fft = fft(A0)
    A_ifft = ifft(A_fft)
    print(f"    ✓ Frequencies on {freqs.device}")
    print(f"    ✓ FFT output on {A_fft.device}")
    print(f"    ✓ IFFT output on {A_ifft.device}")
    
    # Memory usage (if CUDA)
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1e9:.4f} GB")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_gpu_compatibility()

