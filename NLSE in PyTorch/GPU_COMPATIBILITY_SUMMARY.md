# GPU Compatibility Update Summary

## Overview
Both `nlse.py` and `XPM_Identity_Operator.ipynb` have been updated to be fully GPU-compatible. The code now automatically detects and uses GPU (CUDA) if available, with graceful fallback to CPU.

## Changes Made

### 1. nlse.py

#### Device Management
- **Added automatic device detection** at module level:
  ```python
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  ```

#### Updated Functions

**Operators:**
- `dispersion_operator()`: Added `device=A.device` to `torch.fft.fftfreq()` calls
- `split_step_fourier()`: Added `device=A0.device` to tensor creation
- `split_step_fourier_xpm()`: Added `device` parameter to evolution tensors
- `split_step_fourier_xpm_batch()`: Already had device support in line 123-124

**Utility Functions:**
- `time_derivative_fft()`: Added `device=A.device` to frequency tensor creation
- `get_freqs()`: Added optional `device` parameter with default fallback
- `get_energy()`: Added `device=A.device` to tensor creation
- `simulate_and_plot()`: Added `device=A0.device` to linspace

**Plotting Functions (added `.cpu()` calls before `.numpy()`):**
- `plot_inputs_and_target()`
- `plot_intensity_evolution()`
- `plot_intensity_comparison()`
- `plot_spectrum()`
- `plot_spectrum_comparison()`
- `plot_time_derivative()`
- `plot_temporal_waveform()`
- `plot_cowave_evolution()` (already had `.cpu()` calls)
- `analyze_pulse_in_hg_basis()`

**Hermite-Gauss Basis Functions:**
- `hermite_gauss_stable()`: Added `device=t.device` to `torch.zeros_like()`
- `get_hg_basis()`: Added `device=t.device` to basis tensor creation, updated print message to show device
- `time_to_hg()`: Already device-agnostic (inherits from inputs)
- `hg_to_time()`: Already device-agnostic (inherits from inputs)

### 2. XPM_Identity_Operator.ipynb

#### Cell 1 (Imports)
Added device detection and GPU information display:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

#### Cell 4 (Time Grid and HG Basis)
- Added `device=device` parameter to `torch.linspace()` for time grid

#### Cell 7 (Strong Soliton Function)
- Updated docstring to indicate GPU compatibility
- Function automatically inherits device from input tensor `t`

#### Cell 9 (Training Batches)
- Added `.cpu().numpy()` calls to plotting code
- Updated print statement to show tensor device

#### Cell 14 (Training Setup)
- Enhanced print statements to show device information
- Added `device=device` to zeros tensor in loss function
- Updated final print to show device

#### New Cells 19-20 (GPU Monitoring)
Added new cells for GPU memory monitoring:
- Markdown header for GPU Memory Usage section
- Code cell to display GPU memory usage and tensor device information

## Key Features

### Automatic Device Detection
All tensors are automatically placed on the correct device (GPU if available, CPU otherwise) by:
1. Inheriting device from input tensors
2. Using explicit `device` parameter in tensor creation functions
3. Global `device` variable for new tensor creation

### Seamless CPU/GPU Transfer
- All plotting functions automatically move tensors to CPU before converting to NumPy
- No user intervention required - works transparently

### Memory Efficient
- Tensors stay on GPU during computation
- Only moved to CPU for plotting/visualization
- Reduced memory transfers for better performance

## Benefits

1. **Speed**: GPU acceleration for FFT operations and tensor computations
2. **Transparency**: Code works identically on CPU or GPU
3. **Backward Compatible**: Existing code continues to work without modifications
4. **Informative**: Clear device information displayed during execution

## Usage

Simply run the code as before. The system will automatically:
1. Detect if CUDA/GPU is available
2. Place all tensors on GPU if available
3. Perform all computations on GPU
4. Move data to CPU only for plotting

No changes to user code required!

## Performance Tips

For GPU usage:
- Larger batch sizes (B) will benefit more from GPU acceleration
- FFT operations are significantly faster on GPU
- Keep an eye on GPU memory usage with the monitoring cell
- Use `torch.cuda.empty_cache()` if needed to free GPU memory

## Testing

To verify GPU usage:
1. Run Cell 1 - should show "Using device: cuda" if GPU available
2. Run Cell 20 - shows GPU memory usage and confirms tensor devices
3. Monitor GPU utilization using `nvidia-smi` in terminal (if on CUDA)

