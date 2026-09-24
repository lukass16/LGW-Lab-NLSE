"""PyTorch NLSE / XPM solvers, Hermite–Gauss modes, and optional plotting.

Importing this package does not configure matplotlib or print a device.
Notebooks that previously did ``from nlse import *`` still work after
``pip install -e .`` from the repo root.
"""

from nlse.solvers import (
    dispersion_operator,
    drift_dispersion_operator,
    nonlinear_operator,
    nonlinear_operator_xpm,
    split_step_fourier,
    split_step_fourier_xpm,
    drift_split_step_fourier_xpm,
    split_step_fourier_xpm_batch,
    time_derivative_fft,
    get_freqs,
    fft,
    ifft,
    get_energy,
)
from nlse.modes import (
    hermite_gauss_stable,
    get_hg_basis,
    time_to_hg,
    hg_to_time,
    time_to_trunc_hg,
)
from nlse.pulses import gaussian, strong_soliton
from nlse.losses import (
    intensity_loss,
    hg_loss,
    normalized_hg_loss,
    fid_energy_loss,
    fid_phase_loss,
    trace_loss,
    hg_phase_loss,
    HG_LOSSES,
    LOSS_NAMES,
    eval_frobenius_norm,
    eval_trace_fidelity,
    eval_hg_fidelity,
)

import torch as _torch

# Old notebooks use the ``device`` global from ``from nlse import *``.
device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

__all__ = [
    "dispersion_operator",
    "drift_dispersion_operator",
    "nonlinear_operator",
    "nonlinear_operator_xpm",
    "split_step_fourier",
    "split_step_fourier_xpm",
    "drift_split_step_fourier_xpm",
    "split_step_fourier_xpm_batch",
    "time_derivative_fft",
    "get_freqs",
    "fft",
    "ifft",
    "get_energy",
    "hermite_gauss_stable",
    "get_hg_basis",
    "time_to_hg",
    "hg_to_time",
    "time_to_trunc_hg",
    "gaussian",
    "strong_soliton",
    "intensity_loss",
    "hg_loss",
    "normalized_hg_loss",
    "fid_energy_loss",
    "fid_phase_loss",
    "trace_loss",
    "hg_phase_loss",
    "HG_LOSSES",
    "LOSS_NAMES",
    "eval_frobenius_norm",
    "eval_trace_fidelity",
    "eval_hg_fidelity",
    "device",
]

try:
    from nlse.viz import (
        plot_inputs_and_target,
        plot_intensity_evolution,
        plot_intensity_comparison,
        plot_spectrum,
        plot_spectrum_comparison,
        simulate_and_plot,
        plot_time_derivative,
        plot_temporal_waveform,
        plot_cowave_evolution,
        plot_transformation_matrix,
        plot_mode_comparison,
        plot_mode_comparison_hg,
        plot_mode_hg_coeffs,
        analyze_pulse_in_hg_basis,
        analyze,
    )
    __all__ += [
        "plot_inputs_and_target",
        "plot_intensity_evolution",
        "plot_intensity_comparison",
        "plot_spectrum",
        "plot_spectrum_comparison",
        "simulate_and_plot",
        "plot_time_derivative",
        "plot_temporal_waveform",
        "plot_cowave_evolution",
        "plot_transformation_matrix",
        "plot_mode_comparison",
        "plot_mode_comparison_hg",
        "plot_mode_hg_coeffs",
        "analyze_pulse_in_hg_basis",
        "analyze",
    ]
except ImportError:
    # matplotlib is optional for solver-only use
    pass
