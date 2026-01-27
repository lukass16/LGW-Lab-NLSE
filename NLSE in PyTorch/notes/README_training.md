# XPM Unitary Operator Training Script

This directory contains a Python training script for the XPM Unitary Operator that can be run as a batch job.

## Files

- `train_xpm_unitary.py`: Main training script
- `config.yaml`: Configuration file with all training parameters
- `README_training.md`: This file

## Usage

### Basic Usage

```bash
python train_xpm_unitary.py
```

This will:
- Load parameters from `config.yaml`
- Create a unique run directory in `runs/` with timestamp
- Run the training sequence
- Save all plots, parameters, and logs to the run directory

### Command Line Options

```bash
python train_xpm_unitary.py --help
```

Options:
- `--config PATH`: Path to configuration YAML file (default: `config.yaml`)
- `--run-dir PATH`: Custom run directory path (default: auto-generated with timestamp)
- `--no-wandb`: Disable wandb logging

### Example: Custom Config

```bash
python train_xpm_unitary.py --config my_config.yaml
```

### Example: Custom Run Directory

```bash
python train_xpm_unitary.py --run-dir runs/my_experiment
```

### Example: Without Wandb

```bash
python train_xpm_unitary.py --no-wandb
```

## Output Structure

Each run creates a directory with the following structure:

```
runs/run_YYYYMMDD_HHMMSS/
├── config.yaml                    # Copy of the configuration used
├── training_parameters.txt        # Human-readable parameter summary
├── losses.json                    # Training losses (MSE, penalty, total)
├── parameters/
│   ├── theta.npy                  # Trained parameters as numpy array
│   └── theta_state_dict.pt        # Trained parameters as PyTorch state dict
└── plots/
    ├── training_loss.png          # Loss curves
    ├── initial_hg_modes.png       # Initial HG modes visualization
    ├── intensity_evolution_wave_j.png
    ├── intensity_evolution_wave_k.png
    ├── cowave_evolution.png
    └── wave_comparison.png        # Input vs Target vs Final comparison
```

## Wandb Integration

The script integrates with Weights & Biases (wandb) for experiment tracking. To use:

1. Install wandb: `pip install wandb`
2. Login: `wandb login`
3. Configure wandb settings in `config.yaml`:
   ```yaml
   wandb:
     project: "xpm-unitary-operator"
     entity: "your-username"  # Optional
     name: "experiment-name"  # Optional, auto-generated if not set
     tags: ["tag1", "tag2"]   # Optional
     notes: "Experiment description"  # Optional
   ```

If wandb is not installed, the script will continue without wandb logging.

## Batch Job Usage

The script is designed to work in batch job environments:

- Uses non-interactive matplotlib backend (`Agg`)
- All plots are saved to files (no display required)
- Command-line arguments for easy automation
- Comprehensive logging to files

### Example SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=xpm_train
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

python train_xpm_unitary.py --config config.yaml
```

## Dependencies

Required packages:
- numpy
- matplotlib
- torch
- tqdm
- pyyaml
- wandb (optional, for experiment tracking)

The script also requires the `nlse` module to be available in the Python path.

