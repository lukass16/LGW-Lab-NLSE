# LGW-Lab-NLSE

PyTorch split-step Fourier solvers for the nonlinear Schrödinger equation (NLSE) and cross-phase modulation (XPM), used to inverse-design strong pulses that make a fiber or ring resonator act as a target unitary on Hermite–Gauss (HG) modes.

## Layout

| Path | Contents |
|---|---|
| `nlse/` | The library: solvers, HG modes, pulses, losses, plotting |
| `approaches/xpm_unitary/` | Fiber XPM → target unitary: `train.py`, `analyze.py`, configs |
| `approaches/ring_resonator/` | Ring-resonator variants: `train_scaling.py`, `train_arbitrary.py`, configs |
| `jobs/` | SLURM launchers |
| `examples/` | Curated notebooks |
| `notebooks/` | Scratch work |
| `archive/` | Old code, notebooks and frozen results |

## Setup

```bash
pip install -e ".[train]"
```

## Run

Run from the repo root:

```bash
python -m approaches.xpm_unitary.train --config approaches/xpm_unitary/configs/fiber/config.yaml
python -m approaches.xpm_unitary.analyze --run-dir runs/<name>

python -m approaches.ring_resonator.train_scaling --config approaches/ring_resonator/configs/config_ring_scaling_small.yaml
python -m approaches.ring_resonator.train_arbitrary --config approaches/ring_resonator/configs/config_ring_arbitrary_trace.yaml
```

Output goes to `runs/<name>/` (gitignored). Set `WANDB_API_KEY` in your environment, or pass `--no-wandb`.

How XPM training works, and which loss to pick, is described in [`approaches/xpm_unitary/README.md`](approaches/xpm_unitary/README.md).

## Library

```python
from nlse import split_step_fourier_xpm_batch, get_hg_basis, time_to_hg, hg_to_time, strong_soliton
from nlse.losses import trace_loss, hg_phase_loss
```
