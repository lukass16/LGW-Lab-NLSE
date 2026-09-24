# XPM unitary training

Find a strong pulse A_k such that, through XPM in a fiber, every weak input HG mode comes out as the target unitary U says it should.

```bash
python -m approaches.xpm_unitary.train \
  --config approaches/xpm_unitary/configs/fiber/config.yaml \
  --transformation identity --loss-fn hg_phase    # both optional; override the config
python -m approaches.xpm_unitary.analyze --run-dir runs/<name>
```

## What one training step does

1. **Inputs.** The weak inputs are the first B HG modes (`batch_size`), scaled by `amplitude_downscale`.
2. **Parameters.** theta holds the HG coefficients of the strong pulse. It starts as a soliton.
3. **Forward pass.** All B inputs are propagated together with `split_step_fourier_xpm_batch`, all sharing the same strong pulse.
4. **Realized matrix.** Each output is projected onto the first B HG modes. Row i of V is the output for input mode i.
5. **Loss.** The objective is `w_mse * loss(V, U) + w_pen * penalty`. The penalty keeps the strong pulse inside the central `mask_percent` of the time window.

Every iteration also logs three loss-independent metrics, so runs that used different losses can be compared:

- `eval_hg_fidelity`
- `eval_trace_fidelity`
- `eval_frobenius_norm`

The best theta is kept and saved.

Optional switches (config `training:` section):

- `optimizer: adam | sgd | lbfgs`
- `learning_schedule: cosine_warm_restarts`
- `curriculum_learning: true`: uses `trace`, then switches to `hg_phase` at `curriculum_switch`, optionally with a different `curriculum_optimizer`.

## Losses (`nlse/losses.py`)

Ordered from weakest to strictest. Here ⟨V_i, U_i⟩ = Σ_j V̄_ij U_ij.

| `loss_fn` | Definition | Sees | Blind to |
|---|---|---|---|
| `basic` | mean over i, t of (\|A_j,i(t)\|² − \|y_i(t)\|²)² | output intensity in time | all phases |
| `hg` | (1/B²) Σ_ij (\|V_ij\|² − \|U_ij\|²)² | power in every element | all phases |
| `normalized_hg` | `hg` after normalizing each row | how each row's power is distributed | phases, row norms |
| `fid_energy` | −(1/B) Σ_i (Σ_j \|V_ij\| \|U_ij\|)² | magnitude overlap per row | all phases |
| `fid_phase` | −(1/B) Σ_i \|⟨V_i, U_i⟩\|² | relative phases within a row | one phase per row |
| `trace` | −\|Tr(V†U) / B\|² | relative phases between rows | one global phase |
| `hg_phase` | (1/B²) ‖V − U‖²_F | every element, with phase | nothing |

For a diagonal target such as the identity, each row has only one nonzero entry, so `fid_phase` is effectively phase-blind. Use `trace` or `hg_phase` there.

## Configs

- `configs/fiber/`: identity, beamsplitter, permutation and arbitrary targets
- `configs/analysis/`: batch-size and loss ablations
- `configs/hg-loss/`: optimizer and learning-rate sweeps

SLURM launchers are in `jobs/xpm/`.

## Output: `runs/<name>/`

```
config.yaml  train.log  losses.json  summary.json
checkpoints/best_checkpoint.pt
parameters/theta.npy  transformation_matrix.npy  actual_unitary_matrix.npy
plots/unitary_comparison.png
```
