#!/bin/bash

# Submits one independent SLURM job per config in configs/analysis/trace-spm-off/
# using the regular (non-priority) GPU partition.
# These configs use the trace loss + Adam optimizer with the
# test_spm_dispersion_off flag enabled (strong pulse propagated with
# beta2_k = gamma_k = 0; initial pulse still seeded from the configured values).
# Usage: bash jobs/regular/launch_analysis_trace_spm_off.sh   (or from anywhere)

# SCRIPT_DIR = where this launcher and its run script live (jobs/regular/)
# PROJECT_DIR = repo root, two levels up; exported to SLURM so the run script
# can `cd` there before invoking python with config paths relative to root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/regular_run.sh"
RUNS_DIR="$PROJECT_DIR/runs"

mkdir -p "$RUNS_DIR"

CONFIGS=(
    configs/analysis/trace-spm-off/bs1.yaml
    configs/analysis/trace-spm-off/bs2.yaml
    configs/analysis/trace-spm-off/bs4.yaml
    configs/analysis/trace-spm-off/bs8.yaml
    configs/analysis/trace-spm-off/bs16.yaml
)

for cfg in "${CONFIGS[@]}"; do
    job_id=$(sbatch \
        --export=ALL,CONFIG="$cfg",PROJECT_DIR="$PROJECT_DIR" \
        --output="$RUNS_DIR/slurm_%j_out.txt" \
        --error="$RUNS_DIR/slurm_%j_err.txt" \
        "$JOB_SCRIPT" | awk '{print $NF}')
    echo "Submitted $cfg -> job $job_id"
done
