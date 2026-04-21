#!/bin/bash

# Submits one independent SLURM job per config in configs/analysis/
# using the regular (non-priority) GPU partition.
# Usage: bash launch_analysis.sh

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$PROJECT_DIR/regular_run.sh"
RUNS_DIR="$PROJECT_DIR/runs"

mkdir -p "$RUNS_DIR"

CONFIGS=(
    configs/analysis/hg-phase/bs1.yaml
    configs/analysis/hg-phase/bs2.yaml
    configs/analysis/hg-phase/bs4.yaml
    configs/analysis/hg-phase/bs8.yaml
    configs/analysis/hg-phase/bs16.yaml
    configs/analysis/trace/bs1.yaml
    configs/analysis/trace/bs2.yaml
    configs/analysis/trace/bs4.yaml
    configs/analysis/trace/bs8.yaml
    configs/analysis/trace/bs16.yaml
)

for cfg in "${CONFIGS[@]}"; do
    job_id=$(sbatch \
        --export=ALL,CONFIG="$cfg",PROJECT_DIR="$PROJECT_DIR" \
        --output="$RUNS_DIR/slurm_%j_out.txt" \
        --error="$RUNS_DIR/slurm_%j_err.txt" \
        "$JOB_SCRIPT" | awk '{print $NF}')
    echo "Submitted $cfg -> job $job_id"
done
