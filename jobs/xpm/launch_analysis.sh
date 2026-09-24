#!/bin/bash

# Submits one independent SLURM job per analysis config on the regular GPU partition.
# Usage: bash jobs/xpm/launch_analysis.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/regular_run.sh"
RUNS_DIR="$PROJECT_DIR/runs"

mkdir -p "$RUNS_DIR"

CONFIGS=(
    approaches/xpm_unitary/configs/analysis/hg-phase/bs1.yaml
    approaches/xpm_unitary/configs/analysis/hg-phase/bs2.yaml
    approaches/xpm_unitary/configs/analysis/hg-phase/bs4.yaml
    approaches/xpm_unitary/configs/analysis/hg-phase/bs8.yaml
    approaches/xpm_unitary/configs/analysis/hg-phase/bs16.yaml
    approaches/xpm_unitary/configs/analysis/trace/bs1.yaml
    approaches/xpm_unitary/configs/analysis/trace/bs2.yaml
    approaches/xpm_unitary/configs/analysis/trace/bs4.yaml
    approaches/xpm_unitary/configs/analysis/trace/bs8.yaml
    approaches/xpm_unitary/configs/analysis/trace/bs16.yaml
)

for cfg in "${CONFIGS[@]}"; do
    job_id=$(sbatch \
        --export=ALL,CONFIG="$cfg",PROJECT_DIR="$PROJECT_DIR" \
        --output="$RUNS_DIR/slurm_%j_out.txt" \
        --error="$RUNS_DIR/slurm_%j_err.txt" \
        "$JOB_SCRIPT" | awk '{print $NF}')
    echo "Submitted $cfg -> job $job_id"
done
