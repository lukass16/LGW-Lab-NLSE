#!/bin/bash

# Submits one independent SLURM job per config in configs/analysis/
# Usage: bash launch_analysis.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/priority_run.sh"

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
    job_id=$(sbatch --export=ALL,CONFIG="$cfg" "$JOB_SCRIPT" | awk '{print $NF}')
    echo "Submitted $cfg -> job $job_id"
done
