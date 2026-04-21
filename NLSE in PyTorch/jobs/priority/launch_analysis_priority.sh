#!/bin/bash

# Submits one independent SLURM job per config in configs/analysis/
# using the priority GPU partition.
# Usage: bash jobs/priority/launch_analysis_priority.sh   (or from anywhere)

# SCRIPT_DIR = where this launcher and its run script live (jobs/priority/)
# PROJECT_DIR = repo root, two levels up; exported to SLURM so the run script
# can `cd` there before invoking python with config paths relative to root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/priority_run.sh"
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
