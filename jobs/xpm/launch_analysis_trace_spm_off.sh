#!/bin/bash

# Trace-loss + Adam sweep with test_spm_dispersion_off enabled.
# Usage: bash jobs/xpm/launch_analysis_trace_spm_off.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/regular_run.sh"
RUNS_DIR="$PROJECT_DIR/runs"

mkdir -p "$RUNS_DIR"

CONFIGS=(
    approaches/xpm_unitary/configs/analysis/trace-spm-off/bs1.yaml
    approaches/xpm_unitary/configs/analysis/trace-spm-off/bs2.yaml
    approaches/xpm_unitary/configs/analysis/trace-spm-off/bs4.yaml
    approaches/xpm_unitary/configs/analysis/trace-spm-off/bs8.yaml
    approaches/xpm_unitary/configs/analysis/trace-spm-off/bs16.yaml
)

for cfg in "${CONFIGS[@]}"; do
    job_id=$(sbatch \
        --export=ALL,CONFIG="$cfg",PROJECT_DIR="$PROJECT_DIR" \
        --output="$RUNS_DIR/slurm_%j_out.txt" \
        --error="$RUNS_DIR/slurm_%j_err.txt" \
        "$JOB_SCRIPT" | awk '{print $NF}')
    echo "Submitted $cfg -> job $job_id"
done
