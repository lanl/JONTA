#!/bin/bash -l
#SBATCH --job-name=jonta-invariant-cases
#SBATCH --constraint=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --account=m5276
#SBATCH --qos=regular
#SBATCH --output=jonta-invariant-cases-%j.out

set -euo pipefail

REPO_ROOT="${JONTA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
JONTA_ENV="${JONTA_ENV:-$REPO_ROOT/.venv}"
OUTPUT_ROOT="${INVARIANT_CASE_ROOT:-$REPO_ROOT/benchmark_results/one_d/invariant_cases_gpu}"
export JONTA_ENV OUTPUT_ROOT
mkdir -p "$OUTPUT_ROOT"

cd "$REPO_ROOT"
module load python
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
srun --nodes=4 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=closest \
  bash -lc '
    set -euo pipefail
    case_index="$SLURM_PROCID"
    electric_fields=(0 1 10 100 1000 10000 100000 1000000 10000000 100000000)
    timesteps=(5.12e-7 1.28e-7 3.2e-8 8e-9 2e-9 5e-10 1.25e-10 1e-10)
    total_cases=$((${#electric_fields[@]} * ${#timesteps[@]}))
    case_stride="$SLURM_NTASKS"
    while (( case_index < total_cases )); do
      field_index=$((case_index / ${#timesteps[@]}))
      dt_index=$((case_index % ${#timesteps[@]}))
      field="${electric_fields[$field_index]}"
      dt="${timesteps[$dt_index]}"
      case_dir="$OUTPUT_ROOT/case_e${field}_dt${dt}"
      mkdir -p "$case_dir"
      "$JONTA_ENV/bin/python" benchmarks/one_d/invariant_conservation/run.py \
        --electric-fields "$field" \
        --dts "$dt" \
        --output-dir "$case_dir"
      case_index=$((case_index + case_stride))
    done
  '

echo "case outputs: $OUTPUT_ROOT"
