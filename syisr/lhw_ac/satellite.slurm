#!/usr/bin/env bash
#SBATCH --job-name=lhw-sat
#SBATCH --partition=node
#SBATCH --nodelist=node02
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
root=${1:?run root}
cd "$root/satellite-calibration"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export RAW_RUN_ROOT="$root"
export SATELLITE_CANDIDATE_COUNT=${2:-30}
/opt/slurm/slurm-22.05.11/bin/srun --mpi=none --kill-on-bad-exit=1 --output=beam-%t.log bash -c '/opt/soft/anaconda3/bin/python satellite_voltage.py "$RAW_RUN_ROOT" --beam "$SLURM_PROCID" --count "$SATELLITE_CANDIDATE_COUNT"'
