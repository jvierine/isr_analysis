#!/usr/bin/env bash
#SBATCH --job-name=lhw-vgate
#SBATCH --partition=node
#SBATCH --nodelist=node02
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:40:00
set -euo pipefail
root=${1:?output root}
original=${2:?original root}
export PATH=/opt/slurm/slurm-22.05.11/bin:/opt/soft/anaconda3/bin:$PATH
export LC_ALL=C LANG=C OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$root/source/syisr/lhw_ac"
conda run -n base python -m unittest test_reduction
srun --mpi=none --kill-on-bad-exit=1 /opt/soft/anaconda3/bin/python partial_voltage.py "$original" "$root/blocks" 32
conda run -n base python export_movie.py "$root/blocks" "$root/beam-geometry.h5" "$root/spectra.h5" --integration-minutes 30 --range-gates 4
date -u >"$root/COMPLETE_PARTIAL"
