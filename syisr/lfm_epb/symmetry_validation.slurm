#!/usr/bin/env bash
#SBATCH --job-name=lfm-symmetry
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
code=${1:?code}; root=${2:?root}
((SLURM_NTASKS<=16)) || exit 1
mpi=/opt/soft/anaconda3/envs/py311
cd "$code"
"$mpi/bin/mpirun" -np "$SLURM_NTASKS" "$mpi/bin/python" syisr/lfm_epb/symmetry_validation.py "$root"
