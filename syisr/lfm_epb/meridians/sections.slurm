#!/bin/bash
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=epb-meridians
set -euo pipefail
export PATH=/opt/slurm/slurm-22.05.11/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${1:?code}"
/opt/soft/anaconda3/envs/py311/bin/mpirun -np "$SLURM_NTASKS" /opt/soft/anaconda3/envs/py311/bin/python syisr/lfm_epb/meridians/sections.py "${2:?root}"
