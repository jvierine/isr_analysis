#!/bin/bash
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=epb-cutouts
set -euo pipefail
export PATH=/opt/slurm/slurm-22.05.11/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${1:?code}"
/opt/soft/anaconda3/envs/py311/bin/mpirun -np "$SLURM_NTASKS" /opt/soft/anaconda3/envs/py311/bin/python syisr/lfm_epb/meridians/cutouts.py "${2:?root}" /home/juha/src/isr_analysis/syisr/target/release/lfm_epb_cavity
