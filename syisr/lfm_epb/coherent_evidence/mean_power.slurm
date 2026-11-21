#!/bin/bash
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --job-name=epb-mean-power
set -euo pipefail
export PATH=/opt/slurm/slurm-22.05.11/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
code=${1:?code};out=${2:?out};mpi=/opt/soft/anaconda3/envs/py311
"$mpi/bin/mpirun" -np "$SLURM_NTASKS" "$mpi/bin/python" "$code/syisr/lfm_epb/coherent_evidence/mean_power.py" /gpfs/pub/data_other/juha/lfm_epb/20260301/multibeam/parts "$out/mean_power"
