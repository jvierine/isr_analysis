#!/usr/bin/env bash
#SBATCH --job-name=lfm-checks
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
set -euo pipefail
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
code=${1:?code};root=${2:?root}
cd "$code"
/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python syisr/lfm_epb/cavity_checks.py "$root"
date -u > "$root/CHECKS_COMPLETE"
