#!/usr/bin/env bash
#SBATCH --job-name=lhw-origin
#SBATCH --partition=node
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
root=${1:?run root}
export PATH=/opt/soft/anaconda3/bin:$PATH LC_ALL=C LANG=C OPENBLAS_NUM_THREADS=1
conda run -n base python "$root/source/syisr/lhw_ac/audit_range_origin.py" "$root" "$root/range-origin-audit"
