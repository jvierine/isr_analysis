#!/usr/bin/env bash
#SBATCH --job-name=lhw-verify
#SBATCH --partition=node
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
export PATH=/opt/soft/anaconda3/bin:$PATH LC_ALL=C LANG=C OPENBLAS_NUM_THREADS=1
root=${1:?root}
conda run -n base python "$root/source/syisr/lhw_ac/verify_raw.py" "$root"
conda run -n base python "$root/source/syisr/lhw_ac/display_input.py" "$root"
