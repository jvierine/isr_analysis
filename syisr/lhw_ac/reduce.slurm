#!/usr/bin/env bash
#SBATCH --job-name=lhw-coadd
#SBATCH --partition=node
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
root=${1:?run root}
export PATH=/opt/soft/anaconda3/bin:$PATH LC_ALL=C LANG=C OPENBLAS_NUM_THREADS=1
cd "$root/source/syisr/lhw_ac"
conda run -n base python -m unittest test_reduction
conda run -n base python export_movie.py "$root/blocks" "$root/beam-geometry.h5" "$root/spectra.h5" --integration-minutes 30 --range-gates 4 --range-delay-correction-us 640
conda run -n base python verify_reduction.py "$root"
conda run -n base python display_input.py "$root" spectra.h5
date -u >"$root/COMPLETE_REDUCTION"
