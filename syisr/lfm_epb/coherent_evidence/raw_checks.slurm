#!/bin/bash
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=epb-raw-checks
set -euo pipefail
code=${1:?code};out=${2:?out}
/opt/soft/anaconda3/bin/conda run -n base python "$code/syisr/lfm_epb/coherent_evidence/raw_checks.py" /gpfs/pub/data_other/juha/lfm_epb/20260301/multibeam/parts "$out/raw_checks.h5"
