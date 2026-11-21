#!/bin/bash
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --job-name=epb-clock-audit
set -euo pipefail
code=${1:?code};out=${2:?out}
/opt/soft/anaconda3/bin/conda run -n base python "$code/syisr/lfm_epb/coherent_evidence/clock_register.py" "$out/clock_registration.h5"
