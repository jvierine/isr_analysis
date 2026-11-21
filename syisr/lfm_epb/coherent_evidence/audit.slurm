#!/bin/bash
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=epb-cal-audit
set -euo pipefail
code=${1:?code};out=${2:?out}
mkdir -p "$out"
/opt/soft/anaconda3/bin/conda run -n base python "$code/syisr/lfm_epb/coherent_evidence/audit_l2.py" /gpfs/pub/data/L2/SYISR_results/h5/L2/2026/OSAYA_TISR01_IFIT_L2_DAY_20260301000000_V01.00_N_LP_15sec.h5 "$out/l2_audit.h5"
