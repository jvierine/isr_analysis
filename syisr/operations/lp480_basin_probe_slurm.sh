#!/usr/bin/env bash
# Isolated basin checks: eight native solves, within a single64-core allocation.
set -euo pipefail
month=${1:?}; output=${2:?}; script=${3:?}
export PATH=/opt/slurm/slurm-22.05.11/bin:$PATH
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:?}
(( SLURM_NTASKS*RAYON_NUM_THREADS==64 ))
export SYISR_LP480_TOP_KM=1800 SYISR_LP480_PROFILE_CURVATURE=1 SYISR_TOPSIDE_TEMPERATURE_SLOPE=1
export SYISR_LP480_H_TIME_SIGMA=0.1
unset SYISR_MULTISTART SYISR_LP480_H_NEIGHBOR_START SYISR_FIXED_TIME_CONTEXT
unset SYISR_NEAREST_INITIAL_PROFILE SYISR_DATA_STABILITY_STOP SYISR_LBFGS SYISR_DIAGONAL_STEP SYISR_WARMSTART_DIR
mkdir -p "$output"
exec 8>"$output/probe.lock"; flock -n 8
trap 'date -u >"$output/FAILED"' ERR
bin=$month/bin/syisr_lp480
py=(/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python)
"${py[@]}" "$script" stage "$month" "$output" "$bin" >"$output/stage.log" 2>&1
srun --mpi=none --ntasks="$SLURM_NTASKS" --cpus-per-task="$RAYON_NUM_THREADS" --cpu-bind=cores --kill-on-bad-exit=1 \
 "${py[@]}" "$script" run "$month" "$output" "$bin" >"$output/run.log" 2>&1
date -u >"$output/COMPLETE"
