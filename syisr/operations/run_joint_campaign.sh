#!/bin/bash
set -euo pipefail
base=/nfs/urdr/scratch/juha/isr_analysis/syisr-ac-joint-v3-20261116
input=/nfs/urdr/scratch/juha/isr_analysis/syisr-ac-native-v2-20261116/run
while ps -p 1048146 -o args= | grep -F "$base/syisr_joint_v3" >/dev/null; do sleep 30; done
"$base/syisr_joint_v3" "$input" validate 0 "$base/pilot-v3"
# Superseded pilot only; never stop unrelated work.
if ps -p 1036281 -o args= | grep -F "$base/syisr_joint_v2" >/dev/null; then kill -TERM 1036281; fi
mkdir -p "$base/full-run/joint-fits" "$base/full-run/joint-failures"
ln -sfn "$input/campaign.h5" "$base/full-run/campaign.h5"
ln -sfn joint-fits "$base/full-run/fits"
export RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SYISR_JOINT_PROGRESS=1
cd "$base"
exec mpirun -np 24 --map-by slot:PE=2 --bind-to core "$base/syisr_joint_v3" "$input" all 48 "$base/full-run"
