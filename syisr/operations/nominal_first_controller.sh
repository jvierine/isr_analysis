#!/bin/bash
set -euo pipefail
b=/nfs/urdr/scratch/juha/isr_analysis/syisr-ac-joint-v3-20261116
a="$b/performance-audit"
bin="$a/syisr_joint_fused_fd"
out="$b/nominal-sweep"
old=$(cat "$a/paused-pids.txt")
# The independent workers retain their current bootstrap state in RAM.
xargs -r kill -STOP < "$a/paused-pids.txt"
watch=$(pgrep -u j -f "^bash $a/auto-resume.sh$" || true)
if [ -n "$watch" ];then kill -TERM $watch;fi
child=''
cleanup(){
 set +e
 if [ -n "$child" ];then kill -TERM "$child" 2>/dev/null;wait "$child" 2>/dev/null;fi
 for i in $(seq 1 15);do if ! pgrep -u j -f "^$bin .* all 0 $out" >/dev/null;then break;fi;sleep 1;done
 pgrep -u j -f "^$bin .* all 0 $out" | xargs -r kill -KILL
 xargs -r kill -CONT < "$a/paused-pids.txt"
 date -u > "$a/nominal-sweep-finished.txt"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
mkdir -p "$out/joint-fits"
ln -sfn /nfs/urdr/scratch/juha/isr_analysis/syisr-ac-native-v2-20261116/run/campaign.h5 "$out/campaign.h5"
ln -sfn joint-fits "$out/fits"
export RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 SYISR_FUSED_JOINT=1 SYISR_JOINT_PROGRESS=1
export SYISR_JOINT_SEED_ROOT="$b/full-run/joint-nominal"
unset SYISR_FAST_JOINT SYISR_JOINT_BENCH_FILE SYISR_JOINT_REFERENCE
# A hard two-hour bound prevents an unattended suspended bootstrap campaign.
timeout --signal=TERM --kill-after=30s 7200 mpirun -np 24 --map-by slot:PE=2 --bind-to core "$bin" /nfs/urdr/scratch/juha/isr_analysis/syisr-ac-native-v2-20261116/run all 0 "$out" &
child=$!
echo NOMINAL_MPI_PID=$child
wait "$child"
