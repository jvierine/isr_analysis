#!/bin/bash
set -euo pipefail
b=${1:?output-root}
input=${2:?input-root}
bin="$b/code/syisr/target/release"
export SYISR_H_MODEL=iri_sigmoid SYISR_H_SHAPE_ROOT=${3:?shape-root} SYISR_MAX_ALTITUDE_KM=1300 SYISR_FUSED_JOINT=1
export RAYON_NUM_THREADS=5 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SYISR_JOINT_SEED_ROOT=${4:?independent-fit-seed-root}
unset SYISR_FAST_JOINT SYISR_JOINT_BENCH_FILE SYISR_JOINT_REFERENCE
printf 'Waiting for the shared-beam pilot\n'
while [ ! -f "$b/pilot-finished.txt" ]; do
 if ! pgrep -u j -f "^bash $b/pilot.sh" >/dev/null && ! pgrep -u j -f "^$bin/syisr_shared " >/dev/null;then echo 'Pilot is not running; full campaign not started';exit 12;fi
 sleep 20
done
"$bin/syisr_shared_review" "$b/pilot" none > "$b/logs/pilot-review.log"
if ! grep -q '"nonconverged_windows": 0' "$b/pilot/review.json";then echo 'Pilot did not converge; holding campaign for inspection';exit 13;fi
previous=none
bestcost=1e300
beststage=none
for sweep in $(seq 0 6); do
 stage="$b/sweep-$sweep"
 mkdir -p "$stage"
 ln -sfn "$input/campaign.h5" "$stage/campaign.h5"
 ln -sfn joint-fits "$stage/fits"
 date -u
 printf 'Starting sweep %s, 10 MPI ranks x 5 threads = 50 cores; previous=%s\n' "$sweep" "$previous"
 mpirun -np 10 --map-by slot:PE=5 --bind-to core "$bin/syisr_shared" "$input" all "$stage" "$previous" 400 > "$b/logs/sweep-$sweep.log" 2>&1
 "$bin/syisr_shared_review" "$stage" "$previous" > "$b/logs/review-$sweep.log"
 cat "$stage/review.json"
 cost=$(awk -F: '/global_huber_plus_temporal_objective/{gsub(/[, ]/,"",$2);print $2}' "$stage/review.json")
 if awk -v a="$cost" -v b="$bestcost" 'BEGIN{exit !(a<b)}';then bestcost="$cost";beststage="$stage";ln -sfn "$stage" "$b/best";fi
 ln -sfn "$stage" "$b/latest"
 if [ "$sweep" -gt 0 ] && grep -q '"nonconverged_windows": 0' "$stage/review.json" && awk -F: '/maximum_fraction_change_from_previous_sweep/{gsub(/[, ]/,"",$2);exit !($2+0<0.005)}' "$stage/review.json";then echo 'Temporal sweep fixed point reached';break;fi
 previous="$stage"
done
printf 'Best complete sweep: %s, whole-day penalized objective: %s\n' "$beststage" "$bestcost"
date -u > "$b/finished.txt"
