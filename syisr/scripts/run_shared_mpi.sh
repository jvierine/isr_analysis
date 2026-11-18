#!/bin/bash
set -euo pipefail
b=${1:?output-root}
old=${2:?independent-run-root-containing-shapes-and-nominal}
input=${3:?native-input-root}
cd "$b/code"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$b/code/syisr/target}
CARGO_BUILD_JOBS=8 /home/j/.cargo/bin/cargo build --offline --release --manifest-path syisr/Cargo.toml --bin syisr_shared --bin syisr_shared_seed --bin syisr_shared_review > "$b/logs/build.log" 2>&1
for n in syisr_shared syisr_shared_seed syisr_shared_review;do cp "$CARGO_TARGET_DIR/release/$n" "$b/bin/$n";done
export SYISR_H_MODEL=iri_sigmoid SYISR_H_SHAPE_ROOT="$old/shapes" SYISR_MAX_ALTITUDE_KM=1300 SYISR_FUSED_JOINT=1
export RAYON_NUM_THREADS=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SYISR_JOINT_SEED_ROOT="$old/nominal/joint-fits"
unset SYISR_FAST_JOINT SYISR_JOINT_BENCH_FILE SYISR_JOINT_REFERENCE
mkdir -p "$b/seed-reference"
ln -sfn "$SYISR_JOINT_SEED_ROOT" "$b/seed-reference/joint-fits"
"$b/bin/syisr_shared_seed" "$input" "$SYISR_JOINT_SEED_ROOT" "$b/seed-reference"
previous="$b/seed-reference"
bestcost=1e300
for sweep in $(seq 0 6);do
 stage="$b/sweep-$sweep"
 mkdir -p "$stage"
 ln -sfn "$input/campaign.h5" "$stage/campaign.h5"
 ln -sfn joint-fits "$stage/fits"
 ln -sfn "$stage" "$b/running"
 printf 'FULL_DAY_START sweep=%s MPI=32 THREADS=4 CORES=128 max_iterations=unlimited checkpoint_interval=25 previous=%s
' "$sweep" "$previous"
 date -u
 date -u > "$b/started-128.txt"
 mpirun -np 32 --map-by slot:PE=4 --bind-to core "$b/bin/syisr_shared" "$input" all "$stage" "$previous" 0 > "$b/logs/sweep-$sweep.log" 2>&1
 "$b/bin/syisr_shared_review" "$stage" "$previous" > "$b/logs/review-$sweep.log"
 cat "$stage/review.json"
 cost=$(awk -F: '/global_huber_plus_temporal_objective/{gsub(/[, ]/,"",$2);print $2}' "$stage/review.json")
 if awk -v a="$cost" -v b="$bestcost" 'BEGIN{exit !(a<b)}';then bestcost="$cost";ln -sfn "$stage" "$b/best";fi
 ln -sfn "$stage" "$b/latest"
 if grep -q '"nonconverged_windows": 0' "$stage/review.json" && awk -F: '/maximum_fraction_change_from_previous_sweep/{gsub(/[, ]/,"",$2);exit !($2+0<0.005)}' "$stage/review.json";then echo 'TEMPORAL_FIXED_POINT_REACHED';break;fi
 previous="$stage"
done
date -u > "$b/finished.txt"
