#!/usr/bin/env bash
# One UTC day,64cores. Later days depend on successful temporal completion.
set -euo pipefail
code=${1:?}; month=${2:?}; day=${3:?}; baseline=${4:?}; previous=${5:?}; raw=${6:?}
root=$month/days/$day
export PATH=/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:$PATH
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HDF5_DIR=/opt/soft/anaconda3 RUSTFLAGS=-C\ link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib
export RANKS=${SLURM_NTASKS:?} RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:?}
(( RANKS*RAYON_NUM_THREADS==64 )) || exit 2
export SYISR_HEADER_UTC_OFFSET_S=28800 SYISR_ALLOW_PARTIAL_FINAL_FRAME=1 SYISR_QUARANTINE_CORRUPT_FRAMES=1
export SYISR_LP480_TOP_KM=1800 SYISR_LP480_PROFILE_CURVATURE=1 SYISR_TOPSIDE_TEMPERATURE_SLOPE=1
export SYISR_LP480_H_TIME_SIGMA=0.1 SYISR_LP480_H_NEIGHBOR_START=1
export SYISR_MULTISTART=1
export SYISR_INITIAL_PROFILE_DIR=$previous/final SYISR_NEAREST_INITIAL_PROFILE=1 SYISR_FIXED_TIME_CONTEXT=$previous/final
unset SYISR_LP480_PULSE_COVARIANCE SYISR_LBFGS SYISR_DIAGONAL_STEP SYISR_LP480_LATEST_FIRST
mkdir -p "$root"/{logs,decoded/products,input,bin} "$month/bin" "$month/indices"
exec 8>"$root/day.lock"; flock -n 8
trap 'echo "FAILED stage=${stage:-setup} line=$LINENO" >"$root/FAILED"' ERR
test -f "$previous/COMPLETE"
test ! -f "$root/COMPLETE" || exit 0
stage=build
cd "$code"; git rev-parse HEAD >"$root/code-commit.txt"
if [[ ! -f $month/BUILD_COMPLETE ]]; then
 CARGO_BUILD_JOBS=4 cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin syisr_lp480 >"$root/logs/build.log" 2>&1
 CARGO_BUILD_JOBS=4 cargo test --release --locked --offline --manifest-path syisr/Cargo.toml --lib lp480 >"$root/logs/tests.log" 2>&1
 cp syisr/target/release/syisr_lp480 "$month/bin/"; git rev-parse HEAD >"$month/code-commit.txt"
 date -u >"$month/BUILD_COMPLETE"
fi
cp "$month/bin/syisr_lp480" "$root/bin/"; bin=$root/bin/syisr_lp480
py=(/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python)
stage=utc-index
"${py[@]}" "$code/syisr/operations/lp480_month.py" stage "$month" --day "$day" --base "$baseline" --previous "$previous" --binary "$bin" --raw-root "$raw" >"$root/logs/index.log" 2>&1
if [[ -f $root/NO_OBSERVATIONS ]]; then
 ln -sfn "$(readlink -f "$previous/final")" "$root/final"; date -u >"$root/COMPLETE"
else
 stage=screened-products
 srun --mpi=none --ntasks="$RANKS" --cpus-per-task="$RAYON_NUM_THREADS" --cpu-bind=cores --kill-on-bad-exit=1 \
  "$bin" prepare "$root/decoded/index.h5" "$root/decoded/products" >"$root/logs/prepare.log" 2>&1
 stage=compact
 ln -sfn "$(readlink -f "$baseline/input/background.h5")" "$root/decoded/background.h5"
 "$bin" compact "$root/decoded" "$root/input" >"$root/logs/compact.log" 2>&1
 ln -sfn "$month/references" "$root/input/references"
 cp "$baseline/input/calibration.h5" "$baseline/input/calibration_metadata.h5" "$root/input/"
 stage=temporal-fits
 bash "$code/syisr/operations/lp480_controller.sh" "$root" "$bin" >"$root/logs/controller.log" 2>&1
 stage=snapshot
 "$bin" snapshot "$root/final" "$root/snapshot" >"$root/logs/snapshot.log" 2>&1
fi
stage=catalog
"${py[@]}" "$code/syisr/operations/lp480_month.py" catalog "$month" --base "$baseline" --binary "$bin" >"$root/logs/catalog.log" 2>&1
date -u >"$root/COMPLETE"
