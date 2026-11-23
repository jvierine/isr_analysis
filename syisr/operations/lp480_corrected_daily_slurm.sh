#!/usr/bin/env bash
# Limited corrected-background retry. One explicitly allocated day; never release old jobs.
set -euo pipefail
code=${1:?}; campaign=${2:?}; day=${3:?}; original=${4:?}; receiver=${5:?}; previous=${6:-none}
root=$campaign/days/$day
export PATH=/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:$HOME/texlive/bin/x86_64-linux:$PATH
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HDF5_DIR=/opt/soft/anaconda3 RUSTFLAGS=-C\ link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib
export RANKS=${SLURM_NTASKS:?} RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:?}
(( RANKS*RAYON_NUM_THREADS==64 || RANKS*RAYON_NUM_THREADS==128 ))
export SYISR_LP480_TOP_KM=1800 SYISR_LP480_PROFILE_CURVATURE=1 SYISR_TOPSIDE_TEMPERATURE_SLOPE=1
export SYISR_LP480_H_TIME_SIGMA=0.1 SYISR_LP480_H_NEIGHBOR_START=1 SYISR_MULTISTART=1
export SYISR_LP480_PULSE_COVARIANCE=1 SYISR_PLOT_UTC=1
export SYISR_INITIAL_PROFILE_DIR=${SYISR_INITIAL_PROFILE_DIR:-$original/days/$day/final}
unset SYISR_NEAREST_INITIAL_PROFILE SYISR_FIXED_TIME_CONTEXT SYISR_LBFGS SYISR_DIAGONAL_STEP SYISR_WARMSTART_DIR
if [[ $previous != none ]]; then
 test -f "$previous/COMPLETE"
 export SYISR_FIXED_TIME_CONTEXT=$previous/final
fi
mkdir -p "$root"/{logs,decoded/products,input,bin} "$campaign/bin"
exec 8>"$root/day.lock";flock -n 8
trap 'echo "FAILED stage=${stage:-setup} line=$LINENO" >"$root/FAILED"' ERR
test ! -f "$root/COMPLETE" || exit 0
stage=build
cd "$code";git rev-parse HEAD >"$root/code-commit.txt"
if [[ ! -f $campaign/BUILD_COMPLETE ]]; then
 CARGO_BUILD_JOBS=4 cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin syisr_lp480 >"$root/logs/build.log" 2>&1
 CARGO_BUILD_JOBS=4 cargo test --release --locked --offline --manifest-path syisr/Cargo.toml --lib lp480 >"$root/logs/tests.log" 2>&1
 cp syisr/target/release/syisr_lp480 "$campaign/bin/"
 date -u >"$campaign/BUILD_COMPLETE"
fi
cp "$campaign/bin/syisr_lp480" "$root/bin/";bin=$root/bin/syisr_lp480
stage=validated-index
cp "$receiver/daily-index/$day.h5" "$root/decoded/index.h5"
stage=pulse-covariance
srun --mpi=none --ntasks="$RANKS" --cpus-per-task="$RAYON_NUM_THREADS" --cpu-bind=cores --kill-on-bad-exit=1 \
 "$bin" prepare "$root/decoded/index.h5" "$root/decoded/products" >"$root/logs/prepare.log" 2>&1
stage=corrected-background
cp "$receiver/affine-calibration/reference/background.h5" "$root/decoded/background.h5"
"$bin" compact "$root/decoded" "$root/input" >"$root/logs/compact.log" 2>&1
ln -sfn "$(readlink -f "$original/references")" "$root/input/references"
cp "$original/days/$day/input/calibration.h5" "$original/days/$day/input/calibration_metadata.h5" "$root/input/"
stage=plasma-fit
bash "$code/syisr/operations/lp480_controller.sh" "$root" "$bin" >"$root/logs/controller.log" 2>&1
# Controller COMPLETE means numerical temporal convergence; export gets its own marker.
stage=export
"$bin" snapshot "$root/final" "$root/snapshot" >"$root/logs/snapshot.log" 2>&1
if /gpfs/pub/data_other/juha/syisr-month-render-20261122/target/release/lp480_maps maps "$root/snapshot" "$root/plots" >"$root/logs/plots.log" 2>&1; then
 date -u >"$root/PLOTS_COMPLETE"
else
 # Completed numerical data remain usable if the optional renderer is incompatible.
 date -u >"$root/PLOTS_PENDING"
fi
date -u >"$root/EXPORT_COMPLETE"
