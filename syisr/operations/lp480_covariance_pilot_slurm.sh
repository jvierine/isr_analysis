#!/usr/bin/env bash
# Isolated raw pulse-covariance pilot: do not replace accepted campaign products.
set -euo pipefail
code=${1:?}; campaign=${2:?}; decoded=${3:?}; diagnostic=${4:?}; output=${5:?}
export PATH=/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:$PATH
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HDF5_DIR=/opt/soft/anaconda3 RUSTFLAGS=-C\ link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:?}
export SYISR_LP480_TOP_KM=1800 SYISR_LP480_PROFILE_CURVATURE=1 SYISR_TOPSIDE_TEMPERATURE_SLOPE=1
export SYISR_LP480_H_TIME_SIGMA=0.1 SYISR_LP480_H_NEIGHBOR_START=1
unset SYISR_DIAGONAL_STEP SYISR_LBFGS SYISR_DATA_STABILITY_STOP
mkdir -p "$output"/{logs,bin,fits}
exec 8>"$output/pilot.lock"; flock -n 8
trap 'date -u >"$output/FAILED"' ERR
cd "$code"; git rev-parse HEAD >"$output/code-commit.txt"
CARGO_BUILD_JOBS=4 cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin syisr_lp480 >"$output/logs/build.log" 2>&1
CARGO_BUILD_JOBS=4 cargo test --release --locked --offline --manifest-path syisr/Cargo.toml --lib lp480 >"$output/logs/tests.log" 2>&1
cp syisr/target/release/syisr_lp480 "$output/bin/"
"$output/bin/syisr_lp480" noise-covariance-inputs "$decoded" "$campaign/input" "$diagnostic" "$output/input" >"$output/logs/prepare.log" 2>&1
srun --mpi=none --ntasks="$SLURM_NTASKS" --cpus-per-task="$RAYON_NUM_THREADS" --cpu-bind=cores --kill-on-bad-exit=1 \
 "$output/bin/syisr_lp480" fit "$output/input" all "$output/fits" "$campaign/final" 0 >"$output/logs/fit.log" 2>&1
date -u >"$output/COMPLETE"
