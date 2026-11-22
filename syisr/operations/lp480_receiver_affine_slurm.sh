#!/usr/bin/env bash
# Held-out daytime receiver validation only; never optimize H or plasma profiles.
set -euo pipefail
code=${1:?}; receiver=${2:?}; month=${3:?}
root=$receiver/affine-calibration
export PATH=/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:$PATH
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=2
export HDF5_DIR=/opt/soft/anaconda3 RUSTFLAGS=-C\ link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib
export SYISR_LP480_TOP_KM=1800
test -f "$month/USER_PAUSED_H_FITTING"
mkdir -p "$root"/{logs,bin}
exec 8>"$root/check.lock";flock -n 8
trap 'date -u >"$root/FAILED"' ERR
cd "$code";git rev-parse HEAD >"$root/code-commit.txt"
CARGO_BUILD_JOBS=4 cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin syisr_lp480 >"$root/logs/build.log" 2>&1
CARGO_BUILD_JOBS=4 cargo test --release --locked --offline --manifest-path syisr/Cargo.toml --lib lp480 >"$root/logs/tests.log" 2>&1
cp syisr/target/release/syisr_lp480 "$root/bin/";bin=$root/bin/syisr_lp480
quiet=/gpfs/pub/data_other/juha/syisr-covariance-pilot-20261121/input
variation=/gpfs/pub/data_other/juha/syisr-solar-noise-dwells-20261120/noise-response-refined.h5
baseline=/gpfs/pub/data_other/juha/syisr-storm-lp480-20240510-16-time-v3-20261120/input
"$bin" receiver-affine-reference "$quiet" "$variation" "$root/reference" >"$root/logs/calibration.log" 2>&1
"$bin" receiver-copy-covariance "$quiet" "$root/heldout-covariance" >"$root/logs/reexport.log" 2>&1
"$bin" receiver-report "$root/heldout-covariance" "$baseline" "$quiet" "$root/heldout-reports" "$root/reference/background.h5" >"$root/logs/heldout.log" 2>&1
for date in 20240520 20240531; do
 "$bin" receiver-copy-covariance "$receiver/covariance/$date" "$root/covariance-$date"
 "$bin" receiver-report "$root/covariance-$date" "$month/days/$date/input" "$quiet" "$root/reports-$date" "$root/reference/background.h5"
done >"$root/logs/nighttime.log" 2>&1
date -u >"$root/COMPLETE"
