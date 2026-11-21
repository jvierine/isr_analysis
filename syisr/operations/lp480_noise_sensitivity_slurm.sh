#!/usr/bin/env bash
# Read-only receiver-background sensitivity, never a production replacement.
set -euo pipefail
code=${1:?}; decoded=${2:?}; response=${3:?}; output=${4:?}; shift 4
export PATH=$HOME/.cargo/bin:$PATH
export HDF5_DIR=/opt/soft/anaconda3 RUSTFLAGS=-C\ link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
mkdir -p "$output"/{logs,bin}
exec 8>"$output/diagnostic.lock"; flock -n 8
trap 'date -u >"$output/FAILED"' ERR
cd "$code"; git rev-parse HEAD >"$output/code-commit.txt"
CARGO_BUILD_JOBS=2 cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin syisr_lp480 >"$output/logs/build.log" 2>&1
CARGO_BUILD_JOBS=2 cargo test --release --locked --offline --manifest-path syisr/Cargo.toml --lib lp480_fit >"$output/logs/tests.log" 2>&1
cp syisr/target/release/syisr_lp480 "$output/bin/"
"$output/bin/syisr_lp480" noise-sensitivity "$decoded" "$response" "$output/noise-sensitivity.h5" "$@" >"$output/logs/noise.log" 2>&1
date -u >"$output/COMPLETE"
