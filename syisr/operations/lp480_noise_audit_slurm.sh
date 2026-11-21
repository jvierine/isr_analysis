#!/usr/bin/env bash
# Read-only ten-second noise diagnostic; production products remain unchanged.
set -euo pipefail
code=${1:?}; decoded=${2:?}; output=${3:?}; shift 3
export PATH=/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:$PATH
export HDF5_DIR=/opt/soft/anaconda3 RUSTFLAGS=-C\ link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
mkdir -p "$output"/{logs,bin}
exec 8>"$output/diagnostic.lock"; flock -n 8
trap 'date -u >"$output/FAILED"' ERR
cd "$code"; git rev-parse HEAD >"$output/code-commit.txt"
CARGO_BUILD_JOBS=2 cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin syisr_lp480 >"$output/logs/build.log" 2>&1
CARGO_BUILD_JOBS=2 cargo test --release --locked --offline --manifest-path syisr/Cargo.toml --lib lp480_fit >"$output/logs/tests.log" 2>&1
cp syisr/target/release/syisr_lp480 "$output/bin/"
mode=${SYISR_NOISE_AUDIT_MODE:-noise-audit}
case "$mode" in noise-audit|noise-coherence) ;; *) exit 2 ;; esac
"$output/bin/syisr_lp480" "$mode" "$decoded" "$output/$mode.h5" "$@" >"$output/logs/noise.log" 2>&1
date -u >"$output/COMPLETE"
