#!/bin/bash
# Main compute commands; extraction/cleaning must run on the node with raw data.
set -euo pipefail
export PATH=/home/j/.cargo/bin:$PATH
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-64}
export CARGO_TARGET_DIR=/home/j/src/isr-analysis-syisr/target
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
ROOT=/nfs/urdr/scratch/juha/ruisdap/syisr/hplus/2023-11-15-night-zenith
test "$(findmnt -n -o SOURCE /nfs/urdr/scratch)" = 10.228.2.29:/scratch
cargo build --release --manifest-path "$HERE/Cargo.toml" --bins
BIN=$CARGO_TARGET_DIR/release
"$BIN/hplus_validate" "$ROOT/synthetic-validation.json"
"$BIN/hplus_fit" "$ROOT/night-zenith.h5" "$ROOT/hourly"
# hplus_clean night-zenith.h5 clean-16.h5 16 runs on the raw disk host.
# Transfer only the compact clean product back to this real NFS directory.
"$BIN/hplus_fit" "$ROOT/clean-16.h5" "$ROOT/clean-profile" clean
"$BIN/hplus_boot" "$ROOT/clean-16.h5" "$ROOT/clean-stability.json"
"$BIN/hplus_sensitivity" "$ROOT/clean-16.h5" "$ROOT/h-temperature-sensitivity.json"
"$BIN/hplus_msis_check" /home/j/src/guisdap26/crates/ruisdap-syisr/config/atmosphere.json "$ROOT/msis-collision-check.json"
"$BIN/hplus_export" "$ROOT"
