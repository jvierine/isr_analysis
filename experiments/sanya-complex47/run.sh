#!/bin/bash
# Run on avaruus; bulk HDF5 products stay there. Requires original Memo 43/46 inputs.
set -euo pipefail
export PATH=/home/j/.cargo/bin:$PATH
export CARGO_TARGET_DIR=/home/j/src/isr-analysis-rust-lpi/target
export RAYON_NUM_THREADS=128
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
ROOT=/home/j/sanya-sami2-pe-reference
IRI=/home/j/src/isr-analysis-rust-lpi/experiments/sanya-profile/published_profile.txt
BIN=$CARGO_TARGET_DIR/release
MODEL=$HERE
if [ -d "$HERE/model" ]; then MODEL=$HERE/model; fi
cargo build --release --manifest-path "$MODEL/Cargo.toml" --bins
"$BIN/complex_validate" "$ROOT/beam-32.h5" "$ROOT/complex-validation.json"
for kind in original smoothed; do
 if [ "$kind" = original ]; then INPUT=beam-32.h5; else INPUT=beam-smooth-log025.h5; fi
 CACHE=$ROOT/hot-cache-complex-$kind.h5
 OUT=/home/j/ruisdap-sanya-complex-$kind-25m
 if [ "$kind" = smoothed ]; then "$BIN/complex_validate" "$ROOT/$INPUT" "$ROOT/complex-validation-smoothed.json"; fi
 "$BIN/complex_build" "$ROOT/$INPUT" "$IRI" "$CACHE" 1 0.5
 if [ "$kind" = original ]; then
  "$BIN/complex_cache_check" "$ROOT/hot-cache.h5" "$CACHE" "$ROOT/$INPUT" "$IRI" "$ROOT/complex-cache-check.json"
  "$BIN/complex_compare" "$IRI" "$CACHE" "$ROOT/complex-comparison.json"
  ROOT_SCAN_HZ=100 FULL_QUADRATURE=1 "$BIN/complex_compare" "$IRI" "$CACHE" "$ROOT/complex-comparison-refined.json"
 fi
 "$BIN/complex_full_profile" "$IRI" "$CACHE" "$OUT" 0.025
 # prepare is the unchanged Memo 43 range/instrument folding program.
 "$BIN/prepare" "$OUT" /home/j/ruisdap-sanya-complex-$kind-figdata
 sha256sum "$ROOT/$INPUT" "$CACHE" "$OUT/profile.h5"
done
