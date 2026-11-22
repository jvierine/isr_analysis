#!/usr/bin/env bash
#64-core receiver-only diagnostics. Plasma fitting remains explicitly paused.
set -euo pipefail
code=${1:?}; month=${2:?}; output=${3:?}
export PATH=/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:$PATH
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HDF5_DIR=/opt/soft/anaconda3 RUSTFLAGS=-C\ link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:?}
(( SLURM_NTASKS*RAYON_NUM_THREADS==64 ))
export SYISR_HEADER_UTC_OFFSET_S=28800 SYISR_ALLOW_PARTIAL_FINAL_FRAME=1 SYISR_QUARANTINE_CORRUPT_FRAMES=1 SYISR_LP480_TOP_KM=1800
mkdir -p "$output"/{logs,bin}
exec 8>"$output/receiver.lock";flock -n 8
trap 'date -u >"$output/FAILED"' ERR
cd "$code";git rev-parse HEAD >"$output/code-commit.txt"
CARGO_BUILD_JOBS=4 cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin syisr_lp480 >"$output/logs/build.log" 2>&1
CARGO_BUILD_JOBS=4 cargo test --release --locked --offline --manifest-path syisr/Cargo.toml --lib lp480 >"$output/logs/tests.log" 2>&1
cp syisr/target/release/syisr_lp480 "$output/bin/"; bin=$output/bin/syisr_lp480
script=$code/syisr/operations/lp480_receiver_check.py
py=(/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python)
run(){ srun --mpi=none --ntasks="$SLURM_NTASKS" --cpus-per-task="$RAYON_NUM_THREADS" --cpu-bind=cores --kill-on-bad-exit=1 "$@"; }
"${py[@]}" "$script" receiver-stage "$month" "$output" "$bin" >"$output/logs/receiver-stage.log" 2>&1
export SYISR_LP480_PULSE_COVARIANCE=1
for date in 20240520 20240531; do
 run "$bin" prepare "$output/covariance/$date/index.h5" "$output/covariance/$date/products" >"$output/logs/covariance-$date.log" 2>&1
done
"${py[@]}" "$script" reports "$month" "$output" "$bin" >"$output/logs/reports.log" 2>&1
date -u >"$output/RECEIVER_REPORTS_COMPLETE"
run "${py[@]}" "$script" indices "$month" "$output" "$bin" >"$output/logs/index.log" 2>&1
"${py[@]}" "$script" stage "$month" "$output" "$bin" >"$output/logs/stage.log" 2>&1
run "${py[@]}" "$script" stored "$month" "$output" "$bin" >"$output/logs/stored-tail.log" 2>&1
date -u >"$output/STORED_SPECTRA_COMPLETE"
run "$bin" receiver-raw-tail "$output/all-may-index.h5" "$output/raw-tail" >"$output/logs/raw-tail.log" 2>&1
date -u >"$output/COMPLETE"
