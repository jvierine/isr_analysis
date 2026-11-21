#!/usr/bin/env bash
#SBATCH --job-name=lhw-AC15m
#SBATCH --partition=node
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=06:00:00
set -euo pipefail
root=${1:?run root}
export PATH=/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:/opt/soft/anaconda3/bin:$PATH
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=2
export HDF5_DIR=/opt/soft/anaconda3
export RUSTFLAGS="-C target-cpu=native -C link-arg=-Wl,-rpath,$HDF5_DIR/lib"
mkdir -p "$root"/{logs,blocks,bin}
trap 'date -u >"$root/FAILED"' ERR
cd "$root/source"
CARGO_BUILD_JOBS=8 cargo build --release --locked --offline --manifest-path syisr/lhw_ac/Cargo.toml --bin process >"$root/logs/build.log" 2>&1
CARGO_BUILD_JOBS=8 cargo test --release --locked --offline --manifest-path syisr/lhw_ac/Cargo.toml --lib >"$root/logs/tests.log" 2>&1
cp syisr/lhw_ac/target/release/process "$root/bin/process"
sha256sum "$root/bin/process" >"$root/binary-sha256.txt"
"$root/bin/process" index /gpfs/pub/data/L0/2026/20261119 SY_2026_11_19_19_39_03 "$root/index.h5" >"$root/logs/index.log" 2>&1
srun --mpi=none --ntasks=32 --cpus-per-task=2 --cpu-bind=cores --kill-on-bad-exit=1 "$root/bin/process" worker "$root/index.h5" syisr/m32-reference.bc "$root/blocks" >"$root/logs/decode.log" 2>&1
conda run -n base python syisr/lhw_ac/export_movie.py "$root/blocks" "$root/beam-geometry.h5" "$root/15min-step5min.h5" >"$root/logs/export.log" 2>&1
date -u >"$root/COMPLETE"
