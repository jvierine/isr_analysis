#!/usr/bin/env bash
#SBATCH --job-name=lfm-epb
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# Scheduler selects the next available node; no node pinning.
set -euo pipefail
code=${1:?code checkout}; root=${2:?output root}; source=${3:?L1 beam directory}
start=${4:-2026-03-01T10:00:00Z}; stop=${5:-2026-03-01T16:00:00Z}
export PATH=/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:$PATH
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export HDF5_DIR=/opt/soft/anaconda3
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$HDF5_DIR/lib"
mkdir -p "$root"/{logs,bin}
exec 8>"$root/run.lock"
flock -n 8 || exit 1
trap 'date -u >"$root/FAILED"' ERR
cd "$code"
git rev-parse HEAD >"$root/code-commit.txt"
CARGO_BUILD_JOBS=8 cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin syisr_lfm_epb >"$root/logs/build.log" 2>&1
cargo test --release --locked --offline --manifest-path syisr/Cargo.toml --bin syisr_lfm_epb >"$root/logs/tests.log" 2>&1
cp syisr/target/release/syisr_lfm_epb "$root/bin/"
sha256sum "$root/bin/syisr_lfm_epb" >"$root/binary-sha256.txt"
# py311 is the server's existing MPI environment; scientific compute is Rust.
# Base conda remains the default for non-MPI Python preparation and reduction.
mpi=/opt/soft/anaconda3/envs/py311
"$mpi/bin/mpirun" -np "$SLURM_NTASKS" "$mpi/bin/python" syisr/lfm_epb/mpi_driver.py \
 "$root/bin/syisr_lfm_epb" "$source" "$root" --start "$start" --stop "$stop" >"$root/logs/process.log" 2>&1
/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python syisr/lfm_epb/combine.py "$root" --start "$start" --stop "$stop" >"$root/logs/combine.log" 2>&1
/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python syisr/lfm_epb/export_plot.py "$root/zenith_rti.h5" "$root/zenith_rti_display.h5"
date -u >"$root/COMPLETE"
