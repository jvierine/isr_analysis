#!/usr/bin/env bash
#SBATCH --job-name=lfm-all-beams
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
export PATH=/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:$PATH
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
((SLURM_NTASKS<=16 && SLURM_CPUS_PER_TASK==1)) || exit 1
code=${1:?code};root=${2:?root};source=${3:?source}
mkdir -p "$root"/{logs,bin}
exec 8>"$root/run.lock";flock -n 8 || exit 1
cd "$code"
export HDF5_DIR=/opt/soft/anaconda3 RUSTFLAGS='-C link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib'
CARGO_BUILD_JOBS=8 cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin syisr_lfm_epb
cargo test --release --locked --offline --manifest-path syisr/Cargo.toml --lib lfm_geometry
cp syisr/target/release/syisr_lfm_epb "$root/bin/"
mpi=/opt/soft/anaconda3/envs/py311
"$mpi/bin/mpirun" -np "$SLURM_NTASKS" "$mpi/bin/python" syisr/lfm_epb/mpi_driver.py "$root/bin/syisr_lfm_epb" "$source" "$root" --all-beams --start 2026-03-01T11:00:00Z --stop 2026-03-01T14:00:00Z >"$root/logs/pulses.log" 2>&1
"$mpi/bin/mpirun" -np "$SLURM_NTASKS" "$mpi/bin/python" syisr/lfm_epb/multibeam_reduce.py "$root" >"$root/logs/reduce.log" 2>&1
date -u >"$root/COMPLETE"
