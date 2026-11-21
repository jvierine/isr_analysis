#!/usr/bin/env bash
#SBATCH --job-name=lfm-cavity
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
export PATH=/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:$PATH
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
((SLURM_NTASKS<=16 && SLURM_CPUS_PER_TASK==1)) || exit 1
code=${1:?code};root=${2:?root};mkdir -p "$root/cavity"
exec 8>"$root/cavity/run.lock";flock -n 8 || exit 1
cd "$code"
export HDF5_DIR=/opt/soft/anaconda3 RUSTFLAGS='-C link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib'
CARGO_BUILD_JOBS=8 cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin lfm_epb_cavity
cargo test --release --locked --offline --manifest-path syisr/Cargo.toml --bin lfm_epb_cavity
cp syisr/target/release/lfm_epb_cavity "$root/cavity/native"
"$root/cavity/native" prepare "$root" "$root/cavity/observations.h5"
mpi=/opt/soft/anaconda3/envs/py311
"$mpi/bin/mpirun" -np "$SLURM_NTASKS" "$mpi/bin/python" syisr/lfm_epb/cavity_mpi.py "$root/cavity/native" "$root"
