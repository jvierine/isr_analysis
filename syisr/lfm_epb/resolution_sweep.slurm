#!/usr/bin/env bash
#SBATCH --job-name=lfm-resolution
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
export PATH="/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:$HOME/texlive/bin/x86_64-linux:$PATH"
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
(( SLURM_NTASKS <= 16 && SLURM_CPUS_PER_TASK == 1 )) || exit 1
code=${1:?checkout};root=${2:?output};parts=${3:?pulse products}
mkdir -p "$root"/logs
exec 8>"$root/run.lock";flock -n 8 || exit 1
cd "$code"
mpi=/opt/soft/anaconda3/envs/py311
# 12:20-13:10 UTC = 20:20-21:10 Beijing, surrounding the supplied framing.
/opt/soft/anaconda3/bin/conda run --no-capture-output -n py311 python syisr/lfm_epb/resolution_sweep.py prepare "$root" --parts "$parts" --start 1772367600 --stop 1772370600
"$mpi/bin/mpirun" -np "$SLURM_NTASKS" "$mpi/bin/python" syisr/lfm_epb/resolution_sweep.py sweep "$root"
export HDF5_DIR=/opt/soft/anaconda3
export RUSTFLAGS='-C link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib'
CARGO_BUILD_JOBS=8 cargo build --release --locked --offline --manifest-path syisr/plots/Cargo.toml --bin lfm_resolution
syisr/plots/target/release/lfm_resolution "$root" "$root/figures"
date -u >"$root/COMPLETE"
