#!/bin/bash
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=epbrti-seed
set -euo pipefail
export PATH=$HOME/.cargo/bin:/opt/slurm/slurm-22.05.11/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HDF5_DIR=/opt/soft/anaconda3 RUSTFLAGS='-C link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib'
export CARGO_TARGET_DIR=/home/juha/src/isr_analysis/syisr/target
cd "${1:?code}"
cargo build -j3 --release --manifest-path syisr/Cargo.toml --bin lfm_epb_editor_seed
export SYISR_CAVITY_DEPLETION=.2 SYISR_CAVITY_ITERATIONS=8000
/opt/soft/anaconda3/envs/py311/bin/mpirun -np 3 /opt/soft/anaconda3/envs/py311/bin/python syisr/lfm_epb/meridians/editor_seed.py "${2:?root}"
