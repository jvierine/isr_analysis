#!/bin/bash
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=epb-prediction
set -euo pipefail
export PATH=/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${1:?code}"
export HDF5_DIR=/opt/soft/anaconda3
export RUSTFLAGS='-C link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib'
export CARGO_TARGET_DIR=/home/juha/src/isr_analysis/syisr/target
cargo build -j 8 --release --manifest-path syisr/Cargo.toml --bin lfm_epb_cavity
/opt/soft/anaconda3/envs/py311/bin/mpirun -np "$SLURM_NTASKS" /opt/soft/anaconda3/envs/py311/bin/python syisr/lfm_epb/meridians/predict_rtis.py "${2:?root}" "$CARGO_TARGET_DIR/release/lfm_epb_cavity"
