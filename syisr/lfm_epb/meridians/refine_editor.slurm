#!/bin/bash
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --job-name=epb-detail
set -euo pipefail
export PATH=$HOME/.cargo/bin:/opt/slurm/slurm-22.05.11/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HDF5_DIR=/opt/soft/anaconda3 RUSTFLAGS='-C link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib'
export CARGO_TARGET_DIR=/home/juha/src/isr_analysis/syisr/target
cd "${1:?code}"
cargo build -j12 --release --manifest-path syisr/Cargo.toml --bin lfm_epb_editor_seed
/opt/soft/anaconda3/envs/py311/bin/mpirun -np 12 /opt/soft/anaconda3/envs/py311/bin/python syisr/lfm_epb/meridians/refine_editor.py "${2:?root}" "$CARGO_TARGET_DIR/release/lfm_epb_editor_seed"
