#!/bin/bash
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=epb-thermal-null
set -euo pipefail
code=${1:?code};out=${2:?out}
cd "$code"
export PATH=$HOME/.cargo/bin:$PATH HDF5_DIR=/opt/soft/anaconda3
export RUSTFLAGS='-C link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib'
export CARGO_TARGET_DIR=/home/juha/src/isr_analysis/syisr/target
export CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin lfm_coherent_thermal_null
SYISR_LH_ANGLES=9 "$CARGO_TARGET_DIR/release/lfm_coherent_thermal_null" "$out/thermal_null_angles09.h5"
SYISR_LH_ANGLES=33 "$CARGO_TARGET_DIR/release/lfm_coherent_thermal_null" "$out/thermal_null_angles33.h5"

SYISR_LH_ANGLES=33 "$CARGO_TARGET_DIR/release/lfm_coherent_thermal_null" "$out/thermal_fullband_angles33.h5" fullband
