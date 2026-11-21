#!/usr/bin/env bash
#SBATCH --job-name=lfm-morphology
#SBATCH --partition=node
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
export PATH=/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:$HOME/texlive/bin/x86_64-linux:$PATH
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
((SLURM_NTASKS<=16 && SLURM_CPUS_PER_TASK==1)) || exit 1
code=${1:?code};root=${2:?root}
exec 9>"$root/morphology.lock";flock -n 9 || exit 1
cd "$code"
export HDF5_DIR=/opt/soft/anaconda3 RUSTFLAGS='-C link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib'
git rev-parse HEAD >"$root/morphology-code-commit.txt"
CARGO_BUILD_JOBS=8 cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin lfm_epb_geometry --bin lfm_epb_lift
cargo test --release --locked --offline --manifest-path syisr/Cargo.toml --lib lfm_geometry
if [[ -d $root/cavity && ! -d $root/cavity_centerline_initial ]];then mv "$root/cavity" "$root/cavity_centerline_initial";fi
syisr/target/release/lfm_epb_geometry "$root" "$root/geometry.h5"
bash syisr/lfm_epb/cavity.slurm "$code" "$root"
syisr/target/release/lfm_epb_lift "$root/cavity/all_beams.h5" "$root/cavity/volume.h5"
CARGO_BUILD_JOBS=8 cargo build --release --locked --offline --manifest-path syisr/plots/Cargo.toml --bin lfm_epb_views --bin lfm_resolution
syisr/plots/target/release/lfm_epb_views "$root" "$root/figures"
study=/gpfs/pub/data_other/juha/lfm_epb/20260301/resolution_study
/opt/soft/anaconda3/bin/conda run --no-capture-output -n py311 python syisr/lfm_epb/resolution_sweep.py assess "$study"
syisr/plots/target/release/lfm_resolution "$study" "$study/figures"
date -u >"$root/MORPHOLOGY_COMPLETE"
