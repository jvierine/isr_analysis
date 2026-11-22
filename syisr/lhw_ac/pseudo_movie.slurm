#!/usr/bin/env bash
#SBATCH --job-name=lhw-stitch
#SBATCH --partition=node
#SBATCH --nodelist=node02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:40:00
set -euo pipefail
root=${1:?run root}
src="$root/pseudo-source/syisr/lhw_ac"
out="$root/pseudo-perpendicular"
mkdir -p "$out/qa"
export PATH="$root/bin:/home/juha/texlive/bin/x86_64-linux:/home/juha/.cargo/bin:/opt/soft/anaconda3/bin:$PATH"
export LC_ALL=C LANG=C OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=2
export SOURCE_GIT_COMMIT=$(cat "$root/pseudo-source-commit.txt")
export HDF5_DIR=/opt/soft/anaconda3 RUSTFLAGS="-C target-cpu=native -C link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib"
cd "$src"
conda run -n base python -m unittest test_reduction
conda run -n base python compose_sectors.py "$root/spectra.h5" "$out/sectors.h5" --satellite-calibration "$root/satellite-calibration/calibration.h5" --elevation-offset-deg=-0.72 --altitude-top=500 --feature-cuts
conda run -n base python verify_composite.py "$root/spectra.h5" "$out/sectors.h5"
cd plot
CARGO_BUILD_JOBS=4 cargo build --release --locked --offline --bin composite
movie="$out/syisr-lhw-ac-stitched-sectors-30min-4gate.mp4"
target/release/composite "$out/sectors.h5" "$movie"
ffmpeg -v error -progress "$out/decode.txt" -i "$movie" -f null -
for spec in first:0 middle:12.8125 last:25.5; do
 tag=${spec%:*};t=${spec#*:}
 ffmpeg -v error -y -ss "$t" -i "$movie" -frames:v 1 "$out/qa/$tag.png"
done
conda run -n base python "$src/check_composite_movie.py" "$out"
date -u >"$out/COMPLETE"
