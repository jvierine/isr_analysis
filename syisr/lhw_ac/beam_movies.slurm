#!/usr/bin/env bash
#SBATCH --job-name=lhw-beams
#SBATCH --partition=node
#SBATCH --nodelist=node02
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
set -euo pipefail
root=${1:?run root}
src="$root/beam-source/syisr/lhw_ac"
out="$root/individual-beams"
mkdir -p "$out/qa"
export PATH="/opt/slurm/slurm-22.05.11/bin:$root/bin:/home/juha/texlive/bin/x86_64-linux:/home/juha/.cargo/bin:/opt/soft/anaconda3/bin:$PATH"
export LC_ALL=C LANG=C OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=2
export SOURCE_GIT_COMMIT=$(cat "$root/beam-source-commit.txt")
export HDF5_DIR=/opt/soft/anaconda3 RUSTFLAGS="-C target-cpu=native -C link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib"
cd "$src"
conda run -n base python prepare_beam_movies.py "$root/spectra.h5" "$out"
conda run -n base python verify_beam_movies.py "$root/spectra.h5" "$out"
cd plot
CARGO_BUILD_JOBS=4 cargo build --release --locked --offline --bin composite
export BEAM_RENDERER="$src/plot/target/release/composite" BEAM_OUTPUT="$out"
srun --mpi=none --cpu-bind=cores --kill-on-bad-exit=1 --output="$out/render-%t.log" bash -c '
 alt=$((150+SLURM_PROCID*50))
 "$BEAM_RENDERER" "$BEAM_OUTPUT/beam-${alt}km.h5" "$BEAM_OUTPUT/syisr-ac-${alt}km-corrected.mp4"
'
: >"$out/concat.txt"
for alt in 150 200 250 300 350; do
 movie="$out/syisr-ac-${alt}km-corrected.mp4"
 ffmpeg -v error -progress "$out/decode-${alt}.txt" -i "$movie" -f null -
 printf "file '%s'\n" "$movie" >>"$out/concat.txt"
 for spec in first:0 middle:12.8125 last:25.5; do
  tag=${spec%:*};t=${spec#*:}
  ffmpeg -v error -y -ss "$t" -i "$movie" -frames:v 1 "$out/qa/${alt}-${tag}.png"
 done
done
ffmpeg -v error -y -f concat -safe 0 -i "$out/concat.txt" -c copy -movflags +faststart "$out/syisr-ac-all-beams-corrected.mp4"
ffmpeg -v error -progress "$out/decode-all.txt" -i "$out/syisr-ac-all-beams-corrected.mp4" -f null -
conda run -n base python "$src/check_beam_movies.py" "$out"
date -u >"$out/COMPLETE"
