#!/usr/bin/env bash
#SBATCH --job-name=lhw-movies
#SBATCH --partition=node
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
root=${1:?run root}
export PATH="$root/bin:/home/juha/texlive/bin/x86_64-linux:/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:/opt/soft/anaconda3/bin:$PATH"
export PYTHONPATH="$root/media-python${PYTHONPATH:+:$PYTHONPATH}"
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=2
export HDF5_DIR=/opt/soft/anaconda3 RUSTFLAGS="-C target-cpu=native -C link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib"
mkdir -p "$root"/{movies/qa,logs,bin}
if [[ -f $root/FAILED_RENDER ]]; then mv "$root/FAILED_RENDER" "$root/logs/failed-render-before-$SLURM_JOB_ID.txt"; fi
trap 'date -u >"$root/FAILED_RENDER"' ERR
encoder=$(find "$root/media-python/imageio_ffmpeg/binaries" -maxdepth 1 -type f -name 'ffmpeg-*' -print -quit)
test -n "$encoder" && test -x "$encoder"
ln -sfn "$encoder" "$root/bin/ffmpeg"
ffmpeg -version >"$root/logs/ffmpeg-version.log"
cd "$root/source/syisr/lhw_ac/plot"
CARGO_BUILD_JOBS=8 cargo build --release --locked --offline >"$root/logs/render-build.log" 2>&1
cp target/release/syisr-lhw-ac-movie "$root/bin/renderer"
sha256sum "$root/bin/renderer" >"$root/renderer-binary-sha256.txt"
sha256sum src/main.rs >"$root/renderer-source-sha256.txt"
export TASK_ROOT="$root"
export MOVIE_LABEL=${2:-15min_step5min}
read middle last < <(python -c 'import h5py,sys;h=h5py.File(sys.argv[1]);n=len(h["window_start_utc_s"]);print(n/16,(n-1)/8)' "$root/movie-input.h5")
srun --mpi=none --ntasks=5 --cpus-per-task=4 --cpu-bind=cores --kill-on-bad-exit=1 --output="$root/logs/render-%t.log" bash -c '
 alt=$((150+SLURM_PROCID*50))
 "$TASK_ROOT/bin/renderer" "$TASK_ROOT/movie-input.h5" "$TASK_ROOT/movies/figures" "beam=$SLURM_PROCID" colormap=inferno cuts "movie=$TASK_ROOT/movies/syisr_lhw_ac_${alt}km_${MOVIE_LABEL}.mp4"
'
: >"$root/movies/concat.txt"
for alt in 150 200 250 300 350; do
 movie="$root/movies/syisr_lhw_ac_${alt}km_${MOVIE_LABEL}.mp4"
 ffmpeg -v error -progress "$root/logs/decode-${alt}km.txt" -i "$movie" -f null -
 printf "file '%s'\n" "$movie" >>"$root/movies/concat.txt"
 for spec in first:0 middle:$middle last:$last; do
  tag=${spec%:*}; instant=${spec#*:}
  ffmpeg -v error -y -ss "$instant" -i "$movie" -frames:v 1 "$root/movies/qa/${alt}km-${tag}.png"
 done
done
joined="$root/movies/syisr_lhw_ac_20261119_all_beams_${MOVIE_LABEL}.mp4"
ffmpeg -v error -y -f concat -safe 0 -i "$root/movies/concat.txt" -c copy -movflags +faststart "$joined"
ffmpeg -v error -progress "$root/logs/decode-all.txt" -i "$joined" -f null -
conda run -n base python "$root/source/syisr/lhw_ac/check_movies.py" "$root" "$MOVIE_LABEL"
date -u >"$root/COMPLETE_RENDER"
