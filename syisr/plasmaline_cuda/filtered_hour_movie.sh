#!/usr/bin/env bash
# Native independently normalized channels; RF splice at 440 MHz, no wrapping.
set -euo pipefail
shard="${1:-0}"
shards="${2:-3}"
sigma="${3:-50}"
root=/gpfs/pub/data_other/juha/plasmaline-20261125/dual-channel-hour
src=/home/juha/src/plasmaline-cuda-20261125/syisr/plasmaline_cuda
export LD_LIBRARY_PATH="/opt/soft/anaconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/home/juha/texlive/bin/x86_64-linux:$PATH"
if test "$sigma" = 50; then movie="$root/gaussian-sigma50"; else movie="$root/gaussian-f$sigma-r$sigma"; fi
mkdir -p "$movie/frames"
for index in $(seq 0 59); do
  if test "$((index%shards))" -ne "$shard"; then continue; fi
  epoch=$((1795564800+index*60))
  printf -v frame '%05d' "$index"
  output="$movie/frames/frame-$frame.png"
  if test -s "$output"; then continue; fi
  while ! test -s "$root/$epoch/436000000/matched-step2us.h5" || ! test -s "$root/$epoch/444000000/matched-step2us.h5"; do sleep 5; done
  display="$movie/minute-$epoch/display"
  plots="$movie/minute-$epoch/plots"
  mkdir -p "$display" "$plots"
  limits=("$root/movie/minute-1795564800/display/profile-01-native.h5")
  "$src/rust_extract/target/release/splice_channels" "$root/$epoch" "$display" "${limits[@]}" "$sigma" "$sigma"
  "$src/rust_render/target/release/plasma-rustplotlib" "$display" "$plots" rf video
  cp "$plots/profile-01-rustplotlib-preview.png" "$output.partial"
  mv "$output.partial" "$output"
  echo "FRAME COMPLETE $index $epoch"
done
if test "$shard" -ne 0; then exit 0; fi
while test "$(find "$movie/frames" -maxdepth 1 -name 'frame-*.png' | wc -l)" -lt 60; do sleep 5; done
"$src/rust_extract/target/release/dual_manifest" "$root"
/opt/soft/source/ffmpeg-7.0.2-amd64-static/ffmpeg -hide_banner -loglevel warning -y -framerate 4 -i "$movie/frames/frame-%05d.png" -vf fps=24 -c:v libx264 -threads 4 -preset fast -crf 16 -pix_fmt yuv420p -movflags +faststart "$movie/plasmaline-20261125-0000-0100-filtered-sigma$sigma.mp4"
echo "MOVIE COMPLETE: 60 filtered one-minute integrations; Gaussian sigma $sigma native bins"
