#!/usr/bin/env bash
# Native independently normalized channels; RF splice at 440 MHz, no wrapping.
set -euo pipefail
shard="${1:-0}"
shards="${2:-24}"
sigma="${3:-50}"
viewmax="${4:-500}"
vmin="${5:--0.1}"
color_tag="${vmin//-/m}"
color_tag="${color_tag//./p}"
root=/gpfs/pub/data_other/juha/plasmaline-20261125/dual-channel-hour
src=/home/juha/src/plasmaline-cuda-20261125/syisr/plasmaline_cuda
export LD_LIBRARY_PATH="/opt/soft/anaconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/home/juha/texlive/bin/x86_64-linux:$PATH"
if test "$sigma" = 50; then movie="$root/gaussian-sigma50"; else movie="$root/gaussian-f$sigma-r$sigma"; fi
products="$movie"
first_hour="$movie/range$viewmax-vmin$color_tag"
movie="$root/snapshot-0000-0337-f10-r10-range500-vminm0p1"
mkdir -p "$movie/frames"
# Render the new minutes while the already-rendered first hour is reused.
for index in $(seq 60 216) $(seq 0 59); do
  if test "$((index%shards))" -ne "$shard"; then continue; fi
  epoch=$((1795564800+index*60))
  printf -v frame '%05d' "$index"
  output="$movie/frames/frame-$frame.png"
  if test -s "$output"; then continue; fi
  if test "$index" -lt 60; then
    while ! test -s "$first_hour/frames/frame-$frame.png"; do sleep 3; done
    ln "$first_hour/frames/frame-$frame.png" "$output"
    echo "FRAME REUSED $index $epoch"
    continue
  fi
  while ! test -s "$root/$epoch/436000000/matched-step2us.h5" || ! test -s "$root/$epoch/444000000/matched-step2us.h5"; do sleep 5; done
  display="$products/minute-$epoch/display"
  plots="$movie/minute-$epoch/plots"
  mkdir -p "$display" "$plots"
  limits=("$root/movie/minute-1795564800/display/profile-01-native.h5")
  if ! test -s "$display/profile-01-native.h5"; then
    "$src/rust_extract/target/release/splice_channels" "$root/$epoch" "$display" "${limits[@]}" "$sigma" "$sigma"
  fi
  "$src/rust_render/target/release/plasma-rustplotlib" "$display" "$plots" rf video "$viewmax" "$vmin"
  cp "$plots/profile-01-rustplotlib-preview.png" "$output.partial"
  mv "$output.partial" "$output"
  echo "FRAME COMPLETE $index $epoch"
done
if test "$shard" -ne 0; then exit 0; fi
while test "$(find "$movie/frames" -maxdepth 1 -name 'frame-*.png' | wc -l)" -lt 217; do sleep 5; done
/opt/soft/source/ffmpeg-7.0.2-amd64-static/ffmpeg -hide_banner -loglevel warning -y -framerate 4 -i "$movie/frames/frame-%05d.png" -vf fps=24 -c:v libx264 -threads 4 -preset fast -crf 16 -pix_fmt yuv420p -movflags +faststart "$movie/plasmaline-20261125-0000-0337-filtered-sigma10-range500-vminm0p1.mp4"
echo "MOVIE COMPLETE: 217 consecutive one-minute integrations through 03:37 UTC"
