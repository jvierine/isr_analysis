#!/usr/bin/env bash
# First continuous hour, one real 60 s integration per frame. No interpolation.
set -euo pipefail
product_root=/gpfs/pub/data_other/juha/plasmaline-20261125
source_root=/home/juha/src/plasmaline-cuda-20261125/syisr/plasmaline_cuda
movie_root="$product_root/movie-0000-0101"
extract="$source_root/rust_extract/target/release/block_display"
render="$source_root/rust_render/target/release/plasma-rustplotlib"
limits="$product_root/five-profiles-step2us/display_settings_frequency_normalized.h5"
ffmpeg=/opt/soft/source/ffmpeg-7.0.2-amd64-static/ffmpeg
export LD_LIBRARY_PATH="/opt/soft/anaconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/home/juha/texlive/bin/x86_64-linux:$PATH"
mkdir -p "$movie_root/frames"
for index in $(seq 0 60); do
  epoch=$((1795564800+index*60))
  printf -v frame '%05d' "$index"
  minute_root="$movie_root/minute-$epoch"
  output="$movie_root/frames/frame-$frame.png"
  if test -s "$output"; then continue; fi
  input="$product_root/$epoch/matched-step2us.h5"
  while ! test -s "$input"; do
    if grep -q "^FAILED $epoch:" "$product_root"/campaign-shard*.log; then
      echo "ACTION REQUIRED: minute $epoch failed processing; refusing to invent a video frame" >&2
      exit 3
    fi
    sleep 5
  done
  mkdir -p "$minute_root/input" "$minute_root/display" "$minute_root/plots"
  ln -sfn "$input" "$minute_root/input/block-$epoch.h5"
  "$extract" "$minute_root/input" "$minute_root/display" "$limits"
  "$render" "$minute_root/display" "$minute_root/plots" -6000000
  cp "$minute_root/plots/profile-01-rustplotlib-preview.png" "$output.partial"
  mv "$output.partial" "$output"
  echo "FRAME COMPLETE $index $epoch $output"
done
"$ffmpeg" -hide_banner -loglevel warning -y -framerate 4 -start_number 0 \
  -i "$movie_root/frames/frame-%05d.png" -vf fps=24 \
  -c:v libx264 -preset medium -crf 16 -pix_fmt yuv420p -movflags +faststart \
  "$movie_root/plasmaline-20261125-0000-0101.mp4"
echo "MOVIE COMPLETE $movie_root/plasmaline-20261125-0000-0101.mp4"
