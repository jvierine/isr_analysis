#!/usr/bin/env bash
#SBATCH --job-name=lhw-transfer
#SBATCH --partition=node
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
root=${1:?root}
ffmpeg="$root/bin/ffmpeg"
master="$root/movies/syisr_lhw_ac_20261119_all_beams_15min_step5min.mp4"
out="$root/movies/syisr_lhw_ac_20261119_all_beams_15min_step5min_presentation.mp4"
"$ffmpeg" -v error -y -i "$master" -an -c:v libx264 -preset medium -crf 24 -threads 8 -pix_fmt yuv420p -movflags +faststart "$out"
"$ffmpeg" -v error -progress "$root/logs/presentation-decode.txt" -i "$out" -f null -
"$ffmpeg" -v error -y -ss 13 -i "$out" -frames:v 1 "$root/movies/qa/presentation-150km-middle.png"
export PATH=/opt/soft/anaconda3/bin:$PATH LC_ALL=C LANG=C
conda run -n base python -c 'import pathlib,sys,h5py,hashlib; r=pathlib.Path(sys.argv[1]);d=dict(x.split("=",1) for x in (r/"logs/presentation-decode.txt").read_text().splitlines() if "=" in x);assert d["progress"]=="end" and int(d["frame"])==3120 and abs(int(d["out_time_us"])/1e6-130)<.05;p=r/"movies/syisr_lhw_ac_20261119_all_beams_15min_step5min_presentation.mp4";h=h5py.File(r/"movies/movies.h5","a");h.attrs["presentation_sha256"]=hashlib.sha256(p.read_bytes()).hexdigest();h.attrs["presentation_encoding"]="1920x1080 H264 CRF24,24fps;3120frames130seconds;full decode passed";h.close();print("PASS presentation",p.stat().st_size)' "$root"
date -u >"$root/COMPLETE_PRESENTATION"
