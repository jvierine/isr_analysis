#!/usr/bin/env bash
# Limited corrected-background retry. One explicitly allocated day; never release old jobs.
set -euo pipefail
code=${1:?}; campaign=${2:?}; day=${3:?}; original=${4:?}; receiver=${5:?}; previous=${6:-none}
root=$campaign/days/$day
export PATH=/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:$HOME/texlive/bin/x86_64-linux:$PATH
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HDF5_DIR=/opt/soft/anaconda3 RUSTFLAGS=-C\ link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib
export RANKS=${SLURM_NTASKS:?} RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:?}
(( RANKS*RAYON_NUM_THREADS==64 || RANKS*RAYON_NUM_THREADS==128 ))
export SYISR_LP480_TOP_KM=1800 SYISR_LP480_PROFILE_CURVATURE=1 SYISR_TOPSIDE_TEMPERATURE_SLOPE=1
export SYISR_LP480_H_TIME_SIGMA=0.1 SYISR_LP480_H_NEIGHBOR_START=1 SYISR_MULTISTART=1
export SYISR_LP480_PULSE_COVARIANCE=1 SYISR_PLOT_UTC=1
# Independent starts after gaps; close boundary controls enter through the
# measured temporal neighbor only. Never import stale same-time old fits.
unset SYISR_INITIAL_PROFILE_DIR SYISR_NEAREST_INITIAL_PROFILE SYISR_FIXED_TIME_CONTEXT SYISR_LBFGS SYISR_DIAGONAL_STEP SYISR_WARMSTART_DIR INITIAL_PRIOR
mkdir -p "$root"/{logs,decoded/products,input,bin} "$campaign/bin"
exec 8>"$root/day.lock";flock -n 8
trap 'echo "FAILED stage=${stage:-setup} line=$LINENO" >"$root/FAILED"' ERR
test ! -f "$root/DAILY_COMPLETE" || exit 0
rm -f "$root/FAILED"
stage=build
cd "$code"
if [[ -f $root/COMPLETE ]]; then git rev-parse HEAD >"$root/export-code-commit.txt"; else git rev-parse HEAD >"$root/code-commit.txt"; fi
if [[ ! -f $campaign/BUILD_COMPLETE ]]; then
 CARGO_BUILD_JOBS=4 cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin syisr_lp480 >"$root/logs/build.log" 2>&1
 CARGO_BUILD_JOBS=4 cargo test --release --locked --offline --manifest-path syisr/Cargo.toml --lib lp480 >"$root/logs/tests.log" 2>&1
 cp syisr/target/release/syisr_lp480 "$campaign/bin/"
 date -u >"$campaign/BUILD_COMPLETE"
fi
# Login and compute nodes report the same libc release but differ in a
# backported TLS symbol. Never reuse plotting artifacts linked on the login node.
if [[ ! -f $campaign/COMPUTE_PLOT_BUILD_COMPLETE ]]; then
 test -n "${SLURM_JOB_ID:-}"
 CARGO_TARGET_DIR="$campaign/compute-plot-target" CARGO_BUILD_JOBS=4 cargo build --release --locked --offline --manifest-path syisr/plots/Cargo.toml --bin lp480_diagnostics --bin lp480_maps --bin lp480_residual_time >"$root/logs/compute-diagnostic-build.log" 2>&1
 for name in lp480_diagnostics lp480_maps lp480_residual_time; do cp "$campaign/compute-plot-target/release/$name" "$campaign/bin/"; done
 date -u >"$campaign/COMPUTE_PLOT_BUILD_COMPLETE"
fi
cp "$campaign/bin/syisr_lp480" "$root/bin/";bin=$root/bin/syisr_lp480
if [[ ! -f $root/COMPLETE ]]; then
stage=validated-index
supported_jobs=$(/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python - "$receiver/daily-index/$day.h5" "$root/decoded/index.h5" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

source, output = map(Path, sys.argv[1:])
with h5py.File(source) as src:
    index = json.loads(bytes(src["json_utf8"][:]))

# The decoder requires 64 pulses and drops individual blocks shorter than 64.
supported, excluded = [], []
for job in index["jobs"]:
    counts = Counter()
    for part in job["parts"]:
        counts[part["block"]] += part["pulses"]
    if max(counts.values(), default=0) >= 64:
        supported.append(job)
    else:
        excluded.append({"job": job, "reason": "No block has 64 indexed pulses"})
        print(f"INSUFFICIENT_PULSE_SUPPORT {job['id']} block_counts={dict(counts)}", file=sys.stderr)

index["jobs"] = supported
index["excluded_insufficient_pulse_support"] = excluded
temporary = output.with_suffix(".pending.h5")
with h5py.File(temporary, "w") as dst:
    dst.create_dataset("json_utf8", data=np.frombuffer(json.dumps(index).encode(), dtype="u1"))
temporary.replace(output)
print(len(supported))
PY
)
[[ $supported_jobs =~ ^[0-9]+$ ]]
if (( supported_jobs == 0 )); then
 date -u >"$root/NO_OBSERVATIONS"
 date -u >"$root/COMPLETE"
 date -u >"$root/DAILY_COMPLETE"
 exit 0
fi
stage=temporal-boundary
if [[ $previous != none ]]; then
 /opt/soft/anaconda3/bin/conda run --no-capture-output -n base python "$code/syisr/operations/lp480_day_context.py" "$root" "$previous"
else
 /opt/soft/anaconda3/bin/conda run --no-capture-output -n base python "$code/syisr/operations/lp480_day_context.py" "$root"
fi
source "$root/day-context.env"
stage=pulse-covariance
srun --mpi=none --ntasks="$RANKS" --cpus-per-task="$RAYON_NUM_THREADS" --cpu-bind=cores --kill-on-bad-exit=1 \
 "$bin" prepare "$root/decoded/index.h5" "$root/decoded/products" >"$root/logs/prepare.log" 2>&1
stage=corrected-background
cp "$receiver/affine-calibration/reference/background.h5" "$root/decoded/background.h5"
"$bin" compact "$root/decoded" "$root/input" >"$root/logs/compact.log" 2>&1
ln -sfn "$(readlink -f "$original/references")" "$root/input/references"
cp "$original/days/$day/input/calibration.h5" "$original/days/$day/input/calibration_metadata.h5" "$root/input/"
stage=plasma-fit
bash "$code/syisr/operations/lp480_controller.sh" "$root" "$bin" >"$root/logs/controller.log" 2>&1
else
 source "$root/day-context.env"
fi
# Controller COMPLETE means numerical temporal convergence; export gets its own marker.
stage=export
"$bin" snapshot "$root/final" "$root/snapshot" >"$root/logs/snapshot.log" 2>&1
if "$campaign/bin/lp480_maps" maps "$root/snapshot" "$root/plots" >"$root/logs/plots.log" 2>&1; then
 date -u >"$root/PLOTS_COMPLETE";rm -f "$root/PLOTS_PENDING"
else
 # Completed numerical data remain usable if the optional renderer is incompatible.
 date -u >"$root/PLOTS_PENDING"
fi
date -u >"$root/EXPORT_COMPLETE"

stage=diagnostics
if [[ -x $campaign/bin/lp480_diagnostics ]] && bash "$code/syisr/operations/lp480_diagnostics.sh" "$campaign/bin/lp480_diagnostics" "$root" "$root/diagnostics" >"$root/logs/diagnostics.log" 2>&1; then
 date -u >"$root/DIAGNOSTICS_GENERATED_REVIEW_REQUIRED";rm -f "$root/DIAGNOSTICS_PENDING"
else
 date -u >"$root/DIAGNOSTICS_PENDING"
fi

stage=short-lag-time
test -x "$campaign/bin/lp480_residual_time"
beams=$(/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python -c 'import glob,h5py,json,sys; beams=set();
for p in glob.glob(sys.argv[1]+"/*.h5"):
 with h5py.File(p) as f:w=json.loads(bytes(f["json_utf8"][:]))
 beams.update((round(b["azimuth_deg"],2),round(b["elevation_deg"],2)) for b in w["beams"])
for az,el in sorted(beams):print(az,el)' "$root/snapshot")
test -n "$beams"
while read -r az el; do
 key=$az-$el
 /opt/soft/anaconda3/bin/conda run --no-capture-output -n base python "$code/syisr/operations/lp480_export_short_lags.py" "$campaign" "$root/short-lag-$key.h5" "$day" --az "$az" --el "$el"
 "$campaign/bin/lp480_residual_time" "$root/short-lag-$key.h5" "$root/short-lag-time/$key" >"$root/logs/short-lag-$key.log" 2>&1
 test -s "$root/short-lag-time/$key/first10-residual-time.png"
done <<<"$beams"
# Required plots must succeed before starting the next day. Numerical fits and
# their COMPLETE marker remain preserved if only a renderer needs recovery.
test -f "$root/PLOTS_COMPLETE"
test ! -f "$root/DIAGNOSTICS_PENDING"
date -u >"$root/DAILY_COMPLETE"
rm -f "$root/FAILED"
