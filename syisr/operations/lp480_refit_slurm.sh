#!/usr/bin/env bash
# Reuse screened observations and fitted states; never reuse the old objective's
# optimizer damping/history or label old outputs as fits under a changed prior.
# sbatch [resources] SCRIPT CODE BASE_RUN OUTPUT_RUN [UTC_EPOCH ...]
# With epochs, run a validation pilot. Without epochs, run all temporal sweeps.
set -euo pipefail
code=${1:?code}; base=${2:?base run}; root=${3:?output run}; shift 3
export PATH=/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:$PATH
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HDF5_DIR=/opt/soft/anaconda3
export RUSTFLAGS="-C link-arg=-Wl,-rpath,$HDF5_DIR/lib"
export RANKS=${SLURM_NTASKS:?} RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:?}
export SYISR_LP480_TOP_KM=1800 SYISR_LP480_PROFILE_CURVATURE=1
export SYISR_TOPSIDE_TEMPERATURE_SLOPE=1 SYISR_DATA_STABILITY_STOP=1
export SYISR_LP480_LATEST_FIRST=1
export SYISR_INITIAL_PROFILE_DIR=$base/fits/initial
export SYISR_WARMSTART_DIR=$base/fits/initial
export INITIAL_PRIOR=$base/fits/initial
mkdir -p "$root"/{logs,bin,fits/initial}
exec 8>"$root/refit.lock"
flock -n 8 || { echo 'Refit already active'; exit 1; }
trap 'date -u >"$root/FAILED"' ERR
ln -sfn "$base/input" "$root/input"
cd "$code"
git rev-parse HEAD >"$root/code-commit.txt"
CARGO_BUILD_JOBS=4 cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin syisr_lp480 >"$root/logs/build.log" 2>&1
CARGO_BUILD_JOBS=4 cargo test --release --locked --offline --manifest-path syisr/Cargo.toml --lib lp480_fit >"$root/logs/tests.log" 2>&1
cp syisr/target/release/syisr_lp480 "$root/bin/syisr_lp480"
bin=$root/bin/syisr_lp480
sha256sum "$bin" >"$root/binary-sha256.txt"
if (( $# )); then
 for time in "$@"; do
  srun --mpi=none --ntasks=1 --cpus-per-task="$RAYON_NUM_THREADS" --cpu-bind=cores \
   "$bin" fit "$root/input" "$time" "$root/fits/initial" "$INITIAL_PRIOR" 0 \
   >"$root/logs/pilot-$time.log" 2>&1 &
 done
 failed=0
 for pid in $(jobs -p); do wait "$pid" || failed=1; done
 (( failed == 0 ))
 date -u >"$root/PILOT_COMPLETE"
else
 bash syisr/operations/lp480_controller.sh "$root" "$bin" >"$root/logs/controller.log" 2>&1
fi
