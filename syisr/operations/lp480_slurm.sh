#!/usr/bin/env bash
# Submit on SYISR with sbatch; source stays in the shared main checkout.
# Usage: sbatch --output=RUN/logs/slurm-%j.log SCRIPT CODE RUN RAW_DAY...
#SBATCH --job-name=syisr-Hplus
#SBATCH --partition=node
#SBATCH --nodelist=node01
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --mem=180G
#SBATCH --time=7-00:00:00
set -euo pipefail
code=${1:?code checkout}; root=${2:?run directory}; shift 2
(( $# > 0 )) || { echo 'Specify raw day directories'; exit 1; }
export PATH=/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:$PATH
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HDF5_DIR=${HDF5_DIR:-/usr}
export RANKS=${SLURM_NTASKS:?} RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:?}
export SYISR_TOPSIDE_TEMPERATURE_SLOPE=1 SYISR_MULTISTART=1
export SYISR_HEADER_UTC_OFFSET_S=28800
mkdir -p "$root"/{logs,raw,decoded/products,input,bin}
exec 8>"$root/preparation.lock"
flock -n 8 || { echo 'Preparation already active'; exit 1; }
trap 'echo "FAILED stage=${stage:-setup} line=$LINENO"; date -u >"$root/FAILED"' ERR
stage=build
cd "$code"
git rev-parse HEAD >"$root/code-commit.txt"
CARGO_BUILD_JOBS=8 cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin syisr_lp480 >"$root/logs/build.log" 2>&1
cp syisr/target/release/syisr_lp480 "$root/bin/syisr_lp480"
bin=$root/bin/syisr_lp480
sha256sum "$bin" >"$root/binary-sha256.txt"
stage=index
for dir in "$@"; do
 test -d "$dir"
 day=${dir##*/}
 while IFS= read -r -d '' f; do
  ln -sfn "$f" "$root/raw/${day}-${f##*/}"
 done < <(find "$dir" -maxdepth 1 -type f -name '*.dat*' -print0)
done
if [[ ! -f $root/decoded/index.h5 ]]; then
 "$bin" index "$root/raw" "$root/decoded/index.h5" >"$root/logs/index.log" 2>&1
fi
stage=screened-products
srun --mpi=none --ntasks="$RANKS" --cpus-per-task="$RAYON_NUM_THREADS" --cpu-bind=cores --kill-on-bad-exit=1 \
 "$bin" prepare "$root/decoded/index.h5" "$root/decoded/products" >"$root/logs/prepare.log" 2>&1
stage=quiet-background
if [[ ! -f $root/decoded/background.h5 ]]; then
 "$bin" background "$root/decoded" "$root/decoded/background.h5" >"$root/logs/background.log" 2>&1
fi
stage=compact
if [[ ! -f $root/COMPACT_COMPLETE ]]; then
 "$bin" compact "$root/decoded" "$root/input" >"$root/logs/compact.log" 2>&1
 date -u >"$root/COMPACT_COMPLETE"
fi
test -d "$root/input/references"
stage=density-calibration
if [[ ! -f $root/input/calibration.h5 ]]; then
 l2=()
 for dir in "$@"; do
  day=${dir##*/}
  f=/gpfs/pub/data/L2/SYISR_results/h5/L2/2024/OSAYA_TISR01_IFIT_L2_DAY_${day}000000_V01.00_A_LP_30sec.h5
  test -f "$f"; l2+=("$f")
 done
 "$bin" import-l2 "$root/input" "${l2[@]}" >"$root/logs/l2-reference.log" 2>&1
 "$bin" calibrate "$root/input" unused >"$root/logs/calibrate.log" 2>&1
fi
stage=joint-composition-fits
flock -u 8
bash "$code/syisr/operations/lp480_controller.sh" "$root" "$bin" >"$root/logs/controller.log" 2>&1
stage=complete
echo 'Nominal temporal fit audit complete; parameter uncertainty validation is separate.'
