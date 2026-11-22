#!/usr/bin/env bash
# Receiver-only export/render after the raw audit; no plasma fit or job release.
set -euo pipefail
code=${1:?}; month=${2:?}; root=${3:?}; plotcrate=${4:?}
export PATH=/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:$HOME/texlive/bin/x86_64-linux:$PATH
export LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=2
export HDF5_DIR=/opt/soft/anaconda3 RUSTFLAGS=-C\ link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib
export SYISR_LP480_TOP_KM=1800
export SYISR_RECEIVER_PLOT_THREADS=16
test -f "$root/COMPLETE";test -f "$month/USER_PAUSED_H_FITTING"
mkdir -p "$root/plots" "$root/export-bin"
exec 8>"$root/export.lock";flock -n 8
trap 'date -u >"$root/EXPORT_FAILED"' ERR
cd "$code"
if [[ ${SYISR_RECEIVER_RENDER_ONLY:-0} != 1 ]]; then
git rev-parse HEAD >"$root/export-code-commit.txt"
CARGO_BUILD_JOBS=4 cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin syisr_lp480 >"$root/logs/export-build.log" 2>&1
CARGO_BUILD_JOBS=4 cargo test --release --locked --offline --manifest-path syisr/Cargo.toml --lib lp480 >"$root/logs/export-tests.log" 2>&1
cp syisr/target/release/syisr_lp480 "$root/export-bin/"
/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python "$code/syisr/operations/lp480_receiver_check.py" reports "$month" "$root" "$root/export-bin/syisr_lp480" >"$root/logs/validated-inputs.log" 2>&1
fi
git rev-parse HEAD >"$root/render-code-commit.txt"
cp "$code/syisr/plots/src/bin/lp480_receiver_maps.rs" "$plotcrate/src/bin/"
plot_target=$plotcrate/target
if [[ ${SYISR_RECEIVER_FRESH_PLOT_TARGET:-0} == 1 ]]; then
 plot_target=$root/fresh-plot-target
 unset RUSTC_WRAPPER RUSTC_WORKSPACE_WRAPPER
else
 cargo clean --manifest-path "$plotcrate/Cargo.toml" --target-dir "$plot_target" -p rustplotlib >"$root/logs/plot-cache-refresh.log" 2>&1
fi
rustc -vV >"$root/logs/plot-toolchain.log"
CARGO_BUILD_JOBS=16 cargo build --release --locked --offline --target-dir "$plot_target" --manifest-path "$plotcrate/Cargo.toml" --bin lp480_receiver_maps >"$root/logs/plot-build.log" 2>&1
plot=$plot_target/release/lp480_receiver_maps
"$plot" spectra "$root/stored-tail" "$root/plots/stored" >"$root/logs/plot-stored.log" 2>&1
date -u >"$root/STORED_PLOTS_COMPLETE"
"$plot" spectra "$root/raw-tail" "$root/plots/raw" >"$root/logs/plot-raw.log" 2>&1
for date in 20240520 20240531; do "$plot" acfs "$root/corrected-acfs/$date" "$root/plots/acfs"; done
date -u >"$root/PLOTS_COMPLETE"
