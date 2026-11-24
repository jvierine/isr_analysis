#!/usr/bin/env bash
set -euo pipefail
code=${1:?}; root=${2:?}; month=${3:?}; survey=${4:?}
cd "$code"
export PATH=/opt/slurm/slurm-22.05.11/bin:$HOME/.cargo/bin:$PATH HDF5_DIR=/opt/soft/anaconda3
export RUSTFLAGS='-C link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib'
export SYISR_HEADER_UTC_OFFSET_S=28800 SYISR_ALLOW_PARTIAL_FINAL_FRAME=1 SYISR_QUARANTINE_CORRUPT_FRAMES=1
CARGO_BUILD_JOBS=4 cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin syisr_lp480
bin=$code/syisr/target/release/syisr_lp480
srun --mpi=none --ntasks="${SLURM_NTASKS:?}" --ntasks-per-node="$((SLURM_NTASKS/SLURM_NNODES))" --cpus-per-task="${SLURM_CPUS_PER_TASK:?}" --cpu-bind=cores --kill-on-bad-exit=1 /opt/soft/anaconda3/bin/conda run --no-capture-output -n base python "$code/syisr/operations/lp480_prepare_calendar.py" index "$bin" "$survey" "$root" --month "$month"
/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python "$code/syisr/operations/lp480_prepare_calendar.py" daily "$bin" "$survey" "$root" --month "$month"
date -u >"$root/INDEX_COMPLETE"
