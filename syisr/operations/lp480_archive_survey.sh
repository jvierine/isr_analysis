#!/usr/bin/env bash
set -euo pipefail
code=${1:?}; output=${2:?}; baseline=${3:?}
cd "$code"
export PATH=$HOME/.cargo/bin:$PATH HDF5_DIR=/opt/soft/anaconda3
export RUSTFLAGS='-C link-arg=-Wl,-rpath,/opt/soft/anaconda3/lib'
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
CARGO_BUILD_JOBS=4 cargo build --release --locked --offline --manifest-path syisr/Cargo.toml --bin syisr_lp480_inventory
base=$(/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python -c 'import h5py,json,sys;f=h5py.File(sys.argv[1]);print(json.loads(bytes(f["json_utf8"][:]))["jobs"][0]["parts"][0]["file"])' "$baseline")
syisr/target/release/syisr_lp480_inventory /gpfs/pub/data/L0 "$base" "$output"
