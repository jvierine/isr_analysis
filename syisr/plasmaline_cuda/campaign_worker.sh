#!/usr/bin/env bash
set -euo pipefail
device="$1"
cd /home/juha/src/plasmaline-cuda-20261125/syisr/plasmaline_cuda
args=(campaign.py /gpfs/pub/data/L0/2026/plasmaline/20261125 /gpfs/pub/data_other/juha/plasmaline-20261125 --device "$device" --shard "$device" --shards 3 --range-step-us 2)
/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python -u "${args[@]}" --stop-utc 1795568460
exec /opt/soft/anaconda3/bin/conda run --no-capture-output -n base python -u "${args[@]}"
