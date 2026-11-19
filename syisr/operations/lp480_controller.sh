#!/usr/bin/env bash
# Detached block-coordinate optimization of the identical joint objective.
set -euo pipefail
root=${1:?run root}; bin=${2:?binary}; ranks=${RANKS:-32}; threads=${RAYON_NUM_THREADS:-4}
export RAYON_NUM_THREADS=$threads SYISR_DATA_STABILITY_STOP=1
mkdir -p "$root/logs" "$root/fits"
exec 9>"$root/controller.lock"
flock -n 9 || { echo 'Controller already active'; exit 1; }
run_phase() {
 local selection=$1 out=$2 prior=$3
 mkdir -p "$out"
 mpirun --allow-run-as-root --oversubscribe -np "$ranks" --map-by slot:PE="$threads" --bind-to core \
 "$bin" fit "$root/input" "$selection" "$out" "$prior" 0
 if [[ $prior != none ]]; then
  for f in "$prior"/*.h5; do
   local name=${f##*/}
   [[ $name =~ ^[0-9]+\.h5$ ]] || continue
   [[ -e $out/$name ]] || ln -s "$(readlink -f "$f")" "$out/$name"
  done
 fi
 ln -sfn "$out" "$root/latest"
}
prior=$root/fits/initial
run_phase all "$prior" none >"$root/logs/initial.log" 2>&1
"$bin" audit "$prior" none >"$root/logs/initial-audit.log" 2>&1 || [[ $? == 2 ]]
for ((sweep=1; ; sweep++)); do
 before=$prior
 for parity in even odd; do
  out=$root/fits/sweep-${sweep}-${parity}
  run_phase "$parity" "$out" "$prior" >"$root/logs/sweep-${sweep}-${parity}.log" 2>&1
  prior=$out
 done
 if "$bin" audit "$prior" "$before" >"$root/logs/sweep-${sweep}-audit.log" 2>&1; then
  ln -sfn "$prior" "$root/final"
  date -u >"$root/COMPLETE"
  exit 0
 else
  status=$?; [[ $status == 2 ]] || exit "$status"
 fi
 done
