#!/usr/bin/env bash
# Diagnostic generation is mandatory for production review, independent of the
# numerical controller. Do not label generated plots as a scientific QA pass.
set -euo pipefail
renderer=${1:?lp480_diagnostics executable}; run=${2:?completed run}; output=${3:-$run/diagnostics}
test -f "$run/COMPLETE"
mkdir -p "$output"
rm -f "$output/DIAGNOSTIC_PLOTS_COMPLETE"
"$renderer" "$run/input" "$run/final" "$output"
if [[ ! -s $output/DIAGNOSTIC_PLOTS_COMPLETE ]]; then
 test -s "$output/DIAGNOSTIC_PLOTS_PARTIAL"
 /opt/soft/anaconda3/bin/conda run --no-capture-output -n base python - "$output/coverage.h5" "$run/logs/initial.log" <<'PY'
import json
import sys

import h5py

with h5py.File(sys.argv[1]) as handle:
    coverage = json.loads(bytes(handle["json_utf8"][:]))
with open(sys.argv[2]) as log:
    skipped = {int(line.split()[6].rstrip(":")) for line in log
               if line.startswith("FIT_SKIPPED no supported beam fits at ")}
missing = set(coverage["times_missing_residual_plots"])
if not missing or not missing <= skipped:
    raise SystemExit(f"Unexplained residual coverage gaps: {sorted(missing - skipped)}")
PY
fi
printf '%s\n' 'Review measured background spectra and signed normalized real/imaginary lag residuals before production release. Generated plots are not automatic scientific approval.' >"$output/DIAGNOSTIC_REVIEW_REQUIRED"
