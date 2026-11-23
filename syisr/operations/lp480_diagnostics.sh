#!/usr/bin/env bash
# Diagnostic generation is mandatory for production review, independent of the
# numerical controller. Do not label generated plots as a scientific QA pass.
set -euo pipefail
renderer=${1:?lp480_diagnostics executable}; run=${2:?completed run}; output=${3:-$run/diagnostics}
test -f "$run/COMPLETE"
mkdir -p "$output"
rm -f "$output/DIAGNOSTIC_PLOTS_COMPLETE"
"$renderer" "$run/input" "$run/final" "$output"
test -s "$output/DIAGNOSTIC_PLOTS_COMPLETE"
printf '%s\n' 'Review measured background spectra and signed normalized real/imaginary lag residuals before production release. Generated plots are not automatic scientific approval.' >"$output/DIAGNOSTIC_REVIEW_REQUIRED"
