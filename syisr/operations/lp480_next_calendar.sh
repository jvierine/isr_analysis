#!/usr/bin/env bash
# Validate and deduplicate the next calendar month, then enqueue measured daily
# processing. No external internet/laptop is needed after this job is submitted.
set -euo pipefail
code=${1:?};root=${2:?};month=${3:?};survey=${4:?};previous=${5:?}
bash "$code/syisr/operations/lp480_prepare_calendar.sh" "$code" "$root" "$month" "$survey"
# Plot sources and physical model are unchanged; reuse compute-linked renderers.
mkdir -p "$root/bin"
for name in lp480_maps lp480_diagnostics lp480_residual_time; do cp "$previous/bin/$name" "$root/bin/";done
cp "$previous/COMPUTE_PLOT_BUILD_COMPLETE" "$root/COMPUTE_PLOT_BUILD_COMPLETE"
/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python - "$code" "$root" "$month" "$previous" <<'PY'
import sys,calendar,datetime as dt,subprocess,h5py,json
from pathlib import Path
code,root,month=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
previous=Path(sys.argv[4])
y,m=map(int,month.split('-'));days=[f'{y:04}{m:02}{d:02}' for d in range(1,calendar.monthrange(y,m)[1]+1)]
for day in days:
 with h5py.File(root/'daily-index'/f'{day}.h5') as f:manifest=json.loads(bytes(f['json_utf8'][:]))
 for t in {j['start'] for j in manifest['jobs']}:assert (root/'references'/f'{t}.h5').exists(),f'Missing date-matched IRI reference {t}'
subprocess.run(['/opt/soft/anaconda3/bin/conda','run','--no-capture-output','-n','base','python',str(code/'syisr/operations/lp480_corrected_submit.py'),'--cores','128','--previous-run',str(previous/'days'/(dt.date(y,m,1)-dt.timedelta(days=1)).strftime('%Y%m%d')),str(code),str(root),str(root/'original'),str(root/'receiver'),*days],check=True)
PY
date -u >"$root/DAILY_QUEUE_SUBMITTED"
