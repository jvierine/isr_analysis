"""Detached restart; only orchestration. Scientific processing remains Rust."""
import sys,subprocess as sp,time
from pathlib import Path
sys.path.insert(0,'/home/j/src/isr_analysis_syisr_native_v2_run/syisr/operations')
from remote_driver import ROOT,AV,SSH,node,run,pipe,log
stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
stop=f'''set -e
cd {AV}
if ps -p 3244507 -o args= | grep -F '{AV}/code/syisr/target/release/syisr_ac fit' >/dev/null; then kill -TERM 3244507; fi
for i in $(seq 1 20); do if ! pgrep -f '^{AV}/code/syisr/target/release/syisr_ac fit' >/dev/null; then break; fi; sleep 1; done
if pgrep -f '^{AV}/code/syisr/target/release/syisr_ac fit' >/dev/null;then echo 'Old workers still active';exit 1;fi
mkdir -p archive/{stamp}
for d in fits fit-failures;do if [ -d run/$d ];then mv run/$d archive/{stamp}/$d;fi;done
mkdir -p run/fits run/fit-failures transfer-{stamp}
rm -f run/DECODE_COMPLETE
'''
run(SSH+[stop]);log('Previous MPI stopped; prior fits archived')
run(node(22000,f'test -f {ROOT}/run/DECODE_COMPLETE'))
source=f'''set -e
cd {ROOT}/run
find decoded -maxdepth 1 -regextype posix-extended -regex 'decoded/[0-9]{{6}}.h5' -print > /tmp/syisr-final-list-{stamp}
printf '%s\\n' decode-failures DECODE_COMPLETE >> /tmp/syisr-final-list-{stamp}
tar cf - -T /tmp/syisr-final-list-{stamp}
'''
pipe(node(22000,source),SSH+[f'tar xf - -C {AV}/transfer-{stamp}'])
log('Complete decoded products transferred to staging; temporary references excluded')
launch=f'''set -e
cd {AV}
export PATH=/home/j/.cargo/bin:$PATH
CARGO_BUILD_JOBS=1 cargo test --release --manifest-path code/syisr/Cargo.toml --lib grid::tests > logs/restart-grid-tests-{stamp}.log 2>&1
cat logs/restart-grid-tests-{stamp}.log
mv run/decoded archive/{stamp}/decoded
mv transfer-{stamp}/decoded run/decoded
if [ -d run/decode-failures ];then mv run/decode-failures archive/{stamp}/decode-failures;fi
mv transfer-{stamp}/decode-failures run/decode-failures
mv transfer-{stamp}/DECODE_COMPLETE run/DECODE_COMPLETE
nohup bash run-mpi.sh > logs/mpi-restart-{stamp}.log 2>&1 < /dev/null &
echo RESTART_MPI_PID=$!
'''
run(SSH+[launch]);log('Native/adaptive MPI restarted, 24 ranks x 2 threads, 48 bootstrap draws; complete data staged atomically')
