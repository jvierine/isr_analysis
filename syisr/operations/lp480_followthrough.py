#!/usr/bin/env python3
"""Transport immutable Rust/HDF5 products and detach the MPI controller.
Run from base conda; fitting, reduction and rendering remain Rust operations.
"""
import argparse,os,shlex,subprocess as sp,sys,time,tarfile
from pathlib import Path
p=argparse.ArgumentParser(__doc__)
p.add_argument('local',type=Path);p.add_argument('node_runner',type=Path)
p.add_argument('--node-root',required=True);p.add_argument('--remote-root',required=True)
p.add_argument('--code-root',required=True);p.add_argument('--previous',required=True)
a=p.parse_args();a.local.mkdir(parents=True,exist_ok=True)
ssh=['ssh','-o','ProxyJump=none','-o','ProxyCommand=none','-o','ControlPath=none','-o','BatchMode=yes','-o','ConnectTimeout=10','j@avaruus.uit.no']
def remote(s,**kw):return sp.run(ssh+[s],check=True,**kw)
def save_state(s):(a.local/'transport-status.txt').write_text(s+'\n')
script=a.local/'collect-pooled.sh'
script.write_text(f'set -eu\nroot={shlex.quote(a.node_root)}\ntest -f "$root/CALIBRATION_INPUT_READY" || exit 3\ncd "$root/compact"\ntar czf - index.h5 background.h5 products\n')
archive=a.local/'input.tar.gz'
if not (a.local/'INPUT_COLLECTED').exists():
 while True:
  save_state('Waiting for complete pooled-screen products')
  try:
   with archive.with_suffix('.partial').open('wb') as out:
    r=sp.run([sys.executable,str(a.node_runner),'22000',str(script)],stdout=out,timeout=1200)
   if r.returncode==0:
    with tarfile.open(archive.with_suffix('.partial')) as t:
     if len([n for n in t.getnames() if n.endswith('.h5') and 'products/'in n])!=550:raise ValueError('Incomplete input archive')
    archive.with_suffix('.partial').replace(archive);(a.local/'INPUT_COLLECTED').touch();break
  except (sp.SubprocessError,OSError,ValueError) as e:print(type(e).__name__,str(e),flush=True)
  time.sleep(60)
while not (a.local/'MPI_STARTED').exists():
 try:
  save_state('Transferring complete screened inputs to avaruus')
  root=shlex.quote(a.remote_root);code=shlex.quote(a.code_root);previous=shlex.quote(a.previous)
  with archive.open('rb') as f:remote(f'mkdir -p {root}/input {root}/logs; tar xzf - -C {root}/input',stdin=f,timeout=1200)
  remote(f'''set -eu
root={root}; code={code}
ln -sfn /nfs/urdr/scratch/juha/isr_analysis/lp480-20261119/input/references "$root/input/references"
cd "$code"; git pull --ff-only
CARGO_BUILD_JOBS=8 /home/j/.cargo/bin/cargo build --release --manifest-path syisr/Cargo.toml --bin syisr_lp480 >"$root/logs/build.log" 2>&1
RAYON_NUM_THREADS=8 "$code/syisr/target/release/syisr_lp480" calibrate "$root/input" unused >"$root/logs/calibrate.log" 2>&1
nohup env RANKS=32 RAYON_NUM_THREADS=4 INITIAL_PRIOR={previous} bash "$code/syisr/operations/lp480_controller.sh" "$root" "$code/syisr/target/release/syisr_lp480" >"$root/logs/controller.log" 2>&1 </dev/null &
echo $! >"$root/controller.pid"
echo MPI_CONTROLLER_STARTED
''',timeout=1200)
  (a.local/'MPI_STARTED').touch();save_state('Detached 32-rank x4-thread MPI controller started; check remote logs and convergence audits')
 except (sp.SubprocessError,OSError) as e:print(type(e).__name__,str(e),flush=True);time.sleep(60)
