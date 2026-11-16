#!/usr/bin/env python3
"""Transport/orchestration only; scientific computation is Rust.
Runs on the work Linux laptop in a detached session, independent of the Mac.
"""
import subprocess as sp,sys,time,shlex,os,traceback
from pathlib import Path
ROOT='/home/cubuntu/syisr-full-ac-20261116'
AV='/nfs/urdr/scratch/juha/isr_analysis/syisr-ac-20261116'
LOCAL=Path(__file__).resolve().parents[2]
NODE=['/home/j/anaconda3/bin/conda','run','--no-capture-output','-n','base','python','/home/j/src/compuntu/scripts/connect-node.py']
SSH=['ssh','-o','BatchMode=yes','-o','ControlPath=none','-o','ServerAliveInterval=30','j@avaruus.uit.no']
def node(port,command):return NODE+[str(port),command]
def run(cmd,capture=False):return sp.run(cmd,check=True,stdout=sp.PIPE if capture else None,text=capture)
def pipe(a,b):
 with sp.Popen(a,stdout=sp.PIPE)as p:
  sp.run(b,stdin=p.stdout,check=True)
  if p.wait()!=0:raise RuntimeError('source pipeline failed')
def log(s):print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),s,flush=True)
def main():
 log('Building/test Rust on22002; rawdecode on22000; AvaruusMPI capped48cores plus1plotter')
 build=f'''set -e
cd {ROOT}/code/syisr
export HDF5_DIR=/home/cubuntu/syisr-ac-pilot-20261116/deps/root/usr/lib/x86_64-linux-gnu/hdf5/serial
export LD_LIBRARY_PATH=/home/cubuntu/syisr-ac-pilot-20261116/deps/root/usr/lib/x86_64-linux-gnu
export CARGO_TARGET_DIR=/home/cubuntu/syisr-ac-pilot-20261116/model/target
export CARGO_BUILD_JOBS=4
/home/cubuntu/.cargo/bin/cargo test --release --lib > {ROOT}/tests.log 2>&1
/home/cubuntu/.cargo/bin/cargo build --release --bin syisr_ac > {ROOT}/build.log 2>&1
'''
 run(node(22002,build))
 run(node(22000,f'mkdir -p {ROOT}/bin {ROOT}/run/decoded {ROOT}/logs'))
 pipe(node(22002,'cat /home/cubuntu/syisr-ac-pilot-20261116/model/target/release/syisr_ac'),node(22000,f'cat > {ROOT}/bin/syisr_ac && chmod +x {ROOT}/bin/syisr_ac'))
 phase=LOCAL/'syisr/m32-reference.bc'
 with phase.open('rb')as f:sp.run(node(22000,f'cat > {ROOT}/run/m32-reference.bc'),stdin=f,check=True)
 raw=f'''set -e
cd {ROOT}/run
bin={ROOT}/bin/syisr_ac
if [ ! -s campaign.h5 ];then "$bin" index /mnt/syisr/juha_data/AC/SY/20251006 "$PWD/m32-reference.bc" /mnt/syisr/juha_data/L2_4parameters/OSAYA_TISR01_IFIT_L2_DAY_20251006000000_V01.00_N_AC_01min.h5 "$PWD";fi
"$bin" decode "$PWD" 8
'''
 with (LOCAL/'raw-stage.sh').open('w')as f:f.write(raw)
 with (LOCAL/'raw-stage.sh').open('rb')as f:sp.run(node(22000,f'cat > {ROOT}/raw-stage.sh'),stdin=f,check=True)
 run(node(22000,f'nohup bash {ROOT}/raw-stage.sh > {ROOT}/logs/raw.log 2>&1 < /dev/null & echo RAW_PID=$!'))
 run(SSH+[f'mkdir -p {AV}/code {AV}/run/decoded {AV}/logs'])
 pipe(['tar','czf','-','--exclude=target','--exclude=.git','-C',str(LOCAL),'syisr'],SSH+[f'tar xzf - -C {AV}/code'])
 with phase.open('rb')as f:sp.run(SSH+[f'cat > {AV}/run/m32-reference.bc'],stdin=f,check=True)
 run(SSH+[f'export PATH=/home/j/.cargo/bin:$PATH; export CARGO_BUILD_JOBS=4; cd {AV}/code/syisr; cargo test --release --lib > {AV}/logs/tests.log 2>&1 && cargo build --release --bin syisr_ac > {AV}/logs/build.log 2>&1'])
 log('Rust builds/tests passed; awaiting fullraw manifest')
 while True:
  try:
   run(node(22000,f'test -s {ROOT}/run/campaign.h5'))
   pipe(node(22000,f'cat {ROOT}/run/campaign.h5'),SSH+[f'cat > {AV}/run/campaign.h5.partial && mv {AV}/run/campaign.h5.partial {AV}/run/campaign.h5'])
   break
  except sp.CalledProcessError:time.sleep(15)
 mpi=f'''set -e
export RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd {AV}/run
mpirun -np 24 --map-by slot:PE=2 --bind-to core {AV}/code/syisr/target/release/syisr_ac fit "$PWD" 48
'''
 sp.run(SSH+[f'cat > {AV}/run-mpi.sh'],input=mpi.encode(),check=True)
 run(SSH+[f'nohup bash {AV}/run-mpi.sh > {AV}/logs/mpi.log 2>&1 < /dev/null & echo MPI_PID=$!'])
 log('MPI detached24ranks×2threads=48cores; rawdecode8threads on separatehost')
 receipts=LOCAL/'transferred-dwells.txt'
 copied=set(receipts.read_text().splitlines())if receipts.exists()else set()
 while True:
  listing=run(node(22000,f'find {ROOT}/run/decoded -maxdepth 1 -name "[0-9]*.h5" -printf "%f\\n"'),True).stdout.splitlines()
  fresh=sorted(set(listing)-copied)
  if fresh:
   entries=' '.join(shlex.quote('decoded/'+x)for x in fresh)
   pipe(node(22000,f'tar cf - -C {ROOT}/run {entries}'),SSH+[f'tar xf - -C {AV}/run'])
   copied.update(fresh);receipts.write_text('\n'.join(sorted(copied))+'\n');log(f'Transferred{len(copied)}decoded dwells')
  done=sp.run(node(22000,f'test -f {ROOT}/run/DECODE_COMPLETE'),stdout=sp.DEVNULL).returncode==0
  if done:
   pipe(node(22000,f'tar cf - -C {ROOT}/run DECODE_COMPLETE decode-failures'),SSH+[f'tar xf - -C {AV}/run'])
   # A partial/failed rawdwell is explicitly recorded, never silently markedvalid.
   log('Allrawdwells attempted and transferred; MPI continues independently on Avaruus')
   break
  time.sleep(60)
 (LOCAL/'TRANSPORT_COMPLETE').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
if __name__=='__main__':
 try:main()
 except Exception:traceback.print_exc();sys.exit(1)
