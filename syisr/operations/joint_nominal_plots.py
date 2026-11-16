"""Render intermediate nominal joint fits every30minutes; never label intervals complete."""
import subprocess as sp,sys,time,os
from pathlib import Path
sys.path.insert(0,'/home/j/src/isr_analysis_syisr_native_v2_run/syisr/operations')
from remote_driver import SSH,pipe,log
root=Path('/home/j/src/isr_analysis_syisr_native_v2_run')
remote='/nfs/urdr/scratch/juha/isr_analysis/syisr-ac-joint-v3-20261116/full-run'
cache=root/'joint-nominal-cache';cache.mkdir(exist_ok=True)
(cache/'joint-nominal').mkdir(exist_ok=True)
if not (cache/'fits').exists():(cache/'fits').symlink_to('joint-nominal',target_is_directory=True)
sp.run(['/home/j/.cargo/bin/cargo','build','--release','--manifest-path',str(root/'syisr/plots/Cargo.toml'),'--bin','syisr-progress-plot','-j','2'],check=True)
env=dict(os.environ,SYISR_NOMINAL_ONLY='1')
while True:
 try:
  pipe(SSH+[f'cd {remote} && tar chf - campaign.h5 joint-nominal'],['tar','xf','-','-C',str(cache)])
  sp.run([str(root/'syisr/plots/target/release/syisr-progress-plot'),str(cache),str(root/'joint-nominal-progress')],env=env,check=True)
  log('Updated nominal joint-profile maps; uncertainty refits pending')
 except Exception as e:log(str(e))
 time.sleep(1800)
