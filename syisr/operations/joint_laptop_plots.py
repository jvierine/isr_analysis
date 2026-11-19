"""Detached Rust plotting on the laptop with TeX; no dependency on Mac."""
import sys,subprocess as sp,time
from pathlib import Path
sys.path.insert(0,'/home/j/src/isr_analysis_syisr_native_v2_run/syisr/operations')
from remote_driver import AV,SSH,pipe,log
root=Path('/home/j/src/isr_analysis_syisr_native_v2_run')
AV='/nfs/urdr/scratch/juha/isr_analysis/syisr-ac-joint-v3-20261116'
(root/'joint-plot-cache').mkdir(exist_ok=True)
sp.run(['/home/j/.cargo/bin/cargo','build','--release','--manifest-path',str(root/'syisr/plots/Cargo.toml'),'--bin','syisr-progress-plot','-j','2'],check=True)
cache=root/'joint-plot-cache';cache.mkdir(exist_ok=True)
while True:
 try:
  pipe(SSH+[f'cd {AV}/full-run && tar chf - campaign.h5 fits'],['tar','xf','-','-C',str(cache)])
  sp.run([str(root/'syisr/plots/target/release/syisr-progress-plot'),str(cache),str(root/'joint-progress')],check=True)
  log('Updated native/adaptive maps on work laptop')
 except Exception as e:log(str(e))
 time.sleep(1800)
