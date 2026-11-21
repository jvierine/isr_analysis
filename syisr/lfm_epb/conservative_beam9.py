"""Promote the existing all-other-beams fit to a conservative cavity view.
Drift and smoothing are fixed at their earlier selected values. This does not
repeat velocity selection. All science data remain signed and archived.
"""
from pathlib import Path
import argparse,shutil,subprocess,os
import numpy as np
import h5py
p=argparse.ArgumentParser();p.add_argument('code',type=Path);p.add_argument('root',type=Path);a=p.parse_args()
r=a.root/'conservative_no_beam9';c=r/'cavity';c.mkdir(parents=True,exist_ok=True)
if not (r/'geometry.h5').exists():(r/'geometry.h5').symlink_to(a.root/'geometry.h5')
for src in (a.root/'cavity').glob('*.h5'):
    if src.name in ('all_beams.h5','volume.h5'):continue
    dest=c/src.name
    if not dest.exists():dest.symlink_to(src)
shutil.copyfile(a.root/'cavity/holdout_beam08.h5',c/'all_beams.h5')
with h5py.File(c/'all_beams.h5','a') as h:
    h.attrs['excluded_beam']=9;h.attrs['scope']='Bounded cavity fitted to all eligible measurements of beams1-8,10,11. Velocity and smoothing fixed to previously selected all-beam values.'
with h5py.File(a.root/'cavity/all_beams.h5') as h,h5py.File(c/'all_beams.h5') as g:
    d=h['contrast'][()];e=g['contrast'][()];s=h['support_weight'][()];t=g['support_weight'][()]
    mask=(s>.005*s.max())&(t>.005*t.max())
    rms=np.sqrt(np.mean((d[mask]-e[mask])**2))
    with h5py.File(c/'exclusion_sensitivity.h5','w') as out:out['supported_contrast_rms_change']=rms;out['common_supported_node_count']=mask.sum()
    print('EXCLUDE_BEAM9_SUPPORTED_RMS',rms,flush=True)
env=dict(os.environ,SYISR_MEMO_FIGURES='1')
subprocess.run([str(a.code/'syisr/target/release/lfm_epb_lift'),str(c/'all_beams.h5'),str(c/'volume.h5')],check=True)
subprocess.run([str(a.code/'syisr/plots/target/release/lfm_epb_views'),str(r),str(r/'figures'),'diagnostics-only'],env=env,check=True)
(r/'COMPLETE').write_text('Conservative fixed-drift cavity excludes beam9 entirely\n')
