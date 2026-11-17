"""Empirical restart-relative registration; preserve spectra and original axes.

The five later narrowing features share a 70.16 km offset, unlike any beam
permutation. Round the relative correction to native 10 us sampling: 470 us.
This is an empirical registration, not an independently measured hardware delay.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import h5py
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('audit',type=Path);q=p.parse_args()
 q.output.mkdir(parents=True,exist_ok=True)
 with h5py.File(q.audit) as a:
  assert np.array_equal(a['best_beam_permutation'][:],np.arange(5))
  assert abs(float(a.attrs['common_offset_best_fit_km'])-70.16)<1
  late=a['later_peak_range_km'][:]-a['beam_geometry'][:,3]-470*.149896229
  assert np.max(abs(late))<5,late
  print('Later five-beam residuals after relative correction (km):',late,flush=True)
 shutil.copy2(q.audit,q.output/'beam-association-audit.h5')
 for session,total in [('190252',160.),('205701',630.)]:
  parent=q.source/f'session_{session}/range-doppler.h5'
  dest=q.output/f'session_{session}/restart_registered';dest.mkdir(parents=True,exist_ok=True)
  target=dest/'range-doppler.h5';shutil.copy2(parent,target)
  with h5py.File(parent) as original,h5py.File(target,'r+') as h:
   r=original['slant_range_km'][:];h['nominal_uncorrected_slant_range_km']=r;h['slant_range_km'][:]=r-total*.149896229
   h.attrs['background_range_km']=np.asarray(original.attrs['background_range_km'])-total*.149896229
   h.attrs['fixed_signed_absolute_limit']=15.;h.attrs['fixed_signed_normalized_limit']=2.
   h.attrs['range_delay_correction_us']=total;h.attrs['range_delay_correction_km']=total*.149896229
   h.attrs['restart_relative_delay_us']=total-160.
   h.attrs['range_delay_status']='PROVISIONAL: early 160 us baseline retained; later additional 470 us empirical registration from five-beam perpendicular narrowing following 20:50 Beijing restart. Header configuration changed; exact hardware cause and absolute timing remain unverified.'
   h.attrs['range_reference']='pulse-centred monostatic slant range with run-specific provisional registration'
   h.attrs['parent_product']=str(parent.resolve());h.attrs['parent_sha256']=hashlib.sha256(parent.read_bytes()).hexdigest()
   h.attrs['registration_audit_sha256']=hashlib.sha256(q.audit.read_bytes()).hexdigest()
   h.attrs['registration_program']='syisr-restart-register';h.attrs['registration_version']='0.1.0'
   h.attrs['registration_git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip()
   # Axis-only change: compare every scientific dataset, including signed PSD,
   # exposure, satellite screening, time, geometry and normalization arrays.
   def check(name,obj):
    if isinstance(obj,h5py.Dataset) and name!='slant_range_km':
     assert np.array_equal(obj[()],h[name][()],equal_nan=obj.dtype.kind in 'fc'),name
   original.visititems(check)
   assert np.allclose(h['slant_range_km'][:]+total*.149896229,r,rtol=0,atol=1e-12)
  print('Verified axis-only correction',session,total,'us',flush=True)

if __name__=='__main__':main()
