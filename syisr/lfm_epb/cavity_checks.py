"""Run native numerical checks, summarize sensitivities and retain HDF5 evidence."""
import argparse
from pathlib import Path
import os,subprocess
import h5py
import numpy as np

p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root);out=root/'cavity';binary=out/'native'
with h5py.File(out/'selection.h5') as h:vx,vy,lam=h['selected_config'][()]
subprocess.run([str(binary),'inject',str(out/'observations.h5'),str(out/'injection.h5'),str(vx),str(vy),str(lam)],check=True)
subprocess.run([str(binary),'fit',str(out/'observations.h5'),str(out/'all_beams_refined.h5'),str(vx),str(vy),str(lam),'-1'],check=True,env={**os.environ,'SYISR_CAVITY_ITERATIONS':'10000'})
with h5py.File(out/'all_beams.h5') as h:
    d=h['contrast'][()];w=h['support_weight'][()];p0=h['predicted_ratio'][()];mask=w>.005*w.max()
with h5py.File(out/'all_beams_refined.h5') as h:
    q=h['predicted_ratio'][()];r=h['contrast'][()];ok=np.isfinite(p0)&np.isfinite(q)
    pred_diff=np.sqrt(np.mean((p0[ok]-q[ok])**2));field_diff=np.sqrt(np.mean((d[mask]-r[mask])**2))
with h5py.File(out/'injection.h5') as h:
    ii=h['contrast'][()];truth=h['injected_contrast'][()];iw=h['support_weight'][()];im=iw>.005*iw.max()
    injection_rms=np.sqrt(np.mean((ii[im]-truth[im])**2));injection_min=ii[im].min();true_min=truth.min()
with h5py.File(out/'numerical_checks.h5','w') as h:
    for key,value in [('prediction_refinement_rms',pred_diff),('supported_field_refinement_rms',field_diff),('injection_supported_rms',injection_rms),('injection_minimum_recovered_contrast',injection_min),('injection_minimum_true_contrast',true_min)]:h[key]=value
    h.attrs['injection']='noise-free forward/recovery test under the fitted model and finite-beam geometry; not physical validation or uncertainty coverage'
print('NUMERIC prediction_RMS',pred_diff,'field_RMS',field_diff,'injection_RMS',injection_rms,'injection_minimum',injection_min,flush=True)
names=['point_beam_sensitivity','include_perp_sensitivity','strict_perp_sensitivity','straight_B_sensitivity','drift_minus25','drift_plus25']
with h5py.File(out/'sensitivity_summary.h5','w') as result:
    for name in names:
        with h5py.File(out/(name+'.h5')) as h:
            q=h['contrast'][()];pp=h['predicted_ratio'][()];ok=np.isfinite(p0)&np.isfinite(pp);diag=h['diagnostics'][()]
            values=[np.sqrt(np.mean((d[mask]-q[mask])**2)),np.sqrt(np.mean((p0[ok]-pp[ok])**2)),np.sqrt(diag[3]/diag[4]),diag[4]]
            result[name]=values
            print('SENSITIVITY',name,*values,flush=True)
    result.attrs['columns']='supported contrast RMS difference, prediction RMS difference, training RMSE, eligible count'
