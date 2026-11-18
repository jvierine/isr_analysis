#!/usr/bin/env python3
"""Independent HDF5/RTS audit. Use base conda Python; inference is in Rust.
Numerical products are HDF5. Source provenance stays in metadata.
"""
import argparse,json,subprocess,hashlib
from pathlib import Path
import h5py,numpy as np

def meta(path):
 with h5py.File(path) as f:return json.loads(bytes(f.attrs['metadata_json']))

def state(path):
 with h5py.File(path) as f:return f['mean'][:],f['covariance'][:],f['columns'][:]

def audit(root,session,binaries,fit_directory):
 out=root/session['directory']/fit_directory;n=len(session['windows'])
 subprocess.run([str(binaries/'profile-validate'),str(out),','.join(map(str,range(n)))],check=True,stdout=(out/'independent-likelihood-replay.log').open('w'),stderr=subprocess.STDOUT)
 data=[];smooth_errors=[];cov_errors=[];minimum_eigenvalue=[];outside=[]
 for i in range(n):
  d=out/'windows'/f'{i:03d}';m=meta(d/'profile.h5');fit=m['fit'];columns=np.array(fit['filter_columns']);x=np.array(fit['transformed_profile'])[columns]
  pf=np.array(fit['filter_covariance']).reshape((len(columns),len(columns)),order='F')
  xs,ps,sc=state(d/'smoothed.h5');np.testing.assert_array_equal(columns,sc)
  eigen=np.linalg.eigvalsh(ps);minimum_eigenvalue.append(eigen.min())
  assert eigen.min()>-1e-9*max(1.,eigen.max())
  assert np.linalg.eigvalsh(pf-ps).min()>-1e-7*max(1.,np.linalg.norm(pf,2))
  if i==n-1:
   expected_x,expected_p=x,pf
  else:
   prediction,pp,pc=state(out/'windows'/f'{i+1:03d}'/'prediction.h5');next_x,next_p,nc=state(out/'windows'/f'{i+1:03d}'/'smoothed.h5')
   np.testing.assert_array_equal(columns,pc);np.testing.assert_array_equal(columns,nc)
   gain=np.linalg.solve(pp.T,pf.T).T
   expected_x=x+gain@(next_x-prediction)
   expected_p=pf+gain@(next_p-pp)@gain.T
  smooth_errors.append(np.max(abs(xs-expected_x)))
  cov_errors.append(np.linalg.norm(ps-expected_p)/max(1.,np.linalg.norm(ps)))
  np.testing.assert_allclose(xs,expected_x,rtol=2e-8,atol=2e-8)
  np.testing.assert_allclose(ps,expected_p,rtol=2e-7,atol=2e-8)
  for j,c in enumerate(columns):
   lo,hi=[(np.log(1e-4),np.log(100.)),(np.log(.3),np.log(5.)),(np.log(.3),np.log(12.)),(-1.5,1.5),(0.,1.),(0.,1.)][c%6]
   if xs[j]<lo-1e-6 or xs[j]>hi+1e-6:outside.append([i,int(c),float(xs[j])])
  with h5py.File(d/'profile.h5') as f:ng=len(f['observed_slant_range_km'])
  data.append([i,fit['normalized_gradient'],fit['data_chi_square'],fit['penalty'],fit['temporal_penalty'],fit['data_chi_square']/(ng*119-len(columns))])
 with h5py.File(out/'end-to-end-audit.h5','w') as f:
  f.attrs['method']='Independent saved-likelihood/prior replay and NumPy RTS equations; signed source products and masks checked; covariance PSD/contraction verified. Does not validate physical-model correctness or total uncertainty.'
  f.attrs['columns']='window,normalized_gradient,data_chi_square,spatial_penalty,temporal_penalty,reduced_weighted_data_chi_square'
  f.create_dataset('window_metrics',data=data)
  f.create_dataset('rts_mean_max_abs_error',data=smooth_errors)
  f.create_dataset('rts_covariance_relative_error',data=cov_errors)
  f.create_dataset('smoothed_covariance_min_eigenvalue',data=minimum_eigenvalue)
  f.create_dataset('out_of_bound_smoothed_coordinates',data=np.array(outside).reshape(-1,3))
 return dict(session=session['directory'],windows=n,maximum_gradient=max(v[1] for v in data),median_reduced_data_chi_square=float(np.median([v[-1] for v in data])),smoothed_out_of_bound_coordinates=len(outside),max_rts_mean_replay_error=max(smooth_errors))

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);p.add_argument('--sessions',type=int,nargs='+');p.add_argument('--fit-directory',default='temporal-flat');p.add_argument('--bin-directory',type=Path,default=Path('target/release'));a=p.parse_args();a.bin_directory=a.bin_directory.resolve();m=json.loads((a.root/'plan.json').read_text())
 summaries=[]
 for s in m['sessions']:
  if a.sessions is not None and s['index'] not in a.sessions:continue
  summary=audit(a.root,s,a.bin_directory,a.fit_directory);summaries.append(summary);print(json.dumps(summary),flush=True)
 with h5py.File(a.root/(a.fit_directory+'-audit.h5'),'w') as f:
  f.create_dataset('session_summary_json',data=np.frombuffer(json.dumps(summaries).encode(),dtype=np.uint8))
  for k in ['windows','maximum_gradient','median_reduced_data_chi_square','smoothed_out_of_bound_coordinates','max_rts_mean_replay_error']:f.create_dataset(k,data=[s[k] for s in summaries])
if __name__=='__main__':main()
