"""Independent robust temporal cost replay; no Gaussian RTS assumptions."""
from pathlib import Path
import json,subprocess
import h5py,numpy as np
from scipy.linalg import solve_triangular
root=Path('/Volumes/LaCie/SYISR_analysis/ulp2ms-20261118/session-00/day-20230821-robust-no-he')
paths=sorted(root.glob('windows/*/profile.h5'));metrics=[]
for p in paths:
 with h5py.File(p) as f:
  m=json.loads(bytes(f.attrs['metadata_json']));fit=m['fit'];x=np.array(fit['transformed_profile']);cols=np.array(fit['filter_columns']);n=len(cols)
  cov=np.array(fit['filter_covariance']).reshape(n,n,order='F');mineig=np.linalg.eigvalsh(cov).min();assert mineig>0
  prior=m['temporal_prior']['product'];cost=0
  if prior:
   with h5py.File(prior) as pr:
    z=solve_triangular(np.linalg.cholesky(pr['covariance'][:]),x[pr['columns'][:]]-pr['mean'][:],lower=True);d=float(pr['robust_delta'][()]);cost=np.sum(2*d*d*(np.hypot(1,z/d)-1))
  err=abs(cost-fit['temporal_penalty']);assert err<1e-7,(p,err)
  ng=len(f['observed_slant_range_km']);red=fit['data_chi_square']/(ng*119-n)
  for c in cols:
   lo,hi=[(np.log(1e-4),np.log(100)),(np.log(.3),np.log(5)),(np.log(.3),np.log(12)),(-1.5,1.5),(0,1),(0,1)][c%6]
   assert lo-1e-9<=x[c]<=hi+1e-9
  metrics.append([int(p.parent.name),fit['normalized_gradient'],red,fit['temporal_penalty'],err,mineig])
with h5py.File(root/'robust-cost-audit.h5','w') as f:
 f.attrs['columns']='window,normalized_gradient,reduced_data_chi_square,robust_temporal_penalty,independent_temporal_replay_absolute_error,conditional_covariance_min_eigenvalue'
 f.attrs['method']='Independent NumPy Cholesky whitening and pseudo-Huber cost replay; bounds and conditional covariance positivity; physical reliability requires separate checks.'
 f['metrics']=metrics
print('Audited',len(paths),'windows; max temporal cost replay error',max(v[4] for v in metrics),'median reduced data chi square',np.median([v[2] for v in metrics]),flush=True)
