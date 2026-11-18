#!/usr/bin/env python3
"""Robust joint temporal MAP by forward/reverse block-coordinate descent.
Each time's full-covariance likelihood occurs once in the joint objective.
Each adjacent edge has a pseudo-Huber process penalty, counted once globally.
No posterior-as-prior recursion or Gaussian RTS is used. Scientific arrays: HDF5.
"""
import argparse,json,os,subprocess,shutil,datetime as dt
from pathlib import Path
import h5py,numpy as np
from temporal_campaign import metadata,GRID_SHORT,atomic
C=np.array([4,10,17,23,29])
def sigma(c):
 if c%6<4:return [.7,.4,.5,.2][c%6]
 return {4:150/1450,10:np.log(2)/np.log(15),17:.2,23:200/1050,29:np.log(2)/np.log(10)}[c]
def state(p):return np.array(metadata(p)['fit']['transformed_profile'])
def objective(m):
 f=m['fit'];return sum(f.get('data_objective',f['data_chi_square']) if k=='data_chi_square' else f[k] for k in ['data_chi_square','penalty','temporal_penalty'])
def seed(path,source,shape,warm=False):
 x=state(source)
 if warm:
  x[1::6]=np.log(1.2);x[2::6]=np.log(2.5);x[3::6]=0
  x[C]=[(1000-550)/1450,np.log(100/20)/np.log(15),.02/.5,(850-550)/1050,np.log(150/50)/np.log(10)]
 with h5py.File(source) as f:z=f['slant_range_km'][:]
 with h5py.File(path,'w') as f:
  f['slant_range_km']=z
  f.attrs['metadata_json']=np.frombuffer(json.dumps({'composition_shape':shape,'fit':{'transformed_profile':x.tolist()},'initializer_only':True}).encode(),dtype=np.uint8)
def prior(path,neighbours,columns,mids,t,sig,delta):
 means=[];variances=[]
 for j,p in neighbours:
  means.extend(state(p)[columns]);variances.extend((np.array([sigma(c) for c in columns])*sig)**2*abs(mids[t]-mids[j])/1800)
 with h5py.File(path,'w') as f:
  f['columns']=np.tile(columns,len(neighbours)).astype('u8');f['mean']=means;f['covariance']=np.diag(variances);f['robust_delta']=delta
  f.attrs['method']='Independent adjacent random-walk edges; process covariance only; pseudo-Huber standardized innovations'
def global_cost(paths,mids,columns,sig,delta):
 total=0.;xs=[]
 for p in paths:
  m=metadata(p)['fit'];total+=m.get('data_objective',m['data_chi_square'])+m['penalty'];xs.append(state(p))
 for i in range(1,len(xs)):
  z=(xs[i][columns]-xs[i-1][columns])/(np.array([sigma(c) for c in columns])*sig*np.sqrt((mids[i]-mids[i-1])/1800))
  total+=np.sum(2*delta**2*(np.hypot(1,z/delta)-1))
 return float(total)
def main():
 ap=argparse.ArgumentParser(__doc__);ap.add_argument('root',type=Path);ap.add_argument('--fit-directory',default='day-20230821-robust-no-he');ap.add_argument('--initial-fit-directory',default='day-20230821-corrected');ap.add_argument('--data-delta',type=float,default=3);ap.add_argument('--fit-helium',action='store_true');ap.add_argument('--probe',type=int);ap.add_argument('--threads',type=int,default=8);ap.add_argument('--sweeps',type=int,default=4);ap.add_argument('--sigma',type=float,default=2);ap.add_argument('--delta',type=float,default=1.5);a=ap.parse_args()
 root=a.root.resolve();old=root/a.initial_fit_directory;out=root/a.fit_directory;out.mkdir(exist_ok=True)
 plan=json.loads((root/'plan.json').read_text());
 if plan['layout']['samples']!=1194:raise ValueError('This robust driver currently supports the validated 2023 1194-sample layout only')
 if min(a.sigma,a.delta,a.data_delta,a.threads,a.sweeps)<=0:raise ValueError('Positive scales, threads and sweep count required')
 ws=plan['windows'];mids=np.array([(w['utc_start']+w['utc_end'])/2 for w in ws]);paths=[out/'windows'/f'{i:03d}'/'profile.h5' for i in range(len(ws))]
 shape=metadata(old/'windows/016/profile.h5')['composition_shape'];shutil.copy2(old/'shape.h5',out/'shape.h5')
 with h5py.File(old/'windows/016/profile.h5') as f:knots=f['slant_range_km'][:]
 cols=np.array([i for i in range(6*len(knots)) if i%6<4 or i in (C if a.fit_helium else C[:2])]);gain=json.loads((old/'calibration.json').read_text())['calibration_adu2']
 env=os.environ.copy();env.update(RAYON_NUM_THREADS=str(a.threads),OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',SYISR_FIX_HELIUM_ZERO='0' if a.fit_helium else '1',SYISR_ROBUST_DATA_DELTA=str(a.data_delta),SYISR_RECEIVER_MODEL='delta',SYISR_OBSERVED_MAX_KM='1650',SYISR_PROFILE_OPTIMIZER='bfgs',SYISR_COMPOSITION_SHAPE=str(out/'shape.h5'))
 configuration=dict(method='Nonlinear joint robust temporal MAP; forward/reverse coordinate minimization; no Gaussian RTS',process_sigma_multiplier=a.sigma,pseudo_huber_delta=a.delta,calibration_adu2=gain,absolute_calibration=False,helium_fixed_zero=not a.fit_helium,robust_data_delta=a.data_delta,measurement_covariance='unchanged full covariance',uncertainty='conditional per-time curvature; joint temporal uncertainty not computed')
 if (out/'configuration.json').exists():
  previous_config=json.loads((out/'configuration.json').read_text())
  if previous_config!=configuration:raise ValueError('Scientific configuration changed; choose a new fit directory')
 atomic(out/'configuration.json',configuration)
 def fit(t,passnum,reverse=False,probe=False):
  d=paths[t].parent;d.mkdir(parents=True,exist_ok=True);tag=f'pass-{passnum:02d}';neighbours=[]
  if probe:
   if t>0:neighbours.append((t-1,old/'windows'/f'{t-1:03d}'/'profile.h5'))
  else:
   for j in [t-1,t+1]:
    if 0<=j<len(paths) and paths[j].exists():neighbours.append((j,paths[j]))
  e=env.copy();raw=root/'windows'/f'{t:03d}';e.update(SYISR_PARTIAL_PRODUCT=str(raw/'partial.h5'),SYISR_STRUCTURED_COVARIANCE=str(raw/'covariance.h5'))
  if neighbours:
   pr=d/f'{tag}-prior.h5';prior(pr,neighbours,cols,mids,t,a.sigma,a.delta);e['SYISR_TEMPORAL_PRIOR']=str(pr)
  else:e.pop('SYISR_TEMPORAL_PRIOR',None)
  current=paths[t] if paths[t].exists() else (neighbours[0][1] if neighbours else old/'windows'/f'{t:03d}'/'profile.h5')
  seeds=[current]
  if passnum==0 or probe:
   warm=d/f'{tag}-warm-seed.h5';seed(warm,current,shape,True);seeds.append(warm)
  results=[]
  for k,s in enumerate(seeds):
   product=d/f'{tag}-candidate-{k}.h5'
   if not product.exists():
    e['SYISR_INITIAL_PROFILE']=str(s)
    with (d/f'{tag}-candidate-{k}.log').open('w') as log:subprocess.run([str(Path('target/release/profile-fit').resolve()),str(raw/'lpm.h5'),'none',str(product),GRID_SHORT,'1',str(gain),'700','25'],env=e,stdout=log,stderr=log,check=True)
   m=metadata(product)
   if m['fit']['status']=='converged':results.append((objective(m),product))
  if not results:raise RuntimeError(f'No converged candidate {t}')
  results.sort();best=results[0][1]
  if paths[t].exists():shutil.copy2(paths[t],d/f'{tag}-previous.h5')
  pending=paths[t].with_name('profile.pending.h5');shutil.copy2(best,pending);pending.replace(paths[t]);atomic(d/f'{tag}-selection.json',dict(selected=str(best),candidates=[dict(objective=c,path=str(p)) for c,p in results]))
  print('SELECT',passnum,t,results[0][0],best.name,flush=True)
 if a.probe is not None:fit(a.probe,90,probe=True);return
 history_path=out/'objective-history.json'
 history=json.loads(history_path.read_text())['joint_objective'] if history_path.exists() else []
 if len(history)>=a.sweeps:
  print('Requested sweeps already completed; existing status retained',flush=True);return
 for sweep in range(len(history),a.sweeps):
  partial=any(p.parent.joinpath(f'pass-{sweep:02d}-selection.json').exists() for p in paths)
  before=[state(p) if p.exists() else None for p in paths]
  order=range(len(ws)) if sweep%2==0 else reversed(range(len(ws)))
  for t in order:
   marker=paths[t].parent/f'pass-{sweep:02d}-selection.json'
   if not marker.exists():fit(t,sweep)
   atomic(out/'status.json',dict(stage='optimizing',sweep=sweep,last_window=t,total=len(ws)))
  cost=global_cost(paths,mids,cols,a.sigma,a.delta)
  if history and cost>history[-1]+1e-6*max(1,abs(history[-1])):raise RuntimeError('Joint robust objective increased')
  history.append(cost)
  change=max([np.max(abs((state(p)-x)[cols])/(np.array([sigma(c) for c in cols])*a.sigma)) for p,x in zip(paths,before) if x is not None] or [float('inf')])
  atomic(out/'objective-history.json',dict(joint_objective=history,maximum_standardized_change=float(change)))
  subprocess.run(['target/release/export-temporal',str(root),'filtered',out.name],check=True)
  subprocess.run(['target/release/plot-campaign',str(out/'filtered-campaign.h5'),str(out/f'pass-{sweep:02d}'),'--show-all'],check=True,env={**env,'SYISR_PLOT_MAX_KM':'1650'})
  if not partial and sweep>=1 and change<.05 and abs(history[-1]-history[-2])/max(1,abs(history[-2]))<1e-3:break
 atomic(out/'status.json',dict(stage='complete' if change<.05 else 'sweep_limit_reached',sweeps=len(history),joint_objective=history,total=len(ws),maximum_standardized_change=float(change)))
if __name__=='__main__':main()
