#!/usr/bin/env python3
"""MPI orchestration of measured covariance and forward-filter/reverse-RTS ULP fits.
Numerical inference and smoothing are Rust. Run with base conda Python.
"""
import argparse,datetime as dt,json,os,subprocess,traceback,time,tempfile,shutil
from pathlib import Path
import h5py,numpy as np
GRID_SHORT='80,100,150,200,250,300,350,400,500,600,700,800,1000,1250,1500,1700,1900'
GRID_LONG='80,100,150,200,250,300,350,400,500,600,700,800,1000,1250,1500,1800,2150,2550,3000,3550,4200,5000,5500'

def metadata(p):
 with h5py.File(p) as f:return json.loads(bytes(f.attrs['metadata_json']))

def atomic(p,v):
 q=p.with_suffix('.tmp.json');q.write_text(json.dumps(v,indent=2));q.replace(p)

def command(args,name,argv,log,extra=None):
 env=os.environ.copy()
 for k in ['OMPI_COMM_WORLD_RANK','OMPI_COMM_WORLD_SIZE','SYISR_INITIAL_PROFILE','SYISR_TEMPORAL_PRIOR','SYISR_STRUCTURED_COVARIANCE','SYISR_COVARIANCE_SHRINKAGE','SYISR_BOTTOMSIDE_CHAPMAN']:env.pop(k,None)
 env.update(RAYON_NUM_THREADS=str(args.threads),VECLIB_MAXIMUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SYISR_COVARIANCE_NUGGET='0.1')
 if extra:env.update({k:str(v) for k,v in extra.items()})
 with log.open('w') as f:subprocess.run([str(args.bin_directory/name),*map(str,argv)],env=env,stdout=f,stderr=f,check=True)

def prepare(args,s,w):
 root=args.root/s['directory'];d=root/'windows'/f"{w['index']:03d}"
 while args.wait and not (d/'done.json').exists():time.sleep(5)
 cv=d/'covariance-validation';cv.mkdir(exist_ok=True)
 if (d/'covariance.h5').exists():return
 command(args,'export-profile-observations',[root,w['index']],d/'export.log')
 if s['layout']['samples']==3782:
  old=Path('/Users/jvi019/ruisdap/local/syisr-full-range-correlated-profile-20251217-18/windows')/f"{w['index']:03d}"/'covariance.h5'
  if old.exists():
   (d/'covariance.h5').symlink_to(old);return
 selected=None
 for nugget in [0.1,0.2,0.35,0.5,0.65,0.8]:
  current=cv/f'nugget-{nugget:.2f}';current.mkdir(exist_ok=True)
  command(args,'structured-noise-cv',[root,current,w['index']],d/f'covariance-check-{nugget:.2f}.log',dict(SYISR_COVARIANCE_NUGGET=nugget))
  m=json.loads((current/f"window-{w['index']:03d}.json").read_text())
  if m['selected_score_gain_vs_diagonal']>0 and m['selected_heldout_energy']<2:
   selected=current;break
 if selected is None:raise ValueError('Covariance validation failed for all tested regularization levels')
 # Bound SSD staging to one covariance file per rank; the retained product
 # is external. A contiguous copy avoids thousands of random HDF5 chunk writes
 # on the spinning external disc. The temporary directory is always removed.
 with tempfile.TemporaryDirectory(prefix='syisr-covariance-') as temporary:
  staging=Path(temporary)
  command(args,'prepare-structured-covariance',[root,selected,staging,w['index']],d/'covariance-prepare.log')
  original=staging/'windows'/f"{w['index']:03d}"/'covariance.h5'
  pending=d/'covariance.copying.h5';shutil.copy2(original,pending);pending.replace(d/'covariance.h5')


def fit(args,s):
 root=args.root/s['directory'];out=root/args.fit_directory;out.mkdir(exist_ok=True)
 maximum=1650 if s['layout']['samples']==1194 else 5000;grid=GRID_SHORT if maximum==1650 else GRID_LONG
 shape=dict(altitude_km=[500.,2000.],scale_height_km=[100.,100.],exponential_scale_km=args.skew_scale_km,source=f'Fixed exponential scale-height skew L={args.skew_scale_km}km; sigmoid H crossing/local width and Gaussian He fitted jointly; no IRI reference')
 with h5py.File(out/'shape.h5','w') as f:f.attrs['metadata_json']=np.frombuffer(json.dumps(shape).encode(),dtype=np.uint8)
 env=dict(SYISR_RECEIVER_MODEL='delta',SYISR_OBSERVED_MAX_KM=maximum,SYISR_PROFILE_OPTIMIZER='bfgs',SYISR_COMPOSITION_SHAPE=out/'shape.h5')
 windows=s['windows'];reference=out/'calibration.h5'
 # Use an immutable reference gain shared by the August configuration; supplied
 # after its first reference is solved. December retains its previous fixed gain.
 gain_path=args.root/('calibration-2023.json' if maximum==1650 else 'calibration-2025.json')
 if gain_path.exists():gain=json.loads(gain_path.read_text())['calibration_adu2']
 elif maximum==5000:gain=25899.08075225199
 else:
  # This rank owns a session; provisional gain anchored at its local-noon window.
  mid=lambda w:(w['utc_start']+w['utc_end'])/2
  ref=min(windows,key=lambda w:abs(((mid(w)/3600+8)%24)-12))
  rd=root/'windows'/f"{ref['index']:03d}"
  while args.wait and not (rd/'covariance.h5').exists():
   if (rd/'covariance-failed.json').exists():raise ValueError(f'Covariance failed: {rd}')
   time.sleep(5)
  e={**env,'SYISR_PARTIAL_PRODUCT':rd/'partial.h5','SYISR_STRUCTURED_COVARIANCE':rd/'covariance.h5'}
  if not reference.exists():command(args,'profile-fit',[rd/'lpm.h5','none',reference,grid,1,'auto',args.iterations,25],out/'calibration.log',e)
  m=metadata(reference)
  if m.get('receiver_model')!='delta':raise ValueError('Old calibration uses an unvalidated signal response; choose a fresh fit directory')
  if m['fit']['status']!='converged':raise ValueError('Reference calibration did not converge')
  gain=m['fit']['calibration_adu2']
  # Per-recording gains are explicit until external absolute calibration exists.
 atomic(out/'calibration.json',dict(calibration_adu2=gain,absolute_calibration=False,normalization='provisional F peak 1e12 m^-3',scope='recording session; same gain in its forward/reverse analysis'))
 previous=None;previous_mid=None;complete=[]
 for w in windows:
  index=w['index'];raw=root/'windows'/f'{index:03d}';dest=out/'windows'/f'{index:03d}';dest.mkdir(parents=True,exist_ok=True)
  product=dest/'profile.h5';mid=(w['utc_start']+w['utc_end'])/2
  while args.wait and not (raw/'covariance.h5').exists():
   if (raw/'covariance-failed.json').exists():raise ValueError(f'Covariance failed: {raw}')
   time.sleep(5)
  if not (raw/'covariance.h5').exists():raise ValueError(f'Missing validated covariance {raw}')
  prior=dest/'prediction.h5'
  e={**env,'SYISR_PARTIAL_PRODUCT':raw/'partial.h5','SYISR_STRUCTURED_COVARIANCE':raw/'covariance.h5'}
  if previous:
   command(args,'temporal-step',['predict',previous,prior,mid-previous_mid,args.process_sigma_multiplier],dest/'predict.log')
   e.update(SYISR_TEMPORAL_PRIOR=prior,SYISR_INITIAL_PROFILE=previous)
  # First step uses the ordinary broad spatial prior. Calibration is fixed.
  if product.exists():
   try:metadata(product)
   except Exception:
    product.rename(dest/f"unreadable-{len(list(dest.glob('unreadable-*.h5'))):03d}.h5")
  if product.exists() and metadata(product).get('receiver_model')!='delta':raise ValueError('Legacy receiver-response result cannot be resumed; choose a fresh fit directory')
  if product.exists() and metadata(product)['fit']['status']!='converged':
   backup=dest/f"previous-{len(list(dest.glob('previous-*.h5'))):03d}.h5";product.rename(backup);e['SYISR_INITIAL_PROFILE']=backup
  if not product.exists():command(args,'profile-fit',[raw/'lpm.h5','none',product,grid,1,gain,args.iterations,25],dest/'fit.log',e)
  m=metadata(product)
  # Retry a numerical iteration limit using its own full state; same data/prior.
  for retry in range(2):
   if m['fit']['status']=='converged':break
   backup=dest/f'retry-{retry:02d}.h5';product.rename(backup);e['SYISR_INITIAL_PROFILE']=backup
   # A temporal prior supplies the initial mean by default; request previous state
   # as initial guess while retaining exactly the same prior in the objective.
   e['SYISR_TEMPORAL_RESTART']='1'
   command(args,'profile-fit',[raw/'lpm.h5','none',product,grid,1,gain,args.iterations,25],dest/f'retry-{retry:02d}.log',e);m=metadata(product)
  if m['fit']['status']!='converged':raise ValueError(f'Window {index} failed convergence')
  previous=product;previous_mid=mid;complete.append(w)
  atomic(out/'status.json',dict(stage='forward',completed=len(complete),total=len(windows),last_window=index))
  print('FORWARD',s['directory'],index,flush=True)
 # RTS uses the saved prediction from the forward filter; each lag likelihood
 # enters once. This is not a second independent fit or averaging two fits.
 next_state=None
 for w in reversed(complete):
  index=w['index'];d=out/'windows'/f'{index:03d}';result=d/'smoothed.h5'
  if next_state is None:command(args,'temporal-step',['copy-state',d/'profile.h5',result],d/'smooth.log')
  else:
   next_dir=out/'windows'/f"{index+1:03d}"
   command(args,'temporal-step',['smooth',d/'profile.h5',next_dir/'prediction.h5',next_state,result],d/'smooth.log')
  next_state=result
 for mode in ['filtered','smoothed']:
  command(args,'export-temporal',[root,mode,args.fit_directory],out/f'export-{mode}.log')
  command(args,'plot-campaign',[out/f'{mode}-campaign.h5',out/f'{mode}-all','--show-all'],out/f'plot-{mode}.log',dict(SYISR_PLOT_MAX_KM=2500))
  command(args,'plot-campaign',[out/f'{mode}-campaign.h5',out/f'{mode}-screened'],out/f'plot-{mode}-screened.log',dict(SYISR_PLOT_MAX_KM=2500))
 atomic(out/'status.json',dict(stage='complete',forward_windows=len(complete),reverse_windows=len(complete),total=len(windows),gain=gain,process_sigma_multiplier=args.process_sigma_multiplier,exponential_scale_km=args.skew_scale_km))

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['prepare','fit']);p.add_argument('root',type=Path);p.add_argument('--bin-directory',type=Path,default=Path('target/release'));p.add_argument('--fit-directory',default='temporal-flat');p.add_argument('--sessions',type=int,nargs='+');p.add_argument('--wait',action='store_true');p.add_argument('--threads',type=int,default=1);p.add_argument('--iterations',type=int,default=600);p.add_argument('--skew-scale-km',type=float,default=1500.);p.add_argument('--process-sigma-multiplier',type=float,default=1.)
 a=p.parse_args();a.root=a.root.resolve();a.bin_directory=a.bin_directory.resolve()
 rank=int(os.environ.get('OMPI_COMM_WORLD_RANK',0));size=int(os.environ.get('OMPI_COMM_WORLD_SIZE',1));plan=json.loads((a.root/'plan.json').read_text());failures=[]
 if a.sessions is not None:plan['sessions']=[s for s in plan['sessions'] if s['index'] in a.sessions]
 jobs=[(s,w) for s in plan['sessions'] for w in s['windows']] if a.mode=='prepare' else [(s,None) for s in plan['sessions']]
 for s,w in jobs[rank::size]:
  try:prepare(a,s,w) if w else fit(a,s)
  except Exception as e:
   if w is not None:atomic(a.root/s['directory']/'windows'/f"{w['index']:03d}"/'covariance-failed.json',dict(error=str(e)))
   failures.append(dict(session=s['directory'],window=w['index'] if w else None,error=str(e),traceback=traceback.format_exc()));print('FAILED',failures[-1],flush=True)
 atomic(a.root/f'{a.mode}-rank-{rank:02d}.json',dict(failures=failures))
 if failures:print(f'Rank {rank}: saved {len(failures)} failures for review; other MPI ranks continue',flush=True)
if __name__=='__main__':main()
