"""Depletion-focused forward RTIs. Raw signed observations are never clipped.
One-sided Huber caps positive excess-echo influence, not its stored amplitude.
Field alignment, common 120 m/s advection and lambda=100 are conditional assumptions.
"""
from pathlib import Path
import os,sys,subprocess
import numpy as np,h5py
from mpi4py import MPI
from scipy import sparse
root=Path(sys.argv[1]); native=Path(sys.argv[2]); out=root/'depletion_prediction';out.mkdir(exist_ok=True)
comm=MPI.COMM_WORLD
assert comm.size<=16
with h5py.File(root/'cavity/observations.h5') as h:
    obs=h['observations'][()];eligible=h['eligible'][()];stats=h['beam_statistics'][()]
with h5py.File(root/'three_meridians/drift_profile.h5') as h:vx=float(h['pooled_speed_m_s'][()])/1000
# Beam9 excluded a priori as in existing conservative analysis, but forward-predicted.
if comm.rank==0:
    with h5py.File(out/'observations.h5','w') as h:
        h['observations']=obs;h['eligible']=eligible*(obs[:,1]!=8);h['beam_statistics']=stats
comm.Barrier()
jobs=[('baseline',-1,None),('depletion',-1,.2),('sensitivity_low',-1,.1),('sensitivity_high',-1,.4)]+[(f'heldout_{b}',99+b,.2) for b in range(1,12) if b!=9]
for j in range(comm.rank,len(jobs),comm.size):
    name,fold,delta=jobs[j];env=dict(os.environ)
    if delta is not None:env['SYISR_CAVITY_DEPLETION']=str(delta)
    else:env.pop('SYISR_CAVITY_DEPLETION',None)
    subprocess.run([str(native),'fit',str(out/'observations.h5'),str(out/f'{name}.h5'),str(vx),'0','100',str(fold)],env=env,check=True)
    print('COMPLETE',name,flush=True)
comm.Barrier()
if comm.rank:sys.exit(0)
# Project all observations, including excluded beam9, through finite angular beam.
n=len(obs);rr=[];cc=[];vv=[];w=np.outer([1/6,2/3,1/6],[1/6,2/3,1/6]).ravel()
for q in range(9):
    x=(obs[:,13+2*q]-vx*(obs[:,0]-1772369100.)+1300)/10;y=(obs[:,14+2*q]+800)/10
    ix=np.floor(x).astype(int);iy=np.floor(y).astype(int);good=(ix>=0)&(ix<260)&(iy>=0)&(iy<160);fx=x-ix;fy=y-iy
    for dx,dy,a in [(0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)]:
        rr.append(np.flatnonzero(good));cc.append((iy[good]+dy)*261+ix[good]+dx);vv.append(w[q]*a[good])
A=sparse.csr_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(n,261*161))
geomgood=np.isclose(np.asarray(A.sum(axis=1)).ravel(),1)
pred={};support={}
for name,fold,delta in jobs:
    with h5py.File(out/f'{name}.h5') as h:c=h['contrast'][()].ravel();s=h['support_weight'][()].ravel();d=h['diagnostics'][()]
    valid=geomgood & (np.asarray(A@(s>.005*s.max()).astype(float)).ravel()>.99)
    pred[name]=np.where(valid,1+A@c,np.nan);support[name]=valid
    print('FIT_DIAGNOSTIC',name,d,flush=True)
held=np.full(n,np.nan)
for b in range(1,12):
    ix=obs[:,1]==b-1;held[ix]=pred['depletion' if b==9 else f'heldout_{b}'][ix]
with h5py.File(out/'rti_predictions.h5','w') as h:
    h['observations']=obs;h['original_eligible']=eligible;h['beam9_training_excluded']=True;h['speed_m_s']=vx*1000
    for name in ['baseline','depletion','sensitivity_low','sensitivity_high']:h[name+'_ratio']=pred[name]
    h['heldout_ratio']=held
    metrics=[]
    for b in range(1,12):
        ix=(obs[:,1]==b-1)&(eligible>0)&np.isfinite(held)&(np.abs(obs[:,31])>=2);low=ix&(obs[:,6]<.7)
        known=ix&(obs[:,6]<=1.2);true=known&(obs[:,6]<.7);guess=known&(held<.7)
        metrics.append([b,ix.sum(),low.sum(),np.sqrt(np.mean((held[low]-obs[low,6])**2)) if low.any() else np.nan,(true&guess).sum()/max((true|guess).sum(),1)])
    h['beam_count_depleted_count_depleted_rmse_iou']=metrics
    h.attrs['model']='Finite9point beam, IGRF field-aligned frozen relative echo deficit, common drift, 10km grid. One-sided Huber delta0.2 in echo-ratio units; model ratio0..1. Positive excess residual has bounded influence. Raw data retain signed values.'
    h.attrs['validation']='Wholebeam holdout, fixed previously selected drift and lambda100: conditional predictive diagnostic, not independent model selection. Beam9 never trained. Only supported samples shown; nearperp and ratio>=6 original flags remain excluded from fit and metrics.'
    h.attrs['limitations']='Quiet reference may contain contamination. Enhanced echo can hide a true depletion. No calibrated Ne; no spectral coherent-scatter separation. No inferred parallel length. Range/time response inherited from inversion product; angular averaging explicit.'
    h.attrs['metrics']='Depletion threshold0.7; IoU on observed ratio<=1.2, separately exclude unknown enhancement; empirical diagnostic, not a physical truth mask.'
print('PREDICTIVE_METRICS',np.array(metrics),flush=True)
