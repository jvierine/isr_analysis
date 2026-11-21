"""Independent NumPy replay of selected raw decoded lags; run in Slurm."""
from pathlib import Path
import argparse,json
import h5py,numpy as np
P=argparse.ArgumentParser();P.add_argument('root',type=Path);q=P.parse_args();root=q.root
codes=np.array([[1 if s=='+' else -1 for s in line.strip()] for line in (root/'source/syisr/m32-reference.bc').read_text().splitlines() if line.strip()])
with h5py.File(root/'index.h5') as h:
 files=json.loads(bytes(h['files_json_utf8'][:]));times=h['cycle_time_utc_s'][:];beam=h['beam_index'][:];offset=h['frame_offset'][:];ids=h['file_index'][:];meta=json.loads(bytes(h['metadata_json_utf8'][:]))
with h5py.File(root/'blocks/0000.h5') as h:
 t0,t1=h['interval_utc_s'][:];g=h['beam_0'];stored=g['acf_adu2'][:];bad=g['rejected_cycle_gate'][:];noise=g['far_reference_acf_adu2'][:];r=h['slant_range_km'][:]
chosen=np.flatnonzero((beam==0)&(times[:,0]>=t0)&(times[:,1]<t1));n=meta['samples'];assert len(chosen)==len(bad)
x=np.empty((len(chosen),64,n),np.complex128);handles={}
for j,ci in enumerate(chosen):
 for c in range(64):
  fi=int(ids[ci,c]);
  if fi not in handles:handles[fi]=open(files[fi],'rb')
  f=handles[fi];f.seek(int(offset[ci,c])+1024);v=np.frombuffer(f.read(n*8),'<i4').reshape(n,2).astype('f4').astype('f8');x[j,c]=v[:,1]+1j*v[:,0]
for f in handles.values():f.close()
checks=[]
for target in [169.6,304.5,500.]:
 gate=int(np.argmin(abs(r-target)));keep=bad[:,gate]==0
 assert keep.sum()>0
 for lag in [1,2,3,20,62]:
  branches=[(a,l) for a,l in [(0,lag//2),(1,(lag+1)//2)] if l>0];value=0j;bias=0.
  for a,l in branches:
   b=np.arange(32-l);k=2*gate+2*b+a;phase=codes[:,b]*codes[:,b+l];factor=(2 if lag%2 else 1)/(len(branches)*64*(32-l));products=x[keep][:,:,k+lag]*x[keep][:,:,k].conj();value+=np.einsum('cpb,pb->',products,phase)*factor/keep.sum();bias+=phase.sum()*factor
  value-=complex(*noise[lag])*bias;actual=complex(*stored[gate,lag]);relative=abs(value-actual)/max(abs(value),abs(actual),1.)
  assert relative<1e-7,(gate,lag,value,actual,relative);checks.append([gate,lag,relative])
with h5py.File(root/'15min-step5min.h5') as h:
 freq=h['frequency_hz'][:];nf=len(freq);dt=1e-5
 for b in range(5):
  for j in [0,104,207]:
   ac=h[f'beam_{b}/decoded_acf_adu2'][j,20];acf=ac[:,0]+1j*ac[:,1];v=np.zeros(nf,complex);v[0]=acf[0].real
   for l in range(1,len(acf)):v[l]=acf[l]*(1-l/len(acf));v[-l]=v[l].conj()
   expected=np.fft.fftshift(np.fft.fft(v).real)*dt;got=h[f'beam_{b}/background_subtracted_psd_adu2_per_hz'][j,20];assert np.allclose(got,expected,rtol=1e-6,atol=max(abs(expected))*1e-7)
with h5py.File(root/'verification.h5','w') as h:
 h['raw_decode_checks_gate_lag_relative_error']=checks;h.attrs['result']='passed';h.attrs['scope']='independent NumPy raw replay: beam0 first5min, three ranges, five lags; spectrum replay five beams and first/middle/last windows'
print('PASS raw gated ACF and spectrum replay; max relative lag error',np.max(np.array(checks)[:,2]),flush=True)
