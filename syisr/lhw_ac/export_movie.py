"""15-minute, five-minute-step signed spectra from immutable gated ACF blocks.
Zero lag is the decoder's short-lag extrapolation, not broad measured zero lag.
"""
from pathlib import Path
import argparse,json,hashlib
import h5py,numpy as np
A=argparse.ArgumentParser();A.add_argument('blocks',type=Path);A.add_argument('geometry',type=Path);A.add_argument('output',type=Path);q=A.parse_args()
files=sorted(q.blocks.glob('[0-9][0-9][0-9][0-9].h5'));assert len(files)>=3
with h5py.File(q.geometry) as h:geo=h['beam_geometry'][:]
with h5py.File(files[0]) as h:r=h['slant_range_km'][:];lags=h['lag_s'][:]
ng=len(r);nl=len(lags);nf=512;dt=1e-5;freq=np.fft.fftshift(np.fft.fftfreq(nf,dt));nwin=len(files)-2
starts=[]
for f in files:
 with h5py.File(f) as h:starts.append(h['interval_utc_s'][0])
assert np.allclose(np.diff(starts),300)
def transform(acf):
 a=np.zeros((*acf.shape[:-1],nf),complex);a[...,0]=acf[...,0].real
 for l in range(1,nl):
  v=acf[...,l]*(1-l/nl);a[...,l]=v;a[...,-l]=v.conj()
 return np.fft.fftshift(np.fft.fft(a,axis=-1).real,axes=-1)*dt
q.output.parent.mkdir(parents=True,exist_ok=True)
with h5py.File(q.output,'w') as out:
 out['window_start_utc_s']=starts[:-2];out['window_end_utc_s']=np.array(starts[:-2])+900;out['slant_range_km']=r;out['frequency_hz']=freq;out['lag_s']=lags;out['beam_geometry']=geo
 out.attrs.update(schema='syisr-lhw-ac-movie-v1',integration_wall_s=900,step_wall_s=300,sample_rate_hz=100000,triangular_window_zero_us=630.,zero_lag='(4 Re R10us - Re R20us)/3; signed local range-kernel approximation',range_status='nominal header667us leading delay; no assumed uncoded-pulse restart correction',frequency_sign='positive IQ phase rotation maps positive Doppler; physical velocity sign not calibrated',normalization='signed decoded signal spectrum / positive received-support voltage power before background subtraction; not unit-area signal normalization')
 hashes=[]
 for b in range(5):
  g=out.create_group(f'beam_{b}');s=g.create_dataset('background_subtracted_psd_adu2_per_hz',(nwin,ng,nf),dtype='f4',chunks=(1,ng,nf),compression='gzip',compression_opts=2)
  u=g.create_dataset('unscreened_psd_adu2_per_hz',s.shape,dtype='f4',chunks=s.chunks,compression='gzip',compression_opts=2)
  received=g.create_dataset('psd_adu2_per_hz',s.shape,dtype='f4',chunks=s.chunks,compression='gzip',compression_opts=2)
  received.attrs['role']='Flat proxy carrying measured positive support power for normalization; not a measured spectral shape'
  ac=g.create_dataset('decoded_acf_adu2',(nwin,ng,nl,2),dtype='f8',chunks=(1,ng,nl,2),compression='gzip',compression_opts=2)
  count=g.create_dataset('accepted_cycles',(nwin,ng),dtype='u4');pwr=g.create_dataset('normalization_received_power_adu2',(nwin,ng),dtype='f8')
  cache=[]
  for i,f in enumerate(files):
   with h5py.File(f) as h:
    v=h[f'beam_{b}/acf_adu2'][:];uv=h[f'beam_{b}/unscreened_acf_adu2'][:];counts=h[f'beam_{b}/accepted_cycles'][:];power=h[f'beam_{b}/received_support_power_adu2'][:];nc=len(h[f'beam_{b}/cycle_time_utc_s'])
   cache.append((v[...,0]+1j*v[...,1],uv[...,0]+1j*uv[...,1],counts,power,nc))
   if len(cache)<3:continue
   c=sum(x[2] for x in cache);ns=sum(x[4] for x in cache)
   total=sum(np.nan_to_num(x[0])*x[2][:,None] for x in cache);uns=sum(x[1]*x[4] for x in cache)/ns
   with np.errstate(invalid='ignore',divide='ignore'):
    v=total/c[:,None];power=sum(np.nan_to_num(x[3])*x[2] for x in cache)/c
   sp=transform(v);usp=transform(uns);j=i-2
   valid=np.isfinite(v[:,0].real);assert np.allclose(np.sum(sp[valid],axis=1)*(freq[1]-freq[0]),v[valid,0].real,rtol=1e-6,atol=1e-6)
   s[j]=sp;u[j]=usp;received[j]=np.broadcast_to(power[:,None]/100000,sp.shape);ac[j]=np.stack([v.real,v.imag],axis=-1);count[j]=c;pwr[j]=power;cache.pop(0)
  targets=np.array([geo[b,3],geo[b,3]+900*.149896229]);cuts=[]
  for target,name in zip(targets,['perpendicular_cut_psd_adu2_per_hz','reference_cut_psd_adu2_per_hz']):
   hi=np.searchsorted(r,target);assert 0<hi<len(r);w=(target-r[hi-1])/(r[hi]-r[hi-1]);v=s[:,hi-1,:]*(1-w)+s[:,hi,:]*w;g[name]=v;cuts.append(v)
  g['cut_slant_range_km']=targets
  vals=np.concatenate(cuts,axis=1);low=np.nanmin(vals,axis=1);high=np.nanmax(vals,axis=1);span=np.maximum(high-low,1e-20);g['cut_linear_limits']=np.stack([np.minimum(0,low)-.05*span,high+.1*span],axis=1)
  positive=vals[np.isfinite(vals)&(vals>0)];assert len(positive);g['cut_log_limits']=[10**np.floor(np.log10(np.quantile(positive,.01))),10**np.ceil(np.log10(positive.max()*1.01))]
  print('EXPORT beam',b,'windows',nwin,flush=True)
 # Shared fixed display limits, excluding only nonfinite values from statistics.
 raw=[];norm=[]
 for b in range(5):
  g=out[f'beam_{b}'];v=g['background_subtracted_psd_adu2_per_hz'][::4,::3,:];p=g['normalization_received_power_adu2'][::4,::3][:,:,None];raw.extend(abs(v[np.isfinite(v)]).ravel().tolist());nn=v/p*1e4;norm.extend(abs(nn[np.isfinite(nn)]).ravel().tolist())
 out.attrs['fixed_signed_absolute_limit']=float(np.quantile(raw,.9975));out.attrs['fixed_signed_normalized_limit']=max(.05,float(np.quantile(norm,.9975)))
 out['processing_json_utf8']=np.frombuffer(json.dumps({'processor':'syisr-lhw-ac','version':'0.1.0','integration_s':900,'step_s':300,'source_blocks':[str(f) for f in files],'spectral_window':'w_l=1-l/63, l=0..62; first null ~1587Hz, FWHM~1406Hz; FFT bins195.3125Hz not resolution','negative_estimates':'retained; no PSD clipping','screening':'coded-Doppler/burst/clipping masks, common complete-cycle/gate rejection; unscreened spectra retained'}).encode(),dtype='u1')
print('COMPLETE',q.output,flush=True)
