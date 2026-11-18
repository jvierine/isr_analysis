"""Replay saved background subtraction and inspect signed lag spectra."""
from pathlib import Path
import json
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/Volumes/LaCie/SYISR_analysis/ulp2ms-20261118/session-00');out=root/'range-doppler'
fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained',sharex=True,sharey=True)
freq=np.fft.fftshift(np.fft.fftfreq(512,1e-5));sel=abs(freq)<=15000
records=[]
for row,idx in enumerate([1,16,24]):
 with h5py.File(root/'windows'/f'{idx:03d}'/'lpm.h5') as f:
  z=lambda k:f[k+'_real_adu2'][:]+1j*f[k+'_imag_adu2'][:]
  before=z('acf_before_background');after=z('acf');bg=z('background_acf');counts=f['block_retained_pulses'][:];r=f['slant_range_km'][:]
  m=json.loads(bytes(f.attrs['metadata_json']))
 correction=(200-np.arange(61))/200
 b=np.nansum(bg*counts[:,None],axis=0)/counts.sum()
 replay=before-b[None,:]/correction
 err=np.max(abs(replay-after))/np.max(abs(before));print(idx,'subtraction relative replay error',err,flush=True)
 spectra=[]
 for acf in [before,after]:
  g=acf*correction;g=g*(.5+.5*np.cos(np.pi*np.arange(61)/61))
  a=np.zeros((len(r),512),complex);a[:,:61]=g;a[:,-60:]=g[:,1:][:,::-1].conj()
  spectra.append(np.fft.fftshift(np.fft.ifft(a,axis=1).real,axes=1)*512/1e5)
 scale=np.median(np.abs(spectra[0][r>1400][:,abs(freq)>20000]))
 for col,s in enumerate(spectra):
  v=np.arcsinh(s/scale)/np.log(10)
  im=axes[row,col].pcolormesh(freq[sel]/1000,r,v[:,sel],cmap='RdBu_r',vmin=-2,vmax=2,shading='nearest')
  axes[row,col].set_ylim(1000,1800);axes[row,col].set_ylabel(f'Window {idx:03d}\nSlant range (km)')
 records.append((idx,r,np.array(spectra),scale,err))
axes[0,0].set_title('Before top-gate subtraction');axes[0,1].set_title('After top-gate subtraction')
for ax in axes[-1]:ax.set_xlabel('Doppler frequency (kHz)')
fig.colorbar(im,ax=axes,label='asinh(signed spectrum / background scale) / ln(10)',shrink=.8)
fig.savefig(out/'top-gate-subtraction-check.png',dpi=150)
with h5py.File(out/'top-gate-subtraction-check.h5','w') as f:
 f.attrs['method']='Actual retained-pulse lag products; undo pulse-overlap factor before Hermitian lag transform; cosine lag taper; signed spectra retained. This is a diagnostic lag spectrum, not a deconvolved plasma spectrum.'
 f.create_dataset('frequency_hz',data=freq)
 for idx,r,s,scale,err in records:
  g=f.create_group(f'{idx:03d}');g['slant_range_km']=r;g['before_after_signed_psd_adu2_hz']=s;g.attrs['display_scale']=scale;g.attrs['relative_subtraction_replay_error']=err
