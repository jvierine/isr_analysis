"""Movie of signed lag spectra from the exact top-gate-subtracted estimator."""
from pathlib import Path
import datetime as dt,json,subprocess
import h5py,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
root=Path('/Volumes/LaCie/SYISR_analysis/ulp2ms-20261118/session-00');out=root/'range-doppler-subtracted';out.mkdir(exist_ok=True)
plan=json.loads((root/'plan.json').read_text());spectra=[];intervals=[];pulsecounts=[];acfmeans=[]
freq=np.fft.fftshift(np.fft.fftfreq(512,1e-5));sel=abs(freq)<=15000
lag=np.arange(61);correction=(200-lag)/200;taper=.5+.5*np.cos(np.pi*lag/61)
for w in plan['windows']:
 with h5py.File(root/'windows'/f"{w['index']:03d}"/'lpm.h5') as f:
  blocks=f['block_acf_real_adu2'][:]+1j*f['block_acf_imag_adu2'][:]
  counts=f['block_retained_pulses'][:];tt=f['time_utc_s'][:];r=f['slant_range_km'][:]
  bt=np.array([np.mean(tt[i*200:min((i+1)*200,len(tt))]) for i in range(len(counts))])
  for lo in np.arange(w['utc_start'],w['utc_end'],300):
   hi=min(lo+300,w['utc_end']);ii=(bt>=lo)&(bt<hi)&(counts>0)
   if not np.any(ii):continue
   acf=np.sum(blocks[ii]*counts[ii,None,None],axis=0)/counts[ii].sum()
   g=acf*correction*taper
   h=np.zeros((len(r),512),complex);h[:,:61]=g;h[:,-60:]=g[:,1:][:,::-1].conj()
   # Saved ACF is x[k] conj(x[k+lag]), so use positive-exponent lag transform.
   s=np.fft.fftshift(np.fft.ifft(h,axis=1).real,axes=1)*512/1e5
   spectra.append(s);intervals.append((lo,hi));pulsecounts.append(counts[ii].sum());acfmeans.append(acf)
 print('averaged',w['index'],flush=True)
spectra=np.array(spectra);intervals=np.array(intervals)
noise=np.median(abs(spectra[:,r>1400][:,:,abs(freq)>20000]));vmax=np.percentile(abs(spectra[:,:,sel]),99.8)
with h5py.File(out/'range-doppler-subtracted.h5','w') as f:
 f.attrs['method']='Retained-pulse-count weighted averages of saved 10-second block complex lag products after exact top-gate subtraction. Undo pulse-overlap correction before cosine-tapered Hermitian lag transform. All retained pulses used; nominal five-minute bins assigned by block midpoint. Signed estimates preserved. These lag spectra are not independently range-resolved or deconvolved plasma spectra; lag-dependent range response and 2 ms pulse remain.'
 f.attrs['frequency_convention']='Positive-exponent lag transform of x[k] conj(x[k+lag]); same signed frequency as forward voltage FFT; physical velocity sign unverified.'
 f.attrs['generator']='syisr/ulp2ms/operations/animate_subtracted_spectra.py';f.attrs['top_gate_slant_range_km']=1844.1733053869998
 f.attrs['display_linthresh_adu2_hz']=noise;f.attrs['display_vmax_adu2_hz']=vmax
 for k,v in [('signed_psd_adu2_hz',spectra),('frequency_hz',freq),('slant_range_km',r),('utc_interval_s',intervals),('retained_pulses',pulsecounts),('acf_real_adu2',np.real(acfmeans)),('acf_imag_adu2',np.imag(acfmeans))]:f.create_dataset(k,data=v,compression='gzip')
fig,ax=plt.subplots(figsize=(11,7),layout='constrained')
im=ax.pcolormesh(freq[sel]/1000,r,spectra[0][:,sel],shading='nearest',cmap='RdBu_r',norm=SymLogNorm(linthresh=noise,vmin=-vmax,vmax=vmax,base=10))
ax.set_xlabel('Doppler frequency (kHz)');ax.set_ylabel('Pulse-centroid slant range (km)');ax.set_xlim(-15,15);ax.set_ylim(r[0],r[-1])
fig.colorbar(im,ax=ax,label='Signed lag spectrum (ADU²/Hz; symmetric log scale)')
for i,s in enumerate(spectra):
 im.set_array(s[:,sel]);a,b=[dt.datetime.fromtimestamp(t,dt.timezone.utc) for t in intervals[i]]
 ax.set_title(f'SYISR 2 ms pulse | {a:%d %b %Y %H:%M}–{b:%H:%M} UTC\nTop-gate background subtracted; five-minute averages after pulse rejection',fontsize=12)
 fig.savefig(out/f'frame-{i:04d}.png',dpi=120)
 if i in (0,len(spectra)//2,len(spectra)-1):fig.savefig(out/f'preview-{i:04d}.png',dpi=150)
plt.close(fig)
with (out/'ffmpeg.log').open('w') as log:subprocess.run(['/opt/anaconda3/bin/ffmpeg','-y','-framerate','6','-i',str(out/'frame-%04d.png'),'-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'range-doppler-subtracted.mp4')],check=True,stdout=log,stderr=log)
for p in out.glob('frame-*.png'):p.unlink()
print(out/'range-doppler-subtracted.mp4',flush=True)
