"""Raw-voltage diagnostic movie; provenance stored in HDF5, not rendered."""
from pathlib import Path
import json,datetime as dt,subprocess
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/Volumes/LaCie/SYISR_analysis/ulp2ms-20261118')
out=ROOT/'session-00/range-doppler';out.mkdir(exist_ok=True)
plan=json.loads((ROOT/'session-00/plan.json').read_text())
nfft=128;hop=32;fs=1e5;starts=np.arange(30,1194-nfft+1,hop);hann=np.hanning(nfft)
freq=np.fft.fftshift(np.fft.fftfreq(nfft,1/fs));sel=np.abs(freq)<=15000
# Rectangular transmit-pulse midpoint correction: fast-time echo delay minus 1 ms.
ranges=(2068+10*(starts+(nfft-1)/2)-1000)*1e-6*299792.458/2
rows=[];times=[];counts=[];fractions=[]
for w in plan['windows']:
 d=ROOT/'session-00/windows'/f"{w['index']:03d}"
 spec=json.loads((d/'selection.json').read_text())
 with h5py.File(d/'lpm.h5') as f:
  means=f['removed_stationary_voltage_real'][:]+1j*f['removed_stationary_voltage_imag'][:];keep=f['pulse_keep'][:]>0;tt=f['time_utc_s'][:]
 spans=[];base=0
 for s in spec['segments']:
  n=s['stop_frame']-s['first_frame'];spans.append((base,base+n,s));base+=n
 files={}
 for lo in np.arange(w['utc_start'],w['utc_end'],300):
  hi=min(lo+300,w['utc_end']);inds=np.flatnonzero((tt>=lo)&(tt<hi))
  if len(inds)==0:continue
  chosen=inds[np.linspace(0,len(inds)-1,min(512,len(inds))).astype(int)]
  accum=np.zeros((2,len(starts),nfft))
  for p in chosen:
   a,b,s=next(x for x in spans if x[0]<=p<x[1]);path=s['path']
   if path not in files:files[path]=open(path,'rb')
   fh=files[path];fh.seek(s['byte_offset']+(s['first_frame']+int(p)-a)*spec['frame_bytes']+1024)
   iq=np.frombuffer(fh.read(1194*8),dtype='<i4').reshape(-1,2);v=iq[:,1].astype(float)+1j*iq[:,0]
   for j,x in enumerate((v,v-means[int(p)//200])):
    cut=x[starts[:,None]+np.arange(nfft)]*hann
    accum[j]+=np.abs(np.fft.fftshift(np.fft.fft(cut,axis=1),axes=1))**2/(fs*np.sum(hann**2))
  rows.append(accum/len(chosen));times.append((lo,hi));counts.append(len(chosen));fractions.append(np.mean(keep[inds]))
 for fh in files.values():fh.close()
 print('averaged window',w['index'],flush=True)
power=np.array(rows);times=np.array(times)
with h5py.File(out/'range-doppler.h5','w') as f:
 f.attrs['method']='Uniform deterministic sample of up to 512 pulses per five-minute interval, BEFORE pulse rejection; incoherent Hann 128-sample fast-time PSD averages. Second product removes saved per-native-sample complex mean of nominal 10-second processing blocks. No background subtraction. x=second int32 + i first int32; forward FFT frequency sign; velocity sign unverified. Pulse-centroid slant range, uncalibrated hardware delay. Range resolution limited by 2 ms transmit pulse and 1.28 ms analysis window; not independent altitude gates.'
 f.attrs['generator']='syisr/ulp2ms/operations/animate_range_doppler.py';f.attrs['sample_rate_hz']=fs
 for k,v in [('psd_adu2_hz',power),('frequency_hz',freq),('slant_range_km',ranges),('utc_interval_s',times),('sampled_pulses',counts),('pipeline_retained_fraction',fractions)]:f.create_dataset(k,data=v,compression='gzip')
# A fixed common reference and fixed colour limits for the entire movie.
ref=np.median(power[:,1,:,-32:]);db=10*np.log10(np.maximum(power,1e-30)/ref)
fig,axes=plt.subplots(1,2,figsize=(12,6),sharex=True,sharey=True,layout='constrained')
ims=[]
for ax,title in zip(axes,['Recorded voltages','Stationary complex mean removed']):
 im=ax.pcolormesh(freq[sel]/1000,ranges,db[0,0][:,sel],shading='nearest',cmap='magma',vmin=-6,vmax=30);ims.append(im)
 ax.set_title(title,fontsize=12);ax.set_xlabel('Doppler frequency (kHz)');ax.set_xlim(-15,15);ax.set_ylim(ranges[0],ranges[-1])
axes[0].set_ylabel('Pulse-centroid slant range (km)')
fig.colorbar(ims[0],ax=axes,label='PSD / fixed background reference (dB)',shrink=.85)
for i in range(len(power)):
 for j,im in enumerate(ims):im.set_array(db[i,j][:,sel])
 a,b=[dt.datetime.fromtimestamp(t,dt.timezone.utc) for t in times[i]]
 fig.suptitle(f'SYISR 2 ms pulse | {a:%d %b %Y %H:%M}–{b:%H:%M} UTC\nFive-minute averages; before pulse rejection; {counts[i]} sampled pulses',fontsize=13)
 fig.savefig(out/f'frame-{i:04d}.png',dpi=120)
 if i in (0,len(power)//2,len(power)-1):fig.savefig(out/f'preview-{i:04d}.png',dpi=150)
plt.close(fig)
subprocess.run(['ffmpeg','-y','-framerate','6','-i',str(out/'frame-%04d.png'),'-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'range-doppler-day.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=(out/'ffmpeg.log').open('w'))
for p in out.glob('frame-*.png'):p.unlink()
print(out/'range-doppler-day.mp4',flush=True)
