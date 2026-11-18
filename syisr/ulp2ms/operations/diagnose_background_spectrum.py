from pathlib import Path
import json,struct
import h5py,numpy as np
root=Path('/Volumes/LaCie/SYISR_analysis/ulp2ms-20261118');d=root/'session-00/windows/001';spec=json.loads((d/'selection.json').read_text())
with h5py.File(d/'lpm.h5') as f:
 keep=f['pulse_keep'][:]>0
 means=f['removed_stationary_voltage_real'][:]+1j*f['removed_stationary_voltage_imag'][:]
indices=np.flatnonzero(keep);indices=indices[np.linspace(0,len(indices)-1,min(2048,len(indices))).astype(int)]
spans=[];start=0
for s in spec['segments']:
 n=s['stop_frame']-s['first_frame'];spans.append((start,start+n,s));start+=n
nfft=256;win=np.hanning(nfft);normalization=1e5*np.sum(win**2);raw=[];centered=[]
files={}
for pulse in indices:
 lo,hi,s=next(span for span in spans if span[0]<=pulse<span[1]);path=s['path']
 if path not in files:files[path]=open(path,'rb')
 offset=s['byte_offset']+(s['first_frame']+int(pulse)-lo)*spec['frame_bytes'];f=files[path];f.seek(offset+1024);iq=np.frombuffer(f.read(1194*8),dtype='<i4').reshape(-1,2);v=iq[:,1].astype(float)+1j*iq[:,0]
 mean=means[int(pulse)//200]
 for dest,value in [(raw,v[-nfft:]),(centered,(v-mean)[-nfft:])]:dest.append(np.abs(np.fft.fftshift(np.fft.fft(value*win)))**2/normalization)
for f in files.values():f.close()
raw=np.mean(raw,axis=0);centered=np.mean(centered,axis=0);freq=np.fft.fftshift(np.fft.fftfreq(nfft,1e-5));floor=np.array([np.median(centered[max(0,i-10):min(nfft,i+11)]) for i in range(nfft)])
ratio=centered/np.maximum(floor,1e-30);selected=np.argsort(ratio)[-8:][::-1]
print('Top local-line contrasts',list(zip(freq[selected].tolist(),ratio[selected].tolist(),centered[selected].tolist())))
print('Raw peak frequency',freq[np.argmax(raw)],'centered peak',freq[np.argmax(centered)],'median PSD',np.median(centered))
with h5py.File(root/'single-step-diagnosis/background-spectrum.h5','w') as f:
 f.attrs['method']='2048 retained pulses sampled through one30min window; final256 recorded fast-time samples; Hann PSD; raw and per-native-sample block-complex-mean-removed voltages; signed frequency for x=I+iQ; relative spectral features do not identify transmitter or calibrated receiver transfer'
 f.attrs['sample_rate_hz']=1e5;f.attrs['retained_pulses_used']=len(indices)
 for k,v in [('frequency_hz',freq),('raw_psd_adu2_hz',raw),('centered_psd_adu2_hz',centered),('local_median_psd_adu2_hz',floor),('local_line_contrast',ratio)]:f.create_dataset(k,data=v)
