"""Compare raw, voltage-whitened and ideal-code spectra at identical delays."""
import pathlib,sys,json
import h5py
import numpy as np

root=pathlib.Path(sys.argv[1]);raw=pathlib.Path(sys.argv[2])
with h5py.File(root/'selection.h5') as selection:
    matched=selection['matched_path'].asstr()[1];diag=selection['diagnostic_path'].asstr()[1]
with h5py.File(matched) as m:
    gain=m['whitening_gain_unshifted'][:];good=np.flatnonzero(m['pulse_accepted'][:]);codes=m['ideal_code_phase_signs'][:];fs=m.attrs['sample_rate_hz']
with h5py.File(diag) as d:
    names=d['source_filename'].asstr()[:];offsets=d['source_byte_offset'][:];phases=d['pulse_phase'][:];info=json.loads(d.attrs['header_json'])
indices=good[np.linspace(0,len(good)-1,min(256,len(good))).astype(int)]
ranges=np.array([150.,230.,470.]);starts=np.rint((ranges/.149896229-info['front'])*fs*1e-6).astype(int)
nt=5120;nf=8192;frequency=np.fft.fftshift(np.fft.fftfreq(nf,1/fs));spectra={k:[] for k in ['raw','whitened','matched']};even=[];iq=[]
for i in indices:
    with (raw/names[i]).open('rb',buffering=0) as f:
        f.seek(int(offsets[i])+1024);qi=np.frombuffer(f.read(info['n']*8),dtype='<i4').reshape(-1,2)
    v=qi[:,1].astype(float)+1j*qi[:,0];z=np.fft.ifft(np.fft.fft(v,n=len(gain))*gain)
    code=np.repeat(codes[int(phases[i])],160)
    even.append([np.mean(abs(v[::2])**2),np.mean(abs(v[1::2])**2)])
    iq.append([np.mean(v.real**2),np.mean(v.imag**2),np.mean(v.real*v.imag)])
    for key,voltage,c in [('raw',v,1),('whitened',z,1),('matched',z,code)]:
        spectra[key].append(np.array([np.fft.fftshift(np.fft.fft(voltage[j:j+nt]*c,n=nf)) for j in starts]))
with h5py.File(root/'alias-audit.h5','w') as out:
    out['ranges_km']=ranges;out['frequency_hz']=frequency;out['selected_ledger_indices']=indices
    out['even_odd_power']=even;out['iq_moments']=iq
    for key,rows in spectra.items():
        x=np.asarray(rows);p=np.mean(abs(x)**2,axis=0)/nt
        cross=np.mean(x*np.conj(np.roll(x,nf//2,axis=-1)),axis=0)/nt
        coherence=abs(cross)**2/(p*np.roll(p,nf//2,axis=-1))
        out[key+'_power']=p;out[key+'_8MHz_shift_coherence']=coherence
        print(key,'peak frequencies MHz by range',[[round(frequency[k]/1e6,4) for k in np.argsort(row)[-5:][::-1]] for row in p],flush=True)
        print(key,'max 8MHz coherence',np.max(coherence,axis=1),flush=True)
    out.attrs.update(source_matched=matched,source_diagnostic=diag,interpretation='Rectangular 320us windows at identical delays. No code multiplication in raw/whitened spectra; no frequency averaging.',fs_hz=fs)
print('even/odd mean power ratio',np.mean(even,axis=0)[0]/np.mean(even,axis=0)[1])
print('I/Q moments',np.mean(iq,axis=0))
