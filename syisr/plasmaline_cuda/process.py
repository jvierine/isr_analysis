"""Diagnostic (not AC decoded) 60 s SYISR spectra. HDF5 only; all times UTC.

Raw header clocks use the established SYISR Beijing-clock convention, retained
alongside converted UTC. This new campaign's clock and RF offset need confirmation.
"""
import argparse
import ctypes as ct
import datetime as dt
import json
import pathlib
import struct
import subprocess
import time
import h5py
import numpy as np
from scipy.ndimage import median_filter


def header(b):
    u16=lambda o: struct.unpack_from('<H',b,o)[0]
    u32=lambda o: struct.unpack_from('<I',b,o)[0]
    if len(b)!=1024 or u32(0)!=0xaaaa5555 or u32(1020)!=0xaa5555aa:
        raise ValueError('invalid frame sentinel')
    rates={0:4e6,1:1e5,2:2e5,3:4e5,4:20e6,5:16e6}
    date=dt.datetime(2000+b[13],b[12],b[15],b[14],b[17],b[16],tzinfo=dt.timezone.utc)
    if u16(18)>=40000: raise ValueError('invalid fractional clock')
    return dict(time=date.timestamp()+u16(18)*25e-6-28800,
                fs=rates[b[36]&15],waveform=b[36]>>4,n=u32(768),front=u16(766),
                pulse=u16(988),ipp=u16(980),code=u16(936),phase=u16(942),
                az=u16(856)*360/65536,el=u16(858)*360/65536,
                frequency=(430+b[32]*.2)*1e6,beam_count=b[764])


def layout(path):
    with path.open('rb',buffering=0) as f:
        b=f.read(1024);h=header(b)
        matches=[]
        for extra in range(9):
            stride=1024+8*(h['n']+extra)
            f.seek(stride)
            try: h2=header(f.read(1024))
            except (ValueError,KeyError): continue
            if abs(h2['time']-h['time']-h['ipp']*1e-6)<1e-4: matches.append(extra)
        if len(matches)!=1: raise ValueError(f'unsupported stride: {path}: {matches}')
        h['extra_complex_samples']=matches[0];h['stride']=1024+8*(h['n']+matches[0])
    if h['beam_count']!=1 or (h['fs'],h['pulse'],h['ipp'],h['waveform'],h['code'])!=(16e6,320,10000,3,50):
        raise ValueError(f'unsupported recording layout: {h}')
    return h


class GPU:
    def __init__(self,n,batch,device=0):
        self.lib=ct.CDLL(str(pathlib.Path(__file__).resolve().with_name('libspectra.so')))
        self.lib.pl_error.restype=ct.c_char_p
        self.lib.pl_create.argtypes=[ct.c_int,ct.c_int,ct.c_int,ct.POINTER(ct.c_void_p)]
        self.lib.pl_power.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_void_p]
        self.lib.pl_destroy.argtypes=[ct.c_void_p]
        self.ptr=ct.c_void_p();self.n=n;self.batch=batch
        self.check(self.lib.pl_create(device,n,batch,ct.byref(self.ptr)))
    def check(self,result):
        if result: raise RuntimeError(self.lib.pl_error().decode())
    def power(self,qi):
        x=np.zeros((self.batch,self.n,2),dtype=np.int32)
        x[:len(qi)]=qi
        y=np.empty((self.batch,self.n),dtype=np.float32)
        self.check(self.lib.pl_power(self.ptr,x.ctypes.data,y.ctypes.data))
        return y[:len(qi)]
    def close(self): self.lib.pl_destroy(self.ptr)


def frames(paths,start,stop,reference):
    keys=['fs','n','front','pulse','ipp','code','stride','frequency']
    last=None
    for path in paths:
        h0=layout(path)
        count=path.stat().st_size//h0['stride']
        if h0['time']>=stop or h0['time']+count*h0['ipp']*1e-6<start: continue
        if any(h0[k]!=reference[k] for k in keys): raise ValueError('layout changed')
        with path.open('rb',buffering=0) as f:
            for j in range(count):
                offset=j*h0['stride'];f.seek(offset)
                raw=f.read(1024);h=header(raw)
                if not start<=h['time']<stop: continue
                if any(h[k]!=reference[k] for k in keys if k!='stride'): raise ValueError('within-file layout changed')
                if h['phase']>=64: raise ValueError('unsupported code phase')
                if last is not None and h['time']<=last: raise ValueError('duplicate/backward clock')
                last=h['time']
                qi=np.frombuffer(f.read(h['n']*8),dtype='<i4').reshape(-1,2)
                if len(qi)!=h['n']: raise ValueError('short frame')
                yield h,qi,path.name,offset,raw


def catalogue(raw,destination=None):
    if destination is not None and destination.exists():
        with h5py.File(destination) as src:
            if src.attrs['raw_directory']!=str(raw): raise ValueError('catalogue raw path mismatch')
            return [(float(t),raw/name) for t,name in zip(src['first_header_time_utc_s'][:],src['filename'].asstr()[:])]
    indexed=[]
    for path in raw.glob('*dat*'):
        with path.open('rb',buffering=0) as stream: h=header(stream.read(1024))
        indexed.append((h['time'],path))
    indexed.sort(key=lambda x:x[0])
    if destination is not None:
        destination.parent.mkdir(parents=True,exist_ok=True)
        tmp=destination.with_suffix('.partial.h5')
        with h5py.File(tmp,'w') as out:
            out['first_header_time_utc_s']=[t for t,_ in indexed]
            out['filename']=np.asarray([p.name for _,p in indexed],dtype=h5py.string_dtype())
            out['file_size_bytes']=[p.stat().st_size for _,p in indexed]
            out.attrs.update(raw_directory=str(raw),status='First-header catalogue only; frame integrity not certified')
        tmp.replace(destination)
    return indexed


def run(a):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=pathlib.Path(__file__).resolve().parent,text=True).strip()
    # Header chronology matters: filenames are session-start times, not frame times.
    indexed=catalogue(a.raw,getattr(a,'catalogue',None))
    paths=[p for _,p in indexed];ref=layout(paths[0]);start=a.start if a.start is not None else ref['time']
    stop=start+60.;n=a.nfft;nr=ref['n']//n
    paths=[p for t,p in indexed if t<stop and t+(p.stat().st_size//ref['stride'])*ref['ipp']*1e-6>=start]
    gpu=GPU(n,nr,a.device);blocks=[];ledger=[];beg=time.time()
    # A single minute is ~3.6 GB in float power. Stage HDF5 to bound RAM.
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.output.with_suffix('.partial.h5')
    with h5py.File(temporary,'w') as out:
        stage=out.create_dataset('_stage',shape=(0,nr,n),maxshape=(None,nr,n),chunks=(1,nr,n),dtype='f4')
        for h,qi,name,offset,raw in frames(paths,start,stop,ref):
            p=gpu.power(qi[:nr*n].reshape(nr,n,2))
            i=len(stage);stage.resize(i+1,axis=0);stage[i]=p
            blocks.append(np.median(p[nr*3//4:],axis=0))
            ledger.append((h,name,offset,raw))
            if (i+1)%500==0: print(f'staged {i+1} pulses ({time.time()-beg:.1f}s)',flush=True)
        if len(stage)<2: raise ValueError('insufficient pulses')
        # Robust background from last quarter of recorded ranges. A diagnostic
        # reference, not an assertion of signal-free noise or absolute calibration.
        # Divide by ln(2) to account for median exponential periodogram power.
        baseline=np.median(blocks,axis=0)/np.log(2)
        smooth=median_filter(baseline,size=33,mode='wrap')
        freq_ok=(baseline>0)&np.isfinite(baseline)&(baseline<8*smooth)
        safe=np.where(baseline>0,baseline,1)
        rawsum=np.zeros((nr,n));clean=np.zeros_like(rawsum);counts=np.zeros((nr,n),dtype='u4')
        rejected=[];intermittent=[];pulse_ok=[];narrowband_burst=[]
        masks=out.create_dataset('accepted_cell_mask_packed',shape=(len(stage),nr,(n+7)//8),dtype='u1',compression='gzip')
        for i in range(len(stage)):
            p=stage[i];white=p/safe
            burst=bool(np.any(np.median(white[nr*3//4:],axis=0)[freq_ok]>8))
            good_pulse=(np.median(white[:,freq_ok])<4*np.log(2)) and not burst
            narrowband_burst.append(burst)
            good=(white<30)&freq_ok[None,:]&good_pulse
            masks[i]=np.packbits(good,axis=-1)
            rawsum+=p;clean+=np.where(good,p,0);counts+=good
            rejected.append(1-good.mean());pulse_ok.append(good_pulse)
            intermittent.append(float(np.mean(white[:,freq_ok]>=30)))
        mean=np.divide(clean,counts,out=np.full_like(clean,np.nan),where=counts>0)
        out['raw_mean_power_adu2']=rawsum/len(stage)
        out['accepted_mean_power_adu2']=mean
        out['whitened_power_ratio']=mean/safe
        out['signed_excess_power_adu2']=mean-baseline
        out['accepted_counts']=counts
        out['passband_reference_adu2']=baseline;out['frequency_accepted']=freq_ok
        out['frequency_offset_hz']=np.fft.fftshift(np.fft.fftfreq(n,1/ref['fs']))
        out['receive_delay_us']=ref['front']+(np.arange(nr)*n+(n-1)/2)/ref['fs']*1e6
        out['apparent_slant_range_km']=out['receive_delay_us'][:]*.149896229
        out['pulse_time_utc_s']=[h['time'] for h,_,_,_ in ledger]
        out['pulse_phase']=[h['phase'] for h,_,_,_ in ledger]
        out['pulse_azimuth_deg']=[h['az'] for h,_,_,_ in ledger]
        out['pulse_elevation_deg']=[h['el'] for h,_,_,_ in ledger]
        out['pulse_accepted']=pulse_ok;out['rejected_cell_fraction_per_pulse']=rejected
        out['intermittent_cell_fraction_per_pulse']=intermittent
        out['narrowband_reference_burst']=narrowband_burst
        out['source_filename']=np.asarray([name for _,name,_,_ in ledger],dtype=h5py.string_dtype())
        out['source_byte_offset']=[offset for _,_,offset,_ in ledger]
        out['raw_headers']=np.asarray([np.frombuffer(raw,dtype='u1') for _,_,_,raw in ledger])
        del out['_stage']
        out.attrs.update(schema='syisr-plasmaline-diagnostic-v1',decoded=False,
            integration_start_utc_s=start,integration_stop_utc_s=stop,pulses=len(ledger),
            frequency_sign='FFT forward of I+iQ: positive phase rotation is positive frequency; RF sideband sign uncalibrated',
            range_response='Undecoded 320us pulse convolved with 64us Hann STFT support (for nfft=1024); apparent slant range, not altitude or decoded scattering range',
            whitening='Last-quarter-range median per pulse, median across pulses, divided by ln(2); prior to thresholding',
            rfi='Frequency baseline >8x 33-bin median excluded; whitened cells >30 excluded; pulse median >4ln(2) or any reference-range frequency median >8 excluded. Data-dependent censoring can bias weak signals.',
            limitations='No alternating-code decoding, no transmit sample, no absolute calibration; potential signal in background reference; RF offset and new-campaign UTC+8 clock convention unconfirmed.',
            header_json=json.dumps(ref),nfft=n,window='symmetric Hann',
            normalization='abs(FFT(I+iQ * Hann))^2 / sum(Hann^2), ADU^2 per Fourier bin; divide by sample rate for ADU^2/Hz',
            source_commit=commit)
    gpu.close()
    # Repack removes the large staging allocation before atomic publication.
    compact=a.output.with_suffix('.compact.h5')
    with h5py.File(temporary,'r') as src,h5py.File(compact,'w') as dst:
        for k in src: src.copy(k,dst)
        dst.attrs.update(src.attrs)
    compact.replace(a.output);temporary.unlink()
    print(json.dumps(dict(output=str(a.output),pulses=len(ledger),elapsed_s=time.time()-beg)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=pathlib.Path);p.add_argument('output',type=pathlib.Path)
    p.add_argument('--start',type=float);p.add_argument('--nfft',type=int,default=1024);p.add_argument('--device',type=int,default=0);p.add_argument('--catalogue',type=pathlib.Path)
    run(p.parse_args())
