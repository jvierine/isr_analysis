"""Hilliard-style sliding code-conjugate spectra, voltage-whitened first.

Conditional ideal-waveform result: existing m32 table, 32 x 10 us rectangular
bauds, header little_code as phase. No measured transmit sample is available.
"""
import argparse
import ctypes as ct
import hashlib
import json
import pathlib
import subprocess
import time
import h5py
import numpy as np
from process import GPU,frames,layout


class MatchedGPU:
    def __init__(self,ne,nt,nfft,step,nout,codes,gain,device=0):
        self.lib=ct.CDLL(str(pathlib.Path(__file__).with_name('libspectra.so')))
        self.lib.pl_error.restype=ct.c_char_p
        self.lib.pl_match_create.argtypes=[ct.c_int]*6+[ct.c_void_p,ct.c_void_p,ct.POINTER(ct.c_void_p)]
        self.lib.pl_match_power.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_void_p]
        self.lib.pl_match_destroy.argtypes=[ct.c_void_p]
        self.ptr=ct.c_void_p();self.nr=(ne-nt)//step+1;self.nout=nout
        self.check(self.lib.pl_match_create(device,ne,nt,nfft,step,nout,codes.ctypes.data,gain.ctypes.data,ct.byref(self.ptr)))
    def check(self,result):
        if result: raise RuntimeError(self.lib.pl_error().decode())
    def power(self,qi,phase):
        qi=np.ascontiguousarray(qi,dtype=np.int32);out=np.empty((self.nr,self.nout),dtype='f4')
        self.check(self.lib.pl_match_power(self.ptr,qi.ctypes.data,phase,out.ctypes.data));return out
    def close(self): self.lib.pl_match_destroy(self.ptr)


def run(a):
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(a.reference) as ref:
        baseline=ref['passband_reference_adu2'][:];frequency_ok=ref['frequency_accepted'][:]
        f=ref['frequency_offset_hz'][:];start=ref.attrs['integration_start_utc_s'];stop=ref.attrs['integration_stop_utc_s']
        info=json.loads(ref.attrs['header_json']);source_names=ref['source_filename'].asstr()[:]
        # Reject whole IPPs on broadband bursts or >1% contaminated STFT cells.
        good=ref['pulse_accepted'][:]& (ref['intermittent_cell_fraction_per_pulse'][:]<.01)
        expected_time=ref['pulse_time_utc_s'][:]
    ne=info['n'];nt=round(info['fs']*info['pulse']*1e-6);nfft=8192;step=160;nout=1024
    codestring=a.codes.read_text();rows=[s.strip() for s in codestring.splitlines() if s.strip()]
    if len(rows)!=64 or any(len(s)!=32 or set(s)-set('+-') for s in rows): raise ValueError('64x32 phase table required')
    codes=np.repeat(np.asarray([[1 if x=='+' else -1 for x in s] for s in rows],dtype='f4'),nt//32,axis=1)
    nw=1<<(ne-1).bit_length();ff=np.fft.fftfreq(nw,1/info['fs'])
    # Periodic interpolation avoids an artificial discontinuity at the Nyquist seam.
    power=np.interp(ff,f,baseline,period=info['fs'])
    valid=np.interp(ff,f,frequency_ok.astype(float),period=info['fs'])>.999
    gain=np.asarray(np.where(valid,1/np.sqrt(np.maximum(power,np.finfo(float).tiny)),0),dtype='f4')
    # Suppress finite-record FFT-filter edges: a 32us guard on both sides.
    guard=512;nr=(ne-nt)//step+1;range_good=(np.arange(nr)*step>=guard)&(np.arange(nr)*step+nt<=ne-guard)
    gpu=MatchedGPU(ne,nt,nfft,step,nout,codes,gain,a.device)
    detector=GPU(1024,ne//1024,a.device)
    total=np.zeros((nr,nout));squares=np.zeros_like(total);phase_counts=np.zeros(64,dtype='u4')
    paths=sorted({a.raw/name for name in source_names},key=lambda p:layout(p)['time'])
    ledger=[];beg=time.time()
    for i,(h,qi,name,offset,raw) in enumerate(frames(paths,start,stop,info)):
        if i>=len(good) or abs(h['time']-expected_time[i])>1e-7: raise ValueError('reference ledger mismatch')
        if good[i]:
            nr0=ne//1024
            white=detector.power(qi[:nr0*1024].reshape(nr0,1024,2))/baseline
            good[i]=not np.any(np.median(white[nr0*3//4:],axis=0)[frequency_ok]>8)
        if good[i]:
            p=gpu.power(qi,h['phase']);total+=p;squares+=p.astype(float)**2;phase_counts[h['phase']]+=1
        ledger.append((h,name,offset))
        if (i+1)%500==0: print(f'matched {i+1}/{len(good)} pulses ({time.time()-beg:.1f}s)',flush=True)
    if len(ledger)!=len(good): raise ValueError('incomplete reference replay')
    count=int(good.sum())
    if count<2: raise ValueError('insufficient accepted IPPs')
    mean=total/count;sem=np.sqrt(np.maximum(0,(squares-total**2/count)/(count-1)/count))
    mean[~range_good]=np.nan;sem[~range_good]=np.nan
    tmp=a.output.with_suffix('.partial.h5')
    with h5py.File(tmp,'w') as out:
        out['whitened_matched_power_ratio']=mean;out['pulse_scatter_sem']=sem
        out['signed_excess_over_unit_noise']=mean-1
        out['range_valid']=range_good;out['pulse_accepted']=good;out['accepted_phase_counts']=phase_counts
        out['pulse_time_utc_s']=expected_time;out['ideal_code_phase_signs']=np.asarray([[1 if x=='+' else -1 for x in s] for s in rows],dtype='i1')
        out['whitening_gain_unshifted']=gain
        # Average the frequencies of each group of eight native Fourier bins.
        out['frequency_offset_hz']=np.fft.fftshift(np.fft.fftfreq(nfft,1/info['fs'])).reshape(nout,-1).mean(axis=1)
        out['two_way_delay_us']=info['front']+np.arange(nr)*step/info['fs']*1e6
        out['slant_range_km']=out['two_way_delay_us'][:]*.149896229
        out.attrs.update(schema='syisr-plasmaline-ideal-code-match-v1',
            integration_start_utc_s=start,integration_stop_utc_s=stop,accepted_pulses=count,total_pulses=len(good),
            estimator='Sliding ideal-transmit conjugate * voltage-whitened echo; 8192 point forward FFT; average eight adjacent powers; average accepted IPPs',
            waveform_assumption='Existing 64x32 SYISR m32 phase table, 10us rectangular bauds, header little_code phase. Sequence and hardware timing not independently verified.',
            range_response='Conditional ideal waveform: 10us nominal baud resolution, 10us range sampling, frequency-dependent ambiguity and sidelobes; whitening and eight-bin smoothing alter response. Slant range is not altitude.',
            units='Dimensionless relative to diagnostic passband reference; unit noise is an approximate reference, not fitted noise subtraction; not absolute plasma-line power',
            rfi='Voltage Fourier whitening and static frequency notches before template multiplication; reject IPPs with >1% contaminated STFT cells, broadband burst, or any reference-range frequency median >8. No post-decoding outlier clipping.',
            limitations='No transmit sample. Conditional code identity/phase. Data-dependent RFI rejection can bias power and unbalance phase weights. SEM is pulse-scatter only, neglecting correlations and passband uncertainty. Frequency-specific response must be modeled before quantitative interpretation.',
            frequency_sign='Positive phase rotation of I+iQ yields positive frequency; hardware sideband uncalibrated',
            header_clock_offset_to_utc_s=-28800,reference_product=str(a.reference),
            code_sha256=hashlib.sha256(codestring.encode()).hexdigest(),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=pathlib.Path(__file__).parent,text=True).strip())
    tmp.replace(a.output);gpu.close();detector.close();print(json.dumps(dict(output=str(a.output),accepted=count,total=len(good),elapsed_s=time.time()-beg)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=pathlib.Path);p.add_argument('reference',type=pathlib.Path);p.add_argument('output',type=pathlib.Path)
    p.add_argument('--codes',type=pathlib.Path,default=pathlib.Path(__file__).parent.parent/'m32-reference.bc');p.add_argument('--device',type=int,default=0)
    run(p.parse_args())
