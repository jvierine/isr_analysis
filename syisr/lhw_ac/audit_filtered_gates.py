"""Test nonlocal decoded leakage from finite receiver filtering.

The FIR is centred only to separate pulse-shape effects from the unresolved
absolute time origin. Its measured/code-clutter delay is NOT a calibration.
"""
from pathlib import Path
import sys
import h5py
import numpy as np
from scipy.io import loadmat
from reduction import triangular_spectrum

root = Path(sys.argv[1])
codes = np.array([[1 if s=='+' else -1 for s in line.strip()] for line in
                  (root/'source/syisr/m32-reference.bc').read_text().splitlines() if line.strip()])
filt = loadmat('/gpfs/pub/tool/data_process_cyh/process_new/ht_0.05_new.mat')['ht'].ravel()
filt /= filt.sum()
ng = 267
n = 1267
source = 110  # Native gate55, approximately the lowest perpendicular sector.
tx = np.repeat(codes,20,axis=1)
filtered = np.array([np.convolve(row,filt) for row in tx])
origin = np.dot(np.arange(len(filt)),filt**2)/(filt**2).sum()
receivers = []
for wave,centre in [(tx,0.),(filtered,origin)]:
    t = (np.arange(n)-source)*10+centre
    receivers.append(np.array([np.interp(t,np.arange(wave.shape[1]),row,left=0,right=0) for row in wave]))
acf = np.zeros((2,ng,63),complex)
for kind,x in enumerate(receivers):
    for lag in range(1,63):
        branches = [(a,l) for a,l in [(0,lag//2),(1,(lag+1)//2)] if l>0]
        for a,l in branches:
            baud = np.arange(32-l)
            k = 2*np.arange(ng)[:,None]+2*baud+a
            products = x[:,k]*x[:,k+lag]
            phases = codes[:,baud]*codes[:,baud+l]
            factor = (2 if lag%2 else 1)/(len(branches)*64*(32-l))
            acf[kind,:,lag] += np.einsum('cgb,cb->g',products,phases)*factor
    acf[kind,:,0] = (4*acf[kind,:,1].real-acf[kind,:,2].real)/3
psd = triangular_spectrum(acf)
out = root/'range-origin-audit/filtered-gates.h5'
with h5py.File(out,'w') as h:
    h['acf_ideal_filtered'] = np.stack([acf.real,acf.imag],axis=-1)
    h['spectrum_ideal_filtered'] = psd
    h['native_gate'] = np.arange(ng)
    h.attrs['injected_native_gate'] = source/2
    h.attrs['filter_origin_removed_us'] = origin
    h.attrs['scope'] = 'Point scatterer forward waveforms through fractional-lag decode; tests range/lag ghosts from ignored finite filtering, not a timing calibration'
for kind,label in enumerate(['ideal','finite FIR']):
    peak = np.max(abs(psd[kind,:,256]))
    low = abs(psd[kind,:40,256])
    print(label,'central PSD low gates/max',low.max()/peak,'worst gate',np.argmax(low),'gate29/max',abs(psd[kind,29,256])/peak,flush=True)
print('PASS filtered-point response saved',out,flush=True)
