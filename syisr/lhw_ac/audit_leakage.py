"""Fit a code-bearing near-range signal, independent of ion-line width.

Assumes the enormous code-synchronous near-range signal is direct transmitter
leakage. The fitted delay includes any unrecorded transmit start offset.
"""
from pathlib import Path
import sys, hashlib
import numpy as np
import h5py
from scipy.io import loadmat

root = Path(sys.argv[1])
source = Path('/gpfs/pub/tool/data_process_cyh/process_new/ht_0.05_new.mat')
filt = loadmat(source)['ht'].ravel()
filt = filt/filt.sum()
codes = np.loadtxt('/gpfs/pub/tool/data_process_cyh/setting/AC_32.txt')
local = np.array([[1 if x=='+' else -1 for x in l.strip()] for l in
                  (root/'source/syisr/m32-reference.bc').read_text().splitlines() if l.strip()])
assert np.array_equal(codes,local)
tx = np.repeat(codes,20,axis=1)
wave = np.array([np.convolve(c,filt) for c in tx])
with h5py.File(root/'range-origin-audit/raw-trace.h5') as h:
    v = h['voltage_q_i_int32'][:,:,:,:30].astype(float)
    x = v[...,1]+1j*v[...,0]
    delay = h['nominal_receive_delay_us'][:30]
grid = np.arange(-150.,801.)
score = np.zeros((5,3,len(grid)))
for j,extra in enumerate(grid):
    t = delay-extra
    k = np.floor(t).astype(int)
    a = t-k
    valid = (k>=0)&(k+1<wave.shape[1])
    k = np.clip(k,0,wave.shape[1]-2)
    y = (wave[:,k]*(1-a)+wave[:,k+1]*a)*valid
    energy = np.sum(y*y,axis=1)
    # One nuisance complex gain per pulse; code signs and sample positions
    # still have to explain the complete recorded tail of each code member.
    dot = np.sum(x*y,axis=-1)
    explained = np.sum(abs(dot)**2/np.maximum(energy,1e-30),axis=-1)
    score[:,:,j] = explained/np.sum(abs(x)**2,axis=(-1,-2))
best = np.argmax(score,axis=-1)
extra = grid[best]
centroid = np.dot(np.arange(len(filt)),abs(filt)**2)/np.sum(abs(filt)**2)
with h5py.File(root/'range-origin-audit/leakage-transfer.h5','w') as h:
    h['extra_delay_grid_us'] = grid
    h['explained_voltage_energy_fraction'] = score
    h['best_extra_delay_us'] = extra
    h['filter_impulse_response_1us'] = filt
    h['apparent_signal_delay_us'] = extra+centroid
    h.attrs['filter_source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    h.attrs['filter_source'] = str(source)
    h.attrs['code_table_verified'] = 'supplier AC_32.txt equals analysis reference byte-for-sign'
    h.attrs['qualification'] = 'NOT an instrument calibration: apparent delay includes unknown scatterer propagation, transmit timing and receiver transfer; these components cannot be separated from this signal alone'
print('best extra delay us',extra,flush=True)
print('total filter energy-centroid delay us',extra+centroid,flush=True)
print('best explained fractions',np.take_along_axis(score,best[...,None],axis=-1).squeeze(),flush=True)
