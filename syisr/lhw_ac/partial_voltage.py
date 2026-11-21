"""Exclude an early voltage interval and decode only surviving code/lag pairs.

The veto is in recorded sample coordinates, never a time-origin calibration.
Every surviving branch is normalized by its actual retained baud count.
"""
from pathlib import Path
import sys, os, json
import h5py
import numpy as np
from reduction import partial_pairs

original, output = map(Path,sys.argv[1:3])
cut = int(sys.argv[3])
assert cut%2==0 and 2<=cut<=256
rank = int(os.environ.get('SLURM_PROCID','0'))
ranks = int(os.environ.get('SLURM_NTASKS','1'))
with h5py.File(original/'index.h5') as h:
    paths = json.loads(bytes(h['files_json_utf8'][:]))
    times = h['cycle_time_utc_s'][:]
    beam = h['beam_index'][:]
    offsets = h['frame_offset'][:]
    ids = h['file_index'][:]
codes = np.array([[1 if c=='+' else -1 for c in line.strip()] for line in
                  (original/'source/syisr/m32-reference.bc').read_text().splitlines() if line.strip()])
files = sorted((original/'blocks').glob('[0-9][0-9][0-9][0-9].h5'))
maps = {}
output.mkdir(parents=True,exist_ok=True)
for file in files[rank::ranks]:
    target = output/file.name
    if target.exists(): continue
    tmp = target.with_suffix('.partial.h5')
    with h5py.File(file) as src,h5py.File(tmp,'w') as out:
        for name in src:
            if not name.startswith('beam_') or name=='beam_az_el_deg': src.copy(name,out)
        t0,t1 = src['interval_utc_s'][:]
        for b in range(5):
            g = src[f'beam_{b}']
            dst = out.create_group(f'beam_{b}')
            for key in g:
                if key not in ['acf_adu2','unscreened_acf_adu2','received_support_power_adu2']: g.copy(key,dst)
            mask = g['rejected_cycle_gate'][:].astype(bool)
            keep = ~mask
            counts = keep.sum(axis=0)
            selected = np.flatnonzero((beam==b)&(times[:,0]>=t0)&(times[:,1]<t1))
            assert len(selected)==len(mask)
            nc,ng = mask.shape
            x = np.empty((nc,64,64),complex)
            for j,ci in enumerate(selected):
                for c in range(64):
                    fi = int(ids[ci,c])
                    if fi not in maps: maps[fi] = np.memmap(paths[fi],mode='r',dtype='u1')
                    start = int(offsets[ci,c])+1024+cut*8
                    v = np.frombuffer(maps[fi][start:start+64*8],'<i4').reshape(64,2)
                    x[j,c] = v[:,1].astype(float)+1j*v[:,0]
            full = g['acf_adu2'][:]
            uns = g['unscreened_acf_adu2'][:]
            a = full[...,0]+1j*full[...,1]
            u = uns[...,0]+1j*uns[...,1]
            power = g['received_support_power_adu2'][:]
            partial = min(cut//2,ng)
            a[:partial] = np.nan
            u[:partial] = np.nan
            power[:partial] = np.nan
            for lag in range(1,63):
                products = x[:,:,lag:]*x[:,:,:-lag].conj()
                bank = products.sum(axis=0)
                for gate in range(partial):
                    branches = [(baud,k-cut,l) for baud,k,l in partial_pairs(gate,lag,cut)]
                    if not branches: continue
                    total = 0j
                    accepted = 0j
                    for baud,k,l in branches:
                        signs = codes[:,baud]*codes[:,baud+l]
                        assert np.all(signs.sum(axis=0)==0)
                        scale = (2 if lag%2 else 1)/(len(branches)*64*len(baud))
                        contribution = np.sum(bank[:,k]*signs)*scale
                        rejected = np.sum(products[mask[:,gate]][:,:,k]*signs)*scale
                        total += contribution
                        accepted += contribution-rejected
                    u[gate,lag] = total/nc
                    if counts[gate]>0: a[gate,lag] = accepted/counts[gate]
            for gate in range(partial):
                if np.isfinite(a[gate,1:3]).all(): a[gate,0] = (4*a[gate,1].real-a[gate,2].real)/3
                if np.isfinite(u[gate,1:3]).all(): u[gate,0] = (4*u[gate,1].real-u[gate,2].real)/3
                used = min(64,max(0,gate*2+64-cut))
                if used and counts[gate]>0: power[gate] = np.mean(abs(x[keep[:,gate],:,:used])**2)
            for key,v in [('acf_adu2',a),('unscreened_acf_adu2',u)]:
                dst.create_dataset(key,data=np.stack([v.real,v.imag],axis=-1),compression='gzip')
            dst['received_support_power_adu2'] = power
            good = np.isfinite(a)
            length = np.where(good.all(axis=1),63,np.argmin(good,axis=1))
            dst['maximum_used_lag_s'] = np.maximum(0,length-1)*1e-5
        out.attrs['voltage_veto_start_sample'] = 0
        out.attrs['voltage_veto_end_sample'] = cut
        out.attrs['partial_pulse_method'] = 'Only raw pair indices outside early-voltage veto; branch weights use retained baud counts; unsupported lags NaN; no clock shift inferred from clutter'
        out.attrs['source_block'] = str(file)
    tmp.rename(target)
    print('PARTIAL',rank,file.name,flush=True)
