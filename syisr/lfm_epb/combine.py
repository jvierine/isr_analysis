"""Reduce retained pulse products into UTC RTIs; preserve signed power."""
import argparse
import datetime as dt
import subprocess
from pathlib import Path
import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root'); p.add_argument('--start', required=True); p.add_argument('--stop', required=True)
    p.add_argument('--bin-seconds', type=float, default=15.)
    a = p.parse_args()
    start = dt.datetime.fromisoformat(a.start.replace('Z', '+00:00')).timestamp()
    stop = dt.datetime.fromisoformat(a.stop.replace('Z', '+00:00')).timestamp()
    edges = np.arange(start, stop + .001, a.bin_seconds)
    assert edges[-1] == stop
    fields = ['downchirp_power', 'upchirp_power', 'downchirp_signal']
    sums = {}; count = np.zeros(len(edges)-1, dtype=np.int64)
    noise = np.zeros_like(count, dtype=float)
    sources = []; seen = set(); pulse_max = []; pulse_t = []
    pulse_bins=[[] for _ in count]; noise_bins=[[] for _ in count]
    for path in sorted(Path(a.root, 'parts').glob('*.h5')):
        with h5py.File(path) as h:
            t = h['times_unix_s'][()]
            selected = (t >= start) & (t < stop)
            if not selected.any():
                continue
            ns = np.rint(t[selected]*1e6).astype(np.int64)
            if any(x in seen for x in ns) or len(np.unique(ns)) != len(ns):
                raise ValueError('Duplicate pulse timestamp: ' + str(path))
            seen.update(ns)
            idx = np.floor((t[selected]-start)/a.bin_seconds).astype(int)
            ranges = h['range_km'][()]
            if not sums:
                coords = {k: h[k][()] for k in ['range_km', 'range_edges_km', 'altitude_km', 'altitude_edges_km']}
                attrs = dict(h.attrs)
                sums = {k: np.zeros((len(count), len(ranges))) for k in fields}
            for k,v in coords.items():
                np.testing.assert_allclose(v, h[k][()], rtol=0, atol=1e-8)
            np.add.at(count, idx, 1)
            np.add.at(noise, idx, h['noise_power'][()][selected])
            for k in fields:
                np.add.at(sums[k], idx, h[k][()][selected])
            pw=h['downchirp_power'][()][selected]
            nn=h['noise_power'][()][selected]
            sig=pw-nn[:,None]
            for j in np.unique(idx):
                pulse_bins[j].append(pw[idx==j]); noise_bins[j].append(nn[idx==j])
            # Keep pulse maxima as a transient-interference diagnostic, not a rejection mask.
            pulse_max.extend(np.max(sig/nn[:,None],axis=1)); pulse_t.extend(t[selected])
            sources.append(h.attrs['source_file'])
    if not sums:
        raise ValueError('No products')
    out = Path(a.root, 'zenith_rti.h5')
    with h5py.File(str(out)+'.partial', 'w') as h:
        for k,v in attrs.items():
            if k != 'source_file': h.attrs[k]=v
        h.attrs['reduction_git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parent,text=True).strip()
        h.attrs['time_bin_s']=a.bin_seconds
        h.attrs['relative_reference']='Median signed signal at each range over the selected interval; positive signal only for log display'
        for k,v in coords.items(): h[k]=v
        h['time_edges_unix_s']=edges
        h['time_unix_s']=(edges[:-1]+edges[1:])/2
        h['pulse_count']=count
        h['source_files']=np.array(sources,dtype=h5py.string_dtype())
        h['noise_power']=np.divide(noise,count,out=np.full_like(noise,np.nan),where=count>0)
        for k,v in sums.items():
            h[k]=np.divide(v,count[:,None],out=np.full_like(v,np.nan),where=count[:,None]>0)
        h['matched_power']=h['downchirp_power'][()]
        h['matched_signal']=h['downchirp_signal'][()]
        medp=np.full_like(h['matched_power'][()],np.nan)
        meds=medp.copy()
        for j,parts in enumerate(pulse_bins):
            if parts:
                pp=np.concatenate(parts); nn=np.concatenate(noise_bins[j])
                medp[j]=np.median(pp,axis=0)
                meds[j]=np.median(pp-nn[:,None],axis=0)
        h['median_power']=medp
        h['median_signal']=meds
        signal=meds
        h.attrs['matched_chirp']='downchirp; decreasing instantaneous frequency'
        h.attrs['rti_processing_version']='lfm-epb-v4'
        h.attrs['median_method']='median over individual range-cell pulse powers within each UTC bin; signed noise subtraction precedes median; mean products retained; not an unbiased density estimator'
        reference=np.nanmedian(signal,axis=0)
        h['temporal_reference_signal']=reference
        ratio=np.divide(signal,reference[None,:],out=np.full_like(signal,np.nan),where=(signal>0)&(reference[None,:]>0))
        h['relative_signal_db']=10*np.log10(ratio)
        display=10*np.log10(ratio)
        valid=(signal>0.1*h['noise_power'][()][:,None]) & (reference[None,:]>0.1*np.nanmedian(h['noise_power'][()]))
        display[~valid]=np.nan
        h['relative_signal_display_db']=display
        h.attrs['relative_display_mask']='signal and temporal reference SNR must exceed 0.1; scientific arrays retain all signed values'
        h['range_squared_signal']=signal*ranges[None,:]**2
        h['pulse_time_unix_s']=pulse_t
        h['pulse_max_snr']=pulse_max
        h.attrs['complete']=True
    Path(str(out)+'.partial').replace(out)
    print('RTI',out,'files',len(sources),'pulses',int(count.sum()),'empty_bins',int((count==0).sum()),flush=True)


if __name__ == '__main__': main()
