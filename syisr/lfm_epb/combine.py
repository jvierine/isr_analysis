"""Reduce retained pulse products into UTC RTIs; preserve signed power."""
import argparse
import datetime as dt
from pathlib import Path
import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root'); p.add_argument('--start', required=True); p.add_argument('--stop', required=True)
    p.add_argument('--bin-seconds', type=float, default=60.)
    a = p.parse_args()
    start = dt.datetime.fromisoformat(a.start.replace('Z', '+00:00')).timestamp()
    stop = dt.datetime.fromisoformat(a.stop.replace('Z', '+00:00')).timestamp()
    edges = np.arange(start, stop + .001, a.bin_seconds)
    assert edges[-1] == stop
    fields = ['downchirp_power', 'upchirp_power', 'downchirp_signal']
    sums = {}; count = np.zeros(len(edges)-1, dtype=np.int64)
    noise = np.zeros_like(count, dtype=float)
    sources = []; seen = set(); pulse_max = []; pulse_t = []
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
            sig=h['downchirp_signal'][()][selected]
            # Keep pulse maxima as a transient-interference diagnostic, not a rejection mask.
            pulse_max.extend(np.max(sig,axis=1)); pulse_t.extend(t[selected])
            sources.append(h.attrs['source_file'])
    if not sums:
        raise ValueError('No products')
    out = Path(a.root, 'zenith_rti.h5')
    with h5py.File(str(out)+'.partial', 'w') as h:
        for k,v in attrs.items():
            if k != 'source_file': h.attrs[k]=v
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
        signal=h['downchirp_signal'][()]
        reference=np.nanmedian(signal,axis=0)
        h['temporal_reference_signal']=reference
        ratio=np.divide(signal,reference[None,:],out=np.full_like(signal,np.nan),where=(signal>0)&(reference[None,:]>0))
        h['relative_signal_db']=10*np.log10(ratio)
        h['range_squared_signal']=signal*ranges[None,:]**2
        h['pulse_time_unix_s']=pulse_t
        h['pulse_max_signal']=pulse_max
        h.attrs['complete']=True
    Path(str(out)+'.partial').replace(out)
    print('RTI',out,'files',len(sources),'pulses',int(count.sum()),'empty_bins',int((count==0).sum()),flush=True)


if __name__ == '__main__': main()
