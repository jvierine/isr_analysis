"""Transport unmodified fit arrays for native Rust short-lag diagnostics.
Use base conda. No normalization, interpolation, statistics or fitting here.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np

def load(path):
    with h5py.File(path) as f:
        return json.loads(bytes(f['json_utf8'][:]))

def main(a):
    rows=[]
    for day in a.days:
        root=a.campaign/'days'/day
        for p in sorted((root/'snapshot').glob('*.h5')):
            w=load(root/'final'/p.name)
            for b in w['beams']:
                if abs(b['azimuth_deg']-a.az)>0.1 or abs(b['elevation_deg']-a.el)>0.1:
                    continue
                d=load(root/'input/products'/f"{b['id']}.h5")
                indices=[i for i,l in enumerate(b['measurement_lag']) if 1<=l<=10]
                row=dict(time_utc_s=w['start_utc_s'],status=w['status'],azimuth_deg=b['azimuth_deg'],elevation_deg=b['elevation_deg'],altitude_km=[g['altitude_km'] for g in d['gates']])
                for dest,src in [('gate','measurement_gate'),('lag','measurement_lag'),('observed','observed'),('predicted','predicted'),('sigma','assigned_sigma')]:
                    row[dest]=[b[src][i] for i in indices]
                rows.append(row)
    assert rows,'No matching measured beam fits'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=a.output.with_suffix('.partial.h5')
    payload=dict(rows=rows,provenance=f'Unmodified full-fit observed/predicted/assigned-sigma/gate subsets from {a.campaign}; first ten measured LP480 lags. Normalization is performed in native Rust.')
    with h5py.File(temporary,'w') as f:
        f.create_dataset('json_utf8',data=np.frombuffer(json.dumps(payload,separators=(',',':')).encode(),dtype='u1'),compression='gzip')
    temporary.replace(a.output)

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('campaign',type=Path);p.add_argument('output',type=Path);p.add_argument('days',nargs='+')
    p.add_argument('--az',type=float,default=357.99);p.add_argument('--el',type=float,default=90.)
    main(p.parse_args())
