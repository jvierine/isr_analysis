"""Publish a validated fit; never replace seed.h5, current, or snapshots.

The server creates a new revision on first start, updating only untouched cells.
Scientific inputs/results retain provenance in HDF5, outside the interface.
"""
from pathlib import Path
import json
import sys
import h5py
import numpy as np

source = Path(sys.argv[1])
data = Path(sys.argv[2])
public = data/'public'
with h5py.File(data/'seed.h5') as h:
    old = h['state'][()]
    meta = json.loads(h['metadata_json'][()].tobytes())
assert old.shape == (2, 3, 351, 701)
maps = []
report = {}
for group in ['west', 'central', 'east']:
    with h5py.File(source/f'{group}.h5') as h:
        q = 1+h['contrast'][()]
        d = h['diagnostics'][()]
        assert d[2]<1e-5, (group, d)
        assert q.shape == old.shape[2:] and np.isfinite(q).all()
        assert q.min()>=0 and q.max()<=1
        maps.append(q.astype('f4'))
        report[group] = dict(selected_lambda=float(h.attrs['selected_lambda']),
                             heldout_beam=int(h.attrs['heldout_beam']),
                             candidate_lambdas=h['candidate_lambdas'][()].tolist(),
                             metrics=h['heldout_metrics'][()].tolist())
new = np.array([np.array(maps), np.zeros_like(maps)], dtype='f4')
meta['fit_version'] = 'finer-20261123'
with h5py.File(data/'refined_seed.h5', 'w') as h:
    h['metadata_json'] = np.frombuffer(json.dumps(meta).encode(), dtype='u1')
    h.create_dataset('state', data=new, compression='gzip', compression_opts=4)
    h.attrs['method'] = '15s x1.499km signed-linear display pulse-median averages; same quiet baseline, 120m/s eastward drift, finite9ray beam, 2km grid, asymmetricHuber0.2. Whole-beam validation selects smoothing, final fit uses all eligible beams except9.'
    h.attrs['validation_json'] = json.dumps(report)
    h.attrs['migration'] = 'Server updates only cells still equal to original seed with zero manual weight. All changed/painted cells and historical snapshots retained.'
(public/'previous_seed.bin').write_bytes(old[0].astype('<f4').tobytes())
(public/'seed.bin').write_bytes(new[0].astype('<f4').tobytes())
(public/'refinement.json').write_text(json.dumps(report))
print(json.dumps(report, indent=2))
