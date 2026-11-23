"""Select only a close, measured previous-day boundary; HDF5 metadata only."""
import argparse,json
from pathlib import Path
import h5py,numpy as np

def load(p):
    with h5py.File(p) as f:return json.loads(bytes(f['json_utf8'][:]))
def boundary(first,times):
    if first is None or not times:return None
    last=max(times)
    return last if 0<first-last<=3600 and last//86400==first//86400-1 else None

def main(a):
    manifest=load(a.run/'decoded/index.h5')
    first=min((j['start'] for j in manifest['jobs']),default=None)
    times=[]
    if a.previous and (a.previous/'COMPLETE').exists() and not (a.previous/'NO_OBSERVATIONS').exists():
        times=[int(p.stem) for p in (a.previous/'final').glob('*.h5') if p.stem.isdigit()]
    chosen=boundary(first,times)
    if chosen is not None:
        fit=load(a.previous/'final'/f'{chosen}.h5')
        assert fit['beams'] and fit['status'] in ['converged','data_stable_gradient_pending'],'Unaccepted boundary fit'
    with h5py.File(a.run/'day-context.h5','w') as f:
        f.create_dataset('json_utf8',data=np.frombuffer(json.dumps(dict(first_measured_utc_s=first,previous_measured_utc_s=chosen,max_gap_s=3600,connected=chosen is not None,fresh_initialization=chosen is None,reason='Close corrected previous-day measurement only; missing day resets initialization and temporal context')).encode(),dtype='u1'))
    text='' if chosen is None else f"export SYISR_FIXED_TIME_CONTEXT='{a.previous}/final'\n"
    (a.run/'day-context.env').write_text(text)

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);p.add_argument('previous',nargs='?',type=Path);main(p.parse_args())
