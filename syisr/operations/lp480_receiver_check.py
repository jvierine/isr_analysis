"""Receiver/background and spectral diagnostics only; never run plasma fits."""
import argparse
import datetime as dt
import os
from pathlib import Path
import subprocess as sp
import h5py
import numpy as np
import json

def load(p):
    with h5py.File(p) as f:return json.loads(bytes(f['json_utf8'][:]))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(p,'w') as f:f.create_dataset('json_utf8',data=np.frombuffer(json.dumps(x).encode(),dtype='u1'),compression='gzip')
def call(*a):sp.run([str(v) for v in a],check=True)
def receiver_stage(a):
    for date in ['20240520','20240531']:
        time=int(dt.datetime.strptime(date,'%Y%m%d').replace(hour=18,tzinfo=dt.timezone.utc).timestamp())
        m=load(a.month/'days'/date/'decoded/index.h5');m['jobs']=[j for j in m['jobs'] if j['start']==time]
        assert m['jobs'],'missing selected actual observation'
        save(a.out/'covariance'/date/'index.h5',m)

def main(a):
    rank=int(os.environ.get('SLURM_PROCID',0));size=int(os.environ.get('SLURM_NTASKS',1))
    if a.mode=='receiver-stage':
        receiver_stage(a)
    elif a.mode=='indices':
        for i in range(33):
            if i%size!=rank:continue
            name=f'{dt.datetime(2024,4,30)+dt.timedelta(days=i):%Y%m%d}'
            if not (a.raw/name).is_dir():continue
            cache=a.out/'indices'/f'{name}.h5';source=a.month/'indices'/f'{name}.h5'
            if source.exists():
                for s in [source,source.with_suffix('.signature.h5')]:
                    dest=cache.parent/s.name;dest.parent.mkdir(parents=True,exist_ok=True)
                    if not dest.exists():dest.symlink_to(s.resolve())
            call(a.binary,'cached-index',a.raw/name,cache)
    elif a.mode=='stage':
        jobs=[];template=None;quarantined=[];repairs=[]
        for day in range(1,32):
            date=dt.datetime(2024,5,day,tzinfo=dt.timezone.utc);start=int(date.timestamp())
            indices=[a.out/'indices'/f'{(date+dt.timedelta(days=k)):%Y%m%d}.h5' for k in [-1,0,1]]
            indices=[p for p in indices if p.exists()]
            path=a.out/'daily-index'/f'202405{day:02}.h5'
            call(a.binary,'daily-index',path,start,start+86400,*indices)
            m=load(path);template=m;jobs+=m['jobs'];quarantined+=m.get('quarantined_segments',[]);repairs+=m.get('clock_repairs',[])
        assert len({j['id'] for j in jobs})==len(jobs),'duplicate UTC measurement'
        template['jobs']=jobs;template['quarantined_segments']=quarantined;template['clock_repairs']=repairs
        save(a.out/'all-may-index.h5',template)
        sources=[str(p.parent) for p in sorted((a.month/'days').glob('*/input/index.h5'))]
        base=Path('/gpfs/pub/data_other/juha/syisr-storm-lp480-20240510-16-time-v3-20261120/input')
        m=load(base/'index.h5');m['jobs']=[j for j in m['jobs'] if '20240510'<=dt.datetime.fromtimestamp(j['start'],dt.timezone.utc).strftime('%Y%m%d')<='20240516']
        d=a.out/'stored-base';save(d/'index.h5',m)
        if not (d/'products').exists():(d/'products').symlink_to((base/'products').resolve())
        sources.append(str(d));save(a.out/'stored-sources.h5',sources)
        save(a.out/'coverage.h5',{'utc_days':list(range(1,32)),'indexed_beam_windows':len(jobs),'method':'Validated, exactly deduplicated UTC daily manifests; sparse raw spectra and existing screened spectra remain distinct. No plasma parameter fitting.'})
    elif a.mode=='stored':
        for root in load(a.out/'stored-sources.h5'):call(a.binary,'receiver-stored-tail',root,a.out/'stored-tail')
    elif a.mode=='reports':
        quiet=Path('/gpfs/pub/data_other/juha/syisr-covariance-pilot-20261121/input')
        for date in ['20240520','20240531']:
            call(a.binary,'receiver-report',a.out/'covariance'/date,a.month/'days'/date/'input',quiet,a.out/'corrected-acfs'/date)

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['receiver-stage','indices','stage','stored','reports'])
    p.add_argument('month',type=Path);p.add_argument('out',type=Path);p.add_argument('binary',type=Path)
    p.add_argument('--raw',type=Path,default=Path('/gpfs/pub/data/L0/2024'));main(p.parse_args())
