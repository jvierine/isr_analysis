#!/usr/bin/env python3
"""Daily Slurm orchestration and HDF5 metadata only; science computation is Rust.
Run with base conda. Raw clocks, gating and deduplication are handled natively.
"""
import argparse, datetime as dt, fcntl, json, os, subprocess as sp
from pathlib import Path
import h5py
import numpy as np

def load(path):
    with h5py.File(path) as f: return json.loads(bytes(f['json_utf8'][:]))

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.partial.h5')
    with h5py.File(tmp,'w') as f:
        f.create_dataset('json_utf8',data=np.frombuffer(json.dumps(value).encode(),dtype='u1'),compression='gzip')
    tmp.replace(path)

def call(*args): sp.run([str(x) for x in args],check=True)

def stage(a):
    day=dt.datetime.strptime(a.day,'%Y%m%d').replace(tzinfo=dt.timezone.utc)
    root=a.month/'days'/a.day
    indices=[]
    for offset in [-1,0,1]:
        name=(day+dt.timedelta(days=offset)).strftime('%Y%m%d')
        raw=a.raw_root/name
        if not raw.is_dir(): continue
        cache=a.month/'indices'/f'{name}.h5'
        call(a.binary,'cached-index',raw,cache)
        indices.append(cache)
    call(a.binary,'daily-index',root/'decoded/index.h5',int(day.timestamp()),int(day.timestamp())+86400,*indices)
    m=load(root/'decoded/index.h5')
    save(root/'day.h5',dict(day_utc=a.day,measured_windows=sorted({j['start'] for j in m['jobs']}),
         previous_run=str(a.previous),baseline=str(a.base),cores=64,
         method='Daily conditional temporal fit; adjacent measured boundary held fixed; nearest actual fitted profile used only for initialization.'))
    if not m['jobs']: (root/'NO_OBSERVATIONS').touch()

def catalog(a):
    target=a.month/'profiles'; target.mkdir(exist_ok=True)
    base_snapshot=a.month/'base-snapshot'
    if not (a.month/'BASE_SNAPSHOT_COMPLETE').exists():
        call(a.binary,'snapshot',a.base/'final',base_snapshot)
        (a.month/'BASE_SNAPSHOT_COMPLETE').touch()
    completed={f'202405{d:02}':'existing_campaign' for d in range(10,17)}
    sources=[(base_snapshot,set(completed))]
    for root in sorted((a.month/'days').glob('*')):
        if not (root/'COMPLETE').exists(): continue
        completed[root.name]='no_observations' if (root/'NO_OBSERVATIONS').exists() else 'daily_fit_complete'
        if not (root/'NO_OBSERVATIONS').exists(): sources.append((root/'snapshot',{root.name}))
    for directory,allowed in sources:
        for f in directory.glob('*.h5'):
            try: time=int(f.stem)
            except ValueError: continue
            date=dt.datetime.fromtimestamp(time,dt.timezone.utc).strftime('%Y%m%d')
            if date not in allowed: continue
            dest=target/f.name; temp=target/(f.name+'.link')
            if temp.is_symlink(): temp.unlink()
            temp.symlink_to(f.resolve()); os.replace(temp,dest)
    measured=sorted(int(f.stem) for f in target.glob('*.h5') if f.stem.isdigit())
    save(a.month/'coverage.h5',dict(calendar_month='2024-05',completed_days=completed,measured_windows=measured,
        expected_half_hour_slots=31*48,method='Measured fits only; pending days and genuine missing observations are distinct. Daily fits are conditioned on fixed adjacent-day context; global monthly temporal convergence has not been established. Parameter uncertainties remain preliminary.'))

def submit(a):
    a.month.mkdir(parents=True,exist_ok=True)
    with (a.month/'submission.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (a.month/'jobs.h5').exists(), 'Month already submitted; inspect existing jobs before resuming'
        assert (a.base/'COMPLETE').exists(), 'Baseline is not complete'
        assert len(list((a.month/'references').glob('*.h5')))==1488, 'Full month of offline references required'
        order=list(range(17,32))+list(range(9,0,-1))
        records=[]; dependency=None
        save(a.month/'jobs.h5',dict(calendar_month='2024-05',daily_jobs=[],state='submission_started'))
        for day in order:
            name=f'202405{day:02}'; root=a.month/'days'/name; (root/'logs').mkdir(parents=True,exist_ok=True)
            previous=a.base if day in [17,9] else a.month/'days'/f'202405{day-1 if day>=18 else day+1:02}'
            cmd=[str(a.sbatch),'--parsable','--partition=node','--nodes=1','--nodelist=node01',
                 '--ntasks=32','--cpus-per-task=2','--mem=120G','--time=3-00:00:00',
                 f'--job-name=sy24-{day:02}',f'--output={root}/logs/slurm-%j.log']
            if dependency: cmd += [f'--dependency=afterok:{dependency}']
            cmd += [str(a.code/'syisr/operations/lp480_daily_slurm.sh'),str(a.code),str(a.month),name,str(a.base),str(previous),str(a.raw_root)]
            job=sp.check_output(cmd,text=True).strip().split(';')[0]
            assert job.isdigit(),f'Invalid Slurm receipt: {job}'
            records.append(dict(day_utc=name,job_id=int(job),dependency_afterok=dependency,previous_run=str(previous),cores=64))
            save(a.month/'jobs.h5',dict(calendar_month='2024-05',daily_jobs=records,existing_days=list(range(10,17)),
                 source=str(a.code),method='Sequential daily jobs,32ranks x2threads; afterok prevents concurrent daily analysis and propagation after failure.'))
            print('SUBMITTED',name,job,'afterok',dependency,flush=True)
            dependency=job

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['stage','catalog','submit']);p.add_argument('month',type=Path)
    p.add_argument('--day');p.add_argument('--base',type=Path,required=True);p.add_argument('--previous',type=Path)
    p.add_argument('--binary',type=Path);p.add_argument('--code',type=Path)
    p.add_argument('--raw-root',type=Path,default=Path('/gpfs/pub/data/L0/2024'))
    p.add_argument('--sbatch',type=Path,default=Path('/opt/slurm/slurm-22.05.11/bin/sbatch'))
    a=p.parse_args();globals()[a.mode](a)
if __name__=='__main__': main()
