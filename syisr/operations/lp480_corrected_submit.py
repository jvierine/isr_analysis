"""Submit an explicitly limited, sequential explicitly allocated corrected-receiver retry."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess as sp
import h5py
import numpy as np

def save(path,value):
    tmp=path.with_suffix('.partial.h5')
    with h5py.File(tmp,'w') as f:f.create_dataset('json_utf8',data=np.frombuffer(json.dumps(value).encode(),dtype='u1'),compression='gzip')
    os.replace(tmp,path)

def main(a):
    assert a.initial_dependency is None or all(v.isdigit() for v in a.initial_dependency.split(':')),'Invalid dependency list'
    a.campaign.mkdir(parents=True,exist_ok=True)
    with (a.campaign/'submission.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=a.campaign/'jobs.h5';assert not receipt.exists(),'Queue already submitted; inspect receipt before retrying'
        with h5py.File(a.receiver/'affine-calibration/reference/background.h5') as f:b=json.loads(bytes(f['json_utf8'][:]))
        assert b['pulse_covariance'] and len(b['variable_response'])==48
        for day in a.days:
            assert (a.receiver/'daily-index'/f'{day}.h5').exists()
            assert (a.original/'days'/day/'COMPLETE').exists()
        records=[];dependency=a.initial_dependency;previous=a.previous_run if a.previous_run is not None else 'none'
        save(receipt,dict(state='submission_started',days=a.days,jobs=records,cores=a.cores))
        for day in a.days:
            root=a.campaign/'days'/day;(root/'logs').mkdir(parents=True,exist_ok=True)
            cmd=['/opt/slurm/slurm-22.05.11/bin/sbatch','--parsable','--partition=node',f'--nodelist={"node01,node02" if a.cores==128 else "node01"}',f'--nodes={2 if a.cores==128 else 1}',f'--ntasks={a.cores//2}','--cpus-per-task=2','--mem=120G','--time=3-00:00:00',f'--job-name=sy-cov-{day[6:]}',f'--output={root}/logs/slurm-%j.log']
            if a.cores==128:cmd.append('--ntasks-per-node=32')
            if dependency:cmd.append(f'--dependency=afterok:{dependency}')
            cmd += [str(a.code/'syisr/operations/lp480_corrected_daily_slurm.sh'),str(a.code),str(a.campaign),day,str(a.original),str(a.receiver),str(previous)]
            job=sp.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit(),job
            records.append(dict(day_utc=day,job_id=int(job),dependency_afterok=dependency,previous_corrected_day=str(previous),cores=a.cores))
            save(receipt,dict(state='submitted',days=a.days,jobs=records,cores=a.cores,method='Receiver pulse covariance plus independent affine background; old states initialize only; corrected adjacent-day boundary only; one day at a time.'))
            print(day,job,'afterok',dependency,flush=True);dependency=job;previous=root

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['code','campaign','original','receiver']:p.add_argument(name,type=Path)
    p.add_argument('--initial-dependency',type=str);p.add_argument('--previous-run',type=Path);p.add_argument('--cores',type=int,choices=[64,128],default=64);p.add_argument('days',nargs='+');main(p.parse_args())
