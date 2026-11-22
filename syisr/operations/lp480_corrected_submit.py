"""Submit an explicitly limited, sequential64-core corrected-receiver retry."""
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
    a.campaign.mkdir(parents=True,exist_ok=True)
    with (a.campaign/'submission.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipt=a.campaign/'jobs.h5';assert not receipt.exists(),'Queue already submitted; inspect receipt before retrying'
        with h5py.File(a.receiver/'affine-calibration/reference/background.h5') as f:b=json.loads(bytes(f['json_utf8'][:]))
        assert b['pulse_covariance'] and len(b['variable_response'])==48
        for day in a.days:
            assert (a.receiver/'daily-index'/f'{day}.h5').exists()
            assert (a.original/'days'/day/'COMPLETE').exists()
        records=[];dependency=None;previous='none'
        save(receipt,dict(state='submission_started',days=a.days,jobs=records,cores=64))
        for day in a.days:
            root=a.campaign/'days'/day;(root/'logs').mkdir(parents=True,exist_ok=True)
            cmd=['/opt/slurm/slurm-22.05.11/bin/sbatch','--parsable','--partition=node','--nodelist=node01','--nodes=1','--ntasks=32','--cpus-per-task=2','--mem=120G','--time=3-00:00:00',f'--job-name=sy-cov-{day[6:]}',f'--output={root}/logs/slurm-%j.log']
            if dependency:cmd.append(f'--dependency=afterok:{dependency}')
            cmd += [str(a.code/'syisr/operations/lp480_corrected_daily_slurm.sh'),str(a.code),str(a.campaign),day,str(a.original),str(a.receiver),str(previous)]
            job=sp.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit(),job
            records.append(dict(day_utc=day,job_id=int(job),dependency_afterok=dependency,previous_corrected_day=str(previous),cores=64))
            save(receipt,dict(state='submitted',days=a.days,jobs=records,cores=64,method='Receiver pulse covariance plus independent affine background; old states initialize only; corrected adjacent-day boundary only; one day at a time.'))
            print(day,job,'afterok',dependency,flush=True);dependency=job;previous=root

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['code','campaign','original','receiver']:p.add_argument(name,type=Path)
    p.add_argument('days',nargs='+');main(p.parse_args())
