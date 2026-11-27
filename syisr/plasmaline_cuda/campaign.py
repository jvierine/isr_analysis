"""Restartable serial 60 s research products on one GPU; malformed spans fail closed."""
import argparse
import math
import pathlib
import subprocess
import sys
import h5py
import numpy as np
from process import catalogue,layout

p=argparse.ArgumentParser();p.add_argument('raw',type=pathlib.Path);p.add_argument('output',type=pathlib.Path)
p.add_argument('--device',type=int,default=0);p.add_argument('--limit',type=int);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True);cat=a.output/'catalogue.h5';index=catalogue(a.raw,cat)
info=layout(index[0][1]);start=math.ceil(index[0][0]/60)*60
lasttime,lastpath=index[-1];stop=math.floor((lasttime+(lastpath.stat().st_size//info['stride'])*.01)/60)*60
directory=pathlib.Path(__file__).resolve().parent;failures=[];done=[]
for k,t in enumerate(range(start,stop,60)):
    if a.limit is not None and k>=a.limit: break
    target=a.output/str(t);target.mkdir(exist_ok=True)
    diagnostic=target/'diagnostic.h5';matched=target/'matched.h5'
    commands=[]
    if not diagnostic.exists(): commands.append([sys.executable,'-u',str(directory/'process.py'),str(a.raw),str(diagnostic),'--start',str(t),'--catalogue',str(cat),'--device',str(a.device)])
    if not matched.exists(): commands.append([sys.executable,'-u',str(directory/'matched.py'),str(a.raw),str(diagnostic),str(matched),'--device',str(a.device)])
    try:
        for command in commands:
            with (target/'processing.log').open('a') as log: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        with h5py.File(matched) as out:
            if out.attrs['accepted_pulses']<2: raise ValueError('no valid matched result')
        done.append(t);print(f'COMPLETE {t} {matched}',flush=True)
    except (subprocess.CalledProcessError,ValueError,OSError) as e:
        failures.append((t,str(e)));print(f'FAILED {t}: {e}; inspect {target / "processing.log"}',flush=True)
        # A failed minute has no published matched product. Proceed to independent
        # minutes; no framing interpolation, repaired clocks or invented spectra.
    tmp=a.output/'status.partial.h5'
    with h5py.File(tmp,'w') as out:
        out['completed_minute_start_utc_s']=np.asarray(done,dtype='i8')
        out['failed_minute_start_utc_s']=np.asarray([t for t,_ in failures],dtype='i8')
        out['failure_reason']=np.asarray([s for _,s in failures],dtype=h5py.string_dtype())
        out.attrs.update(status='running',intended_start_utc_s=start,intended_stop_utc_s=stop,preliminary=True)
    tmp.replace(a.output/'status.h5')
with h5py.File(a.output/'status.h5','a') as out: out.attrs['status']='finished'
