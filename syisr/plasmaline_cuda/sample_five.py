"""Five evenly spaced 60 s examples, native frequency bins and 2 us steps."""
import concurrent.futures
import math
import pathlib
import subprocess
import sys
import h5py
import numpy as np
from process import catalogue,layout

raw=pathlib.Path(sys.argv[1]);root=pathlib.Path(sys.argv[2]);out=root/'five-profiles-step2us';out.mkdir(exist_ok=True)
directory=pathlib.Path(__file__).resolve().parent;cat=root/'catalogue.h5';index=catalogue(raw,cat);info=layout(index[0][1])
first=math.ceil(index[0][0]/60)*60
last_time,last=index[-1];last=math.floor((last_time+(last.stat().st_size//info['stride'])*.01-60)/60)*60
targets=[int(round(t/60)*60) for t in np.linspace(first,last,5)]

def job(i):
    device=i%4;target=targets[i];errors=[]
    for delta in [0,60,-60,120,-120,180,-180,240,-240,300,-300]:
        t=target+delta
        if not first<=t<=last: continue
        folder=out/f'{i+1:02d}-{t}';folder.mkdir(exist_ok=True)
        diag=folder/'diagnostic.h5';existing=root/str(t)/'diagnostic.h5'
        if existing.exists(): diag=existing
        matched=folder/'matched-step2us.h5'
        commands=[]
        if not diag.exists(): commands.append([sys.executable,'-u',str(directory/'process.py'),str(raw),str(diag),'--start',str(t),'--catalogue',str(cat),'--device',str(device)])
        if not matched.exists(): commands.append([sys.executable,'-u',str(directory/'matched.py'),str(raw),str(diag),str(matched),'--device',str(device),'--range-step-us','2'])
        commands.append([sys.executable,str(directory/'check_product.py'),str(diag),'--matched',str(matched)])
        try:
            for command in commands:
                with (folder/'processing.log').open('a') as log: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            with h5py.File(matched) as f:
                assert f.attrs['range_step_us']==2
                assert np.allclose(np.diff(f['two_way_delay_us'][:]),2)
            print(f'COMPLETE profile {i+1}: requested {target}, used {t}, {matched}',flush=True)
            return i,target,t,str(diag),str(matched)
        except (subprocess.CalledProcessError,AssertionError) as e:
            errors.append(str(e));print(f'RETRY profile {i+1}: {t}: {e}',flush=True)
    raise RuntimeError(f'No valid minute near {target}: {errors}')

# Four GPU workers. The fifth starts when the first worker finishes, so it
# never shares device 0 with profile 1.
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(job,i) for i in range(4)]
    for future in concurrent.futures.as_completed(futures): results.append(future.result())
results.append(job(4));results.sort()
with h5py.File(out/'selection.h5','w') as f:
    f['requested_start_utc_s']=[r[1] for r in results];f['actual_start_utc_s']=[r[2] for r in results]
    for k,j in [('diagnostic_path',3),('matched_path',4)]:f[k]=np.asarray([r[j] for r in results],dtype=h5py.string_dtype())
    f.attrs.update(integration_s=60,range_step_us=2,selection='Five evenly spaced UTC minute starts across coverage; nearest valid minute within five minutes if strict reader fails')
print('ALL FIVE COMPLETE',flush=True)
