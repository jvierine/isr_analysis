"""Restartable full archive, with separate receiver estimators and first-hour reuse."""
import argparse,pathlib,subprocess,sys,math,time
import h5py
p=argparse.ArgumentParser();p.add_argument('--device',type=int,required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=3);a=p.parse_args()
here=pathlib.Path(__file__).resolve().parent
raw=pathlib.Path('/gpfs/pub/data/L0/2026/plasmaline/20261125')
base=pathlib.Path('/gpfs/pub/data_other/juha/plasmaline-20261125')
root=base/'dual-channel-hour';root.mkdir(exist_ok=True)
from process import layout
with h5py.File(base/'catalogue.h5') as f:
    times=f['first_header_time_utc_s'][:];sizes=f['file_size_bytes'][:];names=f['filename'].asstr()[:]
info=layout(raw/names[0]);start=1795564800;stop=math.floor(max(times+sizes//info['stride']*.01)/60)*60
print('PLAN',start,stop,'minutes',(stop-start)//60,'device',a.device,flush=True)
done=[];failed=[];beg=time.time()
for i in range((stop-start)//60):
    if i%a.shards!=a.shard:continue
    epoch=1795564800+i*60
    for center in (436000000,444000000):
        target=root/str(epoch)/str(center);target.mkdir(parents=True,exist_ok=True)
        diag=target/'diagnostic.h5';matched=target/'matched-step2us.h5'
        commands=[]
        if not diag.exists():commands.append([sys.executable,'-u',str(here/'process.py'),str(raw),str(diag),'--start',str(epoch),'--catalogue',str(base/'catalogue.h5'),'--device',str(a.device),'--receiver-center-hz',str(center)])
        if not matched.exists():commands.append([sys.executable,'-u',str(here/'matched.py'),str(raw),str(diag),str(matched),'--device',str(a.device),'--range-step-us','2'])
        try:
            for command in commands:
                with (target/'processing.log').open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            with h5py.File(matched) as f:
                assert f.attrs['receiver_center_hz']==center and f.attrs['total_pulses']<=3001 and f.attrs['accepted_pulses']>0
            print('CHANNEL COMPLETE',epoch,center,flush=True)
        except (subprocess.CalledProcessError,AssertionError,OSError) as error:
            failed.append((epoch,center,str(error)));print('FAILED',epoch,center,str(error),flush=True)
    if all((root/str(epoch)/str(c)/'matched-step2us.h5').exists() for c in (436000000,444000000)):
        done.append(epoch);print('MINUTE COMPLETE',epoch,flush=True)
    status=root/f'full-status-{a.shard}.partial.h5'
    with h5py.File(status,'w') as f:
        f['completed_minute_start_utc_s']=done
        f['failed_minute_start_utc_s']=[t for t,_,_ in failed]
        f['failed_receiver_center_hz']=[c for _,c,_ in failed]
        f['failure_reason']=[str(e) for _,_,e in failed]
        f.attrs.update(status='running',intended_start_utc_s=start,intended_stop_utc_s=stop,elapsed_wall_s=time.time()-beg)
    status.replace(root/f'full-status-{a.shard}.h5')
with h5py.File(root/f'full-status-{a.shard}.h5','a') as f:f.attrs['status']='finished'
