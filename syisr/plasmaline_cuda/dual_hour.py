"""Separate alternating receiver tunings before every estimator; first UTC hour."""
import argparse,pathlib,subprocess,sys
import h5py
p=argparse.ArgumentParser();p.add_argument('--device',type=int,required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=3);a=p.parse_args()
here=pathlib.Path(__file__).resolve().parent
raw=pathlib.Path('/gpfs/pub/data/L0/2026/plasmaline/20261125')
base=pathlib.Path('/gpfs/pub/data_other/juha/plasmaline-20261125')
root=base/'dual-channel-hour';root.mkdir(exist_ok=True)
for i in range(60):
    if i%a.shards!=a.shard:continue
    epoch=1795564800+i*60
    for center in (436000000,444000000):
        target=root/str(epoch)/str(center);target.mkdir(parents=True,exist_ok=True)
        diag=target/'diagnostic.h5';matched=target/'matched-step2us.h5'
        commands=[]
        if not diag.exists():commands.append([sys.executable,'-u',str(here/'process.py'),str(raw),str(diag),'--start',str(epoch),'--catalogue',str(base/'catalogue.h5'),'--device',str(a.device),'--receiver-center-hz',str(center)])
        if not matched.exists():commands.append([sys.executable,'-u',str(here/'matched.py'),str(raw),str(diag),str(matched),'--device',str(a.device),'--range-step-us','2'])
        for command in commands:
            with (target/'processing.log').open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        with h5py.File(matched) as f:
            assert f.attrs['receiver_center_hz']==center and f.attrs['total_pulses']<=3001 and f.attrs['accepted_pulses']>0
        print('CHANNEL COMPLETE',epoch,center,flush=True)
    print('MINUTE COMPLETE',epoch,flush=True)
