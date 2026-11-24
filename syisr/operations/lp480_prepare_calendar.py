"""Calendar staging/metadata. Native Rust validates frames, clocks and duplicates."""
import argparse,calendar,datetime as dt,json,os,subprocess as sp
from pathlib import Path
import h5py,numpy as np

def load(p):
    with h5py.File(p) as f:return json.loads(bytes(f['json_utf8'][:]))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.partial.h5')
    with h5py.File(q,'w') as f:f.create_dataset('json_utf8',data=np.frombuffer(json.dumps(v).encode(),dtype='u1'),compression='gzip')
    q.replace(p)
def main(a):
    year,month=map(int,a.month.split('-'));start=dt.datetime(year,month,1,tzinfo=dt.timezone.utc);nd=calendar.monthrange(year,month)[1]
    dates={(start+dt.timedelta(days=i)).strftime('%Y%m%d') for i in range(-1,nd+1)}
    survey=load(a.survey);groups=[v for v in survey['rows'] if v['acquisition_day'] in dates and v['candidate_files']]
    assert groups,'No configuration-matched acquisition files; do not fit another mode'
    if a.phase=='index':
        rank=int(os.environ.get('SLURM_PROCID','0'));size=int(os.environ.get('SLURM_NTASKS','1'))
        for i,g in enumerate(groups):
            if i%size!=rank:continue
            day=g['acquisition_day'];stage=a.root/'configuration-matched-raw'/day;stage.mkdir(parents=True,exist_ok=True)
            # Validate the entire acquisition, including corrupt or mixed-mode files.
            # The sample survey chooses candidate days, never discards bad bytes.
            source=Path(g['candidate_files'][0]).parent
            names=[p for p in source.iterdir() if p.is_file() and '.dat' in p.name]
            assert len(names)==g['total_files'],'Archive listing changed since survey'
            for p in names:
                link=stage/p.name
                if link.is_symlink():assert link.resolve()==p.resolve()
                else:link.symlink_to(p)
            out=a.root/'indices'/f'{day}.h5';out.parent.mkdir(exist_ok=True)
            sp.run([str(a.binary),'cached-index',str(stage),str(out)],check=True)
    else:
        indices=[a.root/'indices'/f"{g['acquisition_day']}.h5" for g in groups]
        assert all(p.exists() for p in indices),'Incomplete native validation'
        rows=[]
        for i in range(nd):
            day=start+dt.timedelta(days=i);name=day.strftime('%Y%m%d');t=int(day.timestamp());out=a.root/'daily-index'/f'{name}.h5';out.parent.mkdir(exist_ok=True)
            sp.run([str(a.binary),'daily-index',str(out),str(t),str(t+86400),*[str(p) for p in indices]],check=True)
            m=load(out);times=sorted({j['start'] for j in m['jobs']})
            rows.append(dict(day_utc=name,measured_windows=times,beam_windows=len(m['jobs']),unique_indexed_pulses=sum(p['pulses'] for j in m['jobs'] for p in j['parts'])))
        save(a.root/'validated-coverage.h5',dict(calendar_month=a.month,days=rows,header_clock_offset_s=28800,method='Native full frame/header validation and byte-exact duplicate removal; actual header UTC, not folder-date exposure. Corrupt exclusions retained in daily manifests. No invented observations.'))
if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('phase',choices=['index','daily']);p.add_argument('binary',type=Path);p.add_argument('survey',type=Path);p.add_argument('root',type=Path);p.add_argument('--month',required=True);main(p.parse_args())
