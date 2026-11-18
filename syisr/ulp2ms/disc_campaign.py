#!/usr/bin/env python3
"""Inventory supported 2-ms recordings and MPI-process disjoint windows. Base conda.
Configuration/restart state uses JSON; all scientific arrays use HDF5.
"""
import argparse, datetime as dt, json, os, struct, subprocess, shutil, mmap, concurrent.futures, hashlib
from pathlib import Path
import h5py
import numpy as np

def header(path, offset=0):
    with path.open('rb') as f:
        f.seek(offset); b=f.read(1024)
    if len(b)!=1024 or struct.unpack_from('<I',b)[0]!=0xaaaa5555 or struct.unpack_from('<I',b,1020)[0]!=0xaa5555aa:
        raise ValueError(f'Invalid header {path} at {offset}')
    u=lambda o:struct.unpack_from('<H',b,o)[0]
    n=struct.unpack_from('<I',b,768)[0]
    when=dt.datetime(2000+b[13],b[12],b[15],b[14],b[17],b[16],tzinfo=dt.timezone.utc).timestamp()+u(18)*25e-6-28800
    return dict(time=when,samples=n,frame_bytes=1024+8*(n+int(n==3782)),ipp_s=u(980)*1e-6,gate_front_us=u(766),frequency_hz=(430+b[32]*.2)*1e6,waveform=b[36]>>4,sample_rate_code=b[36]&15,pulse_width_us=u(988),elevation_raw=u(858),tx_elevation_raw=u(870),bandwidth=b[37]>>4)

def plan(args):
    args.root.mkdir(parents=True,exist_ok=True)
    if (args.root/'plan.json').exists():raise ValueError('Plan exists; use immutable saved plan')
    sources=[]; excluded=[]
    paths=[p for p in sorted(args.disc.rglob('*')) if p.is_file() and '.dat1-1-' in p.name and not p.name.startswith('._')]
    index_root=args.root/'raw-index';index_root.mkdir(exist_ok=True)
    def scan(p):
        h=header(p)
        if h['pulse_width_us']!=2000:return [],dict(path=str(p),reason='not 2ms pulse',header=h)
        key=(h['samples'],round(h['ipp_s'],6),h['gate_front_us'],h['frequency_hz'])
        if key not in [(1194,.05,2068,430e6),(3782,.04,2001,430.2e6)] or h['waveform']!=4 or h['sample_rate_code']!=1 or h['elevation_raw']!=16384 or h['tx_elevation_raw'] not in ([0,16384] if h['samples']==1194 else [16384]):raise ValueError(f'Unsupported 2ms layout: {p}: {h}')
        cache=index_root/(p.parent.name+'-'+p.name+'.h5')
        if cache.exists():
            with h5py.File(cache) as f:return json.loads(bytes(f.attrs['runs_json'])),None
        size=p.stat().st_size;stride=h['frame_bytes']; offsets=[];anomalies=[]
        if h['samples']==3782:
            if size%stride:raise ValueError(f'Truncated December file {p}')
            offsets=np.arange(0,size,stride,dtype=np.int64)
        else:
            with p.open('rb') as f:
                mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);pos=0
                while True:
                    pos=mm.find(b'\x55\x55\xaa\xaa',pos)
                    if pos<0:break
                    if pos+1024<=size and mm[pos+1020:pos+1024]==b'\xaa\x55\x55\xaa' and struct.unpack_from('<I',mm,pos+768)[0]==1194:offsets.append(pos)
                    pos+=4
                offsets=np.array(offsets,dtype=np.int64)
                lengths=np.diff(np.r_[offsets,size]);valid=lengths==stride
                anomalies=[dict(offset=int(o),bytes_to_next_header=int(n)) for o,n in zip(offsets[~valid],lengths[~valid])]
                offsets=offsets[valid];mm.close()
        if not len(offsets):raise ValueError(f'No complete pulses: {p}')
        boundaries=np.r_[0,np.where(np.diff(offsets)!=stride)[0]+1,len(offsets)]
        runs=[]
        for lo,hi in zip(boundaries[:-1],boundaries[1:]):
            start=header(p,int(offsets[lo]));last=header(p,int(offsets[hi-1]));n=int(hi-lo)
            # Split irregular clock sequences further rather than infer a missing pulse.
            expected=(n-1)*h['ipp_s']
            if abs(last['time']-start['time']-expected)>26e-6:
                times=np.array([header(p,int(o))['time'] for o in offsets[lo:hi]])
                cuts=np.r_[0,np.where(np.abs(np.diff(times)-h['ipp_s'])>26e-6)[0]+1,n]
            else:cuts=[0,n]
            for ca,cb in zip(cuts[:-1],cuts[1:]):
                x=header(p,int(offsets[lo+ca]));y=header(p,int(offsets[lo+cb-1]));count=int(cb-ca)
                runs.append(dict(path=str(p.resolve()),byte_offset=int(offsets[lo+ca]),frames=count,start=x['time'],end=y['time']+h['ipp_s'],bytes=count*stride,layout=x))
        with h5py.File(cache,'w') as f:
            f.attrs['runs_json']=np.frombuffer(json.dumps(runs).encode(),dtype=np.uint8)
            f.attrs['anomalies_json']=np.frombuffer(json.dumps(anomalies).encode(),dtype=np.uint8)
            f.attrs['source_bytes']=size;f.create_dataset('valid_frame_byte_offset',data=offsets,compression='gzip')
        print('INDEX',p.parent.name,p.name,len(offsets),'pulses',len(anomalies),'non-payload headers',flush=True)
        return runs,None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        for runs,excluded_file in pool.map(scan,paths):
            sources.extend(runs)
            if excluded_file:excluded.append(excluded_file)
    # Very short fragments at timestamp resynchronizations cannot support a fit;
    # retain their exact raw location and pulse count in the exclusion inventory.
    short=[s for s in sources if s['frames']<500]
    excluded.extend(dict(path=s['path'],byte_offset=s['byte_offset'],pulses=s['frames'],reason='short timestamp-resynchronization fragment; raw voltages retained') for s in short)
    sources=[s for s in sources if s['frames']>=500]
    sources.sort(key=lambda s:s['start']);segments=[]
    unique=[];seen={};hashes={}
    def checksum(path):
        if path not in hashes:
            digest=hashlib.sha256()
            with open(path,'rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
            hashes[path]=digest.hexdigest()
        return hashes[path]
    for s in sources:
        key=(s['start'],s['end'],s['frames'],s.get('byte_offset',0))
        if key in seen:
            other=seen[key]
            if checksum(s['path'])!=checksum(other['path']):raise ValueError('Overlapping recordings differ in voltage bytes')
            excluded.append(dict(path=s['path'],byte_offset=s.get('byte_offset',0),pulses=s['frames'],reason='byte-identical duplicate recording span',canonical=other['path'],sha256=hashes[s['path']]))
        else:seen[key]=s;unique.append(s)
    sources=unique

    for s in sources:
        key={k:v for k,v in s['layout'].items() if k!='time'}
        if segments and s['start']<segments[-1]['end']-26e-6:raise ValueError('Overlapping source recordings')
        if segments and 0 <= s['start']-segments[-1]['end'] < 60 and key==segments[-1]['layout']:
            segments[-1]['sources'].append(s);segments[-1]['end']=s['end']
        else:segments.append(dict(index=len(segments),start=s['start'],end=s['end'],layout=key,sources=[s]))
    for session in segments:
        windows=[];lo=np.floor(session['start']/1800)*1800
        while lo<session['end']-1e-5:
            hi=lo+1800;pieces=[];pulses=0
            for s in session['sources']:
                ipp=s['layout']['ipp_s']; first=max(0,int(np.ceil((lo-s['start'])/ipp-1e-5)));stop=min(s['frames'],max(0,int(np.ceil((hi-s['start'])/ipp-1e-5))))
                if stop>first:pieces.append(dict(path=s['path'],byte_offset=s.get('byte_offset',0),first_frame=first,stop_frame=stop));pulses+=stop-first
            if pulses: windows.append(dict(index=len(windows),utc_start=max(lo,session['start']),utc_end=min(hi,session['end']),pulses=pulses,segments=pieces))
            lo=hi
        assert sum(w['pulses'] for w in windows)==sum(s['frames'] for s in session['sources'])
        # Merge very short edge integrations into their nearest neighbour.
        while len(windows)>1:
            small=next((i for i,w in enumerate(windows) if w['utc_end']-w['utc_start']<300),None)
            if small is None:break
            i=small;j=i+1 if i==0 else i-1;a,b=sorted([i,j]);wa,wb=windows[a],windows[b]
            merged=dict(index=a,utc_start=wa['utc_start'],utc_end=wb['utc_end'],pulses=wa['pulses']+wb['pulses'],segments=wa['segments']+wb['segments'])
            windows[a:b+1]=[merged]
        for i,w in enumerate(windows):w['index']=i
        session['windows']=windows;session['directory']=f"session-{session['index']:02d}"
        d=args.root/session['directory'];d.mkdir(exist_ok=True)
        (d/'plan.json').write_text(json.dumps(session,indent=2))
    record=dict(sessions=segments,excluded=excluded,clock='Header China civil time UTC+8 converted to UTC; same owner-confirmed clock convention as December recording',integration_s=1800)
    (args.root/'plan.json').write_text(json.dumps(record,indent=2))
    with h5py.File(args.root/'inventory.h5','w') as f:
        f.create_dataset('metadata_json',data=np.frombuffer(json.dumps(record).encode(),dtype=np.uint8),compression='gzip')
        for k in ['start','end','frames','bytes']:f.create_dataset(k,data=[s[k] for s in sources])
        f.create_dataset('path',data=np.array([s['path'] for s in sources],dtype=h5py.string_dtype()))
    for s in segments:print(s['directory'],dt.datetime.fromtimestamp(s['start'],dt.timezone.utc).isoformat(),dt.datetime.fromtimestamp(s['end'],dt.timezone.utc).isoformat(),len(s['windows']),sum(v['bytes'] for v in s['sources'])/1e9,'GB')

def run(args):
    rank=int(os.environ.get('OMPI_COMM_WORLD_RANK',0));ranks=int(os.environ.get('OMPI_COMM_WORLD_SIZE',1))
    record=json.loads((args.root/'plan.json').read_text());jobs=[(s,w) for s in record['sessions'] for w in s['windows']]
    if args.retry_failed:
        def failed(job):
            s,w=job;d=args.root/s['directory']/'windows'/f"{w['index']:03d}"
            return not (d/'done.json').exists() and (d/'process.log').exists() and (d/'process.log').read_text().startswith('Error:')
        jobs=[job for job in jobs if failed(job)]
    for s,w in jobs[rank::ranks]:
        d=args.root/s['directory']/'windows'/f"{w['index']:03d}";d.mkdir(parents=True,exist_ok=True)
        if (d/'done.json').exists():continue
        # Reuse the completely validated December measured products by explicit source links.
        if s['layout']['samples']==3782 and args.december_source:
            old=args.december_source/'windows'/f"{w['index']:03d}"
            if old.exists():
                for n in ['lpm.h5','partial.h5']:
                    p=d/n
                    if not p.exists():p.symlink_to(old/n)
                (d/'done.json').write_text(json.dumps(dict(reused=str(old),window=w['index'])));continue
        raw=d/'selection.json'
        raw.write_text(json.dumps(dict(frame_bytes=s['layout']['frame_bytes'],segments=w['segments'])))
        env=os.environ.copy();env.update(RAYON_NUM_THREADS=str(args.threads),SYISR_PARTIAL_OUTPUT=str(d/'partial.tmp.h5'),SYISR_SCREEN_END_SAMPLE=str(s['layout']['samples']))
        with (d/'process.log').open('w') as log:
            result=subprocess.run([str(args.bin_directory/'process'),str(raw),str(d/'lpm.tmp.h5'),'--screen-bright'],env=env,stdout=log,stderr=log)
        if result.returncode:
            print('FAILED',s['directory'],w['index'],flush=True);continue
        os.replace(d/'lpm.tmp.h5',d/'lpm.h5');os.replace(d/'partial.tmp.h5',d/'partial.h5')
        with h5py.File(d/'lpm.h5') as f:m=json.loads(bytes(f.attrs['metadata_json']))
        if m['pulse_count']!=w['pulses']:raise ValueError('Pulse conservation failed')
        (d/'done.json').write_text(json.dumps(dict(window=w['index'],pulses=w['pulses'],retained=m['retained_pulse_count'])))
        print('DONE',s['directory'],w['index'],flush=True)

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['plan','process']);p.add_argument('disc',type=Path);p.add_argument('root',type=Path);p.add_argument('--bin-directory',type=Path,default=Path('target/release'));p.add_argument('--threads',type=int,default=2);p.add_argument('--retry-failed',action='store_true');p.add_argument('--december-source',type=Path)
    args=p.parse_args();args.bin_directory=args.bin_directory.resolve()
    (plan if args.action=='plan' else run)(args)
if __name__=='__main__':main()
