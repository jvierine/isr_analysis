"""Production decode of coherent point fixtures and captured satellite code cycles.

The worker requires two cycles per 300 s interval. Each test's captured or
synthetic 64-member cycle is duplicated, with synthetic header clocks solely
to satisfy that interface. Original captured clocks and voltages stay in HDF5.
No independent-sample claim is made for those duplicates.
"""
import argparse,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone,timedelta
import h5py,numpy as np
from sgp4.api import Satrec
from satellite_crossings import observe

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--processor',type=Path);p.add_argument('--output-name',default='range-chain-audit');q=p.parse_args();root=q.root.resolve();old=root.parent/'syisr-lhw-ac-20261119-15m-step5m-v01';out=root/q.output_name;out.mkdir(exist_ok=True)
codes=np.array([[1 if s=='+' else -1 for s in line.strip()] for line in (root/'source/syisr/m32-reference.bc').read_text().splitlines() if line.strip()])
with h5py.File(old/'index.h5') as h:
    beam=h['beam_index'][:];bounds=h['cycle_time_utc_s'][:];off=h['frame_offset'][:];fi=h['file_index'][:];paths=json.loads(bytes(h['files_json_utf8'][:]))
maps={}
def frame(ci,c):
    f=int(fi[ci,c]);o=int(off[ci,c])
    if f not in maps:maps[f]=np.memmap(paths[f],mode='r',dtype='u1')
    head=bytearray(maps[f][o:o+1024]);n=int.from_bytes(head[768:772],'little')+1
    raw=np.frombuffer(maps[f][o+1024:o+1024+n*8],'<i4').reshape(n,2)
    return head,raw[:,1].astype(float)+1j*raw[:,0]
heads=np.empty((5,64,1024),'u1');real=[];real_times=[];predictions=[];raw_ipp=None
with h5py.File(root/'satellite-calibration/crossings.h5') as h:
    rows=h['crossings'][:];lines1=h['tle_line1'].asstr()[:];lines2=h['tle_line2'].asstr()[:]
for case in range(2):
    captures=[];clocks=[];rr=[]
    for b in range(5):
        with h5py.File(root/f'satellite-calibration/pointing-beam-{150+50*b}.h5') as h:
            g=h[f'pass_{case:03d}'];j=int(np.argmax(g['peak_ratio'][:]));t=float(g['utc_s'][j]);norad=g.attrs['norad_id'];closest=g.attrs['closest_utc_s']
        candidates=np.flatnonzero(beam==b);ci=candidates[np.argmin(abs(bounds[candidates].mean(axis=1)-t))]
        tt=bounds[ci,0]+np.arange(64)*(bounds[ci,1]-bounds[ci,0])/63
        x=[]
        for c in range(64):
            head,v=frame(ci,c);heads[b,c]=np.frombuffer(head,'u1');x.append(v)
            if case==0 and b==1 and abs(tt[c]-t)<.041:raw_ipp=(head,v,tt[c],norad)
        idx=np.flatnonzero((rows['beam_index']==b)&(rows['norad_id']==norad)&(abs(rows['utc_s']-closest)<.1))[0]
        sat=Satrec.twoline2rv(lines1[idx],lines2[idx]);_,ran,rate,az,el=observe(sat,tt)
        captures.append(x);clocks.append(tt);rr.append(ran)
    real.append(np.array(captures));real_times.append(np.array(clocks));predictions.append(np.array(rr))
assert raw_ipp is not None
target=np.array([-8,0,20,80,160]);sample=np.arange(real[0].shape[-1]);synthetic=np.zeros_like(real[0]);clutter=np.zeros_like(real[0])
for b,source in enumerate(target):
    baud=(sample-source)//2;good=(baud>=0)&(baud<32)
    synthetic[b,:,good]=10000*codes[:,baud[good]].T*np.exp(2j*np.pi*1562.5*sample[good,None]*1e-5)
    # A strong coherent early code-bearing echo; never treated as transmit zero.
    baud=(sample+50)//2;good=(baud>=0)&(baud<32)
    clutter[b,:,good]=100000*codes[:,baud[good]].T
cases={'coherent_point':synthetic,'coherent_point_plus_clutter':synthetic+clutter,'satellite_0':real[0],'satellite_1':real[1]}
env=dict(os.environ,RAYON_NUM_THREADS='5',SLURM_PROCID='0',SLURM_NTASKS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SYISR_AUDIT_RANGE_GATES='570')
program=q.processor or old/'bin/process'
with h5py.File(out/'evidence.h5','w') as evidence:
    evidence['injected_source_sample_offset']=target;evidence['captured_cycle_utc_s']=real_times;evidence['predicted_satellite_range_km']=predictions
    head,v,t,norad=raw_ipp;g=evidence.create_group('raw_ipp');g['voltage_i_q_adu']=np.c_[v.real,v.imag];g['header_bytes']=np.frombuffer(head,'u1');g['header_fast_time_us']=667+np.arange(len(v))*10;g.attrs.update(utc_s=t,norad_id=norad,ipp_us=16000,excluded_voltage_end_sample=32)
    for name,x in cases.items():
        run=out/name;run.mkdir(exist_ok=True);(run/'source').symlink_to(root/'source') if not (run/'source').exists() else None
        raw=run/'fixture.dat1-1-0'
        with raw.open('wb') as f:
            for start in [10,20,310]:
                for c in range(64):
                    for b in range(5):
                        head=bytearray(heads[b,c].tobytes());t=datetime(2026,11,19,20,tzinfo=timezone.utc)+timedelta(seconds=start+(c*5+b)*.016)
                        head[12:18]=bytes([t.month,t.year-2000,t.hour,t.day,t.second,t.minute]);head[18:20]=round(t.microsecond/25).to_bytes(2,'little')
                        f.write(head);f.write(np.rint(np.c_[x[b,c].imag,x[b,c].real]).astype('<i4').tobytes())
        subprocess.run([str(program),'index',str(run),'fixture',str(run/'index.h5')],check=True,env=env)
        subprocess.run([str(program),'worker',str(run/'index.h5'),str(root/'source/syisr/m32-reference.bc'),str(run/'blocks')],check=True,env=env)
        for cut in [0,32]:
            subprocess.run([sys.executable,str(Path(__file__).parent/'partial_voltage.py'),str(run),str(run/f'cut{cut}'),str(cut)],check=True,env=env)
            with h5py.File(run/f'cut{cut}/0000.h5') as h:
                dst=evidence.create_group(f'{name}/cut{cut}');dst['slant_range_km']=h['slant_range_km'][:]-165.78203594884826*.149896229
                for b in range(5):
                    dst[f'beam_{b}_acf']=h[f'beam_{b}/unscreened_acf_adu2'][:]
        print('DECODE COMPLETE',name,flush=True)
    evidence.attrs.update(delay_correction_us=165.78203594884826,description='Production index, worker and partial-voltage driver; duplicated cycles are interface adapters, not independent samples',git_commit=os.environ.get('SOURCE_GIT_COMMIT','working'))
