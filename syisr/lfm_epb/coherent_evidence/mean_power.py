"""Mean signed LFM power for calibration; pulse medians remain diagnostic only.
Independent beams distributed across MPI ranks, scientific products HDF5.
"""
import argparse
from pathlib import Path
import h5py,numpy as np
from mpi4py import MPI

def beam(source,out):
    ts=[];ps=[];ns=[];meta=None
    for path in sorted(source.glob('*.h5')):
        with h5py.File(path) as h:
            if meta is None:
                ze=h['altitude_edges_km'][()];lo=(np.searchsorted(ze,150)//16)*16;hi=(np.searchsorted(ze,950)//16)*16
                rr=h['range_edges_km'][()][lo:hi+1:16];zz=ze[lo:hi+1:16];meta=h['para'][()]
            t=h['times_unix_s'][()];good=(t>=1772362800)&(t<1772373600)
            if not np.any(good):continue
            power=h['downchirp_power'][()][:,lo:hi][good].reshape(good.sum(),-1,16).mean(axis=2,dtype=np.float64)
            noise=h['noise_power'][()][good];ts.append(t[good]);ps.append(power-noise[:,None]);ns.append(noise)
    t=np.concatenate(ts);p=np.concatenate(ps);noise=np.concatenate(ns);ix=np.argsort(t);t=t[ix];p=p[ix];noise=noise[ix]
    if np.any(np.diff(t)<=0):raise ValueError('Duplicate timestamps')
    edges=np.arange(1772362800,1772373601,60.);bins=np.floor((t-edges[0])/60).astype(int);nt=len(edges)-1;nr=p.shape[1]
    mean=np.full((nt,nr),np.nan);median=mean.copy();sem=mean.copy();count=np.zeros(nt,dtype=int);n=np.full(nt,np.nan)
    for j in range(nt):
        q=bins==j;count[j]=q.sum()
        if count[j]:mean[j]=p[q].mean(axis=0);median[j]=np.median(p[q],axis=0);sem[j]=p[q].std(axis=0,ddof=1)/np.sqrt(count[j]);n[j]=noise[q].mean()
    with h5py.File(out/(source.name+'.h5'),'w') as h:
        for k,v in [('time_unix_s',(edges[:-1]+edges[1:])/2),('time_edges_unix_s',edges),('range_km',(rr[:-1]+rr[1:])/2),('altitude_km',(zz[:-1]+zz[1:])/2),('mean_signal',mean),('median_signal',median),('pulse_sem_signal',sem),('pulse_count',count),('noise_power',n),('para',meta)]:h.create_dataset(k,data=v,compression='gzip',shuffle=True)
        h.attrs['source_parts']=str(source);h.attrs['method']='16 linear 0.3747405725km range cells averaged per pulse; per-pulse raw far-range noise subtracted; signed arithmetic pulse mean retained. No logarithm, clipping or median used for calibration.'
        h.attrs['sem_limitation']='Empirical pulse SEM; independence is not demonstrated and coherent/noise correlations can invalidate a formal uncertainty interpretation.'
    print('MEAN_POWER',source.name,mean.shape,'pulse_count',count.sum(),flush=True)
p=argparse.ArgumentParser();p.add_argument('parts');p.add_argument('output');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
comm=MPI.COMM_WORLD
if comm.size>16:raise ValueError('16-core ceiling')
beams=sorted(Path(a.parts).glob('beam_*'))
try:
    for i in range(comm.rank,len(beams),comm.size):beam(beams[i],out)
    comm.Barrier()
    if comm.rank==0:(out/'COMPLETE').write_text('All available beams reduced with arithmetic means\n')
except BaseException:
    import traceback;traceback.print_exc();comm.Abort(1)
