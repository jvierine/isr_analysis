"""MPI sweep of native-pulse range averages and time medians; HDF5 only.

Every candidate averages linear power over range BEFORE taking a pulse median.
Interleaved-pulse repeats quantify empirical repeatability, not ground truth.
"""
import argparse
from pathlib import Path
import subprocess
import h5py
import numpy as np
from mpi4py import MPI

TIME_BINS=(5,15,30,60)
RANGE_GROUPS=(1,2,4,8)


def prepare(parts, output, start, stop):
    all_t=[];all_p=[];all_n=[];source=[];last=None
    for path in sorted(Path(parts).glob('*.h5')):
        with h5py.File(path) as h:
            t=h['times_unix_s'][()]
            ix=np.flatnonzero((t>=start)&(t<stop))
            if not len(ix): continue
            if last is None:
                r=h['range_km'][()]; z=h['altitude_km'][()]
                re=h['range_edges_km'][()];ze=h['altitude_edges_km'][()]
                # Save the whole range axis, including independent quiet/noise diagnostics.
                last=dict(h.attrs)
            else: np.testing.assert_allclose(r,h['range_km'][()],rtol=0,atol=1e-8)
            all_t.append(t[ix]);all_n.append(h['noise_power'][()][ix])
            all_p.append(h['downchirp_power'][ix[0]:ix[-1]+1].astype(np.float32))
            source.append(h.attrs['source_file'])
    if not all_t:raise ValueError('No overlapping pulses')
    t=np.concatenate(all_t);power=np.concatenate(all_p);noise=np.concatenate(all_n)
    order=np.argsort(t);t=t[order];power=power[order];noise=noise[order]
    if np.any(np.diff(t)<=0):raise ValueError('Duplicate/backward pulse clock')
    dr=np.diff(re)
    np.testing.assert_allclose(dr,.3747405725,rtol=0,atol=1e-8)
    with h5py.File(output,'w') as h:
        for k,v in last.items(): h.attrs[k]=v
        h.attrs['start_utc_unix_s']=start;h.attrs['stop_utc_unix_s']=stop
        h.attrs['code_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        h['time_unix_s']=t;h['noise_power']=noise;h['range_km']=r;h['altitude_km']=z
        h['range_edges_km']=re;h['altitude_edges_km']=ze
        h.create_dataset('pulse_power',data=power,compression='gzip',shuffle=True)
        h['source_files']=np.asarray(source,dtype=h5py.string_dtype())
    print('PREPARED',len(t),'pulses',power.shape,flush=True)


def reduce_candidate(cube, output, seconds, group):
    with h5py.File(cube) as h:
        t=h['time_unix_s'][()];p=h['pulse_power'][()];n=h['noise_power'][()]
        re=h['range_edges_km'][()];ze=h['altitude_edges_km'][()]
        start=h.attrs['start_utc_unix_s'];stop=h.attrs['stop_utc_unix_s']
    ng=p.shape[1]//group
    p=p[:,:ng*group].reshape(len(t),ng,group).mean(axis=2,dtype=np.float64)
    re=re[:ng*group+1:group];ze=ze[:ng*group+1:group]
    r=(re[:-1]+re[1:])/2;z=(ze[:-1]+ze[1:])/2
    edges=np.arange(start,stop+1e-3,seconds)
    assert edges[-1]==stop
    idx=np.floor((t-start)/seconds).astype(int);nt=len(edges)-1
    signal=np.full((nt,ng),np.nan);mean=signal.copy();aa=signal.copy();bb=signal.copy()
    noise=np.full(nt,np.nan);count=np.zeros(nt,dtype=np.int64)
    # Alternate pulses globally, so independent subsets sample the same evolution.
    parity=np.arange(len(t))%2
    for j in range(nt):
        ix=np.flatnonzero(idx==j)
        if not len(ix):continue
        s=p[ix]-n[ix,None]
        signal[j]=np.median(s,axis=0);mean[j]=s.mean(axis=0)
        noise[j]=n[ix].mean();count[j]=len(ix)
        for half,out in [(0,aa),(1,bb)]:
            sub=ix[parity[ix]==half]
            if len(sub):out[j]=np.median(p[sub]-n[sub,None],axis=0)
    assert count.min()>0
    # Quiet high-range repeatability and paired F-region morphology, in linear units.
    quiet=(z>=1200)&(z<1500);fregion=(z>=250)&(z<=600)
    da=aa/noise[:,None];db=bb/noise[:,None]
    quiet_difference=(da-db)[:,quiet].ravel()
    split_rms=np.sqrt(np.mean(quiet_difference**2)/2)
    split_mad=np.median(np.abs(quiet_difference-np.median(quiet_difference)))/(.67448975*np.sqrt(2))
    # Compare range-squared paired maps with a common reference normalization.
    va=(da*r[None,:]**2)[:,fregion].ravel()
    vb=(db*r[None,:]**2)[:,fregion].ravel()
    correlation=np.corrcoef(va,vb)[0,1]
    with h5py.File(output,'w') as h:
        for k,v in [('time_edges_unix_s',edges),('range_edges_km',re),('altitude_edges_km',ze),
                    ('range_km',r),('altitude_km',z),('noise_power',noise),('pulse_count',count),
                    ('median_signal',signal),('mean_signal',mean),('split_a_signal',aa),('split_b_signal',bb)]:
            h.create_dataset(k,data=v,compression='gzip',shuffle=True)
        h.attrs['time_bin_s']=seconds;h.attrs['range_bin_km']=np.median(np.diff(re))
        h.attrs['range_group_from_375m']=group;h.attrs['split_half_quiet_rms_snr']=split_rms
        h.attrs['split_half_quiet_mad_snr']=split_mad;h.attrs['split_half_f_region_correlation']=correlation
        h.attrs['source_file']=str(cube)
        h.attrs['method']='linear range averaging before signed noise subtraction and pulse median; no interpolation'
        h.attrs['split_method']='alternating pulses; quiet diagnostic 1200-1500 km; morphology correlation 250-600 km of range-squared signal/noise'
        h.attrs['limitations']='repeatability is not physical truth; correlation includes large-scale morphology; medians carry range-dependent noise bias; no calibrated density'
    print('REDUCED',seconds,group,signal.shape,'repeatability',split_rms,correlation,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','sweep']);p.add_argument('root')
    p.add_argument('--parts');p.add_argument('--start',type=float);p.add_argument('--stop',type=float)
    a=p.parse_args();root=Path(a.root);comm=MPI.COMM_WORLD
    if comm.size>16:raise ValueError('Resolution study limited to 16 MPI ranks')
    if a.mode=='prepare':
        if comm.size!=1:raise ValueError('Preparation is single process')
        root.mkdir(parents=True,exist_ok=True)
        prepare(a.parts,root/'pulse_cube.h5',a.start,a.stop)
    else:
        jobs=[(t,g) for t in TIME_BINS for g in RANGE_GROUPS]
        for i in range(comm.rank,len(jobs),comm.size):
            seconds,group=jobs[i]
            reduce_candidate(root/'pulse_cube.h5',root/f't{seconds:02d}_g{group:02d}.h5',seconds,group)
        comm.Barrier()
        if comm.rank==0:
            with h5py.File(root/'summary.h5','w') as h:
                for seconds,group in jobs:
                    key=f't{seconds:02d}_g{group:02d}'
                    with h5py.File(root/(key+'.h5')) as c:
                        out=h.create_group(key)
                        for k,v in c.attrs.items():out.attrs[k]=v
            print('SWEEP COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback;traceback.print_exc();MPI.COMM_WORLD.Abort(1)
