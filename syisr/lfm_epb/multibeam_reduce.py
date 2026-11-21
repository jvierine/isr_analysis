"""MPI independent-beam medians from pulse power, with linear range grouping."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from mpi4py import MPI

def reduce_one(source,out,start,stop):
    files=sorted(source.glob('*.h5'));ts=[];ps=[];ns=[]
    para=None
    for path in files:
        with h5py.File(path) as h:
            t=h['times_unix_s'][()];ix=np.flatnonzero((t>=start)&(t<stop))
            if not len(ix):continue
            if para is None:
                para=h['para'][()];re=h['range_edges_km'][()];ze=h['altitude_edges_km'][()]
                # Retain aligned 8-cell blocks, covering 150-950 km altitude.
                lo=(np.searchsorted(ze,150)//8)*8
                hi=min((np.searchsorted(ze,950)//8+1)*8,len(ze)-1)
                hi=lo+((hi-lo)//8)*8
                coords={k:h[k][lo:hi] for k in ['range_km','altitude_km']}
                redges=re[lo:hi+1];zedges=ze[lo:hi+1]
            else:np.testing.assert_allclose(para[[6,7,9,10,11,12,13,14,15]],h['para'][()][[6,7,9,10,11,12,13,14,15]],atol=1e-7)
            ts.append(t[ix]);ns.append(h['noise_power'][()][ix])
            ps.append(h['downchirp_power'][ix[0]:ix[-1]+1,lo:hi].astype(np.float32))
    if not ts:raise ValueError('No observations '+str(source))
    t=np.concatenate(ts);p=np.concatenate(ps);n=np.concatenate(ns);order=np.argsort(t)
    t=t[order];p=p[order];n=n[order]
    if np.any(np.diff(t)<=0):raise ValueError('Duplicate pulse timestamps')
    out.mkdir(parents=True,exist_ok=True)
    for seconds,group,name in [(5,1,'display'),(30,8,'inversion')]:
        ng=p.shape[1]//group
        pp=p[:,:ng*group].reshape(len(t),ng,group).mean(axis=2,dtype=np.float64)
        edges=np.arange(start,stop+.001,seconds);idx=np.floor((t-start)/seconds).astype(int)
        sig=np.full((len(edges)-1,ng),np.nan);aa=sig.copy();bb=sig.copy();nn=np.full(len(sig),np.nan);count=np.zeros(len(sig),dtype=np.int64)
        parity=np.arange(len(t))%2
        for j in range(len(sig)):
            ix=np.flatnonzero(idx==j)
            if not len(ix):continue
            sig[j]=np.median(pp[ix]-n[ix,None],axis=0);nn[j]=n[ix].mean();count[j]=len(ix)
            for half,result in [(0,aa),(1,bb)]:
                sub=ix[parity[ix]==half]
                if len(sub):result[j]=np.median(pp[sub]-n[sub,None],axis=0)
        rr=redges[:ng*group+1:group];zz=zedges[:ng*group+1:group]
        with h5py.File(out/(name+'.h5'),'w') as h:
            h.attrs['beam_azimuth_deg']=para[6];h.attrs['beam_elevation_deg']=para[7]
            h.attrs['time_bin_s']=seconds;h.attrs['range_bin_km']=np.median(np.diff(rr))
            h.attrs['method']='linear range averaging before pulse median; per-pulse raw topside noise subtracted; signed estimates retained'
            h.attrs['altitude_convention']='WGS84 geodetic height; site 18.3492N 109.6222E, assumed 50m ellipsoid height'
            h.attrs['source_directory']=str(source)
            for k,v in [('median_signal',sig),('split_a_signal',aa),('split_b_signal',bb),('noise_power',nn),('pulse_count',count),
                        ('range_edges_km',rr),('altitude_edges_km',zz),('range_km',(rr[:-1]+rr[1:])/2),('altitude_km',(zz[:-1]+zz[1:])/2),
                        ('time_edges_unix_s',edges),('time_unix_s',(edges[:-1]+edges[1:])/2),('para',para)]:
                h.create_dataset(k,data=v,compression='gzip',shuffle=True)
            h.attrs['pulse_gap_max_s']=np.max(np.diff(t));h.attrs['pulse_cadence_median_s']=np.median(np.diff(t))
        print(source.name,name,'shape',sig.shape,'pulses',len(t),'empty',sum(count==0),flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('root');a.add_argument('--start',type=float,default=1772362800);a.add_argument('--stop',type=float,default=1772373600);args=a.parse_args()
    comm=MPI.COMM_WORLD
    if comm.size>16:raise ValueError('16 core limit')
    root=Path(args.root);beams=sorted((root/'parts').glob('beam_*'))
    for i in range(comm.rank,len(beams),comm.size):reduce_one(beams[i],root/'beams'/beams[i].name,args.start,args.stop)
    comm.Barrier()
if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback;traceback.print_exc();MPI.COMM_WORLD.Abort(1)
