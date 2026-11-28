"""Exclude raw-time bursts and only admit complete, uncontaminated code windows."""
import numpy as np

def intervals(mask):
    edge=np.diff(np.r_[False,np.asarray(mask,dtype=bool),False].astype('i1'))
    return np.column_stack((np.flatnonzero(edge==1),np.flatnonzero(edge==-1)))

def detect(power_2us,threshold=4.):
    p=np.asarray(power_2us,dtype='f8')
    finite=p[np.isfinite(p)&(p>0)]
    if not len(finite):raise ValueError('no usable noise floor')
    floor=float(np.median(finite))
    return (~np.isfinite(p))|(p>threshold*floor),floor

def accepted_windows(bad,ne,nt,step,bin_samples=32,guard_samples=256,edge_samples=512):
    """No partial-code estimates: reject every window intersecting guarded bursts."""
    bad=np.asarray(bad,dtype=bool);starts=np.arange((ne-nt)//step+1,dtype='i8')*step
    prefix=np.r_[0,np.cumsum(bad,dtype='i8')]
    lo=np.clip((starts-guard_samples)//bin_samples,0,len(bad))
    hi=np.clip((starts+nt+guard_samples+bin_samples-1)//bin_samples,0,len(bad))
    return (prefix[hi]==prefix[lo])&(starts>=edge_samples)&(starts+nt<=ne-edge_samples)

def guarded_clean_runs(bad,ne,bin_samples=32,guard_samples=256,edge_samples=512):
    bad_runs=intervals(bad)*bin_samples
    bad_runs[:,0]=np.maximum(0,bad_runs[:,0]-guard_samples)
    bad_runs[:,1]=np.minimum(ne,bad_runs[:,1]+guard_samples)
    clean=[];start=edge_samples
    for left,right in bad_runs:
        if left>start:clean.append((start,min(int(left),ne-edge_samples)))
        start=max(start,int(right))
    if start<ne-edge_samples:clean.append((start,ne-edge_samples))
    return np.asarray([(l,r)for l,r in clean if r>l],dtype='i8').reshape(-1,2)
