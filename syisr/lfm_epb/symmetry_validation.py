"""Exploratory nested whole-beam validation of symmetry versus time evolution.
Signed relative-signal observations, finite beam quadrature and fixed previously
selected pattern drift. All candidates are unconstrained linear diagnostics;
these are not the bounded physical reconstruction or a calibrated likelihood.
"""
import argparse
from pathlib import Path
import h5py
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator
from mpi4py import MPI

NX,NY=261,161
N=NX*NY
T0=1772369100.
MODELS=('frozen','along_B','evolving','along_B_and_evolving')

def design(root,exclude_beam=None):
    with h5py.File(root/'cavity/observations.h5') as h:
        obs=h['observations'][()];eligible=h['eligible'][()].astype(bool)
    mask=eligible & (np.abs(obs[:,31])>=2)
    if exclude_beam is not None:mask &= obs[:,1] != exclude_beam-1
    p=obs[mask];nt=len(p);arcs=np.zeros((nt,9))
    with h5py.File(root/'geometry.h5') as g:
        names=sorted(k for k in g if k.startswith('beam_'))
        for b,name in enumerate(names):
            use=p[:,1].astype(int)==b
            with h5py.File(root/'beams'/name/'inversion.h5') as h:ranges=h['range_km'][()]
            index=np.searchsorted(ranges,p[use,2]);index=np.clip(index,0,len(ranges)-1)
            if not np.allclose(ranges[index],p[use,2],atol=1e-7):raise ValueError('Range registration')
            # Stored signed path runs from observation toward reference plane.
            arcs[use]=-g[name]['beam_quadrature_field_line_km'][()][index,:,2]
    with h5py.File(root/'cavity/selection.h5') as h:vx,vy,_=h['selected_config'][()]
    rows=[];cols=[];vals=[];svals=[]
    w=np.outer([1/6,2/3,1/6],[1/6,2/3,1/6]).ravel()
    for q in range(9):
        x=(p[:,13+2*q]-vx*(p[:,0]-T0)+1300)/10
        y=(p[:,14+2*q]-vy*(p[:,0]-T0)+800)/10
        ix=np.floor(x).astype(int);iy=np.floor(y).astype(int)
        good=(ix>=0)&(ix<NX-1)&(iy>=0)&(iy<NY-1)
        fx=x-ix;fy=y-iy
        for dx,dy,a in [(0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)]:
            rr=np.flatnonzero(good);vv=a[good]*w[q]
            rows.append(rr);cols.append((iy[good]+dy)*NX+ix[good]+dx);vals.append(vv);svals.append(vv*arcs[good,q]/500.)
    row=np.concatenate(rows);col=np.concatenate(cols);v=np.concatenate(vals)
    A=sparse.csr_matrix((v,(row,col)),shape=(nt,N))
    S=sparse.csr_matrix((np.concatenate(svals),(row,col)),shape=(nt,N))
    keep=np.abs(np.asarray(A.sum(axis=1)).ravel()-1)<1e-8
    A=A[keep];S=S[keep];p=p[keep];arcs=arcs[keep]
    T=A.multiply(((p[:,0]-T0)/3600)[:,None]).tocsr()
    # Spatial difference regularizer, separately on every coefficient map.
    k=np.arange(N).reshape(NY,NX);left=np.r_[k[:,:-1].ravel(),k[:-1,:].ravel()];right=np.r_[k[:,1:].ravel(),k[1:,:].ravel()]
    rr=np.repeat(np.arange(len(left)),2);cc=np.column_stack([left,right]).ravel()
    D=sparse.csr_matrix((np.tile([1.,-1.],len(left)),(rr,cc)),shape=(len(left),N))
    return p,arcs,A,S,T,(D.T@D).tocsr(),np.flatnonzero(mask)[keep]

def matrices(A,S,T):
    return [A,sparse.hstack([A,S],format='csr'),sparse.hstack([A,T],format='csr'),sparse.hstack([A,S,T],format='csr')]

def solve(X,y,sigma,train,lam,L):
    nfields=X.shape[1]//N
    W=X[train].multiply((1/sigma[train])[:,None]).tocsr();target=y[train]/sigma[train]
    rhs=np.asarray(W.T@target).ravel()
    # Same scale for additional coefficient maps: 500 km or 1 h = unit variation.
    ridge=np.concatenate([np.full(N,.1)]+[np.full(N,10.) for _ in range(nfields-1)])
    reg=sparse.block_diag([L]*nfields,format='csr')
    def mv(x):return W.T@(W@x)+lam*(reg@x)+ridge*x
    normal=LinearOperator((len(rhs),len(rhs)),matvec=mv)
    diag=np.asarray(W.power(2).sum(axis=0)).ravel()+lam*reg.diagonal()+ridge
    pre=LinearOperator(normal.shape,matvec=lambda x:x/diag)
    x,info=cg(normal,rhs,M=pre,rtol=1e-5,atol=0,maxiter=1500)
    if info:raise RuntimeError(f'CG failed: {info}')
    return x,float(np.linalg.norm(mv(x)-rhs)/max(np.linalg.norm(rhs),1e-30))

def main():
    a=argparse.ArgumentParser();a.add_argument('root',type=Path);a.add_argument('--exclude-beam',type=int);args=a.parse_args();root=args.root
    comm=MPI.COMM_WORLD
    if comm.size>16:raise ValueError('16-rank ceiling')
    out=root/('symmetry_validation' if args.exclude_beam is None else f'symmetry_validation_no_beam{args.exclude_beam}');out.mkdir(exist_ok=True)
    p,arcs,A,S,T,L,original=design(root,args.exclude_beam);X=matrices(A,S,T);y=p[:,6]-1;sigma=p[:,7];beams=p[:,1].astype(int)
    # Outer: three entire-beam groups. Inner: remaining beam IDs split in two.
    lambdas=[10.,100.,1000.]
    jobs=[(outer,m,l,inner) for outer in range(3) for m in range(4) for l in lambdas for inner in range(2)]
    for j in range(comm.rank,len(jobs),comm.size):
        outer,m,lam,inner=jobs[j];outertrain=(beams%3)!=outer
        remaining=sorted(set(beams[outertrain]));innerids=remaining[inner::2]
        test=outertrain & np.isin(beams,innerids);train=outertrain & ~test
        x,res=solve(X[m],y,sigma,train,lam,L);pred=X[m][test]@x
        with h5py.File(out/f'inner_{outer}_{m}_{lam:g}_{inner}.h5','w') as h:
            h['sse_count']=[np.sum((pred-y[test])**2),test.sum()];h['normal_relative_residual']=res
        print('INNER',outer,MODELS[m],lam,inner,flush=True)
    comm.Barrier()
    outerjobs=[(outer,m) for outer in range(3) for m in range(4)]
    for j in range(comm.rank,len(outerjobs),comm.size):
        outer,m=outerjobs[j];scores=[]
        for lam in lambdas:
            d=[]
            for inner in range(2):
                with h5py.File(out/f'inner_{outer}_{m}_{lam:g}_{inner}.h5') as h:d.append(h['sse_count'][()])
            d=np.sum(d,axis=0);scores.append(np.sqrt(d[0]/d[1]))
        lam=lambdas[int(np.argmin(scores))];test=beams%3==outer;train=~test
        x,res=solve(X[m],y,sigma,train,lam,L)
        pred=X[m][test]@x
        with h5py.File(out/f'outer_{outer}_{m}.h5','w') as h:
            h['sse_count']=[np.sum((pred-y[test])**2),test.sum()];h['lambda']=lam;h['inner_rmse']=scores;h['normal_relative_residual']=res
            h['observation_index']=original[test];h['predicted_ratio']=1+pred;h['observed_ratio']=1+y[test]
            h['beam']=beams[test];h['time_unix_s']=p[test,0];h['along_B_km']=np.mean(arcs[test],axis=1)
        print('OUTER',outer,MODELS[m],lam,np.sqrt(np.mean((pred-y[test])**2)),flush=True)
    comm.Barrier()
    if comm.rank==0:
        d=np.zeros((3,4,2));selected=np.zeros((3,4));predictions=np.full((4,len(p)),np.nan)
        lookup={int(q):i for i,q in enumerate(original)}
        for outer,m in outerjobs:
            with h5py.File(out/f'outer_{outer}_{m}.h5') as h:
                d[outer,m]=h['sse_count'][()];selected[outer,m]=h['lambda'][()]
                ids=np.array([lookup[int(q)] for q in h['observation_index'][()]])
                predictions[m,ids]=h['predicted_ratio'][()]
        rmse=np.sqrt(d[:,:,0].sum(axis=0)/d[:,:,1].sum(axis=0))
        # Local geometric separation of time and along-field coordinates.
        with h5py.File(root/'cavity/selection.h5') as h:vx,vy,_=h['selected_config'][()]
        u=p[:,4]-vx*(p[:,0]-T0);v=p[:,5]-vy*(p[:,0]-T0);cell=np.column_stack([np.floor(u/30),np.floor(v/30)]).astype(int)
        _,groups=np.unique(cell,axis=0,return_inverse=True);local=[]
        for g in np.unique(groups):
            z=groups==g
            if z.sum()<20 or len(set(beams[z]))<2:continue
            st=np.column_stack([arcs[z].mean(axis=1)/500.,(p[z,0]-T0)/3600.]);st-=st.mean(axis=0)
            eigen=np.linalg.eigvalsh(st.T@st/z.sum());corr=np.corrcoef(st.T)[0,1]
            local.append([z.sum(),len(set(beams[z])),np.ptp(st[:,0])*500,np.ptp(st[:,1])*3600,eigen[0],eigen[1],corr])
        with h5py.File(out/'summary.h5','w') as h:
            h['model_names']=np.array(MODELS,dtype=h5py.string_dtype());h['outer_sse_count']=d;h['outer_rmse']=rmse;h['selected_lambda']=selected
            h['observation_index']=original;h['predicted_ratio']=predictions;h['observed_ratio']=p[:,6];h['beam']=beams;h['time_unix_s']=p[:,0];h['along_B_km']=arcs.mean(axis=1)
            h['local_time_along_B_identifiability']=np.array(local)
            h.attrs['excluded_beam']=args.exclude_beam or 0
            h.attrs['local_columns']='count, beam_count, along_B_span_km, time_span_s, min_eigenvalue, max_eigenvalue, correlation'
            h.attrs['scope']='Exploratory unconstrained linear model comparison. Fixed drift selected earlier from these data. Outer whole-beam groups; lambda selected only using remaining beams. Empirical weights, no calibrated significance or absolute density.'
            h.attrs['models']='c(u,v); c+a*s/500km; c+b*dt/3600s; c+a*s/500km+b*dt/3600s. Finite 9-point angular quadrature. Same 10km spatial regularizer; added coefficient ridge10 versus base0.1.'
        (out/'COMPLETE').write_text('Nested conditional model comparison complete\n')
        print('MODEL_RMSE',dict(zip(MODELS,rmse)),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback;traceback.print_exc();MPI.COMM_WORLD.Abort(1)
