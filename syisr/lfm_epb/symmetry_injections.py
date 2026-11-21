"""Noise-free discrimination check on the actual finite-beam observation geometry.
Not an uncertainty coverage experiment. Uses fixed real-data drift and lambda100.
"""
from pathlib import Path
import sys
import h5py
import numpy as np
from mpi4py import MPI
from symmetry_validation import design,matrices,solve,MODELS,NX,NY
comm=MPI.COMM_WORLD
if comm.size>16:raise ValueError('16-rank limit')
root=Path(sys.argv[1]);out=root/'symmetry_validation';p,arcs,A,S,T,L,original=design(root);X=matrices(A,S,T)
u,v=np.meshgrid(-1300+np.arange(NX)*10,-800+np.arange(NY)*10)
c=(-.6*np.exp(-.5*((u-40)/40)**2-.5*((v+20)/80)**2)).ravel()
truths=[A@c+S@(.4*c),A@c+T@(.4*c)]
beams=p[:,1].astype(int);jobs=[(source,m,fold) for source in range(2) for m in range(4) for fold in range(3)]
for j in range(comm.rank,len(jobs),comm.size):
    source,m,fold=jobs[j];train=beams%3!=fold;test=~train
    x,res=solve(X[m],truths[source],p[:,7],train,100.,L)
    with h5py.File(out/f'injection_{source}_{m}_{fold}.h5','w') as h:
        h['sse_count']=[np.sum((X[m][test]@x-truths[source][test])**2),test.sum()];h['normal_relative_residual']=res
comm.Barrier()
if comm.rank==0:
    result=np.zeros((2,4,3,2))
    for source,m,fold in jobs:
        with h5py.File(out/f'injection_{source}_{m}_{fold}.h5') as h:result[source,m,fold]=h['sse_count'][()]
    rmse=np.sqrt(result[:,:,:,0].sum(axis=2)/result[:,:,:,1].sum(axis=2))
    with h5py.File(out/'injection_summary.h5','w') as h:
        h['truth_names']=np.array(['along_B_only','time_only'],dtype=h5py.string_dtype());h['model_names']=np.array(MODELS,dtype=h5py.string_dtype());h['outer_sse_count']=result;h['outer_rmse']=rmse
        h.attrs['scope']='Noise-free synthetic discrimination on actual geometry, fixed real-data drift and lambda100. Gaussian contrast sigma40x80km, amplitude-0.6, slope0.4 per500km or hour. Does not establish real-data noise, coherent-scatter cleanliness, or general identifiability.'
    print('INJECTION_RMSE',rmse,flush=True)
