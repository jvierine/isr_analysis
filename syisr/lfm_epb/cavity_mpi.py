"""MPI orchestration of native field-aligned advection inversions and beam holdouts."""
import argparse
from pathlib import Path
import subprocess
import h5py
import numpy as np
from mpi4py import MPI


def call(binary,obs,out,vx,vy,lam,fold,straight=False):
    cmd=[str(binary),'fit',str(obs),str(out),str(vx),str(vy),str(lam),str(fold)]
    if straight:cmd.append(straight if isinstance(straight,str) else 'straight')
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    print(p.stdout.strip(),flush=True)
    if p.returncode:raise RuntimeError(str(cmd))


def main():
    p=argparse.ArgumentParser();p.add_argument('binary');p.add_argument('root');a=p.parse_args()
    comm=MPI.COMM_WORLD
    if comm.size>16:raise ValueError('16-rank limit')
    root=Path(a.root);out=root/'cavity';obs=out/'observations.h5'
    configs=[(vx,vy,lam) for vx in [.05,.075,.1,.125,.15,.175,.2] for vy in [-.03,0,.03] for lam in [1.,10.,100.,1000.,10000.]]
    jobs=[(i,*cfg,fold) for i,cfg in enumerate(configs) for fold in range(3)]
    if comm.rank==0:(out/'cross_validation').mkdir(parents=True,exist_ok=True)
    comm.Barrier()
    for j in range(comm.rank,len(jobs),comm.size):
        i,vx,vy,lam,fold=jobs[j]
        call(a.binary,obs,out/'cross_validation'/f'case{i:03d}_fold{fold}.h5',vx,vy,lam,fold)
    comm.Barrier();best=None
    if comm.rank==0:
        scores=[];baselines=[]
        for i,cfg in enumerate(configs):
            diag=[]
            for fold in range(3):
                with h5py.File(out/'cross_validation'/f'case{i:03d}_fold{fold}.h5') as h:diag.append(h['diagnostics'][()])
            diag=np.array(diag);scores.append(np.sqrt(diag[:,3].sum()/diag[:,4].sum()));baselines.append(np.sqrt(diag[:,5].sum()/diag[:,4].sum()))
        best=int(np.argmin(scores))
        with h5py.File(out/'selection.h5','w') as h:
            h['config_vx_vy_lambda']=configs;h['whole_beam_holdout_rmse']=scores;h['reference_only_rmse']=baselines
            h['selected_index']=best;h['selected_config']=configs[best]
            h.attrs['units']='velocity km/s; RMSE in signal/reference ratio'
            h.attrs['selection']='three-fold heldout whole-beam groups, beam index modulo 3; all candidate configurations retained'
        print('SELECTED',best,configs[best],'RMSE',scores[best],'reference',baselines[best],flush=True)
    best=comm.bcast(best,root=0);vx,vy,lam=configs[best]
    final_jobs=[(-1,False,'all_beams'),(-1,True,'straight_B_sensitivity'),(-1,'point','point_beam_sensitivity'),(-1,'include_perp','include_perp_sensitivity'),(-1,'strict_perp','strict_perp_sensitivity'),(-1,'drift_minus25','drift_minus25'),(-1,'drift_plus25','drift_plus25')]+[(100+i,False,f'holdout_beam{i:02d}') for i in range(11)]
    for j in range(comm.rank,len(final_jobs),comm.size):
        fold,straight,name=final_jobs[j]
        delta=-.025 if straight=='drift_minus25' else .025 if straight=='drift_plus25' else 0
        call(a.binary,obs,out/(name+'.h5'),vx+delta,vy,lam,fold,False if delta else straight)
    comm.Barrier()
    if comm.rank==0:
        metrics=[]
        for i in range(11):
            with h5py.File(out/f'holdout_beam{i:02d}.h5') as h:metrics.append(h['diagnostics'][()])
        with h5py.File(out/'validation.h5','w') as h:
            h['whole_beam_holdout_diagnostics']=metrics
            h.attrs['columns']='objective, iterations, max final parameter change, heldout SSE, heldout count, reference-only SSE, training count'
        (out/'COMPLETE').write_text('all native fits and whole-beam diagnostics complete\n')

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback;traceback.print_exc();MPI.COMM_WORLD.Abort(1)
