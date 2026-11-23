"""Independent three-meridian bounded contrast fits, common drift/regularization.
Every map retains its own observation-time distribution. Morphing between these
maps is a conditional visualization, not three simultaneous measured volumes.
"""
from pathlib import Path
import sys,subprocess
import h5py,numpy as np
from mpi4py import MPI
ROOT=Path(sys.argv[1]);OUT=ROOT/'three_meridians';native=ROOT/'cavity/native'
comm=MPI.COMM_WORLD
if comm.size>16:raise ValueError('16rank limit')
GROUPS={'west':[8,7,6],'central':[10,11,5,4],'east':[1,2,3]}
LAMS=[10.,100.,1000.]
with h5py.File(OUT/'drift_profile.h5') as h:speed=float(h['pooled_speed_m_s'][()])/1000
if comm.rank==0:
    with h5py.File(ROOT/'cavity/observations.h5') as h:obs=h['observations'][()];eligible=h['eligible'][()];stats=h['beam_statistics'][()]
    for name,ids in GROUPS.items():
        with h5py.File(OUT/f'{name}_observations.h5','w') as h:
            h['observations']=obs;h['eligible']=eligible*np.isin(obs[:,1]+1,ids);h['beam_statistics']=stats
            h.attrs['beam_ids']=ids;h.attrs['source']=str(ROOT/'cavity/observations.h5')
comm.Barrier()
jobs=[(name,lam,b) for name,ids in GROUPS.items() for lam in LAMS for b in ids]
for j in range(comm.rank,len(jobs),comm.size):
    name,lam,b=jobs[j];path=OUT/f'cv_{name}_{lam:g}_{b}.h5'
    subprocess.run([str(native),'fit',str(OUT/f'{name}_observations.h5'),str(path),str(speed),'0',str(lam),str(99+b)],check=True,stdout=subprocess.DEVNULL)
comm.Barrier();best=None
if comm.rank==0:
    scores=np.zeros((3,3));counts=np.zeros((3,3))
    for g,(name,ids) in enumerate(GROUPS.items()):
        for j,lam in enumerate(LAMS):
            for b in ids:
                with h5py.File(OUT/f'cv_{name}_{lam:g}_{b}.h5') as h:d=h['diagnostics'][()];scores[g,j]+=d[3];counts[g,j]+=d[4]
    best=LAMS[int(np.argmin(scores.sum(axis=0)/counts.sum(axis=0)))]
    with h5py.File(OUT/'section_selection.h5','w') as h:
        h['lambda_grid']=LAMS;h['whole_beam_holdout_rmse']=np.sqrt(scores/counts);h['selected_lambda']=best;h['speed_m_s']=speed*1000
        h.attrs['groups']='west8,7,6; central10,11,5,4; east1,2,3; beam9excluded'
        h.attrs['selection']='Common penalty minimizes pooled whole-beam holdout error across three independent groups. Conditional selection diagnostic, not independent final-test error.'
    print('SECTION_LAMBDA',best,'RMSE',np.sqrt(scores/counts),flush=True)
best=comm.bcast(best,root=0)
for i,(name,ids) in enumerate(GROUPS.items()):
    if i%comm.size!=comm.rank:continue
    subprocess.run([str(native),'fit',str(OUT/f'{name}_observations.h5'),str(OUT/f'{name}_section.h5'),str(speed),'0',str(best),'-1'],check=True)
comm.Barrier()
if comm.rank==0:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from symmetry_validation import design
    p,arc,A,S,T,L,original=design(ROOT,exclude_beam=9,velocity=(speed,0.))
    out=h5py.File(OUT/'sections.h5','w');x=np.arange(261)*10-1300;y=np.arange(161)*10-800
    out['x_km']=x;out['y_km']=y;out['speed_m_s']=speed*1000
    out['reference_epoch_unix_s']=1772369100.;out['geometry_reference_ecef_km']=h5py.File(ROOT/'geometry.h5')['reference_ecef_km'][()]
    maps=[];supports=[];times=[]
    for name,ids in GROUPS.items():
        keep=np.isin(p[:,1]+1,ids);w=1/p[:,7]**2;aa=A[keep]
        support=np.asarray(aa.T@w[keep]).ravel();den=np.maximum(support,1e-30)
        mt=np.asarray(aa.T@(w[keep]*(p[keep,0]-1772369100.))).ravel()/den+1772369100.
        vt=np.asarray(aa.T@(w[keep]*(p[keep,0]-1772369100.)**2)).ravel()/den-(mt-1772369100.)**2
        with h5py.File(OUT/f'{name}_section.h5') as h:contrast=h['contrast'][()];pred=h['predicted_ratio'][()];diagn=h['diagnostics'][()]
        g=out.create_group(name);g['beam_ids']=ids;g['contrast']=contrast;g['support_weight']=support.reshape(161,261)
        g['mean_observation_time_unix_s']=mt.reshape(161,261);g['observation_time_std_s']=np.sqrt(np.maximum(vt,0)).reshape(161,261)
        g['diagnostics']=diagn;g['observed_ratio']=p[keep,6];g['predicted_ratio']=pred[original[keep]];g['observed_altitude_km']=p[keep,3];g['observed_time_unix_s']=p[keep,0]
        maps.append(contrast);supports.append(support.reshape(161,261));times.append(mt.reshape(161,261))
    # Explicit common support and time ordering govern whether morphing is defensible.
    support=np.array(supports);maps=np.array(maps);times=np.array(times)
    masks=support>.005*np.max(support,axis=(1,2))[:,None,None]
    common=masks.all(axis=0);ordered=common&(times[0]<times[1])&(times[1]<times[2])
    out['common_support']=common;out['ordered_common_support']=ordered
    comparisons=[]
    for a,b in [(0,1),(1,2),(0,2)]:
        q=masks[a]&masks[b];comparisons.append([np.sqrt(np.mean((maps[a,q]-maps[b,q])**2)),np.median(times[b,q]-times[a,q]),q.sum()])
    out['pair_difference_rms_time_separation_s_count']=comparisons
    out.attrs['field_model']='Independent bounded fractional echo contrast maps assumed uniform along each IGRF field line within each meridian. No inferred parallel length. Linear blending is visualization, not a measured continuous evolution.'
    out.attrs['support']='Measured weighted finite-beam bilinear support; displayed only above0.5percent of each map maximum. Common and time-ordered common support masks retained.'
    out.attrs['sampling']='Each map uses11:45..13:45UTC; each cell has actual mean time and spread. The three maps are not simultaneous snapshots.'
    out.close();print('COMPARISONS',comparisons,'ORDERED_COMMON',ordered.sum(),'COMMON',common.sum(),flush=True)
