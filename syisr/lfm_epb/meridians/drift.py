"""Altitude-resolved EPB pattern speed from matched west/east beam profiles.
Uses signed echo ratios, native30s profiles, actual IGRF-mapped separation.
Block resampling is a conditional stability diagnostic, not plasma-velocity error.
"""
from pathlib import Path
import sys
import h5py,numpy as np
from mpi4py import MPI
ROOT=Path(sys.argv[1]);OUT=ROOT/'three_meridians';OUT.mkdir(exist_ok=True)
comm=MPI.COMM_WORLD
if comm.size>16:raise ValueError('16 rank ceiling')
SPEED=np.arange(40.,301.,2.)
PAIRS=[(8,1),(7,2),(6,3),(7,11),(11,2)]
HEIGHT=np.arange(250.,701.,50.)
names=sorted(p.name for p in (ROOT/'beams').glob('beam_*'))
def profile(beam,lo,hi):
    name=names[beam-1]
    with h5py.File(ROOT/'beams'/name/'inversion.h5') as h:
        t=h['time_unix_s'][()];z=h['altitude_km'][()];q=h['median_signal'][()];noise=h['noise_power'][()]
    with h5py.File(ROOT/'geometry.h5') as h:uv=h[name]['field_line_reference_plane_km'][()]
    ref=np.median(q[(t>=1772362800)&(t<1772364600)],axis=0)
    good=(z>=lo)&(z<hi)&(ref>.15*np.median(noise))
    if good.sum()<3:return t,np.full(len(t),np.nan),np.array([np.nan,np.nan]),0
    y=np.nanmedian(q[:,good]/ref[good],axis=1)
    return t,np.arcsinh(y),np.median(uv[good,:2],axis=0),int(good.sum())
def correlation(a,b):
    a=a-a.mean(axis=-1,keepdims=True);b=b-b.mean(axis=-1,keepdims=True)
    return np.sum(a*b,axis=-1)/np.sqrt(np.sum(a*a,axis=-1)*np.sum(b*b,axis=-1))
for j in range(comm.rank,len(HEIGHT)-1,comm.size):
    lo,hi=HEIGHT[j:j+2];curves=[];seps=[];boot=[];counts=[]
    rng=np.random.default_rng(20260301+j)
    for left,right in PAIRS:
        t,a,ua,na=profile(left,lo,hi);tb,b,ub,nb=profile(right,lo,hi);delta=ub-ua;seps.append(delta);counts.append([na,nb])
        if not np.isfinite(delta).all() or delta[0]<=0:
            curves.append(np.full(len(SPEED),np.nan));boot.append(np.full((100,len(SPEED)),np.nan));continue
        # Fixed common time support for every tested speed; no extrapolation.
        lag=delta[0]*1000/SPEED
        ix=(t>=1772365500)&(t<1772372700)&(t+lag.max()<=tb[-1])
        base=a[ix];shifted=np.array([np.interp(t[ix]+d,tb,b) for d in lag]);valid=np.isfinite(base)&np.all(np.isfinite(shifted),axis=0)
        base=base[valid];shifted=shifted[:,valid]
        if len(base)<60:
            curves.append(np.full(len(SPEED),np.nan));boot.append(np.full((100,len(SPEED)),np.nan));continue
        curves.append(correlation(np.broadcast_to(base,shifted.shape),shifted))
        # Circular ten-minute blocks preserve local autocorrelation.
        draws=[]
        for _ in range(100):
            ids=np.concatenate([(rng.integers(len(base))+np.arange(20))%len(base) for k in range((len(base)+19)//20)])[:len(base)]
            draws.append(correlation(np.broadcast_to(base[ids],shifted[:,ids].shape),shifted[:,ids]))
        boot.append(draws)
    curves=np.array(curves);boot=np.array(boot)
    # Primary three pairs have matching elevation and symmetric north/south angles.
    score=np.nanmean(curves[:3],axis=0);stability=np.nanmean(boot[:3],axis=0)
    valid=np.isfinite(score).any();best=SPEED[np.nanargmax(score)] if valid else np.nan
    samples=SPEED[np.nanargmax(stability,axis=1)] if valid else np.full(100,np.nan)
    with h5py.File(OUT/f'drift_{int(lo)}.h5','w') as h:
        for k,v in [('speed_m_s',SPEED),('pair_beam_ids',PAIRS),('pair_correlation',curves),('pair_mapped_separation_km',seps),('pair_range_gate_counts',counts),('mean_matched_pair_correlation',score),('block_selected_speed_m_s',samples),('height_bounds_km',[lo,hi]),('best_speed_m_s',best)]:h[k]=v
    print('DRIFT',lo,hi,best,'corr',np.nanmax(score),'block10-90',np.nanpercentile(samples,[10,90]),flush=True)
comm.Barrier()
if comm.rank==0:
    with h5py.File(OUT/'drift_profile.h5','w') as out:
        scores=[];best=[];bounds=[];samples=[]
        for lo in HEIGHT[:-1]:
            with h5py.File(OUT/f'drift_{int(lo)}.h5') as h:
                h.copy('/',out,name=f'altitude_{int(lo)}')
                scores.append(h['mean_matched_pair_correlation'][()]);best.append(h['best_speed_m_s'][()]);bounds.append(h['height_bounds_km'][()]);samples.append(h['block_selected_speed_m_s'][()])
        scores=np.array(scores);qualified=(np.nanmax(scores,axis=1)>.5)&(np.array(best)>SPEED[0])&(np.array(best)<SPEED[-1])
        # Only supported altitude bands can inform pooled translation.
        if not qualified.any():raise ValueError('No interior, correlated drift estimate')
        pooled=np.nanmean(scores[qualified],axis=0);speed=SPEED[np.nanargmax(pooled)]
        out['height_bounds_km']=bounds;out['best_speed_m_s']=best;out['qualified']=qualified;out['pooled_speed_m_s']=speed;out['speed_grid_m_s']=SPEED;out['pooled_score']=pooled;out['stability_quantiles_m_s']=np.nanpercentile(samples,[10,50,90],axis=1).T
        out.attrs['method']='Mean Pearson correlation of asinh signed quiet-normalized profiles for symmetric W/E pairs8-1,7-2,6-3. Actual IGRF cross-field east separation divided by positive speed shifts east profile later. Native30s integration;50km altitude bins; block-resampling10minutes100draws.'
        out.attrs['limitations']='Pattern speed, not Doppler plasma velocity. Equal-height rays can have residual mapped-upward separation; retained per-pair. Block quantiles conditional only. Pooled speed is an approximation, not proof of altitude independence.'
    print('POOLED',speed,flush=True)
