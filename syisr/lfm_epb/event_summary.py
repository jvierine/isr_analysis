"""Conservative altitude-band diagnostics from signed median-pulse RTIs."""
import argparse
from pathlib import Path
import h5py
import numpy as np


p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args()
with h5py.File(a.input) as h:
    z=h['altitude_km'][()];t=h['time_unix_s'][()]
    signal=h['median_signal'][()];noise=h['noise_power'][()]
    hour=(t%86400)/3600+8
    quiet=(hour>=19)&(hour<19.5)
    event=(hour>=20.3)&(hour<21.1)
    bands=np.array([[250,300],[300,350],[350,400],[400,450],[450,500],[500,600],[600,700],[700,900],[900,1200],[1200,1600]])
    curves=[];ref=[];mins=[];min_time=[];ratios=[];maxs=[];starts=[];stops=[]
    for lo,hi in bands:
        v=np.median(signal[:,(z>=lo)&(z<hi)]/noise[:,None],axis=1)
        smooth=np.array([np.median(v[(t>=tj-90)&(t<tj+90)]) for tj in t])
        half_bin=np.median(np.diff(t))/2
        baseline=np.median(v[quiet]);i=np.where(event)[0][np.argmin(smooth[event])]
        low=event&(smooth<0.5*baseline)
        # Report the longest consecutive half-reference interval, not disconnected samples.
        ix=np.where(low)[0];runs=np.split(ix,np.where(np.diff(ix)>1)[0]+1)
        run=max(runs,key=len) if len(ix) else np.array([],dtype=int)
        begin=t[run[0]]-half_bin if len(run) else np.nan
        end=t[run[-1]]+half_bin if len(run) else np.nan
        curves.append(v);ref.append(baseline);mins.append(smooth[i]);min_time.append(t[i]);ratios.append(smooth[i]/baseline if baseline>0.05 else np.nan);maxs.append(np.max(smooth[event]));starts.append(begin);stops.append(end)
        print(f'{lo:4.0f}-{hi:4.0f} km: baseline SNR={baseline:.4f}; event min={smooth[i]:.4f} at {hour[i]:.4f} Beijing h; ratio={ratios[-1]:.3f}; max SNR={maxs[-1]:.4f}',flush=True)
with h5py.File(a.output,'w') as h:
    h.attrs['reference']='19:00-19:30 Beijing local time (11:00-11:30 UTC), 2026-03-01'
    h.attrs['event_search']='20:18-21:06 Beijing time (12:18-13:06 UTC)'
    h.attrs['method']='median over altitude cells of median-pulse signal/noise, then a 3-minute running median for extrema'
    h.attrs['limitation']='relative echo power, not calibrated electron density; near-floor values do not quantify complete depletion; poor baseline SNR ratios omitted'
    h.attrs['source_file']=str(Path(a.input).resolve())
    h['altitude_bands_km']=bands;h['time_unix_s']=t;h['band_signal_to_noise']=curves
    for k,v in [('reference_signal_to_noise',ref),('minimum_signal_to_noise',mins),('minimum_time_unix_s',min_time),('minimum_relative_signal',ratios),('maximum_signal_to_noise',maxs),('half_reference_start_unix_s',starts),('half_reference_end_unix_s',stops)]:h[k]=v
