"""Select ten closest independently detected passes per beam and audit their delays."""
import sys,os
from pathlib import Path
import h5py,numpy as np

root=Path(sys.argv[1]);rng=np.random.default_rng(20261119);allrows=[];stats=[]
with h5py.File(root/'calibration.h5','w') as out:
    for b,target in enumerate([150,200,250,300,350]):
        with h5py.File(root/f'voltage-beam-{target}.h5') as h:
            s=h['summary'][:]
            good=np.isfinite(s[:,4])&(s[:,6]>=5)&(s[:,5]<15)&(s[:,7]>100)
            chosen=s[good][:10]
            assert len(chosen)==10,(target,'insufficient confirmed crossings',len(chosen))
            d=chosen[:,4];median=np.median(d);mad=1.4826*np.median(abs(d-median));ci=np.quantile(np.median(rng.choice(d,(10000,10)),axis=1),[.025,.975])
            stats.append([target,median,mad,*ci]);allrows.extend(np.c_[np.full(10,target),chosen])
            print('BEAM',target,'ten closest confirmed: median us',median,'between-pass MAD',mad,'bootstrap95',ci,flush=True)
            # Preserve the actual data and full diagnostics for all selected passes.
            dst=out.create_group(f'beam_{target}')
            for rank in chosen[:,0]:h.copy(f'pass_{int(rank):03d}',dst)
            best=chosen[np.argmax(chosen[:,6]*np.minimum(chosen[:,7],1e5))];g=h[f'pass_{int(best[0]):03d}']
            t=g['utc_s'][:]-g.attrs['closest_utc_s'];lags=g['recorded_leading_edge_delay_us'][:];pred=g['predicted_range_km'][:]/.149896229;power=g['matched_filter_power_to_median'][:]
            grid=np.arange(-100.,805.,5.);mapped=np.array([np.interp(pred[i]+grid,lags,power[i],left=np.nan,right=np.nan) for i in range(len(t))])
            dst['example_time_from_closest_s']=t;dst['example_excess_delay_us']=grid;dst['example_matched_power_ratio']=mapped
            dst.attrs['example_norad_id']=g.attrs['norad_id'];dst.attrs['example_angle_deg']=g.attrs['closest_angle_deg']
    out['selected_passes']=allrows;out['beam_statistics']=stats
    d=np.array(allrows)[:,5];med=np.median(d);mad=1.4826*np.median(abs(d-med));print('ALL 50',med,'MAD',mad,'equivalent range km',med*.149896229,flush=True)
    out.attrs.update(result='50 independently code- and Doppler-confirmed crossings; ten closest accepted per beam',effective_delay_us=med,between_pass_MAD_us=mad,range_bias_km=med*.149896229,uncertainty_scope='Bootstrap intervals describe pass scatter only, not full ephemeris or waveform systematic uncertainty',selected_columns='nominal_beam_km,rank,norad_id,utc_s,separation_deg,excess_delay_us,delay_MAD_us,accepted_pulses,max_power_to_median',statistics_columns='nominal_beam_km,median_delay_us,between_pass_MAD_us,bootstrap95_low_us,bootstrap95_high_us',git_commit=os.environ.get('SOURCE_GIT_COMMIT','working'))
