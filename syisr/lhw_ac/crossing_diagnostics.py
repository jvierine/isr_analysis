"""Two observed satellite crossings per beam: gain, SNR and residuals.

Gain is an estimated Gaussian TX*RX power pattern using the published
2022 zenith full HPBWs (NS 1.46 deg, EW 1.07 deg), with NS scan projection.
It is not a measured gain pattern of the November 2026 acquisition.
"""
import argparse,os
from pathlib import Path
from datetime import datetime,timezone
import h5py,numpy as np
from scipy.ndimage import gaussian_filter1d
from sgp4.api import Satrec
from satellite_crossings import observe,ELS,AZ

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);q=p.parse_args()
root=q.directory
with h5py.File(root/'calibration.h5') as cal,h5py.File(root/'crossings.h5') as orbital,h5py.File(root/'crossing-diagnostics.h5','w') as out:
    crossings=orbital['crossings'][:];signscores=[[],[]]
    for target in [150,200,250,300,350]:
        for g in cal[f'beam_{target}'].values():
            if not isinstance(g,h5py.Group):continue
            good=g['accepted_pulses'][:].astype(bool);fm=g['measured_doppler_hz'][:][good];fp=-2*g['predicted_range_rate_km_s'][:][good]*430e6/299792.458
            signscores[0].extend(abs(fm-fp));signscores[1].extend(abs(-fm-fp))
    sign=1 if np.median(signscores[0])<np.median(signscores[1]) else -1
    out.attrs.update(measured_frequency_multiplier_to_physical_doppler=sign,delay_correction_us=cal.attrs['effective_delay_us'],gain_model='Estimated two-way Gaussian: one-way full NS HPBW=1.46/sin(el), EW HPBW=1.07 degrees; published 2022 widths, not a measured November 2026 pattern',gain_reference='Yue et al. 2022 doi:10.1029/2022JA030451',snr_definition='signal/noise = ln(2)*target-tracked matched power / median matched background - 1; frequency-bank maximization introduces selection bias near noise floor; strong echoes are unaffected',residual_definition='delay minus satellite prediction minus adopted 165.782 us; physical Doppler = -2 range_rate / wavelength, measured sign fixed globally across all 50 passes',git_commit=os.environ.get('SOURCE_GIT_COMMIT','working'))
    for b,target in enumerate([150,200,250,300,350]):
        groups=[(k,g) for k,g in cal[f'beam_{target}'].items() if isinstance(g,h5py.Group)]
        groups=sorted(groups,key=lambda kg:kg[1].attrs['closest_angle_deg'])
        # Closest confirmed crossings first; retain at least 7 confirmed samples.
        chosen=[(k,g) for k,g in groups if np.sum(g['accepted_pulses'][:])>=7][:2]
        assert len(chosen)==2
        for j,(key,g) in enumerate(chosen):
            t=g['utc_s'][:];t0=g.attrs['closest_utc_s'];norad=g.attrs['norad_id']
            ix=np.flatnonzero((crossings['beam_index']==b)&(crossings['norad_id']==norad)&(abs(crossings['utc_s']-t0)<.1));assert len(ix)==1
            sat=Satrec.twoline2rv(orbital['tle_line1'].asstr()[ix[0]],orbital['tle_line2'].asstr()[ix[0]])
            sep,ran,rate,az,el=observe(sat,t)
            a,e=np.deg2rad([AZ,ELS[b]]);u=np.array([np.sin(a)*np.cos(e),np.cos(a)*np.cos(e),np.sin(e)])
            ea=np.array([np.cos(a),-np.sin(a),0]);ee=np.array([-np.sin(a)*np.sin(e),-np.cos(a)*np.sin(e),np.cos(e)])
            aa,ee_sat=np.deg2rad(az),np.deg2rad(el);v=np.c_[np.sin(aa)*np.cos(ee_sat),np.cos(aa)*np.cos(ee_sat),np.sin(ee_sat)]
            x=np.rad2deg(np.arctan2(v@ea,v@u));y=np.rad2deg(np.arctan2(v@ee,v@u))
            gain=-24.082399653118497*((x/1.07)**2+(y/(1.46/np.sin(e)))**2)
            lags=g['recorded_leading_edge_delay_us'][:];plane=g['matched_filter_power_to_median'][:]
            wanted=ran/.149896229+cal.attrs['effective_delay_us']
            ratio=np.array([np.max(row[abs(lags-d)<30]) for row,d in zip(plane,wanted)])
            snr=np.maximum(np.log(2)*ratio-1,1e-3);snr_db=10*np.log10(snr)
            # Range-correct before estimating envelope peak; smooth only for this marker.
            corrected=snr*(ran/np.median(ran))**4
            smooth=gaussian_filter1d(corrected,2.)
            near=sep[:,b]<3.;peak=np.argmax(np.where(near,smooth,-np.inf));pred_peak=np.argmax(gain)
            good=g['accepted_pulses'][:].astype(bool)
            delay=g['peak_excess_delay_us'][:]-cal.attrs['effective_delay_us']
            doppler=sign*g['measured_doppler_hz'][:]+2*rate*430e6/299792.458
            delay[~good]=np.nan;doppler[~good]=np.nan
            dst=out.create_group(f'beam_{target}/crossing_{j}')
            for name,value in [('time_from_closest_s',t-t0),('predicted_gain_db',gain),('snr_db',snr_db),('delay_residual_us',delay),('doppler_residual_hz',doppler),('cross_azimuth_offset_deg',x),('elevation_offset_deg',y),('confirmed',good.astype('u1'))]:dst[name]=value
            dst.attrs.update(norad_id=norad,closest_utc_s=t0,utc_label=datetime.fromtimestamp(t0,timezone.utc).strftime('%d %b %H:%M:%S UTC'),predicted_peak_s=t[pred_peak]-t0,echo_peak_s=t[peak]-t0,echo_peak_cross_az_deg=x[peak],echo_peak_el_deg=y[peak],closest_angle_deg=g.attrs['closest_angle_deg'],beam_azimuth_deg=AZ,beam_elevation_deg=ELS[b])
            print(target,int(norad),'echo peak offset s',round(t[peak]-t[pred_peak],3),'angular at echo peak',round(x[peak],3),round(y[peak],3),flush=True)
