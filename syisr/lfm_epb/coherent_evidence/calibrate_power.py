"""Conditional L2-anchored apparent ISR density, explicitly not absolute proof.
No density upper cutoff enters calibration. Beam9 gain is transferred from
nonperpendicular controls; the transfer and unmagnetized power law are tested
assumptions, not independently measured hardware calibration.
"""
import argparse
from pathlib import Path
from datetime import datetime,timezone
import h5py,numpy as np
from scipy.interpolate import RegularGridInterpolator

def nmf2_from_fof2(f_mhz):
    return 1.2404426e10*np.asarray(f_mhz)**2

def infer_ne(signal,range_km,te_ti,coefficient):
    return np.asarray(signal)*np.asarray(range_km)**2*(1+np.asarray(te_ti))*coefficient

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('geometry');a=p.parse_args();root=a.root
    l2=h5py.File(root/'l2_audit.h5');iri=h5py.File(root/'iri_assumption.h5');geometry=h5py.File(a.geometry)
    times=l2['time_unix_s_provisional_UTC'][()];pars=l2['Fittedparameter'][()];quality=l2['candidate_quality_mask'][()];geo=l2['GEO'][()];dirs=l2['Beamdir'][()]
    results={};samples={};controls=[]
    for path in sorted((root/'mean_power').glob('*.h5')):
        with h5py.File(path) as h:
            name=path.stem;z=h['altitude_km'][()];r=h['range_km'][()];t=h['time_unix_s'][()];signal=h['mean_signal'][()];med=h['median_signal'][()];noise=h['noise_power'][()];para=h['para'][()]
        ref=iri[name];coords=ref['geodetic_lat_lon_alt'][()];it=iri['time_unix_s'][()];vals=ref['ne_te_ti'][()]
        ratio=RegularGridInterpolator((it,coords[:,2]),vals[:,:,1]/vals[:,:,2],bounds_error=False,fill_value=None)
        tt,zz=np.meshgrid(t,z,indexing='ij');tau=ratio(np.column_stack([tt.ravel(),zz.ravel()])).reshape(signal.shape)
        neiri=RegularGridInterpolator((it,coords[:,2]),vals[:,:,0],bounds_error=False,fill_value=None)(np.column_stack([tt.ravel(),zz.ravel()])).reshape(signal.shape)
        angular=np.hypot((dirs[:,0]-para[6]+180)%360-180,dirs[:,1]-para[7]);b=int(np.argmin(angular));ks=[];matches=[]
        if angular[b]<.05:
            for j in np.flatnonzero((times[b]>=1772362800)&(times[b]<1772364600)):
                k=int(np.argmin(abs(t-times[b,j])))
                if abs(t[k]-times[b,j])>60:continue
                good=quality[b,j]&(geo[b,j,:,2]>=250)&(geo[b,j,:,2]<=500)
                heights=geo[b,j,:,2][good];density=pars[b,j,:,0][good]
                order=np.argsort(heights);heights=heights[order];density=density[order]
                if len(heights)<10:continue
                lp=np.interp(z,heights,density,left=np.nan,right=np.nan)
                # F-region plateau avoids the steep lower edge and LP range smearing.
                use=(z>=300)&(z<=450)&np.isfinite(lp)&(signal[k]>.2*noise[k])&(signal[k]<4*np.maximum(med[k],0)+noise[k])
                vv=lp[use]/(signal[k,use]*r[use]**2*(1+tau[k,use]));ks.extend(vv)
                for iz,kk,nn in zip(np.flatnonzero(use),vv,lp[use]):matches.append([times[b,j],t[k],z[iz],nn,signal[k,iz],tau[k,iz],kk])
        K=np.median(ks) if ks else np.nan
        results[name]={'t':t,'z':z,'r':r,'signal':signal,'median':med,'noise':noise,'tau':tau,'neiri':neiri,'para':para,'K':K,'ks':ks}
        samples[name]=np.array(matches).reshape(-1,7)
        print('CALIBRATION',name,'matched',len(ks),'K',K,'range',np.percentile(ks,[10,90]) if ks else [],flush=True)
    zen='beam_358.0_90.0';Kzen=results[zen]['K'];assert Kzen>0
    for name,d in results.items():
        if name!='beam_358.0_60.0' and np.isfinite(d['K']):controls.append(d['K']/Kzen)
    controls=np.array(controls);nominal=Kzen/np.sin(np.deg2rad(60.))
    o=h5py.File(root/'apparent_density.h5','w');o['zenith_calibration_samples']=samples[zen];o['nonperpendicular_relative_calibration_coefficients']=controls;o['beam9_coefficient_scenarios']=[nominal/10,nominal,nominal*10]
    o.attrs['calibration_samples_columns']='L2_timestamp_UTC_provisional,LFM_timestamp_UTC,height_km,L2_Ne_m^-3,LFM_signed_mean_signal,IRI_Te_Ti,Ne_per_range2_power_temperature'
    o.attrs['clock']='L2 text provisionally interpreted as UTC; no explicit UTC metadata supplied. Two quiet profiles11:02:45,11:19:46 matched to LFM within60s. Clock remains an audit limitation.'
    o.attrs['calibration']='Median coefficient from numerical-quality-screened zenith LP L2 Ne at300..450km11:00..11:30UTC; no absolute calibration metadata supplied. No upper Ne cutoff. LFM impulsive bins excluded only by mean-vs-median and baseline SNR rules.'
    o.attrs['power_law']='P*range_km^2 = Ne/[C*(1+Te/Ti)]. IRI2020 Te/Ti. Standard unmagnetized long-wavelength ion-line law. Near-perpendicular magnetized thermal null and finite-Debye corrections not yet evaluated.'
    o.attrs['beam9_gain']='Nominal coefficient C9=Czen/sin(60deg), ideal projected aperture for a volume-filling thermal target. Separate factor10 up/down stress scenarios are assumptions, not calibrated confidence bounds. Nonperpendicular empirical control coefficients retained.'
    o.attrs['not_absolute_density']='Apparent density conditional on thermal ISR hypothesis and gain transfer. Negative inferred estimates preserved. No calibrated uncertainty or coherent-scatter identification from this product alone.'
    for name,d in results.items():
        coeff=d['K'] if np.isfinite(d['K']) and name!='beam_358.0_60.0' else Kzen/np.sin(np.deg2rad(d['para'][7]))
        apparent=infer_ne(d['signal'],d['r'][None,:],d['tau'],coeff)
        g=o.create_group(name)
        for key,val in [('time_unix_s',d['t']),('altitude_km',d['z']),('apparent_Ne_m3',apparent),('IRI_Ne_m3',d['neiri']),('IRI_Te_Ti',d['tau']),('mean_signal',d['signal']),('median_mean_ratio',d['median']/d['signal']),('signal_noise_ratio',d['signal']/d['noise'][:,None]),('calibration_samples',samples[name])]:g.create_dataset(key,data=val,compression='gzip',shuffle=True)
        g.attrs['adopted_coefficient']=coeff
        if name=='beam_358.0_60.0':
            aspect=np.interp(d['z'],geometry[name]['geodetic_lat_lon_alt'][()][:,2],geometry[name]['aspect_deviation_from_perpendicular_deg'][()])
            g['aspect_from_perpendicular_deg']=aspect
            event=(d['t'][:,None]>=1772365500)&(d['t'][:,None]<1772372700)&(d['z']>=250)&(d['z']<=650)
            persistent=event&(d['signal']/d['noise'][:,None]>100)&(d['median']/d['signal']>.5)
            g['persistent_strong_mask']=persistent
            iz=np.unravel_index(np.argmax(np.where(persistent,apparent,-np.inf)),apparent.shape)
            g['peak_index']=iz;g['persistent_summary']=[persistent.sum(),np.any(persistent,axis=1).sum(),np.min(d['t'][np.any(persistent,axis=1)]),np.max(d['t'][np.any(persistent,axis=1)]),apparent[iz],d['z'][iz[1]],aspect[iz[1]],d['median'][iz]/d['signal'][iz],(apparent/d['neiri'])[iz]]
            print('BEAM9_PERSISTENT',g['persistent_summary'][()], 'peak_UTC',datetime.fromtimestamp(d['t'][iz[0]],timezone.utc).isoformat(),flush=True)
    print('CONTROL_K9_SCENARIOS',controls,[nominal/10,nominal,nominal*10],flush=True)
    o.close();iri.close();geometry.close();l2.close()
if __name__=='__main__':main()
