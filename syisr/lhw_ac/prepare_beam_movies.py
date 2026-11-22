"""Unstitched beam products with exact geometric altitude cuts."""
from pathlib import Path
from datetime import datetime
import sys,importlib.util,hashlib,os,argparse
import numpy as np
import h5py
from scipy.optimize import brentq
from reduction import triangular_spectrum

p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);p.add_argument('--satellite-calibration',type=Path)
q=p.parse_args();source,destination=q.source,q.destination
delay=0.
if q.satellite_calibration:
    with h5py.File(q.satellite_calibration) as cal:
        assert len(cal['selected_passes'])==50
        delay=float(cal.attrs['effective_delay_us'])
destination.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('geo',Path(__file__).resolve().parent.parent/'lower_hybrid/index_recording.py')
geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
site=geo.ecef(18.34,109.62,0);basis=geo.basis(18.34,109.62)
digest=hashlib.sha256(source.read_bytes()).hexdigest()
with h5py.File(source) as h:
    native=h['slant_range_km'][:]-delay*.149896229;freq=h['frequency_hz'][:];geometry=h['beam_geometry'][:]
    for b,row in enumerate(geometry):
        target=int(row[0]);aa,ee=np.deg2rad(row[1:3])
        u=np.array([np.sin(aa)*np.cos(ee),np.cos(aa)*np.cos(ee),np.sin(ee)])@basis
        perp_range=row[3];perp=geo.geod(site+perp_range*u)[2]
        field=geo.field(site+perp_range*u,datetime(2026,11,19,12))
        assert abs(np.dot(u,field)/np.linalg.norm(field))<1e-8
        ref_range=perp_range+900*.149896229;reference=geo.geod(site+ref_range*u)[2]
        cuts=np.array([100.,perp,reference])
        z=np.unique(np.r_[np.arange(80.,501.,2.),cuts])
        ranges=np.array([brentq(lambda r:geo.geod(site+r*u)[2]-height,0,900) for height in z])
        hi=np.searchsorted(native,ranges);assert np.all((hi>0)&(hi<len(native)))
        w=(ranges-native[hi-1])/(native[hi]-native[hi-1])
        a=h[f'beam_{b}/decoded_acf_adu2'][:];a=a[...,0]+1j*a[...,1]
        p=h[f'beam_{b}/normalization_received_power_adu2'][:]
        a=a[:,hi-1]*(1-w[None,:,None])+a[:,hi]*w[None,:,None]
        p=p[:,hi-1]*(1-w)+p[:,hi]*w
        norm=triangular_spectrum(a)/p[...,None]*1e4
        ci=np.array([np.flatnonzero(z==height)[0] for height in cuts])
        values=norm[:,ci]
        low=np.nanmin(values,axis=(1,2));high=np.nanmax(values,axis=(1,2));span=np.maximum(high-low,1e-20)
        positive=values[np.isfinite(values)&(values>0)]
        coords=np.array([geo.geod(site+r*u) for r in ranges])
        enu=np.asarray(geo.ppigrf.igrf(coords[:,1],coords[:,0],coords[:,2],datetime(2026,11,19,12))).reshape(3,-1).T
        fields=np.einsum('ni,nij->nj',enu,np.array([geo.basis(lat,lon) for lat,lon in coords[:,:2]]))
        aspect=np.rad2deg(np.arcsin((fields@u)/np.linalg.norm(fields,axis=1)))
        with h5py.File(destination/f'beam-{target}km.h5','w') as out:
            for key,val in h.attrs.items():out.attrs[key]=val
            out.attrs.update(schema='syisr-lhw-individual-beam-v1',program='syisr-lhw-beam-view',version='0.1.0',git_commit=os.environ['SOURCE_GIT_COMMIT'],source_product=str(source),source_sha256=digest,beam_nominal_target_altitude_km=target,recorded_azimuth_deg=row[1],recorded_elevation_deg=row[2],method='One recorded beam only; WGS84 altitude conversion; ACF interpolation before gate-dependent triangular FFT; predicted perpendicular cut from k dot IGRF B=0, never fitted to a spectral feature')
            out.attrs['satellite_delay_calibration_us']=delay
            if q.satellite_calibration:
                out.attrs['satellite_calibration_sha256']=hashlib.sha256(q.satellite_calibration.read_bytes()).hexdigest()
                out.attrs['range_calibration']='Subtract measured satellite excess two-way delay from recorded gate delays before WGS84 altitude conversion and cut selection'
            for key in ['window_start_utc_s','window_end_utc_s','frequency_hz','lag_s']:h.copy(key,out)
            out['altitude_km']=z;out['beam_slant_range_km']=ranges;out['beam_centre_aspect_deg']=aspect
            out['cut_altitude_km']=cuts;out['cut_slant_range_km']=ranges[ci]
            out['cut_recorded_source_sample_offset']=(ranges[ci]/.149896229+delay-667)/10
            out['cut_linear_limits']=np.stack([np.minimum(0,low)-span*.05,high+span*.1],axis=-1)
            out['cut_log_limits']=[10**np.floor(np.log10(np.quantile(positive,.02))),10**np.ceil(np.log10(positive.max()*1.01))]
            out.create_dataset('normalized_psd_1e4_per_hz',data=norm.astype('f4'),compression='gzip',compression_opts=2)
            out.create_dataset('cut_normalized_psd_1e4_per_hz',data=values.astype('f4'),compression='gzip',compression_opts=2)
            out['normalization_received_power_adu2']=p
            good=np.isfinite(a);length=np.where(good.all(axis=-1),63,np.argmin(good,axis=-1))
            out['maximum_used_lag_s']=np.maximum(0,length-1)*1e-5
            out['cut_maximum_used_lag_s']=np.maximum(0,length[:,ci]-1)*1e-5
        print('BEAM VIEW',target,'cut altitudes',cuts,'cut ranges',ranges[ci],flush=True)
