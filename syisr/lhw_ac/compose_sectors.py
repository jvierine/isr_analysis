"""Altitude-sector composite from already screened, range-averaged complex ACFs.

    Range registration is inherited explicitly, not inferred from spectral peaks.
"""
from pathlib import Path
from datetime import datetime
import argparse, importlib.util, hashlib, os
import h5py
import numpy as np
from scipy.optimize import brentq
from reduction import sector_weights, triangular_spectrum

p = argparse.ArgumentParser()
p.add_argument('input', type=Path)
p.add_argument('output', type=Path)
args = p.parse_args()
spec = importlib.util.spec_from_file_location('geo',Path(__file__).resolve().parent.parent/'lower_hybrid/index_recording.py')
geo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geo)
alt = np.arange(80.,401.,2.)
cuts = np.array([100.,150.,200.,250.,300.,350.])
site = geo.ecef(18.34,109.62,0.)
basis = geo.basis(18.34,109.62)
with h5py.File(args.input) as src:
    geometry = src['beam_geometry'][:]
    native_r = src['slant_range_km'][:]
    freq = src['frequency_hz'][:]
    starts = src['window_start_utc_s'][:]
    ends = src['window_end_utc_s'][:]
    lags = src['lag_s'][:]
    ranges, aspect, fwhm = [], [], []
    for row in geometry:
        aa, ee = np.deg2rad(row[1:3])
        u = np.array([np.sin(aa)*np.cos(ee),np.cos(aa)*np.cos(ee),np.sin(ee)])@basis
        r = np.array([brentq(lambda r: geo.geod(site+r*u)[2]-z,0.,800.) for z in alt])
        points = site+r[:,None]*u
        b = np.array([geo.field(x,datetime(2026,11,19,12)) for x in points])
        aspect.append(np.rad2deg(np.arcsin(np.clip((b@u)/np.linalg.norm(b,axis=1),-1,1))))
        ranges.append(r)
        fwhm.append(1.46/np.sin(ee)/np.sqrt(2))
    ranges, aspect = np.array(ranges), np.array(aspect)
    weights = sector_weights(aspect)
    assert np.all(weights>=0) and np.allclose(weights.sum(axis=0),1)
    available = (ranges>=native_r[0])&(ranges<=native_r[-1])
    weights *= available
    total_weight = weights.sum(axis=0)
    supported = total_weight>0
    weights[:,supported] /= total_weight[supported]
    acf = np.zeros((len(starts),len(alt),len(lags)),complex)
    power = np.zeros((len(starts),len(alt)))
    for b in range(5):
        hi = np.clip(np.searchsorted(native_r,ranges[b]),1,len(native_r)-1)
        w = (ranges[b]-native_r[hi-1])/(native_r[hi]-native_r[hi-1])
        v = src[f'beam_{b}/decoded_acf_adu2'][:]
        v = v[...,0]+1j*v[...,1]
        pwr = src[f'beam_{b}/normalization_received_power_adu2'][:]
        acf += np.where(weights[b][None,:,None]>0,((1-w[None,:,None])*v[:,hi-1]+w[None,:,None]*v[:,hi])*weights[b][None,:,None],0)
        power += np.where(weights[b][None,:]>0,((1-w)*pwr[:,hi-1]+w*pwr[:,hi])*weights[b],0)
    supported &= np.all(np.isfinite(acf[:,:,:3]),axis=(0,2))
    weights[:,~supported] = 0
    acf[:,~supported] = np.nan
    power[:,~supported] = np.nan
    psd = triangular_spectrum(acf)
    assert np.all(power[:,supported]>0)
    norm = psd/power[...,None]*1e4
    assert np.allclose(psd[:,supported].sum(axis=-1)*(freq[1]-freq[0]),acf[:,supported,0].real,rtol=1e-6,atol=1e-6)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(args.output,'w') as out:
        for k,v in src.attrs.items(): out.attrs[k] = v
        out.attrs['schema'] = 'syisr-lhw-sector-composite-v1'
        out.attrs['program'] = 'syisr-lhw-sector-composite'
        out.attrs['version'] = '0.2.0'
        out.attrs['git_commit'] = os.environ.get('SOURCE_GIT_COMMIT','unspecified')
        out.attrs['source_product'] = str(args.input)
        out.attrs['source_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
        out.attrs['method'] = 'WGS84 altitude regrid per recorded beam; adjacent beam-centre signed-aspect interpolation bracketing zero; closest beam outside bracket; complex ACFs and received power blended before triangular FFT and normalization; no seam gain rescaling'
        out.attrs['limitations'] = 'Observed-sector composite, not inversion or a new exact-perpendicular observation; source range registration inherited explicitly and remains provisional; code-bearing clutter does not calibrate zero delay; heights below recorded support are NaN, not extrapolated; 2km display grid does not improve four-native-gate resolution'
        out['altitude_km'] = alt
        out['beam_slant_range_km'] = ranges
        out['beam_centre_aspect_deg'] = aspect
        out['beam_weights'] = weights
        out['has_recorded_sample_support'] = supported
        out['beam_has_recorded_sample_support'] = available
        out['estimated_two_way_full_FWHM_deg'] = fwhm
        out['outside_perpendicular_centre_bracket'] = np.all(aspect>0,axis=0)|np.all(aspect<0,axis=0)
        out['perpendicular_direction_within_estimated_half_power_beam'] = np.any(abs(aspect)<=np.asarray(fwhm)[:,None]/2,axis=0)
        out['frequency_hz'] = freq
        out['window_start_utc_s'] = starts
        out['window_end_utc_s'] = ends
        out['lag_s'] = lags
        out['cut_altitude_km'] = cuts
        for key,value in [('decoded_acf_adu2',np.stack([acf.real,acf.imag],axis=-1)),('signed_psd_adu2_per_hz',psd),('normalization_received_power_adu2',power),('normalized_psd_1e4_per_hz',norm)]:
            out.create_dataset(key,data=value,compression='gzip',compression_opts=2)
        ix = np.searchsorted(alt,cuts)
        out['cut_normalized_psd_1e4_per_hz'] = norm[:,ix,:]
        values = norm[:,ix,:]
        lo = np.min(values,axis=(1,2));hi = np.max(values,axis=(1,2));span = np.maximum(hi-lo,1e-20)
        out['cut_linear_limits'] = np.stack([np.minimum(0,lo)-span*.05,hi+span*.1],axis=-1)
        positive = values[values>0]
        out['cut_log_limits'] = [10**np.floor(np.log10(np.quantile(positive,.02))),10**np.ceil(np.log10(positive.max()*1.01))]
print('PASS composite',acf.shape,flush=True)
