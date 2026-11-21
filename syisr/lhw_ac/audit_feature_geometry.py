"""Noncircular cross-beam test of a feature-based range registration.

Uses relative two-sided spectral-line power at common altitude, with a free
common altitude profile. This is a beam-model diagnostic, not ground timing.
"""
from pathlib import Path
from datetime import datetime
import sys,importlib.util
import numpy as np
import h5py
from scipy.optimize import brentq

root = Path(sys.argv[1])
module = Path(__file__).resolve().parent.parent/'lower_hybrid/index_recording.py'
spec = importlib.util.spec_from_file_location('geo',module)
geo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geo)
z = np.arange(120.,391.,5.)
site = geo.ecef(18.34,109.62,0)
basis = geo.basis(18.34,109.62)
with h5py.File(root/'spectra.h5') as src:
    f = src['frequency_hz'][:]
    r = src['slant_range_km'][:]
    geometry = src['beam_geometry'][:]
    select = (abs(f)>=4500)&(abs(f)<=13000)
    x = abs(f[select])/10000
    continuum = np.stack([np.ones(len(x)),x*x],axis=1)
    template = np.exp(-.5*((abs(f[select])-8000)/800)**2)
    template -= continuum@np.linalg.lstsq(continuum,template,rcond=None)[0]
    template /= np.dot(template,template)
    line,noise = [],[]
    for b in range(5):
        p = src[f'beam_{b}/background_subtracted_psd_adu2_per_hz'][-24:]
        a = p[:,:,select]@template
        line.append(np.mean(a,axis=0))
        noise.append(np.maximum(np.std(a,axis=0)/2,np.median(np.std(a,axis=0))*.1))
line,noise = np.array(line),np.array(noise)
rays,aspect = [],[]
for row in geometry:
    aa,ee = np.deg2rad(row[1:3])
    u = np.array([np.sin(aa)*np.cos(ee),np.cos(aa)*np.cos(ee),np.sin(ee)])@basis
    ray = np.array([brentq(lambda r:geo.geod(site+r*u)[2]-h,0,800) for h in z])
    points = site+ray[:,None]*u
    coords = np.array([geo.geod(v) for v in points])
    enu = np.asarray(geo.ppigrf.igrf(coords[:,1],coords[:,0],coords[:,2],datetime(2026,11,19,12))).reshape(3,-1).T
    b = np.einsum('ni,nij->nj',enu,np.array([geo.basis(a,o) for a,o in coords[:,:2]]))
    rays.append(ray)
    aspect.append(np.rad2deg(np.arcsin((b@u)/np.linalg.norm(b,axis=1))))
rays,aspect = np.array(rays),np.array(aspect)
delays = np.arange(-200.,901.,10.)
widths = np.arange(.5,2.51,.05)
loss = np.zeros((len(delays),len(widths)))
gain = np.sin(np.deg2rad(geometry[:,2]))[:,None]**2
for i,delay in enumerate(delays):
    query = rays+(delay-640)*.149896229
    obs = np.array([np.interp(query[b],r,line[b]) for b in range(5)])*rays**2/gain
    err = np.array([np.interp(query[b],r,noise[b]) for b in range(5)])*rays**2/gain
    wt = 1/np.maximum(err,1e-12)**2
    for j,width in enumerate(widths):
        beam = np.exp(-4*np.log(2)*(aspect/(width/np.sqrt(gain)))**2)
        common = np.maximum(0,(wt*beam*obs).sum(axis=0)/(wt*beam**2).sum(axis=0))
        loss[i,j] = np.sum(wt*(obs-beam*common)**2)/np.sum(wt*obs**2)
best = np.unravel_index(np.argmin(loss),loss.shape)
with h5py.File(root/'feature-geometry-audit.h5','w') as h:
    h['trial_delay_us'] = delays
    h['trial_full_two_way_beamwidth_deg_at_zenith'] = widths
    h['relative_cross_beam_residual'] = loss
    h['altitude_km'] = z
    h['beam_aspect_deg'] = aspect
    h['beam_slant_range_km'] = rays
    h['spectral_feature_power_adu2_per_hz'] = line
    h['spectral_feature_time_variability'] = noise
    h.attrs['scope'] = 'Late24 overlapping30min frames; symmetric8kHz feature minus local quadratic continuum; Gaussian beam and common free altitude profile; no independent instrument timing calibration'
    h.attrs['best_delay_us'] = delays[best[0]]
    h.attrs['best_width_deg'] = widths[best[1]]
print('cross-beam best delay/width/loss',delays[best[0]],widths[best[1]],loss[best],flush=True)
for d in [0,160,480,640]:
    i=np.argmin(abs(delays-d));j=np.argmin(loss[i]);print('candidate',delays[i],'width',widths[j],'loss',loss[i,j],flush=True)
