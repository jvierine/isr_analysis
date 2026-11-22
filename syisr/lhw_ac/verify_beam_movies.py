"""Replay altitude cuts against their original, single-beam ACFs."""
from pathlib import Path
import sys
import h5py
import numpy as np

source,directory=map(Path,sys.argv[1:3])
checks=[]
with h5py.File(source) as s:
    native=s['slant_range_km'][:];geometry=s['beam_geometry'][:]
    for file in sorted(directory.glob('beam-*km.h5')):
        with h5py.File(file) as h:
            delay=float(h.attrs.get('satellite_delay_calibration_us',0.))
            r=native-delay*.149896229
            target=int(h.attrs['beam_nominal_target_altitude_km'])
            b=int(np.flatnonzero(geometry[:,0]==target)[0])
            z=h['altitude_km'][:];cuts=h['cut_altitude_km'][:];cut_r=h['cut_slant_range_km'][:]
            assert cuts[0]==100
            assert abs(cut_r[1]-geometry[b,3])<1e-7
            assert abs(cut_r[2]-cut_r[1]-900*.149896229)<1e-7
            for c,height in enumerate(cuts):
                j=int(np.flatnonzero(z==height)[0])
                np.testing.assert_array_equal(h['cut_normalized_psd_1e4_per_hz'][:,c],h['normalized_psd_1e4_per_hz'][:,j])
                for t in [0,102,204]:
                    hi=np.searchsorted(r,cut_r[c]);w=(cut_r[c]-r[hi-1])/(r[hi]-r[hi-1])
                    a=s[f'beam_{b}/decoded_acf_adu2'][t,hi-1:hi+1];a=a[...,0]+1j*a[...,1]
                    a=(1-w)*a[0]+w*a[1]
                    p=s[f'beam_{b}/normalization_received_power_adu2'][t,hi-1:hi+1];power=(1-w)*p[0]+w*p[1]
                    missing=np.flatnonzero(~np.isfinite(a));nl=int(missing[0]) if len(missing) else 63
                    v=np.zeros(512,complex);v[0]=a[0].real
                    for lag in range(1,nl):v[lag]=a[lag]*(1-lag/nl);v[-lag]=v[lag].conjugate()
                    expected=np.fft.fftshift(np.fft.fft(v).real)*1e-5/power*1e4
                    actual=h['cut_normalized_psd_1e4_per_hz'][t,c]
                    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=np.max(abs(expected))*1e-7)
                    checks.append([target,t,c,height,cut_r[c]])
with h5py.File(directory/'cut-verification.h5','w') as h:
    h['beam_frame_cut_altitude_range_checks']=checks
    h.attrs['result']='PASS: 100km, predicted perpendicular and +900us reference; each cut equals its own single-beam map row and original source ACF replay'
print('PASS individual-beam cut replay',len(checks),flush=True)
