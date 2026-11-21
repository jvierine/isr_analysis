"""Independent source replay of representative stitched altitude spectra."""
from pathlib import Path
import sys
import h5py
import numpy as np

source, product = map(Path, sys.argv[1:3])
rows = []
with h5py.File(source) as s, h5py.File(product) as h:
    r = s['slant_range_km'][:]
    z = h['altitude_km'][:]
    weights = h['beam_weights'][:]
    assert z[0] == 80 and z[-1] == 400
    assert np.all(weights>=0) and np.allclose(weights.sum(axis=0),1)
    assert np.max(np.count_nonzero(weights,axis=0))<=2
    np.testing.assert_array_equal(h['window_start_utc_s'][:],s['window_start_utc_s'][:])
    np.testing.assert_array_equal(h['window_end_utc_s'][:],s['window_end_utc_s'][:])
    for t in [0,len(h['window_start_utc_s'])//2,len(h['window_start_utc_s'])-1]:
        for j in [0,35,60,85,len(z)-1]:
            acf = np.zeros(63,complex)
            power = 0.
            for b in np.flatnonzero(weights[:,j]):
                target = h['beam_slant_range_km'][b,j]
                hi = np.searchsorted(r,target)
                f = (target-r[hi-1])/(r[hi]-r[hi-1])
                a = s[f'beam_{b}/decoded_acf_adu2'][t,hi-1:hi+1]
                p = s[f'beam_{b}/normalization_received_power_adu2'][t,hi-1:hi+1]
                a = a[...,0]+1j*a[...,1]
                acf += weights[b,j]*((1-f)*a[0]+f*a[1])
                power += weights[b,j]*((1-f)*p[0]+f*p[1])
            a = np.zeros(512,complex)
            a[0] = acf[0].real
            for lag in range(1,63):
                a[lag] = acf[lag]*(1-lag/63)
                a[-lag] = a[lag].conjugate()
            expected = np.fft.fftshift(np.fft.fft(a).real)*1e-5/power*1e4
            np.testing.assert_allclose(h['normalized_psd_1e4_per_hz'][t,j],expected,rtol=1e-10,atol=1e-12)
            rows.append([t,j])
    for c,alt in enumerate(h['cut_altitude_km'][:]):
        j = np.flatnonzero(z==alt)[0]
        np.testing.assert_array_equal(h['cut_normalized_psd_1e4_per_hz'][:,c],h['normalized_psd_1e4_per_hz'][:,j])
    assert np.any(h['normalized_psd_1e4_per_hz'][:]<0), 'signed noise estimates lost'
with h5py.File(product.with_name('verification.h5'),'w') as h:
    h['frame_height_index_checks'] = rows
    h.attrs['result'] = 'PASS: source ACF replay, positive-power normalization, UTC conservation, altitude cuts, partition weights and signed estimates'
print('PASS composite source replay and altitude cuts',flush=True)
