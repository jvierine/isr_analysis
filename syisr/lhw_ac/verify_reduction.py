"""Independent replay of time coadd, gate average, FFT and selected cuts."""
from pathlib import Path
import sys
import h5py
import numpy as np

root = Path(sys.argv[1])
checks = []
files = sorted((root/'blocks').glob('[0-9][0-9][0-9][0-9].h5'))
with h5py.File(root/'spectra.h5') as h:
    n = len(h['window_start_utc_s'])
    nb = int(h.attrs['integration_wall_s'])//300
    width = int(h.attrs['range_smoothing_gates'])
    delay = float(h.attrs['range_delay_correction_us'])
    ranges = h['slant_range_km'][:]
    freq = h['frequency_hz'][:]
    assert np.allclose(np.diff(h['window_start_utc_s'][:]), 300)
    assert np.allclose(h['window_end_utc_s'][:]-h['window_start_utc_s'][:], nb*300)
    with h5py.File(files[0]) as block:
        native = block['slant_range_km'][:]-delay*.149896229
    expected_ranges = np.array([np.mean(native[i:i+width]) for i in range(len(ranges))])
    np.testing.assert_allclose(ranges, expected_ranges, atol=1e-10)
    for b in range(5):
        g = h[f'beam_{b}']
        for frame in [0, n//2, n-1]:
            for gate in [20, 80, 140]:
                num = np.zeros((width, 63), complex)
                den = np.zeros(width)
                for file in files[frame:frame+nb]:
                    with h5py.File(file) as block:
                        a = block[f'beam_{b}/acf_adu2'][gate:gate+width]
                        c = block[f'beam_{b}/accepted_cycles'][gate:gate+width]
                    num += np.nan_to_num(a[...,0]+1j*a[...,1])*c[:,None]
                    den += c
                acf = np.mean(num/den[:,None], axis=0)
                actual = g['decoded_acf_adu2'][frame,gate]
                np.testing.assert_allclose(actual[...,0]+1j*actual[...,1], acf, rtol=1e-12, atol=1e-9)
                a = np.zeros(512, complex)
                a[0] = acf[0].real
                for lag in range(1,63):
                    a[lag] = acf[lag]*(1-lag/63)
                    a[-lag] = a[lag].conjugate()
                expected = np.fft.fftshift(np.fft.fft(a).real)*1e-5
                np.testing.assert_allclose(g['background_subtracted_psd_adu2_per_hz'][frame,gate], expected, rtol=1e-6, atol=max(abs(expected))*1e-7)
                checks.append([b,frame,gate])
        for j,key in enumerate(['perpendicular_cut_psd_adu2_per_hz','reference_cut_psd_adu2_per_hz']):
            target = g['cut_slant_range_km'][j]
            if j == 0:
                assert abs(target-h['beam_geometry'][b,3])<1e-9
            else:
                assert abs(target-g['cut_slant_range_km'][0]-900*.149896229)<1e-9
            hi = np.searchsorted(ranges,target)
            w = (target-ranges[hi-1])/(ranges[hi]-ranges[hi-1])
            expected = (1-w)*g['background_subtracted_psd_adu2_per_hz'][:,hi-1]+w*g['background_subtracted_psd_adu2_per_hz'][:,hi]
            np.testing.assert_allclose(g[key][:],expected)
with h5py.File(root/'reduction-verification.h5','w') as out:
    out['beam_frame_gate_checks'] = checks
    out.attrs['result'] = 'PASS: native block replay, range centres, ACF averaging before FFT, cut coordinates and reference offset'
print('PASS reduction replay',len(checks),'spectra plus all cuts',flush=True)
