"""Read-back numerical/I/O invariants for published pilot HDF5 products."""
import argparse
import json
import h5py
import numpy as np

p=argparse.ArgumentParser();p.add_argument('diagnostic');p.add_argument('--matched');a=p.parse_args()
with h5py.File(a.diagnostic) as f:
    n=len(f['frequency_offset_hz']);count=np.zeros_like(f['accepted_counts'][:],dtype='u4')
    for packed in f['accepted_cell_mask_packed']:
        count+=np.unpackbits(packed,axis=-1)[:,:n]
    np.testing.assert_array_equal(count,f['accepted_counts'][:])
    np.testing.assert_allclose(f['signed_excess_power_adu2'][:],f['accepted_mean_power_adu2'][:]-f['passband_reference_adu2'][:],equal_nan=True)
    t=f['pulse_time_utc_s'][:];assert np.all(np.diff(t)>0)
    assert np.all((t>=f.attrs['integration_start_utc_s'])&(t<f.attrs['integration_stop_utc_s']))
    assert np.all(np.diff(f['frequency_offset_hz'][:])>0)
    assert np.all(np.diff(f['apparent_slant_range_km'][:])>0)
    assert np.all(np.isfinite(f['raw_mean_power_adu2'][:]))
    if 'receiver_center_hz' in f.attrs:
        center=f.attrs['receiver_center_hz'];assert center in (436e6,444e6)
        assert np.all(430e6+f['raw_headers'][:,34].astype(float)*.2e6==center),'mixed receiver channels'
        np.testing.assert_array_equal(f['rf_frequency_hz'][:],center+f['frequency_offset_hz'][:])
        assert len(t)<=3001
    summary=dict(pulses=len(t),first_utc_s=t[0],last_utc_s=t[-1],
        pulse_acceptance=float(np.mean(f['pulse_accepted'][:])),frequency_acceptance=float(np.mean(f['frequency_accepted'][:])),
        negative_excess_cells=int(np.sum(f['signed_excess_power_adu2'][:]<0)),
        phase_min=int(np.min(f['pulse_phase'][:])),phase_max=int(np.max(f['pulse_phase'][:])))
    assert summary['negative_excess_cells']>0,'negative excess estimates unexpectedly absent'
    if a.matched:
        with h5py.File(a.matched) as m:
            if 'receiver_center_hz' in f.attrs:
                assert m.attrs['receiver_center_hz']==center
                np.testing.assert_array_equal(m['rf_frequency_hz'][:],center+m['frequency_offset_hz'][:])
                np.testing.assert_array_equal(m['frequency_relative_to_transmit_hz'][:],m['rf_frequency_hz'][:]-430e6)
            assert np.sum(m['pulse_accepted'][:])==m.attrs['accepted_pulses']==np.sum(m['accepted_phase_counts'][:])
            np.testing.assert_array_equal(m['pulse_time_utc_s'][:],t)
            x=m['whitened_matched_power_ratio'][:];valid=m['range_valid'][:]
            assert np.all(np.isfinite(x[valid])) and np.all(np.isnan(x[~valid]))
            assert np.all(x[valid]>=0)
            np.testing.assert_allclose(m['signed_excess_over_unit_noise'][:],x-1,rtol=2e-6,atol=2e-7,equal_nan=True)
            summary.update(matched_accepted=int(m.attrs['accepted_pulses']),matched_median=float(np.median(x[valid])),matched_maximum=float(np.max(x[valid])))
print('PASS: HDF5 read-back, exposure masks, signed excess, UTC ordering, coordinates; '+json.dumps(summary))
