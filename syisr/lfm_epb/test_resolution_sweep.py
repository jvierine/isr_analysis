"""Checks range averaging precedes the nonlinear median and split support is real."""
import tempfile
from pathlib import Path
import unittest
import h5py
import numpy as np
from resolution_sweep import reduce_candidate

class SweepTests(unittest.TestCase):
    def test_linear_grouping_before_median_and_grid_support(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cube=root/'pulse_cube.h5'
            rng=np.random.default_rng(42)
            t=np.arange(0,120,.768);re=80+np.arange(4265)*.3747405725
            power=100+rng.uniform(0,10,(len(t),4264))
            # A nonlinear median example in the first three pulses/range cells.
            power[:3,:2]=[[0,100],[100,0],[100,100]]
            with h5py.File(cube,'w') as h:
                h['time_unix_s']=t;h['pulse_power']=power;h['noise_power']=np.full(len(t),20.)
                h['range_edges_km']=re;h['altitude_edges_km']=re+.05
                h.attrs['start_utc_unix_s']=0.;h.attrs['stop_utc_unix_s']=120.
            output=root/'test.h5';reduce_candidate(cube,output,15,2)
            with h5py.File(output) as h:
                self.assertEqual(h['median_signal'].shape,(8,2132))
                np.testing.assert_allclose(np.diff(h['range_edges_km'][()]),.749481145)
                self.assertEqual(h['pulse_count'][()].sum(),len(t))
                ix=np.flatnonzero(t<15)
                expected=np.median(power[ix].reshape(len(ix),2132,2).mean(axis=2)-20.,axis=0)
                np.testing.assert_allclose(h['median_signal'][0],expected)
                for parity,key in [(0,'split_a_signal'),(1,'split_b_signal')]:
                    sub=ix[ix%2==parity]
                    expected=np.median(power[sub].reshape(len(sub),2132,2).mean(axis=2)-20.,axis=0)
                    np.testing.assert_allclose(h[key][0],expected)
                self.assertTrue(np.isfinite(h.attrs['split_half_quiet_rms_snr']))

if __name__=='__main__':unittest.main()
