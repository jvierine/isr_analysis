"""Numerical/I/O invariants for RTI integration, missing support and duplicates."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np

class CombineTests(unittest.TestCase):
    def test_signed_mean_gap_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'parts').mkdir()
            f=root/'parts'/'test.h5'
            with h5py.File(f,'w') as h:
                h.attrs['source_file']='synthetic'
                h['times_unix_s']=[1772366401.,1772366430.]
                h['range_km']=[100.,103.];h['altitude_km']=[100.05,103.05]
                h['range_edges_km']=[98.5,101.5,104.5]
                h['altitude_edges_km']=[98.55,101.55,104.55]
                h['noise_power']=[3.,3.]
                h['downchirp_power']=[[2.,5.],[0.,7.]]
                h['upchirp_power']=[[2.,5.],[0.,7.]]
                h['downchirp_signal']=[[-1.,2.],[-3.,4.]]
            cmd=[sys.executable,str(Path(__file__).with_name('combine.py')),d,
                 '--bin-seconds','60','--start','2026-03-01T12:00:00Z','--stop','2026-03-01T12:02:00Z']
            subprocess.run(cmd,check=True,capture_output=True)
            with h5py.File(root/'zenith_rti.h5') as h:
                np.testing.assert_equal(h['pulse_count'][()],[2,0])
                np.testing.assert_equal(h['downchirp_signal'][0],[-2.,3.])
                self.assertTrue(np.isnan(h['downchirp_signal'][1]).all())
                np.testing.assert_equal(h['matched_signal'][0],[-2.,3.])
                self.assertTrue(np.isnan(h['relative_signal_db'][0,0]))
            (root/'parts'/'duplicate.h5').write_bytes(f.read_bytes())
            failed=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertIn('Duplicate pulse',failed.stderr)

if __name__=='__main__':unittest.main()
