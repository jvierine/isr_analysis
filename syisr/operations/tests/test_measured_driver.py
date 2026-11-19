"""Exercise optional native SAMI2 forcing without its external source tree."""
from pathlib import Path
import subprocess
import tempfile
import unittest

HOOK = Path(__file__).resolve().parents[1] / 'sami2_measured_driver.f'


class MeasuredDriftHook(unittest.TestCase):
    def test_absent_input_and_nonperiodic_qualified_interpolation(self):
        for table in (False, True):
            expected = 15.0 if table else 7.0
            program = f'''      program check
      real vd
      vd=7.
      call measured_drift(10.,vd)
      if(abs(vd-7.).gt.1e-5) stop 1
      vd=7.
      call measured_drift(23.75,vd)
      if(abs(vd-{expected}).gt.1e-5) stop 2
      vd=7.
      call measured_drift(24.75,vd)
      if(abs(vd-7.).gt.1e-5) stop 3
      vd=7.
      call measured_drift(48.,vd)
      if(abs(vd-7.).gt.1e-5) stop 4
      end
'''
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'check.f').write_text(program)
                if table:
                    (root / 'measured-drift.inp').write_text(
                        '3\n23.5 10.0 1\n24.0 20.0 1\n25.0 50.0 0\n')
                subprocess.run(['gfortran', '-O2', '-std=legacy',
                                '-ffixed-line-length-none', 'check.f',
                                str(HOOK), '-o', 'check'], cwd=root, check=True)
                subprocess.run(['./check'], cwd=root, check=True)


if __name__ == '__main__':
    unittest.main()
