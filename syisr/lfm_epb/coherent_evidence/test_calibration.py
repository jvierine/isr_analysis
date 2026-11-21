"""Numerical invariants for calibration units and preserved signed estimates."""
import unittest
import numpy as np
from calibrate_power import nmf2_from_fof2,infer_ne
class Calibration(unittest.TestCase):
 def test_frequency_density_scaling(self):
  self.assertAlmostEqual(float(nmf2_from_fof2(10)),1.2404426e12)
  self.assertAlmostEqual(float(nmf2_from_fof2(20)/nmf2_from_fof2(10)),4.)
 def test_gain_range_temperature_and_signed_power(self):
  p=np.array([-2.,0.,2.]);ne=infer_ne(p,300.,1.,10.)
  np.testing.assert_allclose(ne,[-3600000.,0.,3600000.])
  np.testing.assert_allclose(infer_ne(p,600.,1.,10.),4*ne)
  np.testing.assert_allclose(infer_ne(p,300.,3.,10.),2*ne)
if __name__=='__main__':unittest.main()
