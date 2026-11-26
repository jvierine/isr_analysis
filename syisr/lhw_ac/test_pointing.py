import argparse,unittest
import numpy as np
from pointing import phase_table_pointing,physical_pointing,add_arguments

class SteeringTests(unittest.TestCase):
 def test_campaign_angles(self):
  for cmd,bore,expected in [((14.996337890625,74.9981689453125),(0.,90.),(14.996338,74.640791)),((149.9798583984375,42.099609375),(158.3,70.),(149.879764,41.385846)),((226.4996337890625,33.59619140625),(221.9,70.),(226.553945,32.605141))]:
   np.testing.assert_allclose(phase_table_pointing(*cmd,boresight=bore),expected,atol=1e-6)
 def test_lower_hybrid(self):
  for el,new in zip([61.58935546875,60.897216796875,60.1776123046875,59.425048828125,58.64501953125],[60.86608851,60.15278131,59.41084987,58.63457364,57.82955543]):
   az,pred=phase_table_pointing(357.9949951171875,el)
   self.assertAlmostEqual(az,357.9949951171875);self.assertAlmostEqual(pred,new,places=7)
   self.assertAlmostEqual(430*np.cos(np.deg2rad(pred)),440*np.cos(np.deg2rad(el)))
 def test_identity_inverse_boresight(self):
  np.testing.assert_allclose(phase_table_pointing(20.,60.,430.,430.),[20.,60.],atol=1e-12)
  q=phase_table_pointing(20.,60.);np.testing.assert_allclose(phase_table_pointing(*q,430.,440.),[20.,60.],atol=1e-12)
  self.assertAlmostEqual(phase_table_pointing(0.,90.)[1],90.)
 def test_no_stacked_correction(self):
  p=argparse.ArgumentParser();add_arguments(p)
  with self.assertRaises(ValueError):physical_pointing(0.,60.,p.parse_args(['--elevation-offset-deg=-.72']))
  np.testing.assert_allclose(physical_pointing(0.,60.,p.parse_args(['--pointing-model','empirical-offset','--elevation-offset-deg=-.72'])),[0.,59.28])
 def test_invalid_steering(self):
  for args in [(0.,1.),(0.,-10.),(0.,60.,440.,0.)]:
   with self.assertRaises(ValueError):phase_table_pointing(*args)
if __name__=='__main__':unittest.main()
