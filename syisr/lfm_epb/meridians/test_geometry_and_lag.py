"""Independent timing/sign and constant-field interpolation invariants."""
import unittest
import numpy as np
from scipy.interpolate import RegularGridInterpolator
class Checks(unittest.TestCase):
 def test_eastward_pattern_delay_sign(self):
  t=np.arange(0,10800,30.);v=125.;dx=250000.
  shape=lambda t:np.exp(-((t-4000)/300)**2)+.6*np.exp(-((t-5100)/180)**2)
  west=shape(t);east=shape(t-dx/v);speeds=np.arange(80,181,1.)
  scores=[]
  for speed in speeds:
   good=(t>1000)&(t<6500)&(t+dx/speed<t[-1]);scores.append(np.corrcoef(west[good],np.interp(t[good]+dx/speed,t,east))[0,1])
  self.assertEqual(speeds[np.argmax(scores)],v)
 def test_common_support_morph_and_constant(self):
  x=np.arange(5.);y=np.arange(6.);f=RegularGridInterpolator((y,x),np.full((6,5),.4))
  self.assertTrue(np.allclose(f([[1.2,3.3],[2.8,.2]]),.4))
  a=np.array([.2,.4]);b=np.array([.5,.7]);self.assertTrue(np.all((.6*a+.4*b)>=a));self.assertTrue(np.all((.6*a+.4*b)<=b))
if __name__=='__main__':unittest.main()
