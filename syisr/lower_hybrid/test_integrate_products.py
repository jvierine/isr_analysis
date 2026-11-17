"""Physical-signal preservation checks for the additional long-integration guard."""
import unittest
import numpy as np
from integrate_products import ambiguity_template,residual_echo_mask

class EchoScreenTests(unittest.TestCase):
    def test_localized_offcentre_echo_but_not_steady_or_common_signal(self):
        ranges=150+np.arange(73)*18.;frequency=(np.arange(256)-128)*390.625
        data=np.ones((6,73,256));template=ambiguity_template()
        data[:,25:42,96:161]+=template[None,:,:] # persistent physical feature
        clean,_=residual_echo_mask(data,ranges,frequency,ranges[33])
        self.assertFalse(clean.any())
        data[2,25:42,148:213]+=20*template
        masked,candidates=residual_echo_mask(data,ranges,frequency,ranges[33])
        self.assertTrue(masked[2].any());self.assertFalse(masked[[0,1,3,4,5]].any())
        self.assertGreater(len(candidates),0)
        common=np.ones((6,73,256));common[2,:,170:177]=100
        self.assertFalse(residual_echo_mask(common,ranges,frequency,ranges[33])[0].any())
    def test_perpendicular_central_ion_line_is_preserved(self):
        ranges=150+np.arange(73)*18.;frequency=(np.arange(256)-128)*390.625
        data=np.ones((6,73,256));data[2,25:42,96:161]+=20*ambiguity_template()
        self.assertFalse(residual_echo_mask(data,ranges,frequency,ranges[33])[0].any())

if __name__=='__main__':unittest.main()
