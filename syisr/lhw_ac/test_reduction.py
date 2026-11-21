import unittest
import numpy as np
from reduction import range_mean, sector_weights, triangular_spectrum


class RangeTests(unittest.TestCase):
    def test_four_gate_centre_and_complex_average(self):
        r = np.arange(8.) * 3 + 100
        acf = (r[:, None] + 1j*np.arange(3)[None, :])
        np.testing.assert_allclose(range_mean(r, 4), [104.5,107.5,110.5,113.5,116.5])
        np.testing.assert_allclose(range_mean(acf, 4), range_mean(r, 4)[:, None]+1j*np.arange(3)[None, :])

    def test_fft_of_mean_preserves_signed_spectra(self):
        rng = np.random.default_rng(4)
        a = rng.normal(size=(12, 63)) + 1j*rng.normal(size=(12, 63))
        window = 1 - np.arange(63)/63
        np.testing.assert_allclose(np.fft.fft(range_mean(a, 4)*window, axis=1), range_mean(np.fft.fft(a*window, axis=1), 4), atol=1e-14)

    def test_sector_partition_and_linear_spectral_blend(self):
        aspect = np.array([[-1.,0.,1.,2.],[-2.,-1.,0.,1.],[-3.,-2.,-1.,0.]])
        weights = sector_weights(aspect)
        np.testing.assert_allclose(weights.sum(axis=0),1)
        np.testing.assert_allclose(weights[:,0],[1,0,0])
        np.testing.assert_allclose(weights[:,1],[1,0,0])
        np.testing.assert_allclose(weights[:,2],[0,1,0])
        np.testing.assert_allclose(weights[:,3],[0,0,1])
        half = sector_weights(np.array([[.5],[-.5],[-1.5]]))
        np.testing.assert_allclose(half[:,0],[.5,.5,0])
        rng = np.random.default_rng(8)
        a = rng.normal(size=(3,63))+1j*rng.normal(size=(3,63))
        a[:,0] = -2
        lhs = triangular_spectrum((a*half).sum(axis=0))
        rhs = (triangular_spectrum(a)*half).sum(axis=0)
        np.testing.assert_allclose(lhs,rhs,atol=1e-18)
        self.assertAlmostEqual(lhs.sum()*(1/(512*1e-5)), -2)


if __name__ == '__main__':
    unittest.main()
