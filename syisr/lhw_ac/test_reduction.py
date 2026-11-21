import unittest
import numpy as np
from reduction import range_mean, sector_weights, triangular_spectrum, partial_pairs
from pathlib import Path


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

    def test_partial_lags_are_not_zero_filled(self):
        a = np.ones((3,63),complex)
        a[1,12:] = np.nan
        a[2] = np.nan
        s = triangular_spectrum(a)
        self.assertTrue(np.isnan(s[2]).all())
        np.testing.assert_allclose(s[:2].sum(axis=1)/(512*1e-5),[1,1])
        self.assertGreater(s[0,256],s[1,256])

    def test_partial_pulse_preserves_point_range(self):
        codes = np.array([[1 if c=='+' else -1 for c in l.strip()] for l in
                          (Path(__file__).resolve().parent.parent/'m32-reference.bc').read_text().splitlines() if l.strip()])
        x = np.zeros((64,160))
        x[:,20:84] = np.repeat(codes,2,axis=1)
        for lag in [1,2,7,20]:
            values = []
            for gate in range(30):
                branches = partial_pairs(gate,lag,32)
                v = 0.
                for baud,k,l in branches:
                    phase = codes[:,baud]*codes[:,baud+l]
                    v += np.sum(x[:,k]*x[:,k+lag]*phase)*(2 if lag%2 else 1)/(len(branches)*64*len(baud))
                values.append(v)
            self.assertEqual(np.argmax(values),10)

    def test_partial_grid_origin_moves_indices_as_well_as_labels(self):
        codes = np.array([[1 if c=='+' else -1 for c in l.strip()] for l in
                          (Path(__file__).resolve().parent.parent/'m32-reference.bc').read_text().splitlines() if l.strip()])
        for source in [-8,0,20]:
            sample = np.arange(200)
            bit = (sample-source)//2
            good = (bit>=0)&(bit<32)
            x = np.zeros((64,200))
            x[:,good] = codes[:,bit[good]]
            for lag in [1,2,7,20]:
                values = []
                for j in range(90):
                    branches = partial_pairs(j-32,lag,32)
                    value = 0.
                    for baud,k,l in branches:
                        value += np.sum(x[:,k]*x[:,k+lag]*codes[:,baud]*codes[:,baud+l])*(2 if lag%2 else 1)/(len(branches)*64*len(baud))
                    values.append(value)
                j = int(np.argmax(values))
                self.assertEqual(j,source//2+32)
                self.assertAlmostEqual((667-640+j*20)*.149896229,(667+source*10)*.149896229)


if __name__ == '__main__':
    unittest.main()
