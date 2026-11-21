import unittest
import numpy as np
from reduction import range_mean


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


if __name__ == '__main__':
    unittest.main()
