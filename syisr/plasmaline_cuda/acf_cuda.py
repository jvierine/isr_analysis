"""Full sample-lag code-matched ACF using batched cuFFT correlations.

A(r,l) = sum_j conj(c[j+l])*c[j]*x[r+j+l]*conj(x[r+j]).
Valid linear correlations only. Range selection follows lag-product formation.
"""
import argparse
import time
import numpy as np


class ACF:
    def __init__(self, ne, code, gain, step=32, batch=16):
        import cupy as cp
        self.cp = cp
        self.code = cp.asarray(code, dtype=cp.complex64)
        self.gain = cp.asarray(gain, dtype=cp.float32)
        self.ne, self.nt, self.step, self.batch = ne, len(code), step, batch
        self.nfft = 1 << (ne+len(code)-2).bit_length()
        self.nwhite = len(gain)
        if ne < len(code) or step < 1:
            raise ValueError('Invalid gate dimensions')

    def estimate(self, voltage):
        cp = self.cp
        echo = cp.fft.ifft(cp.fft.fft(cp.asarray(voltage, dtype=cp.complex64),
                                   n=self.nwhite)*self.gain)[:self.ne]
        result = cp.empty(((self.ne-self.nt)//self.step+1, self.nt), cp.complex64)
        for first in range(0, self.nt, self.batch):
            lags = cp.arange(first, min(self.nt, first+self.batch))[:, None]
            sample = cp.arange(self.ne)[None, :]
            partner = sample+lags
            products = cp.where(partner < self.ne,
                                echo[cp.minimum(partner, self.ne-1)]*echo.conj()[sample], 0)
            j = cp.arange(self.nt)[None, :]
            cj = j+lags
            template = cp.where(cj < self.nt,
                                self.code[cp.minimum(cj,self.nt-1)].conj()*self.code[j], 0)
            # Correlation through zero-padded FFTs. Negative lags do not wrap
            # into the retained valid range. Normalize by contributing pairs.
            spectrum = cp.fft.fft(products, n=self.nfft, axis=1)
            spectrum *= cp.fft.fft(template, n=self.nfft, axis=1).conj()
            decoded = cp.fft.ifft(spectrum, axis=1)
            result[:, first:first+len(lags)] = (
                decoded[:, :self.ne-self.nt+1:self.step]/(self.nt-lags)).T
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples', type=int, default=149436)
    p.add_argument('--code-samples', type=int, default=5120)
    p.add_argument('--range-step-samples', type=int, default=32)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    import cupy as cp
    rng = np.random.default_rng(42)
    # Verification uses nontrivial colored whitening and an independently
    # evaluated direct lag-product sum at every lag of a small fixture.
    n, nt = 73, 13
    code = rng.choice([-1,1], nt).astype('f4')
    z = (rng.normal(size=n)+1j*rng.normal(size=n)).astype('c8')
    gain = np.linspace(.5,1.5,128).astype('f4')
    engine = ACF(n,code,gain,step=3,batch=4)
    actual = cp.asnumpy(engine.estimate(z))
    white = np.fft.ifft(np.fft.fft(z,128)*gain)[:n]
    expected = np.array([[np.sum(code[l:]*code[:nt-l] *
                          white[r+l:r+nt]*white[r:r+nt-l].conj())/(nt-l)
                          for l in range(nt)] for r in range(0,n-nt+1,3)])
    np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-5)
    print('PASS direct complex ACF, all lags, whitening, valid ranges',flush=True)
    nt = a.code_samples
    code = rng.choice([-1,1],32).repeat((nt+31)//32)[:nt]
    z = (rng.normal(size=a.samples)+1j*rng.normal(size=a.samples)).astype('c8')
    nw = 1 << (a.samples-1).bit_length()
    engine = ACF(a.samples,code,np.ones(nw),a.range_step_samples,a.batch)
    engine.estimate(z)
    cp.cuda.Stream.null.synchronize()
    begin = time.perf_counter()
    for _ in range(a.repeats):
        engine.estimate(z)
    cp.cuda.Stream.null.synchronize()
    seconds = (time.perf_counter()-begin)/a.repeats
    print(f'{seconds:.6f} s/IPP, {1/seconds:.3f} IPP/s, '
          f'{.01/seconds:.6f} recording-seconds/wall-second/GPU',flush=True)


if __name__ == '__main__':
    main()
