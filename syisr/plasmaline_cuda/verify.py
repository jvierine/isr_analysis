"""GPU/CPU numerical invariants, including frequency sign and signed Q,I."""
import argparse
import numpy as np
from process import GPU

p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);a=p.parse_args()
n=1024;b=5;rng=np.random.default_rng(36)
x=rng.integers(-10000,10000,(b,n,2),dtype=np.int32)
t=np.arange(n);z=np.rint(5000*np.exp(2j*np.pi*77*t/n))
x[0,:,0]=z.imag;x[0,:,1]=z.real
x[1]=-x[0];x[2]=0
gpu=GPU(n,b,a.device);got=gpu.power(x)
w=np.hanning(n);v=x[:,:,1]+1j*x[:,:,0]
want=np.abs(np.fft.fftshift(np.fft.fft(v*w,axis=-1),axes=-1))**2/np.sum(w*w)
np.testing.assert_allclose(got,want,rtol=3e-5,atol=want.max()*2e-7)
assert got[0].argmax()==n//2+77
np.testing.assert_array_equal(got[0],got[1]);assert np.all(got[2]==0)
np.testing.assert_allclose(got.sum(axis=1),n*np.sum(np.abs(v*w)**2,axis=1)/np.sum(w*w),rtol=2e-6)
gpu.close();print('PASS: CPU/GPU Hann spectra, positive tone sign, signed QI, zero input, Parseval')
