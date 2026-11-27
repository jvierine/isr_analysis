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

from matched import MatchedGPU
ne=10000;nt=320;nf=512;step=160;no=64;nw=1<<(ne-1).bit_length()
codes=np.repeat(rng.choice([-1.,1.],(64,32)).astype('f4'),10,axis=1)
qi=rng.integers(-100,100,(ne,2),dtype=np.int32)
gain=np.asarray(1+.2*np.cos(2*np.pi*np.arange(nw)/nw),dtype='f4');gain[100:105]=0
g=MatchedGPU(ne,nt,nf,step,no,codes,gain,a.device)
got=g.power(qi,17)
echo=np.fft.ifft(np.fft.fft(qi[:,1]+1j*qi[:,0],n=nw)*gain)
expect=[]
for r in range(g.nr):
    z=echo[r*step:r*step+nt]*codes[17]
    expect.append((np.abs(np.fft.fftshift(np.fft.fft(z,n=nf)))**2).reshape(no,-1).mean(axis=1)/nt)
np.testing.assert_allclose(got,expect,rtol=2e-5,atol=np.max(expect)*2e-6)
g.close()
# Ideal coded delayed tone: the correct delay must maximize the summed peak.
ne=4096;nt=320;nf=512;step=16;no=512;nw=4096;delay=640;t=np.arange(nt)
code=codes[0];tone=code*10000*np.exp(2j*np.pi*64*t/nf)
qi=np.zeros((ne,2),dtype='i4');qi[delay:delay+nt,0]=np.rint(tone.imag);qi[delay:delay+nt,1]=np.rint(tone.real)
g=MatchedGPU(ne,nt,nf,step,no,codes,np.ones(nw,dtype='f4'),a.device);got=g.power(qi,0)
assert np.unravel_index(got.argmax(),got.shape)==(delay//step,nf//2+64)
g.close();print('PASS: whiten-before-code GPU/CPU equivalence, frequency notches, delayed ideal-code tone range and frequency')
