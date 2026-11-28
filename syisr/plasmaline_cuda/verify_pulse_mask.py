"""Independent support tests: burst exclusion must never discard clean windows."""
import numpy as np
from pulse_mask import accepted_windows,detect,guarded_clean_runs
bad=np.zeros(1000,dtype=bool);bad[200:210]=True
good=accepted_windows(bad,1000,160,1,bin_samples=1,guard_samples=8,edge_samples=0)
assert good[32] and not good[33] and not good[217] and good[218]
for start,ok in enumerate(good):
    overlap=any(start<right+8 and start+160>left-8 for left,right in [(200,210)])
    assert bool(ok)==(not overlap)
p=np.ones(1000);p[200:210]=100.;original=p.copy();found,floor=detect(p)
np.testing.assert_array_equal(found,bad);np.testing.assert_array_equal(p,original)
assert floor==1.
rng=np.random.default_rng(42);z=rng.normal(size=1000)+1j*rng.normal(size=1000);contaminated=z.copy();contaminated[200:210]+=1e5
code=rng.choice([-1,1],160)
for start in np.flatnonzero(good)[::7]:
    np.testing.assert_array_equal(np.fft.fft(code*z[start:start+160]),np.fft.fft(code*contaminated[start:start+160]))
np.testing.assert_array_equal(guarded_clean_runs(bad,1000,1,8,0),[[0,192],[218,1000]])
print('PASS: guarded burst overlap; full-code support; unchanged clean voltage and spectra; exact clean segments')
