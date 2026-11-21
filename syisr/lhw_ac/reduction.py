"""ACF-domain range averaging; even widths retain their half-gate centre."""
import numpy as np


def partial_pairs(gate, lag, first_voltage_sample):
    branches = []
    for phase, l in [(0,lag//2),(1,(lag+1)//2)]:
        if not l: continue
        baud = np.arange(32-l)
        k = gate*2+baud*2+phase
        good = k>=first_voltage_sample
        if good.any(): branches.append((baud[good],k[good],l))
    return branches


def range_mean(values, width):
    values = np.asarray(values)
    assert 1 <= width <= len(values)
    n = len(values) - width + 1
    return sum(values[k:k+n] for k in range(width)) / width


def triangular_spectrum(acf, dt=1e-5, nfft=512):
    acf = np.asarray(acf)
    nl = acf.shape[-1]
    assert nfft > 2*nl
    rows = acf.reshape(-1,nl)
    good = np.isfinite(rows)
    length = np.where(good.all(axis=1),nl,np.argmin(good,axis=1))
    result = np.full((len(rows),nfft),np.nan)
    for n in np.unique(length):
        if n<3: continue
        select = length==n
        a = np.zeros((select.sum(),nfft),complex)
        a[:,0] = rows[select,0].real
        for lag in range(1,n):
            v = rows[select,lag]*(1-lag/n)
            a[:,lag] = v
            a[:,-lag] = v.conj()
        result[select] = np.fft.fftshift(np.fft.fft(a,axis=-1).real,axes=-1)*dt
    return result.reshape(*acf.shape[:-1],nfft)


def sector_weights(aspect_deg):
    """Partition between adjacent beam centres bracketing perpendicularity.

    Outside their bracket use the closest centre. This interpolates measured
    sectors; it does not synthesize an exactly perpendicular observation.
    """
    aspect = np.asarray(aspect_deg)
    weights = np.zeros_like(aspect)
    for j in range(aspect.shape[1]):
        brackets = [b for b in range(len(aspect)-1)
                    if aspect[b,j]*aspect[b+1,j] <= 0]
        if brackets:
            b = min(brackets, key=lambda b: abs(aspect[b,j])+abs(aspect[b+1,j]))
            den = aspect[b+1,j]-aspect[b,j]
            w = aspect[b+1,j]/den if den != 0 else .5
            weights[b,j], weights[b+1,j] = w, 1-w
        else:
            weights[np.argmin(abs(aspect[:,j])),j] = 1
    return weights
