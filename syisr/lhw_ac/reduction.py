"""ACF-domain range averaging; even widths retain their half-gate centre."""
import numpy as np


def range_mean(values, width):
    values = np.asarray(values)
    assert 1 <= width <= len(values)
    n = len(values) - width + 1
    return sum(values[k:k+n] for k in range(width)) / width


def triangular_spectrum(acf, dt=1e-5, nfft=512):
    acf = np.asarray(acf)
    nl = acf.shape[-1]
    assert nfft > 2*nl
    a = np.zeros((*acf.shape[:-1], nfft), complex)
    a[..., 0] = acf[..., 0].real
    for lag in range(1, nl):
        v = acf[..., lag]*(1-lag/nl)
        a[..., lag] = v
        a[..., -lag] = v.conj()
    return np.fft.fftshift(np.fft.fft(a, axis=-1).real, axes=-1)*dt


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
