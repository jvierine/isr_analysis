"""ACF-domain range averaging; even widths retain their half-gate centre."""
import numpy as np


def range_mean(values, width):
    values = np.asarray(values)
    assert 1 <= width <= len(values)
    n = len(values) - width + 1
    return sum(values[k:k+n] for k in range(width)) / width
