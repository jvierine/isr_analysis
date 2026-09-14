import math
from types import SimpleNamespace

import numpy as np

from single_pulse_satellite_cfar import FFTBank, Mode, ambiguity_power, cfar_peaks


def test_two_nonoverlapping_echoes():
    rng = np.random.default_rng(20260914)
    length = 64
    mode = Mode(tx0=10, tx1=10 + length, clutter_end=100, noise0=900, noise1=980)
    template = (
        rng.choice(np.asarray([-1, 1], dtype=np.float32), length)
        + 1j * rng.choice(np.asarray([-1, 1], dtype=np.float32), length)
    ).astype(np.complex64)
    echo = (
        rng.normal(0, 1 / math.sqrt(2), 1000)
        + 1j * rng.normal(0, 1 / math.sqrt(2), 1000)
    ).astype(np.complex64)
    expected = [(280, 17321.0), (610, -28654.0)]
    n = np.arange(length)
    for start, doppler in expected:
        echo[start : start + length] += 8 * template * np.exp(2j * np.pi * doppler * n / 1e6)

    nfft = 256
    power, frequencies = ambiguity_power(
        echo, template, mode.clutter_end, mode.noise0 - length + 1,
        nfft, 100_000, 1.0, FFTBank(32),
    )
    peaks, _ = cfar_peaks(power, frequencies, mode, 1e-8, 8)
    assert len(peaks) == 2
    got = sorted((mode.clutter_end + p["range_index"], p["doppler_hz"]) for p in peaks)
    for (got_range, got_doppler), (want_range, want_doppler) in zip(got, expected):
        assert abs(got_range - want_range) <= 1
        assert abs(got_doppler - want_doppler) < 1000
