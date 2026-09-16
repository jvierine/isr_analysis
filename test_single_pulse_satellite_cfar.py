import math
import numpy as np

from single_pulse_satellite_cfar import (
    FFTBank,
    Mode,
    ambiguity_power,
    cfar_peaks,
    decimation_geometry,
    integrate_and_decimate,
)


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
    # Keep both targets away from the CFAR map edges and farther apart than a
    # pulse length so each must survive as a distinct echo.
    expected = [(350, 17321.0), (600, -28654.0)]
    n = np.arange(length)
    for start, doppler in expected:
        echo[start : start + length] += 8 * template * np.exp(2j * np.pi * doppler * n / 1e6)

    nfft = 256
    power, frequencies = ambiguity_power(
        echo, template, mode.clutter_end, mode.noise0 - length + 1,
        nfft, 100_000, 1.0, FFTBank(32),
    )
    peaks, _ = cfar_peaks(power, frequencies, mode.pulse_length, 1e-8, 8)
    assert len(peaks) == 2
    got = sorted((mode.clutter_end + p["range_index"], p["doppler_hz"]) for p in peaks)
    for (got_range, got_doppler), (want_range, want_doppler) in zip(got, expected):
        assert abs(got_range - want_range) <= 1
        assert abs(got_doppler - want_doppler) < 1000


def test_integrate_and_decimate_preserves_alignment_and_complex64():
    values = np.arange(28, dtype=np.float32).astype(np.complex64)
    values += 1j * values
    got = integrate_and_decimate(values, factor=8, offset=4)
    want = np.asarray(
        [np.sum(values[4:12]), np.sum(values[12:20]), np.sum(values[20:28])],
        dtype=np.complex64,
    )
    assert got.dtype == np.complex64
    np.testing.assert_array_equal(got, want)


def test_decimation_geometry_and_range_mapping():
    mode = Mode(tx0=76, tx1=645, clutter_end=1000, noise0=7800, noise1=8371)
    offset, length, search_start, search_stop = decimation_geometry(mode, factor=8)
    assert offset == 4
    assert length == 71
    assert offset + 8 * search_start >= mode.clutter_end
    assert offset + 8 * (search_start - 1) < mode.clutter_end
    assert offset + 8 * (search_stop - 1 + length) <= mode.noise0
    raw_delay = offset + 8 * (search_start + 17)
    corrected = raw_delay - mode.tx0 - 11
    assert corrected == 8 * ((search_start + 17) - (mode.tx0 - offset) // 8) - 11


def test_decimated_ambiguity_recovers_range_and_doppler():
    rng = np.random.default_rng(20260916)
    factor = 8
    sample_rate = 1_000_000
    effective_rate = sample_rate / factor
    offset = 4
    integrated_length = 64
    raw_template = np.repeat(
        (rng.standard_normal(integrated_length) + 1j * rng.standard_normal(integrated_length))
        .astype(np.complex64),
        factor,
    )
    template = integrate_and_decimate(raw_template, factor)
    echo_raw = (
        rng.standard_normal(offset + factor * 900)
        + 1j * rng.standard_normal(offset + factor * 900)
    ).astype(np.complex64)
    expected = [(260, 12_000.0), (570, -21_000.0)]
    raw_indices = np.arange(len(raw_template), dtype=np.float32)
    for gate, doppler in expected:
        start = offset + factor * gate
        phase = np.exp(2j * np.pi * doppler * raw_indices / sample_rate).astype(np.complex64)
        echo_raw[start : start + len(raw_template)] += np.complex64(12.0) * raw_template * phase

    echo = integrate_and_decimate(echo_raw, factor, offset)
    nfft = 256
    power, frequencies = ambiguity_power(
        echo,
        template,
        100,
        750,
        nfft,
        60_000,
        noise_power=float(2 * factor),
        fft_bank=FFTBank(32),
        sample_rate_hz=effective_rate,
    )
    peaks, _ = cfar_peaks(
        power,
        frequencies,
        integrated_length,
        pfa=1e-8,
        max_echoes=8,
        training_padding=16,
    )
    assert len(peaks) == 2
    got = sorted((100 + p["range_index"], p["doppler_hz"]) for p in peaks)
    for (got_gate, got_doppler), (want_gate, want_doppler) in zip(got, expected):
        assert abs(got_gate - want_gate) <= 1
        assert abs(got_doppler - want_doppler) < 1_000
