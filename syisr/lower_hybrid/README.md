# SYISR lower-hybrid visit recordings

One range–Doppler MP4 per pointing, with signed background-subtracted power
and a second panel normalized to unit integrated signal power.

The 16 November 2026 recording contains 116,509 pulses over
11:03:59–11:23:24 UTC. The header clock is Beijing time, confirmed by the
data owner. The containing folder is named 20251116, but the filename and
headers identify 2026. Five directions are interleaved pulse by pulse.

Each frame uses 60 seconds of wall-clock data, stepping by 30 seconds.
There are 38 frames per direction; the final full window is anchored at the
recording end. Most frames contain about 1,200 pulses per beam. The output
is 1920×1080 H.264, 19 seconds per movie, with two scientific updates per
second encoded at 24 fps. Color limits are fixed across all times and beams.

## Spectra and ranges

The signed int32 samples are decoded as I+iQ, following the supplied MATLAB
reader. Each 480 µs uncoded pulse is analyzed independently: 48 samples at
100 kHz, a symmetric Hann taper, and a 256-point forward FFT. Frequency
spacing after zero padding is 390.625 Hz; the actual Hann noise-equivalent
bandwidth is about 3.19 kHz. A slow-time FFT between pulses is not used.
The two-sided PSD is |FFT(w x)|² / (fs sum(w²)), in receiver ADU²/Hz.
No absolute calibration or physical toward/away velocity sign is claimed.

Receiver windows overlap by 75%, giving 18 km gate spacing. The transmitted
pulse and receiver window together have roughly 144 km full range support
and 72 km FWHM. These overlapping gates are not independent 18 km range
resolution. Range is pulse-centred monostatic slant range, including the
Hann centroid and half-pulse correction; instrumental group delay remains
uncalibrated.

The faint marker is computed from each recorded pointing, IGRF-14 on the
recording date, WGS84 straight rays, and the provisional Memo 50 site
(18.34° N, 109.62° E). The planned 250/350/450/550/650 km targets map to
actual perpendicular slant ranges 286.8/406.5/529.3/658.2/793.4 km.
The planned geometry is verified against the committed Overleaf Memo 50
product; slant range is never relabelled as altitude.

## Background and normalization

A same-beam, same-window average of quiet candidates among the eight highest
complete gates, centred at 1248–1374 km, is subtracted at every frequency.
This includes a common far-range narrow feature near −2.34 kHz. The identity
of that feature is not established by background subtraction.

Localized positive excesses are excluded from the reference estimate using
integrated-power and per-frequency relative-excess guards. A localized event
contaminated one reference gate in two overlapping windows of the highest
beam. The event remains in the measured spectra. Exact gate-use masks and
the unvetted reference are retained. Quiet top-four and top-twelve reference
controls change the core-range spectrum by at most 0.13% of its peak.
The overlapping gates are not treated as independent noise measurements.

All residuals remain signed. The blue/white/red asinh scale shows negative,
zero, and positive power without clipping stored estimates. The normalized
PSD divides by the signed full-band frequency integral when that integral
is positive. Negative frequency bins remain negative. Its displayed rows
integrate to one. Grey rows have weak/nonpositive net power or belong to the
background reference region. The empirical display guard is three times the
far-range integrated-power scatter, floored at its same-beam temporal median;
it is not a calibrated detection probability. Raw spectra and the unmodified
unit-integral raw spectra are retained as separate datasets.

## Reproduce

Raw recordings and generated products stay outside Git. Use base conda for
Python, with NumPy, SciPy, h5py, ppigrf and Pillow. Rust builds use Cargo.lock;
FFmpeg supplies encoding and complete decode checks. Run from the repository
root, replacing the three input/output paths with your local recording,
committed Memo 50 geometry HDF5, and destination directory:

```sh
cargo build --release --locked --manifest-path syisr/lower_hybrid/Cargo.toml
cargo test --release --locked --manifest-path syisr/lower_hybrid/Cargo.toml --lib
conda run -n base python syisr/lower_hybrid/index_recording.py RAW PLAN OUTPUT/recording-index.h5
syisr/lower_hybrid/target/release/process OUTPUT/recording-index.h5 OUTPUT/range-doppler.h5
conda run -n base python syisr/lower_hybrid/subtract_background.py OUTPUT/range-doppler.h5
conda run -n base python syisr/lower_hybrid/verify.py OUTPUT/recording-index.h5 OUTPUT/range-doppler.h5
conda run -n base python syisr/lower_hybrid/make_movies.py OUTPUT/range-doppler.h5 OUTPUT
conda run -n base python syisr/lower_hybrid/finalize.py OUTPUT
```

The recording index checks every frame sentinel, layout, timestamp, angle,
sampling rate and waveform. Independent NumPy FFT and Parseval checks against
original signed IQ agree with the Rust spectra within 8.1e-8 of peak.
Normalization, pulse counts, signed subtraction, reference-gate selection,
movie frame counts and complete decoding are checked. Decoded first/middle/
last frames from every movie are retained for visual review. These are
measured exploratory spectra; the movies do not independently identify a
lower-hybrid mode or establish its excitation mechanism.
