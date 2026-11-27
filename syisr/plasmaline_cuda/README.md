# SYISR broadband plasma-line CUDA pilot

## Receiver-channel correction (2026-11-27)

Header byte 32 is transmit tuning (430 MHz here); byte 34 is receive tuning,
with `430 + 0.2 * byte` MHz. Byte 34 alternates 30/70, hence 436/444 MHz,
between consecutive 10 ms records. Each receiver has 16 MHz complex bandwidth.
Both records share a code phase; successive phases for one receiver are 20 ms
apart. **The original single-band products and movie mixed these tunings and
are superseded.** Their apparent 8 MHz duplicate is not evidence of aliasing.
The earlier circular carrier-centering display is also superseded: RF axes
must translate without wrapping.

`process.py` now requires explicit receiver selection. `matched.py` refuses a
reference without separate receiver metadata. `dual_hour.py` processes the
first 60 minutes independently per channel before any power statistics,
whitening, RFI decisions, or decoding. `splice_channels` normalizes each decoded
frequency column separately within each receiver. Below 440 MHz it selects the
436 MHz receiver, and at/above 440 MHz the 444 MHz receiver. Native spacing is
unchanged, with 12288 bins across 428–452 MHz (−2 to +22 MHz from transmit),
without duplicate or missing seam bins. Both 436–444 MHz overlap estimates,
normalization factors, and separate pulse counts are retained in HDF5.
`dual_hour_movie.sh` uses the corrected RF grid, common first-minute IQR limits,
native raster rendering, and the compact Rustplotlib layout. Its manifest
requires 120 real channel products spanning 60 consecutive minutes. Exposure
per receiver is accepted records ×10 ms (at most about 30 s per 60 s window).
No range HPF is applied. Original mixed products remain separate for audit.

The following historical pilot notes describe earlier development; receiver
interpretations and mixed products in those notes are superseded above.

The 2026-11-25 archive sampled headers report **16 MHz complex sampling,
320 µs pulse, 10 ms IPP, code identifier 50**, rather than the proposed
640 µs mode. Each frame includes four extra complex pairs after the declared
149436 samples. The reader verifies its stride against the following header.
The user confirmed use of the recorded 320 µs pulse duration. The 32-baud
sequence and phase registration have not been independently verified.

This initial CUDA program computes **undecoded diagnostic** range–frequency
spectra, not alternating-code plasma-line products. Native Q,I signed int32
are transformed as I+iQ with cuFFT, symmetric Hann windows and no range overlap.
The 1024-sample window is 64 µs (15.625 kHz bins); pulse convolution remains.
Apparent slant range is not altitude. No transmit sample or hardware IQ/RF sign
calibration is available. Header time is converted using the established SYISR
UTC+8 convention, pending confirmation for this campaign.

Whitening uses robust power in the final quarter of receive ranges before
frequency, cell and pulse rejection. Keep the raw spectrum, passband reference,
native accepted spectrum, signed excess, counts and packed exact masks.
Data-dependent censoring biases power; these thresholds are initial diagnostics,
not validated science settings. The background can contain real signal.

Build on gpu01:

```sh
nvcc -O3 -std=c++14 -Xcompiler -fPIC -shared spectra.cu -lcufft -o libspectra.so
/opt/soft/anaconda3/bin/conda run -n base python verify.py
/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python process.py \
  /gpfs/pub/data/L0/2026/plasmaline/20261125 \
  /gpfs/pub/data_other/juha/plasmaline-20261125/pilot60.h5
```

Only one GPU is used. Raw data and HDF5 outputs stay outside Git.
The supplied `Downloads/PlasmaLine_CUDA.tar.gz` contains Nate Hilliard's CUDA
7-era sliding conjugate-transmit/echo FFT implementation. `matched.py` and the
`pl_match_*` CUDA API adapt that architecture with persistent plans, bounds
checks, normalization and voltage whitening **before** ideal-code multiplication.
The ideal waveform uses the existing m32 table, 32 x 10 µs bauds, and the header
phase identifier. This is a conditional matched spectrum, not a validated
alternating-code lag decoder; measured transmitter response is unavailable.

```sh
/opt/soft/anaconda3/bin/conda run --no-capture-output -n base python matched.py \
  /gpfs/pub/data/L0/2026/plasmaline/20261125 \
  /gpfs/pub/data_other/juha/plasmaline-20261125/pilot60.h5 \
  /gpfs/pub/data_other/juha/plasmaline-20261125/matched60.h5
```

The numerical test covers independent CPU equivalence for pre-whitening,
frequency notches, template multiplication, FFT and power averaging, plus a
synthetic delayed coded tone's range and frequency. It does not establish the
recorded waveform identity. Unequal code-phase counts after RFI rejection are
stored and can change the range ambiguity response.

Matched products retain 8192 native Fourier bins (1.953125 kHz spacing); the
320 µs pulse sets a broader intrinsic frequency response (~3.125 kHz).
`verify_reader.py` checks stride, exact clock-fragment exclusions without time
repair, and rejection of unsupported modes. Clock checks require first/last
file endpoints to agree with the physical-frame cadence. Files with malformed
framing or unstable endpoints are excluded; unresolved tails are excluded
before a damaged payload can enter the estimate. Raw headers, offsets and
exclusion reasons are retained. Unsupported modes fail the minute.

`campaign.py RAW OUTPUT` runs sequential UTC-aligned 60 s research integrations
on GPU 0 with a cached HDF5 first-header catalogue, restartable per-minute
products, logs and an HDF5 status ledger. It records failed minutes and
continues independent minutes. A catalogue is not a certification of raw
frame integrity. `check_product.py DIAGNOSTIC --matched MATCHED` verifies
stored masks/counts, signed excess, time ordering and coordinates on read-back.

The requested default range step is **2 µs (32 samples)**. Matched sums and
second moments accumulate on the GPU in double precision. `sample_five.py`
selects five evenly spaced minutes. Plotting uses **rustplotlib only** through
`rust_render`; `rust_extract` transports native scalar cells in HDF5, without
range or frequency averaging. The displayed range ends at 800 km. Each decoded
frequency column is divided by its median over valid range gates. Original
decoded power and the normalization factors remain available in HDF5. Inferno
and common Tukey-IQR limits are used. Native PNGs and previews are separate.

The user confirmed that recorded -6 MHz corresponds to the 430 MHz carrier.
Passing `-6000000` to the Rust renderer performs an exact 3072-bin circular
permutation and labels frequency relative to 430 MHz. This does not undo
acquisition aliasing. The 8 MHz-spaced replica is also visible in the raw
pre-decoding STFT; its particular receiver/recording mechanism is unverified.
No range HPF was applied.

`interference_blocks` reconstructs five-second wall-clock blocks of 06:19 UTC
with the same CUDA kernel. The available early 00-10 s interval has about a
tenth of the received ADC power of 40-50 s. These periods are kept separate;
plasma-line recovery in the strongest interference is not established.
Wrong-code controls and raw spectra are diagnostic evidence, not validated
interference-free scientific products.
