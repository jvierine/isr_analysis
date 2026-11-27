# SYISR broadband plasma-line CUDA pilot

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
file endpoints to agree with the physical-frame cadence. Malformed framing
or unstable endpoints fail the minute; missing products remain explicit.

`campaign.py RAW OUTPUT` runs sequential UTC-aligned 60 s research integrations
on GPU 0 with a cached HDF5 first-header catalogue, restartable per-minute
products, logs and an HDF5 status ledger. It records failed minutes and
continues independent minutes. A catalogue is not a certification of raw
frame integrity. `check_product.py DIAGNOSTIC --matched MATCHED` verifies
stored masks/counts, signed excess, time ordering and coordinates on read-back.
