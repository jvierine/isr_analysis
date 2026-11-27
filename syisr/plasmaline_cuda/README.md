# SYISR broadband plasma-line CUDA pilot

The 2026-11-25 archive sampled headers report **16 MHz complex sampling,
320 µs pulse, 10 ms IPP, code identifier 50**, rather than the proposed
640 µs mode. Each frame includes four extra complex pairs after the declared
149436 samples. The reader verifies its stride against the following header.
The 32-baud waveform and sequence have not been verified. Do not run the known
640 µs decoder against these data without resolving the mode difference.

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
Nate Hilliard's original CUDA source was not found in the locally available
Dropbox tree. This pilot is new diagnostic code; adaptation of the original
and the alternating-code decoder remain pending those inputs.
