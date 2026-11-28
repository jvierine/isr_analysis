# SYISR plasma-line processing

Requires Linux, CUDA/cuFFT, and base conda with NumPy and h5py.

Build once, from this directory:

```bash
nvcc -O3 -std=c++14 -Xcompiler -fPIC -shared spectra.cu -lcufft -o libspectra.so
```

Run:

```bash
conda run --no-capture-output -n base python run.py --data /path/to/data --output /path/to/products
```

The entire data directory is analyzed by default, including the final partial
minute. All usable CUDA GPUs are detected automatically. Each GPU processes one minute
at a time, with the 436 and 444 MHz receivers handled separately. Progress is
printed to the terminal; detailed logs are saved beside each product.
Existing products are reused. A failure stops new work.

- `--minutes 60` optionally limits processing to the first hour.
- `--start UNIX_SECONDS` sets the first integration in UTC (default first header).
- `--device 0` restricts processing to one GPU. `CUDA_VISIBLE_DEVICES` is respected.
- Alternatively edit `example.json` and run `python run.py --config example.json`.
  Command-line values override configuration.

HDF5 products use 2 µs range steps and are written to
`OUTPUT/dual-channel-hour/TIME/RECEIVER/`.
This is the sliding code-matched spectral estimator. Pulse-level RFI masking
is under development. The current matched decoder uses whole-IPP rejection.

### Pulsed RFI rejection

Use a fresh output folder to process the 04:49 UTC test case with raw
pulse exclusion:

```bash
conda run -n base python run.py --data /path/to/raw --output /path/to/0449 \
  --start 1795582140 --minutes 1 --mask-pulses
```

This mode excludes full 320 µs decoding windows overlapping a detected raw
pulse plus its time guard. It normalizes frequency columns after decoding.
The HDF5 product records accepted counts by range, accepted counts by code
phase and range, and the packed gate mask. An IPP counted as accepted in a
plot header may contribute only some ranges. Use the per-range counts for
exposure and uncertainty.
