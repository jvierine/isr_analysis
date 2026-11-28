# SYISR plasma-line processing

Requires Linux, CUDA/cuFFT, and base conda with NumPy and h5py.

Build once, from this directory:

```bash
nvcc -O3 -std=c++14 -Xcompiler -fPIC -shared spectra.cu -lcufft -o libspectra.so
```

Run:

```bash
conda run --no-capture-output -n base python run.py --data /path/to/data --output /path/to/products --minutes 60
```

All usable CUDA GPUs are detected automatically. Each GPU processes one minute
at a time, with the 436 and 444 MHz receivers handled separately. Progress is
printed to the terminal; detailed logs are saved beside each product.
Existing products are reused. A failure stops new work.

- `--minutes 60` processes 60 consecutive one-minute integrations (default 1).
- `--start UNIX_SECONDS` sets the first integration in UTC (default first header).
- `--device 0` restricts processing to one GPU. `CUDA_VISIBLE_DEVICES` is respected.
- Alternatively edit `example.json` and run `python run.py --config example.json`.
  Command-line values override configuration.

HDF5 products use 2 µs range steps and are written to
`OUTPUT/dual-channel-hour/TIME/RECEIVER/`.
This is the sliding code-matched spectral estimator. Pulse-level RFI masking
is under development. The current matched decoder uses whole-IPP rejection.
