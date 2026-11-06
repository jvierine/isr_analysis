Rust signal processing and outlier-resistant lag-profile inversion
=================================================================

`outlier-lpi` is the Rust port of the repository's `outlier_lpi.py`,
extracted from RUISDAP's Millstone processing chain. It preserves the signed
complex lag convention `R(l) = x(t) conj(x(t+l))`, explicit measured transmit
waveforms, sample masks, code-matched pulse pairs for coherent ground-clutter
removal, robust outlier tests, sample-variance weighting and retained direct
measurement equations. It rejects singular designs; it does not fill missing
lags, perform a truncated SVD, or clip negative measurements.

`isr-signal` supplies the shared float32/complex64 and float64/complex128
lag-product kernels, runtime-selected AVX implementations and optional CUDA
backend. The GPU implementation is selected explicitly; missing CUDA support
does not silently masquerade as GPU processing.

Both are libraries: experiment timing, code pairing, noise-injection calibration,
physical code energy, range domains and masks must be supplied by the caller.
The existing calibration entry points require an independently measured positive
receiver gain. They must not be used with invented injection temperatures.

Run `cargo test --workspace --release` from this directory. Additional Arecibo
acceptance simulations must establish transmit timing, range response, ground
clutter and heating-related RFI rejection before using a new decoder for
production ion-line fits. A reconstruction grid is not a guarantee of resolution.

For recordings without noise injection, `outlier_lpi::estimate_native` reuses
these same robust product masks and solver in explicitly uncalibrated native
voltage-squared units. It requires `filter_half=0`, ignores the injection gate,
and returns no gain, system temperature or Kelvin-labelled products. Recording
receiver filtering and physical density calibration remain the caller’s explicit
forward-model responsibilities. This mode does not imply that filtering a coded
waveform is separable in range and lag.
