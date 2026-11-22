# SYISR 100 kHz receiver voltage response

`receiver_100khz.h5` preserves all 264 float64 values of `ht` from the
supplied `ht_0.05_new.mat`, without normalization, cropping or resampling.
Source SHA-256, DC gain and timing qualifications are HDF5 attributes.
The original coefficient sum is 10.091453555368808. The energy centroid is
tap 130.96874962361747; the largest coefficient is at tap 131 (zero based).

100 kHz describes the **output voltage sampling rate**, not the FIR tap
rate. The supplied plot and earlier supplier-file analysis indicate 1 us
tap spacing (1 MHz); the MAT file itself contains only `ht`, with no time
axis. This spacing is recorded as inferred, not independently verified.

For full-profile inversion, use these as **voltage** convolution weights
inside the range-ambiguity calculation, before correlation and code decoding.
For y(t) = sum_a h_a x(t-a), the covariance includes every
h_a conj(h_b) R_x(t-a, t+lag-b) term. Applying just a power-spectrum correction
after decoding loses range/code boundary coupling. Apply the same retained
sample mask, partial-pulse decoding and range averaging as used on the data.

`receiver_filter::load(path, interval_s, origin_s)` returns the explicit
time-offset/complex-weight pairs accepted by `exact_response::coded_cell`
and `rectangular_cell`. It normalizes to unit DC gain; the file retains
the original gain. Pass `1e-6` only with the above timing qualification.
No automatic truncation or conversion to output-rate taps is performed.

The origin must be chosen explicitly in the calibrated range convention.
Do not subtract the roughly 131 us filter latency again from a range already
calibrated with satellite echoes. Peak/energy-centroid delay is not necessarily
the matched-filter timing reference. Retain both the chosen origin and the
filter source hash in inversion-product metadata.

Existing `ac_range::stencil` assumes odd, centered taps spaced at 10 us;
the 264 native taps must **not** be passed to it directly. Existing fits
using the noise-derived symmetric receiver model are not retroactively
corrected by adding this file. The loader connects this response to the
explicit-time forward integral; a production stencil replacement and refit
must retain its sub-sample timing and validate against filtered point targets.
