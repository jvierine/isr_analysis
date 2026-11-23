# LP480 production diagnostics

A converged optimizer and plausible parameter map do not establish a valid
receiver correction. Production release requires the following diagnostics for
every measured beam/time, followed by documented scientific review.

Run `lp480_diagnostics INPUT FULL_FITS OUTPUT` using the Rust plotting binary,
or `lp480_diagnostics.sh RENDERER COMPLETED_RUN OUTPUT`. Full fit files are
required: compact plasma-map snapshots deliberately omit the residual arrays
and must fail this check. Re-running diagnostics never changes fits or raw data.

* Background spectra versus UTC: final four measured receive windows, absolute
  Hann periodogram power in ADU² per FFT bin and power divided by the band
  median. Show all frequency bins, including DC; no receiver-background
  subtraction, notch or cosmetic smoothing. These are retained-pulse spectra,
  so screening exclusions still require a separate unscreened raw audit. Far
  range is not proof of plasma-free noise. Compare the actual windows with
  quiet/reference spectra and check acquisition restarts and solar-noise changes.
* For every fitted beam/time, altitude versus lag maps of real and imaginary
  `(model-data)/assigned sigma`. Include ALL measured fit lags (10–470 µs in
  this mode), especially the first few lags relevant to H⁺. Use the actual
  source gates, no zero-lag normalization, no Huber downweighting, no prior
  residuals, and no interpolation across missing products. Display ±5 with
  exceedance counts retained in HDF5. Stored values are not clipped.
* `residual-summary.h5` retains signed sums, squared sums and counts per lag,
  beam and time, allowing coherent bias to be distinguished from scatter.
  Assigned errors are diagonal working weights; these residuals are not
  independent standard-normal samples or validated parameter confidence.

Check for persistent short-lag oscillations, nonzero long-lag pedestals,
altitude-coherent bias and time changes coincident with background spectra.
Inspect individual signed ACFs where structure appears. Do not use smoothing
or larger assigned errors to conceal it. A shared reference creates correlated
errors across gates/times and needs independent background validation.

`DIAGNOSTIC_PLOTS_COMPLETE` certifies generation only. The accompanying
`DIAGNOSTIC_REVIEW_REQUIRED` intentionally withholds production acceptance.
The current storm rerun is a research validation run, not production release.
An absent or failed renderer does not invalidate completed numerical work, but
it prevents calling that work production-ready. No automatic residual threshold
is claimed calibrated by this initial implementation.

New corrected daily submissions build the diagnostic renderer against the
cluster libc in an isolated Cargo target, then run it after numerical completion.
`DIAGNOSTICS_PENDING` explicitly records build/render/coverage failure. Science
can continue to the next day; production acceptance cannot. Already-running
frozen deployments are unchanged and need the standalone diagnostic command.
