# Lower-hybrid alternating-code movies

Gated ACFs and signed, triangular-windowed spectra for the five-beam
19–20 November 2026 recording. The radar interleaves beams pulse by pulse;
64 code members must be gathered separately for each direction.

The supported mode is 640 µs / 32×20 µs, 100 kHz complex sampling and 16 ms IPP.
The movie uses 15-minute wall-clock windows stepped by five minutes, inferno,
and perpendicular / +900 µs reference cuts on logarithmic and linear axes.

Submit `run.slurm RUN` through Slurm on an available node01–03. Build snapshots,
raw data and HDF5 products stay outside Git. `prepare_geometry.py` uses actual
recorded angles. `export_movie.py` combines complete five-minute blocks;
`plot/` streams frames to FFmpeg using Rustplotlib.

Coded-Doppler, burst and clipping screening precedes accumulation. Masks reject
an affected complete cycle/gate support across every code member; persistent
coherent supports are retained. Unscreened ACFs and spectra, masks and retained
counts accompany the accepted estimates. Candidate rejection is not confirmed
satellite identification and can miss weak/persistent targets.

Zero lag is locally extrapolated from the first two decoded lags. Fractional
lag kernels differ, so this is an approximation at sharp range gradients.
No negative power is clipped. Nominal ranges use the header delay; the earlier
uncoded restart correction is not applied. Frequency sign, units, processing
versions and limitations are stored in the HDF5 metadata.

The 30-minute revision uses six native five-minute blocks and a four-cell running
mean of the complex ACF **before** the triangular taper and FFT. Even-width
windows are placed at the mean of their four native range centres (11.99 km
range support, 3.00 km output spacing). Cuts use this averaged range coordinate.
The earlier coordinate-only 640 µs registration was wrong: it shifted full-pulse
source data into partial-pulse gate labels. Those movies are superseded.
Use `partial.slurm RUN ORIGINAL` to add the earlier source gates with the
correct negative recorded-source indices, exclude the first 32 recorded
voltage samples, and normalize only surviving baud/lag pairs. The 640 µs is
then the partial-grid origin G−T, not a second shift of decoded coordinates.
Unsupported lags stay missing; each gate's FFT taper uses its available lags.

The sector composite regrids each beam onto WGS84 altitude before joining
adjacent beam-centre aspect angles. It blends complex ACFs and received power,
then transforms and normalizes; there is no gain rescaling to hide seams.
Outside the beam-centre bracket it uses the closest beam and records coverage
flags. The composite starts at 80 km altitude and retains six altitude cuts.
This is an interpolation of measured sectors, not an exactly perpendicular
observation or a beam-pattern inversion.

The raw-origin audit traced 960 original frames and recovered 30 known-delay
point-echo profiles through the production binary reader/index/decoder without
a 64-sample shift. The supplier code table matches the analysis table. A fit to
the strong code-bearing near-range signal gives an apparent delay near 163 us,
but that signal may be unknown-range ground clutter: it is **not** an absolute
clock calibration. Those checks alone did not calibrate timing. The subsequent 50-pass satellite
calibration and independent production-decoder satellite checks set the range
basis in `pointing-calibration.json`; changing pointing does not alter it.


## Adopted pointing model (26 November 2026)

The default for the November lower-hybrid campaign is the fixed-phase 440 MHz
steering hypothesis evaluated at 430 MHz. For the horizontal Sanya array this
preserves azimuth and scales cos(elevation) by 440/430. The predicted elevations
are 60.866089, 60.152781, 59.410850, 58.634574 and 57.829555 degrees. IGRF14
perpendicular crossings are approximately 202.3, 252.0, 301.5, 351.0 and 400.0 km.
The steering tables/controller were unavailable: these are adopted model
predictions, not independently calibrated directions.

This **replaces** the common empirical −0.72 degree offset. Combining
`--pointing-model phase-table` with any `--elevation-offset-deg` fails. Explicit
`header` and `empirical-offset` modes remain for reproducing comparisons.
Headers are preserved; products store the physical directions, design/operating
frequencies and model version separately. Satellite timing and decoded slant
ranges are unchanged. Individual-beam cuts now use the computed IGRF crossing;
`--feature-cuts` is an explicit legacy visual-height alternative and is not used
by the new individual-beam batch. The composite retains altitude cuts while its
sector weights are recomputed from the predicted beam directions.

The default batches write `individual-phase-table` and `pseudo-phase-table`,
leaving earlier products intact. Set `LHW_SOURCE_TREE` to the deployed Git
checkout when submitting the batches. A better geometric explanation does not
by itself establish that the perpendicular Te/Ti fitting bands are corrected.
