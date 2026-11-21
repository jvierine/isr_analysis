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
A 640 µs empirical time-origin correction subtracts 95.934 km from the initial
header-based ranges: the multi-beam ion-line narrowing was displaced by about
one code length. This is registration to the observed perpendicular feature,
not an independently calibrated hardware delay. Native blocks remain unchanged.
Run `reduce.slurm RUN` followed by `render.slurm RUN 30min_step5min_4gate`.
