# SYISR lower-hybrid visit recordings

One range–Doppler MP4 per pointing, with signed background-subtracted power
and a second panel normalized by integrated absolute spectral magnitude.

The full-night five-minute products use 300 s windows with 150 s steps.
`restart_calibrate.py` preserves the initial run's provisional 160 µs correction
and applies 640 µs total to the post-restart run. The additional 480 µs
(71.95 km) is the user-assumed one-transmit-pulse delay after the restart.
The five-beam narrowing displacement provides a consistency check. It is not an independently measured hardware
delay. Header words at byte offsets 664, 732 and 740 change across the restart;
their hardware meanings remain unknown. Beam IDs and RX/TX angles agree for
all 5,019,393 audited pulses. The deleted 30.8 s test file is excluded.
Original products and signed spectra remain unchanged. Rendered comparisons
use plasma, a minimum of −0.01 on both existing scales, and no quality mask.

To build the corrected full movie, run `restart_calibrate.py` with the original
product root, a separate destination and the beam-association audit HDF5;
run `make_movies.py` for each of its two `restart_registered` products;
then `combine_movies.py ROOT --variant restart_registered` followed by
`concat_altitudes.py ROOT`. Use the base conda environment for these scripts.

The 16 November 2026 recording contains 116,509 pulses over
11:03:59–11:23:24 UTC. The header clock is Beijing time, confirmed by the
data owner. The containing folder is named 20251116, but the filename and
headers identify 2026. Five directions are interleaved pulse by pulse.

Each frame uses 60 seconds of wall-clock data, stepping by 30 seconds.
There are 38 frames per direction; the final full window is anchored at the
recording end. Most frames contain about 1,200 pulses per beam. The output
is 1920×1080 H.264, 19 seconds per movie, with two scientific updates per
second encoded at 24 fps. Color limits are fixed across all times and beams.

The compact full-night movies play eight scientific frames per second (4× the
previous playback speed), retaining every 300 s integration. Rustplotlib frames
stream as raw RGBA through FFmpeg stdin; only decoded QA snapshots and one SVG
preview per beam are saved. `--png-frames` enables the legacy export explicitly.
Timing/calibration details stay in HDF5 metadata, not animation footers.

## Driver and shared code

The instrument-independent estimator is `crates/range-doppler` at the repository
root. The SYISR binary parser stays in `syisr/src/reader.rs`; this driver's
experiment constants and rectangular-pulse ambiguity screen stay in `src/lib.rs`.
`process` validates the supported 480 µs uncoded layout and supplies decoded IQ,
beam assignments, timing and exposure to the shared estimator. Coded experiments
must use their own decoding driver and are rejected here.

`render` reads the experiment's HDF5 products and uses Rustplotlib's public
`AnimationWriter` for video output. Neither the spectral core nor the animation
writer knows SYISR filenames, beam geometry or restart calibration assumptions.

## Spectra and ranges

The signed int32 samples are decoded as I+iQ, following the supplied MATLAB
reader. Each 480 µs uncoded pulse is analyzed independently: 48 samples at
100 kHz, a symmetric Hann taper, and a 256-point forward FFT. Frequency
spacing after zero padding is 390.625 Hz; the actual Hann noise-equivalent
bandwidth is about 3.19 kHz. A slow-time FFT between pulses is not used.
The two-sided PSD is |FFT(w x)|² / (fs sum(w²)), in receiver ADU²/Hz.
No absolute calibration or physical toward/away velocity sign is claimed.

Receiver windows overlap by 75%, giving 18 km gate spacing. The transmitted
pulse and receiver window together have roughly 144 km full range support
and 72 km FWHM. These overlapping gates are not independent 18 km range
resolution. Range is pulse-centred monostatic slant range, including the
Hann centroid and half-pulse correction; instrumental group delay remains
uncalibrated.

The faint marker is computed from each recorded pointing, IGRF-14 on the
recording date, WGS84 straight rays, and the provisional Memo 50 site
(18.34° N, 109.62° E). The planned 250/350/450/550/650 km targets map to
actual perpendicular slant ranges 286.8/406.5/529.3/658.2/793.4 km.
The planned geometry is verified against the committed Overleaf Memo 50
product; slant range is never relabelled as altitude.

## Background and normalization

A same-beam, same-window average of quiet candidates among the eight highest
complete gates, centred at 1248–1374 km, is subtracted at every frequency.
This includes a common far-range narrow feature near −2.34 kHz. The identity
of that feature is not established by background subtraction.

Localized positive excesses are excluded from the reference estimate using
integrated-power and per-frequency relative-excess guards. A localized event
contaminated one reference gate in two overlapping windows of the highest
beam. The event remains in the measured spectra. Exact gate-use masks and
the unvetted reference are retained. Quiet top-four and top-twelve reference
controls change the core-range spectrum by at most 0.13% of its peak.
The overlapping gates are not treated as independent noise measurements.

All residuals remain signed. The plasma asinh scale shows negative,
zero, and positive power without clipping stored estimates. The normalized
PSD divides by the signed full-band frequency integral when that integral
is positive. Negative frequency bins remain negative. These signed unit-integral products and their quality flags remain stored.
Renderer 0.2.2 displays every range gate using the positive integral of the
absolute residual as its normalization, without reversing negative estimates.
Each displayed row has unit integrated absolute magnitude; no grey quality mask
is applied. The empirical display guard is three times the
far-range integrated-power scatter, floored at its same-beam temporal median;
it is not a calibrated detection probability. Raw spectra and the unmodified
unit-integral raw spectra are retained as separate datasets.

## Reproduce

Raw recordings and generated products stay outside Git. Use base conda for
Python, with NumPy, SciPy, h5py, ppigrf and Pillow. Rust builds use Cargo.lock;
FFmpeg supplies encoding and complete decode checks. Run from the repository
root, replacing the three input/output paths with your local recording,
committed Memo 50 geometry HDF5, and destination directory:

```sh
cargo build --release --locked --manifest-path syisr/lower_hybrid/Cargo.toml
cargo test --release --locked --manifest-path syisr/lower_hybrid/Cargo.toml --lib
conda run -n base python syisr/lower_hybrid/index_recording.py RAW PLAN OUTPUT/recording-index.h5
syisr/lower_hybrid/target/release/process OUTPUT/recording-index.h5 OUTPUT/range-doppler.h5
conda run -n base python syisr/lower_hybrid/subtract_background.py OUTPUT/range-doppler.h5
conda run -n base python syisr/lower_hybrid/verify.py OUTPUT/recording-index.h5 OUTPUT/range-doppler.h5
conda run -n base python syisr/lower_hybrid/make_movies.py OUTPUT/range-doppler.h5 OUTPUT
conda run -n base python syisr/lower_hybrid/finalize.py OUTPUT
```

The recording index checks every frame sentinel, layout, timestamp, angle,
sampling rate and waveform. Independent NumPy FFT and Parseval checks against
original signed IQ agree with the Rust spectra within 8.1e-8 of peak.
Normalization, pulse counts, signed subtraction, reference-gate selection,
movie frame counts and complete decoding are checked. Decoded first/middle/
last frames from every movie are retained for visual review. These are
measured exploratory spectra; the movies do not independently identify a
lower-hybrid mode or establish its excitation mechanism.

## Absolute timing audit and provisional copies

The original 533 µs gate-origin interpretation has been checked against the
header span, sample spacing, Hann/pulse centroids and injected point responses.
There is no two-gate indexing error under that interpretation. Independent
native Rust IGRF/WGS84 calculations agree with the Python crossings within 8 m.
The measured central spectral fraction peaks 24–33 km beyond those crossings;
that association alone cannot calibrate the range origin.

A separate diagnostic finds a coherent near-gate falling edge 153–163 µs
after the commanded pulse end in all five current beams. Twelve older 2000 µs
pulse recordings put their edge at 161–171 µs. If this is zero-range transmitter
leakage, it supports a common timing delay of about 160 µs. The exact timing
reference and the possibility of receiver switching/blanking remain unverified.

Separate `provisional_delay160us` copies subtract 23.9834 km from the range
axis. Every spectrum, background estimate and normalization mask is unchanged;
the original products remain available. Each movie labels the correction as
provisional. The geometric dashed markers stay at their independently computed
ranges. After correction the empirical narrowing offsets are about 0–9.2 km,
roughly half the displayed 18 km gate spacing. This agreement does not establish
a hardware calibration or an uncertainty bound.

Run the following after the original products are complete. `CONTROLS` is the
directory of older 2000 µs recordings; these checks fail on unexpected headers.

```sh
conda run -n base python syisr/lower_hybrid/audit_gating.py OUTPUT/recording-index.h5 OUTPUT/range-doppler.h5 OUTPUT/range-gating-audit.h5
syisr/lower_hybrid/target/release/audit-geometry OUTPUT/range-gating-audit.h5
conda run -n base python syisr/lower_hybrid/audit_pulse_edge.py OUTPUT/recording-index.h5 OUTPUT/range-gating-audit.h5 CONTROLS
syisr/lower_hybrid/target/release/plot-gating-audit OUTPUT/range-gating-audit.h5 OUTPUT/range-gating-audit
syisr/lower_hybrid/target/release/plot-pulse-edge OUTPUT/range-gating-audit.h5 OUTPUT/pulse-edge-audit
conda run -n base python syisr/lower_hybrid/candidate_range_delay.py OUTPUT/range-doppler.h5 OUTPUT/range-gating-audit.h5 OUTPUT/provisional_delay160us/range-doppler.h5
conda run -n base python syisr/lower_hybrid/verify.py OUTPUT/recording-index.h5 OUTPUT/provisional_delay160us/range-doppler.h5
conda run -n base python syisr/lower_hybrid/make_movies.py OUTPUT/provisional_delay160us/range-doppler.h5 OUTPUT/provisional_delay160us
conda run -n base python syisr/lower_hybrid/finalize_candidate.py OUTPUT/range-gating-audit.h5 OUTPUT/provisional_delay160us
```

Finalize from a clean committed checkout. Correction/rendering provenance is
recorded separately from the original IQ-processing version.

## Five-minute, complete-file-sequence processing (0.2.0)

Version 0.2 defaults to 300-second integrations and 150-second steps, including
windows across file boundaries. A shorter test recording is explicitly labelled
with its actual duration. Only complete records are decoded; incomplete bytes
are permitted solely at the end of the last file and are counted in metadata.
Every source byte is hashed, including that tail. Header timestamps remain UTC
across midnight. Optional `--cache-dir` on the indexer copies source bytes during
one sequential pass so repeated FFT access can use fast storage; original paths,
per-file hashes and the concatenated-byte hash remain in HDF5.

Candidate satellite screening uses same-beam one-second spectra. Positive
excursions above eight times the temporal median form connected range/frequency
components. A component must contain at least three bins, span no more than
16 range gates, and have cosine similarity at least 0.7 to the calculated
point-target power ambiguity pattern of the rectangular 480 µs transmission
and the actual 48-sample Hann receiver window. The trial range centres span two
gates either side of the peak. Four extra range gates and one extra second guard
the detected component. Entire affected time/range exposures are excluded across
all frequencies. Frequency bins are never selectively clipped. Accepted pulse
counts, exact one-second masks and unscreened PSDs accompany the screened PSDs.
Peaks within 5.08 kHz of zero Doppler and eight gates of the nominal
perpendicular crossing are protected: a bright, localized central ion-line is
not by itself evidence of a satellite. Satellite echoes in this protected
region can consequently remain. The rule is intended to preserve the plasma
signal rather than promise a satellite-free product.

The screen preserves steady narrow spectral features and broad common
interference in synthetic checks. These cuts are conservative morphology
criteria, not calibrated false-alarm probabilities or independent satellite
identifications. Weak or persistent echoes can remain, and real physical
transients resembling point targets can be excluded. The before/after review
plots and exposure fractions must be inspected for each recording.
The primary eight-gate background requires four clean gates. The four-gate
sensitivity control requires two, so a contaminated control gate does not
invalidate an otherwise well-supported primary background estimate.
Normalized display also requires the positive spectral peak to exceed three
times the largest top-four/top-twelve reference difference from the primary
background. This protects weak overnight spectra from misleading normalization;
it is an empirical sensitivity guard, not a calibrated uncertainty interval.

For a complete session, replace the single RAW input in the recipe above with
`--session FILENAME_PREFIX --cache-dir CACHE RAW_DIRECTORY`. The other
processing/subtraction/verification commands accept the resulting multi-file
index. Independent NumPy checks cover each file boundary as well as the first,
middle and final integration, and use the actual retained pulse exposure.

`audit_session_edges.py RAW_DIRECTORY ROOT/pulse-edge-audit.h5` checks sampled
pulses in every file and direction. `prepare_display.py ROOT` supplies shared
signed colour scales across sessions and separate `provisional_delay160us`
products. The 160 µs correction is still explicitly provisional. Run
`make_movies.py` on each corrected product, then `screen_review.py PRODUCT
REVIEW.h5` and `plot-screen-review REVIEW.h5 FIGURE_STEM`. After visual review,
commit/push and run `finalize_batch.py ROOT` to attach program short names,
versions, Git hashes, component source hashes and checked movie hashes.

Renderer 0.2.1 uses the plasma colormap for the spectra and screening reviews.
The signed data, asinh transform, colour limits and compute version 0.2.0 are
unchanged.

Renderer 0.2.2 sets both displayed lower colour limits to -0.01 in their
respective displayed units, retains plasma, and concatenates all five nominal
target beams with `concat_altitudes.py`. Stored signed data are unchanged.
The later run has an unresolved configuration-dependent displacement of the
observed narrowing region. `audit_beam_association.py` checks every recorded
beam identifier and RX/TX angle without using that feature to impose a new
range correction. Header fields alone do not establish actual antenna pointing.

## Longer integrations

`integrate_products.py SOURCE.h5 OUTPUT.h5 --beam 4 --window-s 1800 --step-s 300`
combines six non-overlapping five-minute PSDs with their actual retained exposure
at each range gate. It retains the original one-second exclusions and adds a
conservative block/range ambiguity guard for residual localized echoes. Whole
frequency spectra are excluded together; the central perpendicular ion line is
protected. Original unscreened power, masks, parent indices and exposure counts
remain available. This is satellite-morphology screening, not confirmed object
identification. Background subtraction and normalization follow integration.

Run `subtract_background.py`, `verify.py` with the original recording index, and
`make_movies.py ... --beam 4` for each session. `finish_integrated_movie.py` joins
and publishes the two sessions without integrating across the restart. Incomplete
trailing windows are omitted to retain exact five-minute steps.

## Signed excess normalization and colour comparison

The normalized excess display divides the signed background-subtracted spectrum
by the positive full-band power **before** background subtraction. Its integral
is S/(S+B), rather than being forced to one in noise-only gates. Negative noise
estimates remain signed. The older L1 and signed-unit-area products stay in HDF5
for comparison; they are not the current display normalization.

`compare_colormaps.py FIVE_MINUTE_ROOT THIRTY_MINUTE_ROOT DESTINATION` exports
matched plasma, inferno and nipy_spectral variants with identical normalization,
limits and asinh widths. Each includes 250–550 km at five-minute integration and
650 km at thirty-minute integration. It then concatenates all three variants,
retaining UTC, calibration, satellite screening and every scientific frame.
