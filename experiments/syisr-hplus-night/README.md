# SYISR nighttime zenith H+ fit: first proof of concept

Independent Rust retrieval of H+/He+/O+ fractions from SYISR uncoded 480 us
long-pulse complex lag products. The fitted composition is NOT prescribed by
IRI, and supplied L2 fits are not inputs. This is a local broad-footprint fit,
not a full-profile composition inversion.

Live results: http://avaruus.uit.no/ruisdap/syisr-hplus.html

## Executed measurement

- Sanya monostatic zenith, 2023-11-15 16:00--17:00 UTC. Header clock minus
  eight hours; station 18.3492 N, 109.6222 E, 50 m, from supplied L2 station
  metadata. IGRF-14 gives 25.19 degrees from perpendicular at 700 km.
- 20,625 zenith IPPs; 776 rejected by a conservative whole-IPP range-power
  screen, leaving 19,849 IPPs and 11 beam dwells distributed through the hour.
- At nominal 700 km, H+ fraction = 0.241, with He+ free and O+ the remainder.
  Whole-dwell resampling percentiles are 0.227--0.267, conditional on the model.
- Common ion temperature 651 K, electron temperature 1343 K in that fit.
  Temperature interpretation changes when proton and O+ temperatures differ.
- H=0 was independently refitted, including temperatures, He fraction,
  velocity, signal power and background adjustment. Delta chi-square = 82.6
  under conservative diagonal weights. This is NOT a formal discovery sigma.
- A separate profile allows each dwell its own temperatures/He/power/velocity
  while sharing H fraction. It also favors nonzero H.
- 100 centered whole-dwell null resamples gave no statistic as large as the
  observed one; conditional empirical p=1/101. This is resolution-limited,
  exploratory, and not corrected for selecting a candidate in a survey.

The first 6-hour exploratory scan fitted 66 hour/height cells. Those fits are
retained on NFS but are not promoted as accepted composition measurements.
Above roughly 900 km most fractions in this data are poorly constrained.
Boundary-temperature fits at lower gates are explicitly withheld.

## Forward model and fitting

Fit signed measured lag products, retaining every measured lag 0--400 us.
The imaginary component at lag zero is identically zero, not an independent
measurement. Fractional receive-bin alignment and 17-bin range averaging
keep the pulse-corrected pair center at a common physical altitude. Range is
converted to altitude for the zenith geometry, including the 50 m station.

The forward kernel integrates a 480 us rectangular pulse, a centered voltage
FIR inferred from far-range receiver-noise ACFs, and a local exponential
raw-power gradient. It includes receiver mixing BEFORE correlation. The
nominal footprint is about 100 km plus FIR tails. The local thermodynamic and
composition state is constant within that footprint; the grid is not an
independent fine-range profile.

The Maxwellian multi-ion theory includes an exact magnetized electron
Gordeyev response, cached in temperature/frequency, and a checked large-k*rho
unmagnetized ion limit. This pilot deliberately rejects near-perpendicular
geometries. It does not claim to implement the Milla--Kudeki Coulomb model.
The topside collisionless approximation has no fitted collision rate. A Rust
MSIS density/rate-scale check gives an inherited effective neutral rate near
9.44e-5 /s at 700 km, or 3.78e-8 times the longest lag; that check is not a
species-resolved H/He/O collision operator. Detailed Coulomb physics remains
separate from this proof of concept.

Optimization uses bounded damped Gauss--Newton/Cholesky, multi-starts, and
profile likelihood for H fraction. Signal amplitude and a background
adjustment with an independent reference-error penalty are profiled linearly.
No truncated SVD, missing-lag filling or clipping of signed measurements.

Absolute Ne is uncalibrated. The amplitude is native band power; Debye shape
uses explicit density sensitivity runs rather than pretending amplitude is
calibrated density. Physical Doppler sign is not independently established.

## Sensitivity at 700 km

| Variant | H+ fraction |
|---|---:|
| Main screened all-lag fit | 0.241 |
| Alternative far-range background | 0.250 |
| Debye density 1e10--1e12 m^-3 | 0.241--0.243 |
| Proton/O+ temperature ratio 0.5--2 | 0.241--0.252 |
| Diagnostic omission of 0 and 10 us | 0.183 |

The short-lag response sensitivity is larger than the fixed-model bootstrap
interval. Report approximately 20--25% H+ as the current proof-of-concept
scale, not 24.1% with a total 2% uncertainty. Baseline noise inferred from the
first receive gates over-subtracts the topside and was rejected; two distinct
far-range references are retained. No empirical receiver PSD projection was
needed for the selected hour (negative PSD mass was zero).

The pulse screen uses each guard range's same-hour median, across 400--1000 km
in 50 km steps. Its threshold is ln(13/1e-6)/ln(2) times the median, motivated
by a fully coherent Gaussian exponential-power limit and guard-count union
bound. It is a conservative outlier guard, not an identified-satellite or
range--Doppler matched-filter CFAR catalogue. All pulses are screened before
all range/lag averages. Raw data are read only; rejected pulse timestamps and
flags are saved.

## Validation and uncertainty

- Recover H=0, 0.1, 0.4, 0.8 with He also free through an independently
  constructed finite-pulse/FIR response. Noiseless fraction error <0.005;
  temperatures recovered within 1%. This tests numerical recovery, not
  real-instrument calibration.
- Electron cache versus direct magnetized response: relative error <3.3e-5
  at the selected validation points. Ion-limit G error <0.00113 of G(0).
- 500 versus 250 Hz grid peak-relative spectral difference <4.4e-9 for the
  specified synthetic state; all data-model assumptions remain unchanged.
- Resample complete beam-dwell vectors to retain lag/range correlation and
  observed variability in the stability test. Only 11 dwells are available.
- Pointwise SEMs conservatively allow complete correlation among averaged
  range bins. The optimizer itself uses diagonal lag weights. Full joint
  covariance, response uncertainty and species distribution functions have
  not all been marginalized.

## Programs and storage

`hplus_extract`: bounded copy of existing Rust mean/SEM products for nighttime
zenith windows. `hplus_fit`: hourly mixed-ion fits and sensitivity profiles.
`hplus_clean`: read original raw files and recompute products after pulse
screening. `hplus_boot`: whole-dwell resampling and independent-temperature
joint profiles. `hplus_validate`: synthetic recovery and kernel tests.
`hplus_sensitivity`: proton-temperature controls. `hplus_msis_check`: prescribed
neutral atmosphere scale check. `hplus_export`: named float32 HDF5 datasets.

Authoritative products remain on avaruus:

    /nfs/urdr/scratch/juha/ruisdap/syisr/hplus/2023-11-15-night-zenith/

The real NFS mount must be present before bulk writes. Original raw data are
on Compuntu port 22000 at `/mnt/syisr/juha_data/LP/SY/20231115`. Build on a
machine with the matching HDF5 runtime (or provide that runtime explicitly).
Raw records, observational fits, PNG products and receiver data are NOT
vendored into this source repository. The shared `isr-ion` crate supplies
IGRF and independent pulse-response validation.

## Time–altitude map

`hplus_fit input.h5 output_dir UTC_HOUR` fits an explicitly selected hour.
`hplus_export_night ROOT` assembles the six pulse-screened hourly products
(15:00–21:00 UTC) into named float32 `(time, altitude)` datasets and a compact
JSON product. It uses `clean-profile/fits.json` for hour 16 and
`clean-hourly/HOUR/fits.json` for the other hours.

`plots/src/bin/time_altitude.rs` plots the H+ best fraction and conditional
profile-interval width using Rustplotlib. Failed convergence, temperature
boundaries and poor residual fits are blank. White dots mark conditional
support; undotted best fractions may have little information. No
interpolation fills rejected cells. Hourly integration is chosen for
composition sensitivity, not because the beam dwells themselves last an hour.

The cleaner uses buffered relative seeking when skipping non-selected IPPs;
this preserves byte offsets without discarding and refilling the 4 MB read
buffer at every pulse.

`hplus_timefit input.h5 output_dir UTC_HOUR 15` pools independent beam
dwells into quarter-hour integrations before fitting (all measured lags).
`hplus_export_night ROOT 15` exports the resulting 24 x 11 map. The live
page switches between 15-minute and hourly maps and their HDF5 downloads.
The quarter-hour map has 15 conditional-support cells out of 264; the hourly
map has three out of 66. Most supported cells are near 700 km before 18 UTC.
These are conditional profile checks, not simultaneous confidence bounds
across all cells or total systematic uncertainty.

The buffered-seek change was checked against the original hour-16 cleaner:
all real/imaginary mean and SEM arrays, pulse timestamps and exclusion masks
are bit-identical, including 776 excluded of 20,625 selected pulses.
