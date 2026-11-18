# Drift reconstruction and measured SAMI2 forcing

Requested after the shared-composition fit is fast enough for routine use.
Preserve the physical model and objective; report strict-gradient and
measurement-space stability as different qualifications.

1. Validate LOS velocity polarity. The current spectrum shifts omega by
   minus k times velocity, but that alone does not calibrate the instrument
   I/Q polarity or identify toward/away. Compare supplied L2 velocities and,
   if necessary, an echo whose range-rate sign is independently measurable.
   A synthetic complex-phase test checks software conventions only.
2. Fit east, north and up velocity from all usable beam LOS projections on
   common nominal altitudes/times. Use beam-specific geographic gate geometry;
   never identify slant range with altitude. Retain condition numbers,
   leave-one-beam-out residuals, and robust consistency flags. Do not assume
   one regional vector is valid when residuals indicate spatial gradients.
   Thirty-minute windows pool asynchronous one-minute dwells and assume
   quasi-steady flow; shorter-scale structure needs a separate dwell-level
   retrieval rather than interpolation of the integrated velocities.
3. Use the local IGRF geometry to separate field-aligned and perpendicular
   flow. Deliver geographic east/north/up and magnetic perpendicular
   zonal/meridional components separately. The present inputs contain a
   magnetic-field model, not an independently measured magnetometer time
   series. Restrict electric-drift inference to appropriate F-region heights
   and account for neutral/collisional contributions and weak topside SNR.
4. Inspect the native SAMI2 external-drift interface and its time, sign,
   coordinate and field-line mapping conventions before creating drivers.
   Test measured meridional forcing against Fejer--Scherliess forcing while
   initially keeping HWM14 winds fixed. Check what east-west transport the
   two-dimensional implementation can represent; do not imply that a fixed
   meridional plane captures longitudinal advection without testing it.
5. Compare modeled and fitted H fraction, 0.5 crossing height and transition
   thickness, with the same observation geometry and integration windows.
   Treat drift-driven explanations of boundary excursions as hypotheses,
   with uncertainty and model-response limitations retained.

## Native SAMI2 forcing interface inspected 2026-11-18

The local sami2py checkout exposes `fejer=False`, a 10x2 Fourier coefficient
array, and constant `ve01`, all in m/s. The native `vdrift_model` evaluates
`ve01 + sum(A_i cos(i*pi*SLT/12) + B_i sin(i*pi*SLT/12))` for i=1..10.
This is a periodic solar-local-time driver, not the China civil-time axis
used in the observation figures. `exb` computes SLT from UTC plus the
geographic longitude of the equatorial grid point. It multiplies the scalar
by 100 to obtain cm/s and maps the drift along each field line with
`cos(latitude)^3 / sqrt(1+3*sin(latitude)^2)`. The current radial multiplier
is fixed to one (`altfac=1`), with separate low-altitude attenuation.

Consequences for a measured driver: project the measured vector onto the
model's perpendicular meridional direction and undo the actual field-line
mapping before fitting coefficients. A geographic vertical velocity alone
is insufficient. A daily Fourier fit imposes periodic continuation and
filters fast excursions; use a native time-series interface if that would
hide the fluctuations being tested. These source checks do not establish
absolute instrument LOS polarity. This entry point provides one scalar
meridional-plane forcing; it is not a general two-component electric-field
or longitudinal-advection interface. East-west drift needs a separate model
capability audit before claiming it drives SAMI2 transport.

Source provenance: local scientific working copy under
`ruisdap/local/syisr-ac-latest/sami2py`, `_core.py` and
`fortran/sami2py-1.00.f` (`exb`, `vdrift_model`). Retain code provenance here,
not in scientific figure labels.

## First full-day regional inversion, 2026-11-18

Implemented `drift.rs`, `syisr_drift`, and the Rustplotlib `drift_maps`
renderer. The complete first shared-composition sweep contains 50 measured
windows and 450 beam profiles. Exact stored velocity spline controls are
sampled at common nominal altitudes, 250--1000 km in 10 km increments;
this sampling does not redefine or refit the AC gates. Each ray is rotated
to ENU at its own geographic scattering position. Two independent regional
models fit common local geographic ENU components and common local magnetic
perpendicular-east, perpendicular-north/up, and along-B components. IGRF14
is evaluated at each beam position; the sign is inherited from the AC fit.

At least five beams are required, density must exceed 1e10 m^-3, and LOS
controls near the broad +/-2000 m/s optimization bound are excluded.
Equal initial beam weights and Huber IRLS (1.5, MAD scale floor 5 m/s)
handle beam-level disagreement. The present working objective has no
per-beam statistical errors because the plasma retrieval uncertainties
were disabled. Leave-one-beam-out spread is a conditional consistency
check, not a physical confidence interval. Acceptance requires weighted
condition <15, raw LOS residual RMS <30 m/s, maximum omitted-beam prediction
error <75 m/s, and every component jackknife spread <50 m/s.

First sweep: 3759 altitude/time inversions, 1364 pass magnetic consistency.
At 300/400/500/600/700/800/900/1000 km, the accepted counts are
45/30/33/23/11/4/0/0 (50 windows, except 49 at800,48 at900,45 at1000).
Thus the F-region estimate is substantially better supported than the
upper-range vectors. The raw products and unmasked plots are retained;
masking does not erase failed fits from HDF5. Four tests check exact
synthetic recovery, magnetic/zenith geometry, rejection of a rank-deficient
layout, and flagging/downweighting of a contaminated beam.

Local first-pass products: ruisdap/local/syisr-ac-latest/drifts-first-pass.h5
and drift-plots/. These are provisional regional-flow estimates and must
not yet be fed into SAMI2 as calibrated forcing. Absolute hardware LOS
polarity, spatial uniformity, and changes under the temporal refinement
remain to be checked.

## Altitude regularization and supplied L2 convention check

The user supplied Zhang et al. (2024), F Region Neutral Wind and Electric
Field Measured by SYISR and Evaluation, doi:10.1029/2024JA032514, and Zhang
et al. (2024), Calculation and Evaluation of Neutral Winds in the Lower
Thermosphere Based on SYISR Observations, doi:10.1029/2024JA032994.
The publisher abstracts establish the distinction between multibeam ion
vector reconstruction and momentum-equation inference of neutral winds and
electric fields. The former paper uses150--500km and the latter100--150km.
Our current product is ion velocity, not a momentum-equation neutral-wind
retrieval. Do not apply the F-region interpretation at100--150km: the
present plasma retrieval omits molecular ions there. The detailed full-text
methods were not reliably accessible through the publisher during this pass;
do not claim the new regularizer reproduces either paper's exact algorithm.

`syisr_drift_regularize` jointly fits cubic velocity profiles to all beam
LOS samples over250--650km, with30km spline break spacing and a second
altitude-derivative penalty. Three penalty lengths15,30,60km are tested,
plus a zero-penalty reference. Each validation fold withholds one entire
beam over the profile; selection uses its300--600km Huber prediction loss
(threshold30m/s). There is no time regularization at this step. This changes
the velocity reconstruction, not the expensive original AC likelihood.
All original independent-altitude fits are retained in the HDF5 product.
Above650km the original results and their consistency flags are retained.

All50 windows improve held-out loss over the same spline model without a
curvature penalty. Magnetic median loss ratio0.79588; geographic0.79436.
Magnetic selected lengths:60km in39windows,30km in8,15km in3. These are
penalty scales, not independently resolved altitude bins. The retained input
LOS samples are correlated through the plasma splines, so these are model
prediction checks, not formal likelihood-ratio significances. At300/400/
500/600km,44/41/37/30windows pass the unchanged consistency checks. A
synthetic affine-flow test verifies no artificial curvature bias and exact
prediction of a withheld beam under all tested penalty strengths.

The supplied L2 file was retrieved from compute node22000. `syisr_drift_l2`
compares successful300--600km samples within the same30min AC windows.
All9supported directions have smaller RMS discrepancy for unchanged than
reversed sign. However correlations are only0.018--0.452, and unchanged-sign
RMS discrepancies59--95m/s. This supports the same *relative* convention,
not independent absolute hardware polarity or validated electric fields.
Time-averaged comparison and independent range-rate sign calibration remain
appropriate before measured SAMI2 forcing. The raw paired comparison data
are stored in drift-l2-polarity.h5.
