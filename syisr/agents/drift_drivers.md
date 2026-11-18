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

## Paired measured-driver SAMI2 comparison, 2026-11-18

User requested addition to the ISR high-level data-product skill and comparisons
with Fejer--Scherliess and HWM, including H+ under measured forcing. An existing
local skill was not found; a discoverable skill was created at
`~/.codex/skills/isr-high-level-data-product` with a SYISR-specific reference.

Native model and reference drivers were compiled locally with gfortran from the
existing sami2py scientific working copy. No external SAMI2 source is vendored
into Git. A narrow optional hook and patch helper live in operations/. The native
Fejer routine and HWM14 routine were evaluated at each measurement window center
and altitude. The model's eccentric-dipole magnetic latitude and field direction
were queried using its gtob/vector routines. Qualified IGRF14 perpendicular
velocity components were projected onto the native meridional direction and
divided by its cos^3/sqrt field-line mapping. Median300--500km estimates provide
one equatorial amplitude per30-minute window, requiring10qualified altitude
samples (correlated through the radar splines).49of50windows qualify.

Native time-series input is nonperiodic, in m/s and UT simulation hours from
2025-10-05T00:00UTC. The hook leaves Fejer unchanged before the first measured
interval, outside measured times and across invalid forcing brackets. Linear
interpolation is used between qualified centers; edge quarter-hours are held
only within the measured integration edges. Three otherwise identical cases
were run: baselineFejer, measuredcurrentACsign, and reversed-sign sensitivity.
HWM14, day279seasonal inputs, F107133,F107a154.3,Ap4, chemistry, grid and spinup
were held fixed. The native model repeats its seasonal day in time-dependent
neutral/solar routines; this is not an event-specific solar-history hindcast.
Only the supported perpendicular meridional scalar forcing is imposed. The
model does not simulate the measured zonal advection in its2Dmeridional plane.
HWM neutral winds and projected along-B wind are references, not inferred winds
or predictions of total ion velocity including diffusion/collisions.

All three50hmodel runs completed at~50.0025model hours, producing10915-minute
output records from23.25h. The first ion-density output is bit-identical across
cases, before measured forcing starts. The hook test checks absent input,
unchanged spinup, qualified interpolation, invalid-bracket fallback and lack of
periodic replay. Tests also check conservative linear interpolation and refusal
to extrapolate H50outside the500--1300km comparison domain. All tests pass.

Composition uses one model meridional plane anchored atSanya300km. Barycentric
sampling at18.3492N gives actual model longitudes109.4594--109.5858E over500--
1300km, retained in HDF5. Five-point Simpson integration averages ion densities
over each matching30-minute window before fractions are formed. H+ index1 and
He+ index5 in native one-based ion storage. All4050time/altitude points are valid.
The fitted composition is from the complete first shared sweep; temporal
refinement is a separate retrieval, not silently substituted here.

Results against the fitted H+ field (descriptive, not uncertainty-weighted):
- baselinefraction RMSE0.15161,MAE0.10788; H50RMSE127.93km,MAE103.22km;
- measuredcurrent-signfraction RMSE0.18470,MAE0.12823; H50RMSE125.13km,MAE99.84km;
- reversed-signfraction RMSE0.15814,MAE0.10880; H50RMSE118.48km,MAE93.31km.
H50errors use49in-domain fit crossings; one fitted crossing exceeds1300km and
remains missing rather than extrapolated. The present measured forcing does not
improve the full H+ fraction field despite a small improvement in H50error.
Do not select absolute LOS sign from these model scores. Original velocity
polarity remains unverified, with model/wind physics and retrieval systematics
also unresolved.

Working case root: ruisdap/local/syisr-ac-latest/sami2-measured-drift.
Products: drivers.h5,hplus-response.h5,comparison-summary.json and plots/.
Code: syisr_sami2_driver,sami2_drift_response,sami2_drift_comparison and
operations/patch_sami2_measured_driver.py. Scientific model references:
Scherliess and Fejer1999,doi10.1029/1999JA900025; Drob et al.2015,
doi10.1002/2014EA000089; SAMI2 by Huba,Joyce andFedder2000,
doi10.1029/2000JA000035. For publication include the model acknowledgement
required by the external SAMI2 release: this work uses the SAMI2 ionosphere
model written and developed by the Naval Research Laboratory.
