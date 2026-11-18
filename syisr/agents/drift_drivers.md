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
