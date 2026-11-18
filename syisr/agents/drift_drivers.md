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
