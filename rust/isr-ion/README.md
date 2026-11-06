# Shared ion-line physics

Pure Rust physics and profile inversion, extracted from the RUISDAP source snapshot recorded in `SOURCE_PROVENANCE.json`. Scientific callers provide calibration, receiver response, geometry, atmosphere and measurement covariance explicitly.

## Range response and composition

`pulse_response::rectangular_cell` integrates finite range cells and applies a complex voltage FIR before correlation. Unit tests compare partitioned cells against the uniform-range analytic response.

`composition::dy1985` implements the IRI atomic-ion composition parameterization over 75–1000 km. Its species order is O, H, N, He. `topside::calion` implements CALION v2.5 over 350–2000 km, with species order O, H, He, N. CALION requires mixed invariant/dip latitude and magnetic local time; geographic latitude or solar local time must not silently replace them.

The coefficient values and equations were checked against the IRI-2026 reference in the external, read-only GUISDAP reference tree (`models/iri2026/irifun.f`, routines IONCO1 and CALION). Original Fortran source is not included. Tests contain numerical reference outputs from 1,204 independent single-precision Fortran cases, evaluated on Compuntu. Maximum observed fraction error is 1.086e-5; test tolerance is 2e-5. These tests validate the port, not the empirical model's accuracy for a particular recording.

`geomagnetic::igrf14` evaluates the official NOAA IGRF-14 coefficient series for 2020–2030, including ionospheric heights. Coefficients: https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf14coeffs.txt . Reference evaluator: https://www.ngdc.noaa.gov/IAGA/vmod/igrf14.f . The fixture covers 240 combinations of date, location and height; maximum observed component error is 0.0141 nT. `dipole_longitude` supplies central-dipole longitude, not invariant latitude or L-shell.

## Calibration and acceptance

A gate with `shape_density_m3` fixes Debye spectral shape while the first fitted coordinate becomes a nuisance amplitude. Such output must not be labeled calibrated electron density. Optimizer convergence and conditional curvature errors do not establish scientific acceptance. Receiver assumptions, missing absolute calibration, weak signals and prior-dominated cells remain the adapter's responsibility.
