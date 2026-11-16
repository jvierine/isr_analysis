# Joint AC full-profile inversion

The joint spline inversion is a new calculation; gated-fit maps are not its results.
It consumes the immutable screened, quiet-background-subtracted complex decoded
products, with their actual native/adaptive gate edges and complete-cycle masks.
The earlier independent fits supply starting values and empirical density gain.

All five fields are clamped cubic B-splines: log electron density, log ion and
electron temperatures, signed spectral velocity, and bounded H fraction. Cubic
basis functions are nonnegative and sum to one. The initial interior break spacing
is 15 km below 400 km, increasing linearly to 150 km at 2000 km. This is a chosen
model resolution, not a claim of independently resolved plasma structure. The
source-cell quadrature remains at the native 20-us baud resolution: two exact
10-us subintervals per source cell. Existing gates are never replaced by knots.

The finite coded response is computed before the spectrum is integrated: all 64
phase codes, whole/half-baud decoder coefficients, both receiver-FIR endpoints,
and source cells contributing to neighboring gates are retained. The local Ne,
Ti, Te, velocity, H mixture, density-dependent spectral mass and 1/range^2 factor
vary inside every gate integral. Electron density is empirically calibrated to
supplied L2 density; the density comparison is therefore not independent.

H is zero below 500 km, with a C2 onset over 500–530 km; He is zero. NO+/O2+
are not fitted on the topside. The lower-ionosphere model still lacks molecular
ions; estimates below 200 km are explicitly flagged. Source-cell support extends
past the nominal fit boundaries. Splines have controls only within 100–1000 km;
physical profiles are extrapolated constantly into the source-cell halo, retaining
the geometric 1/range^2 variation. This explicit boundary assumption avoids fitting
unidentifiable halo-only controls. Gates within 30 km of the limits are flagged.

The working likelihood uses signed real and imaginary lag products with empirical
four-code-cycle-block variances. It does not claim a full off-diagonal covariance
likelihood. Uncertainty draws resample the same complete blocks across all
altitudes and lags, separately within each dwell. Every draw refits the full
profile and perturbs a common calibration gain using measured scatter with a
15% floor. Intervals require at least 32 converged draws. They are conditional on
spline spacing, the response model and the retained species model. Calibration
systematics, receiver-response uncertainty and effective resolution still need
sensitivity studies; smooth plots alone are not validation.

HDF5 products retain actual gate edges, knot vectors, transformed control
coordinates, predictions, signed observations, working uncertainties, nominal
optimizer history, bootstrap controls and convergence flags. Code provenance is
kept in source/data metadata and is not printed on scientific figures.

Meaningful tests cover B-spline partition and linear precision, increasing knot
spacing, finite-cell coded response versus explicit quadrature, physical-density
Jacobian dependence on spectral mass/Debye shape, existing AC decoding and
background cancellation, and beam-dependent native/adaptive gate geometry.

Run one nominal pilot with `syisr_joint INPUT_ROOT JOB_ID 0 OUTPUT_ROOT`.
Validate convergence before the MPI campaign. Production draws must be at least
32; 48 are intended. Compute is detached on Avaruus with at most 48 worker cores.
