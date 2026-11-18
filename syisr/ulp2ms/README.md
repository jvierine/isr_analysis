# SYISR 2 ms uncoded long-pulse experiment

This package lives under `syisr/ulp2ms` on `main`. The current December 2025
baseline is the full-range-screened, full-covariance 48-window analysis
(section "Extension through 5000 km"). The sigmoid/Gaussian model is an
optional comparison, described at the end. Earlier sections retain the
history of narrower-domain and diagonal-covariance experiments; their
limitations and results do not describe the final baseline.

Rust processing of the SYISR uncoded 2 ms raw-voltage recording, including
range–Doppler IQR/coherence satellite rejection, lag-profile estimation,
H+/He+/O+ mixture fits, and rustplotlib figures. All scientific products use
HDF5; raw recordings and generated outputs are excluded from Git. The
implementation intentionally supports the layout below and fails on other
sampling, pulse timing, beam geometry, or frame structure.

## Build

### Structured Gaussian voltage covariance validation

`structured-noise-cv` tests a physical covariance target against separate
temporal halves of disjoint adjacent-block differences. The target includes
the exact partial-product index sets, overlap factors, real/imaginary Gaussian
fourth moments, receiver filtering, nonstationary plasma self-noise, and the
single shared highest-gate reference. Reference voltages are independent of
the fitted range gates under this model; observed long-range reference
correlations are checked separately. No fitted plasma residual trains it.

Training-only local voltage ACFs are reconstructed before background
subtraction and overlap correction. Positive spectral projection creates
valid stationary plasma components, mixed with smooth voltage amplitudes
(90-km Gaussian scale). A compact positive-definite lag taper has support
600 us. Signed signal-power estimates and observations are retained; only
the covariance construction projects negative signal spectra. This is an
approximate covariance model, not an additional plasma measurement.

`SYISR_COVARIANCE_NUGGET=0.01` mixes the normalized structured target with
1% identity before the separately tested low-rank empirical correction.
This prevents excessive confidence in small-eigenvalue directions. On all
48 windows, temporal-half validation gives median normalized held-out energy
1.063 (expected 1), and median real adjacent-lag correlation 0.511 predicted
versus 0.524 observed. Seven windows have held-out energy above 1.5; windows
14 and 45 score worse than the diagonal baseline. Their temporal asymmetry
requires further investigation before replacing the campaign likelihood.
Hyperparameters were explored on these validation blocks: these are
development validation results, not independent nested test-set guarantees.
The existing plasma campaign remains unchanged.

Increasing the diagonal component to 10% gives median held-out energy 0.964,
all 48 windows improve their predictive score over the diagonal baseline,
and none has energy outside 0.7..1.5. Alternating-pair tests of windows 14
and 45 with the 1% component still fail (energies 2.39 and 3.75), so a slow
change between temporal halves alone does not explain the discrepancy.
The 10% result supports regularizing covariance directions that the Gaussian
model predicts too confidently. It does not establish exact covariance or
remove the persistent low-altitude mean residual.

```sh
SYISR_COVARIANCE_NUGGET=0.01 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=4 \
  target/release/structured-noise-cv FIT_ROOT OUTPUT_ROOT WINDOW_LIST
target/release/plot-structured-covariance OUTPUT_ROOT
```

`WINDOW_LIST` is a comma-separated list of zero-based windows. The optional
`SYISR_COVARIANCE_INTERLEAVED=1` uses alternating disjoint pairs to diagnose
time dependence; it does not replace the stricter temporal-half test.
`validation.h5` stores signed range/lag correlation kernels. JSON files
contain scalar diagnostics and the selected shrinkage, not scientific data
arrays. Native macOS Accelerate Cholesky and triangular solves support dense
structured whitening; other platforms use nalgebra. Tests compare the
nonstationary prefix sums with direct Gaussian pair sums, validate simulated
complex voltages, and verify whitening of the full structured/empirical
mixture against the identity covariance.

### Residual and noise-covariance audit of the 200-km campaign

The audit uses all 48 completed profiles, 31 observed gates and 60 fitted
nonzero lags (10..600 us), stacking real/imaginary parts into 3720 real
measurements. It does not assume fitted residuals have the covariance of
unfitted noise: parameter estimation, constraints and smoothing change their
distribution. Measurement correlations are instead estimated directly from
the saved 10-second block ACFs, using the actual pulse counts and satellite
mask. Pre-subtraction block products are reconstructed by adding back the
same saved highest-gate background, including overlap scaling.

Disjoint pairs of adjacent 10-second blocks provide a second estimate.
For counts n0,n1, the difference is divided by sqrt(1/n0+1/n1), then its
covariance is divided by total retained pulses to estimate covariance of
the integrated mean. Pair differences are centered; this suppresses slow
variation but does not prove pure thermal noise. Differences are formed
only when both actual adjacent blocks exist. Temporal correlations are
checked separately, without joining over missing blocks.

The individual SEM magnitudes are broadly supported: the campaign median
ratio of differenced to saved variance is about 0.995 (pooled 5th..95th
percentiles about 0.73..1.19). Median detrended 10-second-block temporal
correlation is -0.007. The diagonal *joint* covariance assumption fails:
campaign medians of pair-averaged differenced correlations are approximately
0.53/0.54 for real/imaginary products at 10 us separation; 0.49 at 50 km gate
separation; and 0.24 at 300 km separation for the same lag. The corresponding
300-km correlations before background subtraction are only 0.011/0.001.
The shared reference therefore contributes an observable long-range term.
These correlation averages are distinct from medians over individual pairs;
non-overlapping temporal halves independently retain strong lag correlations.
Range bins are 25 km wide, accepting actual nominal offsets within 6 km.

For stacked real/imaginary products y=a-Wb, the exact covariance decomposition
is `Cy = Ca + W Cb W^T - Cab W^T - W Cba`. Here a includes the actual partial
pair sums and overlap correction, b is the reference ACF, and W repeats that
reference at every range with the appropriate lag scaling. Treating reference
errors as independent at each range loses this term. Local covariance also
contains the overlapping sums, voltage-filter response and plasma self-noise.
Cross terms may be dropped only if their independence is justified.

Fitted residual RMS is about 1.03 SEM in real and 0.99 in imaginary parts
(medians over windows); this does not establish independence. The mean
residual maps retain lower-range structure, especially below ~450--500 km.
For the noon reference, low-gate real residual RMS is 2.10 SEM. Reducing
smoothing weight from 1 to 0.1 only reduces it to 1.98 SEM (diagonal data
chi-square 5649.89 to 5350.71). Thus correlations and smoothing do not fully
explain that misfit; basis, boundaries, receiver/pulse response and background
assumptions remain candidates. The audit does not identify a unique cause.

Empirical full covariances are rank deficient at this block count. Two-fold
covariance prediction on separate temporal halves of disjoint block-pair
differences tests alpha in 0.05,0.1,0.2,0.35,0.5,0.7,1. Training variances and
covariance use no fitted residuals. The target is the training diagonal and
the score is log-determinant plus held-out quadratic energy per coordinate;
alpha=0.7 is typically selected. This improves covariance prediction score
over diagonal weighting but leaves held-out energy ~1.33 per coordinate
(diagonal ~1.06), so it is not a perfectly calibrated final covariance model.
Selection score is a cross-validation diagnostic, not an independent
post-selection significance test. Stronger structural pooling is warranted.

Diagnostic noon refits using full-block empirical covariance at alpha=0.7
and 0.5 converge and replay their saved objectives to ~3e-15 relative error.
Over 500..1400 km, maximum H+ changes are 1.21 and 1.25 percentage points;
maximum Te changes are 12.1% and 13.5%, concentrated toward the upper end.
These are sensitivity tests, not uncertainty intervals. All refits use the
same screened data, gain, physical model and knot grid; the campaign remains
unchanged. Weighted chi-squares under different covariance matrices are not
directly comparable; a common diagonal replay cost is saved separately.
The lower residual pattern remains under both covariance and weaker-smoothing
tests. No assumption that all remaining noise is Gaussian or satellite-free
is made. Same-product real/imaginary correlations are near the finite-sample
estimation scale, but this does not prove every cross-component term vanishes.

```sh
cargo run --release --offline --bin residual-covariance -- \
  ORIGINAL_ROOT PARTIAL_PROFILE_ROOT AUDIT_ROOT
cargo run --release --offline --features plot --bin plot-residual-covariance -- AUDIT_ROOT
cargo test --release --offline --bin residual-covariance
```

The local audit root is
`/Users/jvi019/ruisdap/local/syisr-residual-covariance-20251218`.
`audit.h5` stores signed residuals, variance ratios, covariance kernels and
held-out scores; JSON files summarize diagnostics. Rust/rustplotlib figures
show covariance before/after subtraction and reference/campaign-mean
residuals. White represents zero, not masked data. Synthetic tests verify
white-noise variance calibration and recovery/prediction of known correlated
noise. The existing dense-inverse whitening and profiled-Jacobian tests also
remain applicable. Raw recordings and campaign parameter maps are not changed.

### Full-profile retrieval down to 200 km using partial products

The lower-range extension uses actual raw voltage pairs, not a drawn-on
Chapman continuation. The fitted physical grid is
`80,100,150,200,250,300,350,400,500,600,700,800,1000,1250,1500,1700` km
in slant range; maps display reconstructed profiles from 200 to 1400 km.
For this zenith beam altitude is slant range +0.05 km. The first three
knots are nuisance guards, not a claimed E-region retrieval.

The original native lag means reveal a large startup transient near 1e14 ADU²
over samples 0..16, followed by ring-down; by sample 30 the noon power is
about 4.1e5 ADU² and follows the atmospheric echo. Products therefore use
only early voltage indices `k=max(30,i-l)..i+20`, each paired with `k+l`.
The highest complete gate's complex ACF is subtracted per 10-second block,
then the actual pair average is multiplied by `200/(200-l)`. Partial
products are reconstructed at native gate indices 20,25,..85. Older complete
gates at indices 60..85 are replaced too, since their long-lag sums otherwise
include the startup transient. Native indices >=90 have their full stable
pair support for all 10..600 us lags and reuse the original products.

The observed nominal gate centres are approximately 200..425 km every 25 km,
followed by the original 450..1450 km every 50 km (31 gates, 1860 complex
products). Nominal gate coordinates do not imply a common partial-kernel
centroid: the range-lag kernels retain the actual lag-dependent support.
The operator uses `q=max(0,l+30-i)..l+20` and exactly the same pair-count
normalization as the estimator, including receiver-FIR cross-lags and the
analytic true-range inverse-square integral. No missing pair is invented.
Independent physical-time point-response tests cover truncated support,
the sample-30 startup guard, complex FIRs, and recovery of the complete limit.

Physical integration cells are 80,100,125,150,175,200 km below 200, and the
previous 25 km cells above it. Scattering below 80 km is set to zero. A fixed
30-amu molecular component supplies the lower nuisance-layer ion response:
its fraction is one below 150 km and decreases linearly to zero at 200 km,
replaced by O+. H+/He+ are suppressed in local spectra below 200 km. This
is a prescribed lower-guard mixture, not a retrieved NO+/O2+ composition;
neutral collisions and species-specific temperatures are omitted. These
assumptions should be tested before interpreting the bottomside quantitatively.
The displayed >=200 km region uses the existing H+/He+/O+ mixture. No Chapman
shape is imposed. Gain and the whole-pulse satellite mask are unchanged.
The present campaign uses diagonal SEM weights; correlated-error experiments
remain separate and must not be described as the campaign likelihood.

`partial-products` rereads raw frames locally using the saved plan and exact
`pulse_keep` mask; the raw disk read is serialized to avoid HDD seek thrashing,
while MPI distributes the profile fits over all 14 cores. Only the short
early-voltage prefix is decoded, although disk reads are sequential. All
outputs remain local HDF5, under `local/syisr-partial-profile-20251218`.
`validate-partials` independently recomposes early means from the original
saved native lag means and weighted original backgrounds. `profile-validate`
replays model costs, physical bounds, source mask counts and UTC invariants.
`audit-partial-support` records the exact native indices and valid pair counts
and reconciles descriptive metadata from the initial processing run.

```sh
cargo build --release --offline --features plot --bins
mpirun -np 14 --bind-to none target/release/partial-campaign OLD_ROOT NEW_ROOT
target/release/validate-partials OLD_ROOT NEW_ROOT
target/release/audit-partial-support NEW_ROOT
target/release/profile-validate NEW_ROOT
target/release/profile-campaign collect OLD_ROOT NEW_ROOT REFERENCE_PROFILE \
  80,100,150,200,250,300,350,400,500,600,700,800,1000,1250,1500,1700
target/release/interpolate-campaign NEW_ROOT NEW_ROOT/campaign-interpolated.h5
target/release/plot-campaign NEW_ROOT/campaign-interpolated.h5 \
  NEW_ROOT/time-altitude-all-finite --show-all
```

`--show-all` displays every finite fitted value in completed windows, including
weak signals, large conditional uncertainties, high residual costs and active
bounds. It leaves only genuinely missing/nonfinite results blank. The original
quality flags remain in the products; visibility is not a claim of precision.
Turbo scales are unchanged for plasma parameters, and the signed lag-profile
colour scale is unaffected. Fine display interpolation is not extra resolution.

The completed 200-km partial-product campaign has 48/48 converged windows.
Independent recomposition of 40,992 complex early products agrees to at most
4.96e-13 of a saved SEM; all 768 knot states passed the physical/I/O checks,
with maximum relative data-cost replay error 3.86e-15. The source pulse-mask
counts and common gain were checked in every window. The library suite has
30 passing tests. The noon 200 km node has ne=1.14372e11 m^-3, Ti=988.74 K,
Te=1022.02 K; its fitted density maximum is near 350 km at 1.02373e12 m^-3.
These use the provisional density scale and lower-guard assumptions above.
The final figure is `time-altitude-all-finite.png` in the local campaign root;
its only masking is for missing/nonfinite values, not scientific quality cuts.

### F2 bottomside boundary experiment

`SYISR_BOTTOMSIDE_CHAPMAN=400:100 profile-fit ...` uses a prescribed alpha-
Chapman log density below the indicated slant-range join, with scale height
H in km: `log(ne/Nm) = (1-u-exp(-u))/2`, `u=(R-Rm)/H`. The join must be an
existing interior knot with two upper neighbours. A cubic Hermite interval
above the join matches its zero density slope to the next linear interval's
slope, so density is C1 at both collar endpoints. The upper coefficients
and peak amplitude remain fitted at the same fixed receiver gain. Lower
density knot values are derived rather than independently fitted. Density
curvature stencils touching or below the join are omitted; temperature,
velocity and fraction bases are unchanged. This changes the prior/model
family as well as the number of free density coefficients. There is no
global monotonicity constraint: a prescribed join is not guaranteed to be
the global peak, so profiles must be inspected for secondary maxima.

This is a boundary sensitivity experiment, not a measured bottomside and
not a replacement of the campaign. It still uses observed gates 450..1450 km,
10..600 us, the same screened/background-subtracted ACFs, full range-lag and
inverse-square response, 25 km integration cells and diagonal SEM likelihood.
It does not yet add early partial products, molecular ions, or E/F1 layers.
Below 200 km a plotted Chapman curve is only the assumed F2 continuation;
the forward model's physical grid remains 200..1700 km.

At the 18 December 2025 04:00--04:30 UTC reference, the original nonuniform
fit had data chi-square 2445.24. With the smooth join fixed at 400 km, H=70,
100 and 120 km gave 3264.91, 2652.55 and 2621.23 (2520 real measurements),
respectively. The H=120 case changes density by at most 5.17% and H+ by
1.19 percentage points over 500..1400 km. Thus the smoother boundary carries
a measurable fit penalty and is not validated by appearance. H=50,140,150
and joins 350,375,425 km were also explored. Some lower-join fits reduce the
cost but develop a higher secondary density maximum; they are not evidence
for the prescribed join being a measured peak. Width and peak height are
not claimed as optimized or independently identified. The earlier C0
Chapman-only trials are retained locally as separate products.

`plot-bottomside baseline.h5 trial1.h5 trial2.h5 trial3.h5 output-prefix`
reconstructs the exact analytic/Hermite density, displays assumed parts as
dashed curves, and reports top-side sensitivity. HDF5 trials remain under
`/Users/jvi019/ruisdap/local/syisr-bottomside-20251218/`. Tests check Chapman
peak/slope, Hermite continuity and the full forward Jacobian. Formula reference:
https://doi.org/10.1038/s41598-023-28034-z (Chapman representation, Eq. 1).

### Correlated-error prototype and interpolated display

`SYISR_COVARIANCE_SHRINKAGE=0.1 profile-fit ...` enables a joint empirical
measurement covariance across real/imaginary parts, ranges and lags. Saved
10-second block ACFs already contain the actual trapezoidal gate sums and
per-block highest-gate subtraction. Weighted delete-one-block means use
exactly the existing SEM convention. If `U` contains their centered,
jackknife-scaled deviations divided by saved SEMs (floored at 1 ADU²), then
`C = D [alpha I + (1-alpha) U U^T] D`. `D` is that diagonal SEM matrix.
This captures measured shared-product and shared-background correlations,
conditional on the retained whole-pulse satellite mask. It also includes
block-to-block plasma variability; independence/stationarity of blocks and
covariance estimation uncertainty remain limitations. It is not an exact
instrument-only covariance derived from voltage fourth moments.

With 2520 real observables and at most 180 blocks, the unregularized estimate
is rank deficient. Alpha is an explicit sensitivity parameter, not a fitted
physical quantity or validated optimum. Low-rank symmetric whitening avoids
constructing/inverting a dense 2520-by-2520 matrix. Calibration and all residual
background amplitudes are profiled jointly in the whitened space; simply
whitening the old independently profiled residuals would be incorrect.
The normalized covariance factors and alpha are stored in each profile HDF5.
The validator can replay either diagonal or covariance-weighted objectives;
use an optional window index to validate one experimental window.

The noon nonuniform-grid prototype converged for alpha=0.1 and alpha=0.5,
with data chi-squares 15731.97 and 3285.82 respectively, against 2520 real
residuals before nuisance/profile degrees of freedom. Such sensitivity needs
assessment before treating correlated retrievals as a final campaign result.
The 48-window nonuniform campaign documented below still uses diagonal SEM
weights. Correlation experiments are stored separately under
`local/syisr-correlated-profile-20251218`; they do not replace that campaign.
Tests verify whitening against a dense inverse, the delete-block diagonal,
the correlated profiled Jacobian by finite differences, and recovery of the
old projection when alpha=1. The full library suite has 28 passing tests.

`interpolate-campaign profile-root output.h5` reconstructs the fitted fields
every 10 km over 500..1400 km for display. It interpolates the actual transformed
linear hats, reconstructs helium nonlinearly, and propagates the full cross-knot
conditional parameter covariance. Quality masks are inherited from the nearest
supported knot, with interpolated error cuts; the fine display is not new
spatial resolution. Pass the resulting HDF5 to `plot-campaign` as usual.

The 500 km display cutoff is an analysis choice: the present objective only
uses observed gate centres 450..1450 km, and lower knots were treated as guards.
The first stored complete 0..600 us trapezoidal gate is at 255.723 km; its
range ambiguity includes lower physical altitudes. Extending retrieval toward
200 km requires lower observed gates and a physical grid extending below their
entire FIR-broadened support. A 100 km claim additionally needs a suitable
molecular-ion model and treatment of the partially observed long pulse.

### Nonuniform full-profile grids

#### Extension through 5000 km (17--18 December 2025)

The extended inversion retains the partial-pulse lower profile and uses physical
knots at 80, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 1000,
1250, 1500, 1800, 2150, 2550, 3000, 3550, 4200, 5000 and 5500 km slant
range. The 5500-km knot is an upper nuisance guard. The increasing upper
intervals are 300, 350, 400, 450, 550, 650 and 800 km; the final guard interval
is 500 km. Knot spacing does not establish measurement resolution.

Observation centres retain 25-km sampling at 200--425 km and 50-km sampling at
450--1450 km, followed by 100-km sampling through 3000 km and 200-km sampling
through 5000 km. Each requested centre selects the nearest stored native gate;
the last is 4999.9386 km. All 60 complex lags from 10 to 600 microseconds enter
the fit, with signed values retained. The 25-km quadrature continues to 5500 km;
the exact ambiguity operator rejects truncated pulse/filter support.

The initial extension reused the original pulse mask and diagonal errors.
Full-range covariance validation exposed large brief echoes outside the old
screen (native samples through 1660), so that initial result is diagnostic
only. The final campaign reprocesses all raw windows with the same screen
thresholds extended through native sample 3782. Complete and lower partial
products are now accumulated from the same native blocks and pulse mask.
Signed data remain unchanged in convention; rejected blocks remain missing.

The final likelihood uses the structured Gaussian plus empirical covariance
of all 6840 real coordinates (57 ranges, 60 complex lags), including local
overlap, real/imaginary cross terms, receiver filtering, plasma self-noise,
and the common highest-gate reference. A 10% diagonal component regularizes
the structured target. Temporal-half prediction on disjoint adjacent-block
differences selects the structured/empirical mixture; the full covariance is
then estimated from unfitted block products and held fixed during fitting.
Residual-background amplitudes are profiled jointly after whitening. Saved
Cholesky factors and empirical factors specify the complete covariance and
allow independent cost replay. Covariance selection/estimation uncertainty is
not part of the reported conditional parameter curvature.

The original gain, range response, regularization, and density normalization
are retained. The inferred FIR is re-estimated from each reprocessed window.
Weak signal, active bounds, and receiver/background/calibration systematics
still limit high-range interpretation; covariance weighting cannot remove a
biased forward model.
IGRF14 is continued as an internal-field model to the upper guard; external
magnetic fields are not included. Independent field values and a time-domain
electron-response integral test the extended numerical domain.

Set `SYISR_OBSERVED_MAX_KM=5000` for both fitting and collection. The physical
domain is inferred from the explicit knot list. `SYISR_PARTIAL_ROOT` reuses
the saved partial products for each window; `SYISR_INITIAL_ROOT` supplies
initial states only. `SYISR_PROFILE_ITERATIONS` sets the iteration limit.
Resuming preserves unsuccessful products and initializes from their full
profile. Each MPI rank records failures without aborting other ranks.

The initial extension is retained under
`/Users/jvi019/ruisdap/local/syisr-extended-profile-5000km-20251217-18/`.
The final raw products are in `syisr-full-range-screened-20251217-18` and the
correlated fits in `syisr-full-range-correlated-profile-20251217-18`, under
the same local parent. Use `SYISR_COVARIANCE_ROOT` for the prepared per-window
covariance factors. The inversion still extends to 5000 km; the requested
figures use `SYISR_PLOT_MAX_KM=2500`. Screened density
requires convergence, no temperature-bound flag, reduced weighted chi-square
at most three, conditional log-density error below 0.5, and density resolution
diagonal above 0.1. Temperature and composition panels retain their original
conditional-error thresholds. These are conditional display checks, not
proof of a high-range detection. HDF5 retains rejected and failed estimates.

The completed extended campaign converges in all 48 windows at the unchanged
normalized projected-gradient tolerance of 1e-3 (maximum 0.000998905).
Powell-damped BFGS resolves stagnation in weak upper-profile coordinates;
it changes the optimizer, not the objective or the returned Gauss--Newton
conditional curvature. Of 2,154,424 input pulses, 1,831,788 are retained;
the extended screen rejects 29,780 additional pulses without reaccepting any
previously rejected pulse. Selected temporal-fold normalized energies span
0.804--1.188 (median 0.879), and covariance prediction improves over diagonal
weighting in every window. These are development checks used in covariance
selection, not an independent nested test of total physical uncertainty.
Independent likelihood replay agrees within 1.2e-14 relative error in all
48 windows. The final directory contains `validation.json`,
`end-to-end-audit.h5`, and combined `covariance-validation.h5` diagnostics.

`profile-fit` accepts either a uniform spacing or an explicit comma-separated
list in its fourth positional argument after the executable. Knots are true
slant ranges in km, finite and strictly increasing. Explicit grids can begin
at 50--200 km and end at 1700--6000 km; the legacy uniform domain remains
200--1700 km.
For example `200,300,400,500,600,700,800,1000,1250,1500,1700` retains 100 km
spacing below 800 km and uses 200--250 km spacing above it (the last guard
interval is 200 km). Linear hats and the curvature penalty use actual adjacent
knot distances. The 25 km integration grid is independent of these knots.

`profile-campaign run|collect old-root new-root reference.h5 [grid]` accepts
the same specification. Use a new output root for grid comparisons. A resumed
product is accepted only when converged and its knots match the requested grid.
The reference supplies the fixed gain; the campaign does not recalibrate it.
`compare-grids uniform-root nonuniform-root output-prefix` writes HDF5
differences, a JSON summary and Rust/rustplotlib noon overlays. Comparisons
reconstruct transformed fields, including the conditional-helium conversion.
Map cell boundaries use midpoints between actual nonuniform knot coordinates.
They display knot estimates; cell widths are not averaging-kernel resolutions.

The 18 December 2025 comparison used identical screened ACFs, receiver response,
10--600 us fit lags, 25 km integration cells, fixed gain 25899.08075225199,
and the existing distance-scaled curvature rule with weight one. All 48 windows
converged (one required 320 rather than 200 iterations); 528 physical nodes
passed the independent invariant replay, with maximum relative data-cost
discrepancy 3.95e-15. On 25 km evaluation samples over 500..1400 km, median
absolute grid changes were 1.27% in density, 0.87% in Ti, 1.06% in Te,
0.47 percentage points in H+, and 0.073 points in He+. The H+ 95th percentile
was 4.49 points. These comparisons include weak estimates, with outliers as
large as 41 H+ percentage points; they are sensitivity measures, not errors.
Median change in the data chi-square was +3.85 for 2520 real residuals.
Changing the grid also changes the number and positions of curvature terms;
the smoothing rule is held fixed, not an identical continuous statistical prior.
The reduced parameter count does not by itself establish improved accuracy.

Local products are under
`/Users/jvi019/ruisdap/local/syisr-nonuniform-profile-20251218/` and remain
outside Git. The original uniform-grid campaign is preserved separately.

```sh
cargo build --release --features plot --bins
cargo test --release --all-targets --features plot
```

A native HDF5 library is required (`brew install hdf5` on macOS). Plotting
uses the pinned Git revision of `jvierine/rustplotlib` and its local LaTeX
and SVG export tools. Dependencies must be available at build time; the
compiled processing, fitting, MPI campaign and plotting commands run offline.
No Python or MATLAB runtime is used. The MATLAB reader supplied with the
recording was the byte-layout reference.

## Raw frame format

All integers are little-endian. Byte offsets are **zero-based**. Each frame
contains a 1024-byte header followed by signed 32-bit sample pairs. This
recording stores **3783 complex pairs**, although its header sample count is
3782. The supplied MATLAB reader uses the first 3782; this implementation
skips the extra pair and checks the following header. The frame stride is
therefore **31,288 bytes**, not 31,280. There is no separate data trailer
sentinel: both sentinels below belong to the header.

| Byte offset | Type | Meaning / supported value |
|---:|---|---|
| 0 | u32 | Frame marker `0xAAAA5555` |
| 12 | u8 | Month |
| 13 | u8 | Year minus 2000 |
| 14 | u8 | Hour |
| 15 | u8 | Day |
| 16 | u8 | Second |
| 17 | u8 | Minute |
| 18 | u16 | Fractional-second ticks, each 25 microseconds |
| 32 | u8 | Radar frequency in MHz: `430 + 0.2 * value` |
| 36 | u8 | Low nibble sample-rate code, high nibble waveform code |
| 37 | u8 | High nibble receiver-bandwidth code |
| 40 | u16 | Mode field, retained without interpretation |
| 668 | u16 | Raw code-width field, microseconds |
| 764 | u8 | DBF output beams; must be 1 |
| 766 | u16 | Receive-gate leading edge: 2001 microseconds |
| 768 | u32 | Analysis complex samples: 3782 |
| 856, 858 | u16, u16 | Receive azimuth/elevation; degrees = raw * 360/65536 |
| 868, 870 | u16, u16 | Transmit azimuth/elevation, same conversion |
| 936, 942 | u16, u16 | Big/little-code fields, retained |
| 980 | u16 | IPP: 40,000 microseconds |
| 988 | u16 | Transmit pulse width: 2000 microseconds |
| 1020 | u32 | Header-end marker `0xAA5555AA` |
| 1024 onward | i32 pairs | First word Q, second word I; complex voltage = I + iQ |

The supported byte 36 is `0x41`: sample-rate code 1 = 100 kHz, waveform code
4 = long pulse. Thus native lag spacing is 10 microseconds. Zenith means
both receive and transmit elevation are exactly 16384 = 90 degrees. Frequency
is 430.2 MHz for this campaign. The physical radial-velocity sign has not
been independently calibrated.

The data owner confirmed that header clocks are **Beijing civil time,
UTC+08:00**. The reader retains a header-calendar epoch; the estimator
subtracts eight hours when producing UTC seconds. The recording occasionally
has one-tick timestamp excursions; continuity accepts at most 26 microseconds
around the nominal 40 ms IPP, and otherwise fails. Times in HDF5 are Unix UTC
seconds. Source filename dates are not used as the clock authority.

## Estimator and conventions

For each pulse, form `x[k] * conj(x[k+l])`, with lags `l=0..60` (0–600 us).
This convention has negative ACF phase for a positive-frequency
`exp(+i 2 pi f t)` voltage. Ten-second blocks have 250 pulses; a partial final
block is permitted. Satellite rejection precedes all lag/range products.
Every range and lag uses the same retained whole-pulse mask.

For output receive index `i`, trapezoidal summation averages the products
`x[j-l] * conj(x[j])` over `j=i..i+l+20`. Subtract the complex highest-gate
background at each lag, then divide by fractional pulse overlap
`(200-l)/200`. Background subtraction occurs **before** normalization. The
shown matrix is `Re R(r,l) / Re R(r,0)`. Negative power/ACF estimates remain
signed; near-zero denominators can produce arbitrarily large ratios.

Nominal monostatic slant-range centres in kilometres are

```text
r(i) = [2001 + 10 * (i + (21-200)/2)] * 0.149896229
```

The grid uses indices `60,65,...` through the last complete summed gate.
Its 7.495 km sampling is not independent range resolution. Trapezoidal
summation gives all lags the same centre and approximately 330 km extent,
but **their normalized weighting shapes differ**. Unit-area weights are
stored explicitly. Rectangular kernels omit receiver-filter tails; the fit
forward response additionally includes the inferred voltage filter. Plotting
starts at 300 km to avoid receive-gate recovery in the longest-lag products.

The background reference has `i=3701`, nominal 5713.44 km slant range. Each
block/lag uses its own reference estimate with the same trapezoidal summation
rule. An alternate reference at `i=3451` is retained for sensitivity checks.
This removes additive filtered receiver noise; it does not deconvolve the
signal transfer function. Range-dependent receiver background remains a
known limitation, and the single reference injects common estimation error
into all ranges. Each plasma fit profiles a free residual-background amplitude
along the noise ACF. It is not a general arbitrary background model.

## Satellite rejection

The same procedure used in the accepted pilot is applied per integration:

* A bright-echo guard compares 16-sample range powers against their temporal
  medians. A ratio above 64 rejects the whole pulse; a weaker ratio above 12
  is rejected with >0.92 single-tone power fraction and three-pulse support.
* The main IQR stage forms 128-sample Hann-window spectra, at stride 32
  samples, and averages eight pulses (0.32 s). Each range–frequency bin has
  its own temporal median, Q3 and IQR. Spectral excess above `Q3+6*IQR`
  needs >60% of local positive excess within three of eleven frequency bins
  and nearby time/range support within one Doppler bin.
* An integrated-power `Q3+6*IQR` guard also rejects severe broadband events.
  Pulse/time margins cover echo edges. Scores, spectral references and masks
  are saved. Blocks with fewer than 25 clean pulses are wholly excluded;
  their counts are zero and their block ACFs are NaN. At least three usable
  error blocks are required.

These flags identify radar-echo candidates; they are not orbit assignments
or a guarantee of removing every weak satellite. Raw voltages are unchanged.

## HDF5 data products

All complex arrays are split into real/imaginary datasets. `ADU²` means
squared native signed-integer voltage counts; no absolute calibration is
implied. Dataset dimensions follow HDF5/C row-major order.

### `process` output

Let `P` be input pulses, `B=ceil(P/250)`, `R` output range gates,
`L=61`, `S=3782`, `T=ceil(P/8)`, and `W=47` spectral windows.

| Dataset | Shape | Meaning |
|---|---|---|
| `time_utc_s` | P | Actual input pulse times, UTC |
| `slant_range_km` | R | Pulse-centred monostatic slant range |
| `lag_us` | L | `0,10,...,600` |
| `pulse_keep` | P | 0/1 final common pulse mask |
| `initial_pulse_keep` | P | 0/1 mask before the IQR stage |
| `pulse_max_guard_power_ratio` | P | Initial maximum excess-power ratio |
| `block_retained_pulses` | B | Weights/counts; zero means excluded block |
| `acf_{real,imag}_adu2` | R,L | Corrected signed ACF |
| `acf_before_background_{real,imag}_adu2` | R,L | Overlap-corrected ACF before reference subtraction |
| `block_acf_{real,imag}_adu2` | B,R,L | Per-block corrected ACF; NaN for excluded blocks |
| `background_acf_{real,imag}_adu2` | B,L | Highest-gate background before overlap correction |
| `alternate_background_acf_{real,imag}_adu2` | B,L | Alternate reference, same units |
| `native_lpm_{real,imag}_adu2` | L,S | Native unsummed, unsubtracted pulse-averaged products; invalid pair cells are NaN |
| `acf_{real,imag}_sem_adu2` | R,L | Delete-one-block jackknife SEM with retained-pulse weights |
| `normalized_acf_real` | R,L | Corrected real ACF / corrected real zero lag |
| `range_ambiguity_weights` | L,220 | Unit-area nominal rectangular-pulse range kernels |
| `range_ambiguity_offset_km` | 220 | Kernel coordinates relative to common centre |
| `iqr_spectral_score` | T,W | Maximum spectral-bin excess in IQR units |
| `iqr_peak_frequency_hz` | T,W | Frequency of that maximum |
| `iqr_narrow_excess_fraction` | T,W | Three-bin/local-eleven-bin excess ratio |
| `iqr_spectral_candidate` | T,W | Candidate flag before neighbor support |
| `iqr_power_score` | T,W | Integrated-power excess in IQR units |
| `iqr_spectrum_{median,iqr,q3}_adu2` | W,128 | Temporal spectral references |

The `metadata_json` attribute contains a UTF-8 byte array with the raw header,
UTC interval, counts, reference index/range, sign conventions, units,
processing choices, and limitations. Do not assume this attribute is an HDF5
variable-length string: read its bytes and decode UTF-8.

### Geometry and fit products

`geometry` writes `range_index` (i64), `slant_range_km`, `altitude_km`,
`field_nT`, `angle_from_perpendicular_deg`, plus the IGRF epoch. Only verified
zenith is supported: altitude = slant range + the 0.05 km station elevation.
This is an explicit geometry conversion, not a relabelling of slant range.

`fit` writes a `range_NNN` group for each attempted usable range. Each group
contains `lag_us`, observed/model real/imaginary ACFs in ADU², real/imaginary
SEM arrays, and a conditional H-fraction profile in
`hydrogen_profile_{fraction,delta_chi2}`. Its `metadata_json` byte attribute
contains temperatures, velocity convention, H/He/O fractions, signal and
background amplitudes, chi-square/dof, convergence, local conditional errors,
assumed density and range-gradient model. Root `time_utc_s` contains interval
start/end; `failed_gates_json` lists gates that could not be fitted.

The reused Rust theory has magnetized Maxwellian electrons, a separately
checked large-k-rho unmagnetized ion limit, and H+/He+/O+ sharing one Ti.
It is a collisionless topside approximation and rejects unvalidated
near-perpendicular geometry. Signal amplitude and residual background are
profiled linearly; bounded multistart optimization fits Ti, Te, spectral
velocity, and two independent fractions. Default minimum fitted lag is zero;
pass final argument `1` to fit 10–600 us. Imaginary zero lag is never an
independent observation. A zero-lag-free objective still uses zero-lag power
for its provisional density proxy, gradient, and plot normalization.

Plasma shape is constant inside each broad footprint, with a local
exponential amplitude gradient. These are overlapping broad-volume fits,
not a coupled fine-altitude inversion. Local curvature errors and diagonal
jackknife weights omit cross-lag covariance and systematic uncertainties;
small best-fit He fractions are not automatically significant detections.

### Campaign product

`campaign collect` writes `campaign.h5` with `(altitude,time)` fields
`h_fraction`, `he_fraction`, `o_fraction`, `ne_m3`, `te_k`, `ti_k`,
`reduced_chi2`, `sigma_h`, `sigma_he`, `sigma_ti_k`, `sigma_te_k`, and
`background_adjustment_adu2`. It retains separate `slant_range_km` and
`altitude_km`, UTC edges/centres, input/retained counts, `window_complete`,
`density_valid`, and `quality_flags`. Pending values are NaN.

Quality bits are 1: no convergence; 2: reduced diagonal chi-square >3;
4: temperature bound; 8: weak/nonpositive signal; 16: singular local
covariance; 255: no fitted gate. The display also excludes temperature errors
above 50% and composition errors above 0.15 absolute fraction. Raw results
are preserved. The density panel uses its own positive-power significance
mask and does not require a well-constrained composition fit.

`ne_m3` is currently a **provisional P(r)r² proxy**, with one fixed scale from
a noon reference whose F-region peak is assigned 1e12 m⁻³. It preserves
relative changes across the day, but has no ionosonde calibration or
range-dependent Te/Ti power correction. Set `SYISR_PEAK_POWER_RANGE2` to the
reference maximum P(r)r² to use one scale across separate `fit` commands;
without it each fit file uses its own peak.

## Commands

### Joint full-profile inversion

`profile-fit` jointly fits a physical altitude profile to the **existing signed,
trapezoid-summed ACF product**, reusing its pulse mask and top-gate subtraction.
It is a nonlinear profile inversion, not an inversion of a normalized plot.
The forward operator includes every selected voltage pair, finite rectangular
2 ms TX pulse and inferred 41-tap voltage FIR. FIR cross-terms couple intrinsic
lags from -400 to +1000 us. Only measured 10--600 us lags enter the objective.

The soft-target power factor `(1000 km/R)^2` is integrated analytically inside
every constant-weight geometric interval, **before summing ranges**. Plasma
spectra vary over physical quadrature cells (default 25 km). Six transformed
fields are linearly interpolated between 100 km knots from 200 to 1700 km:
log density, log Ti, log Te, common velocity, H fraction and He/(1-H).
Observations use 21 centres near 450--1450 km, at about 50 km spacing. The
outer profile knots are guard regions, not independent measurements.

Unlike the old normalized spectrum/amplitude fit, the profile model retains
the unnormalized spectral integral, so local echo strength is proportional
to `ne * spectral_mass / R^2`, including temperature/composition dependence.
One common gain and one signed residual-noise amplitude per observed range
are profiled analytically. With `auto` gain, one assumed peak knot is fixed
to 1e12 m^-3 and other knots capped there. The anchor is selected on the
100 km reference grid from positive zero-lag P(r)r² and explicitly included
when comparing knot spacings; the executed noon reference uses 400 km.
Use that fitted gain for all other times. Density remains provisionally
scaled, but is a jointly inferred profile rather than the old P(r)r² proxy.

An explicit second-derivative penalty uses weight 1 by default and scales
`[0.5,0.25,0.3,0.15,0.15,0.1]` over 100 km. Data chi-square and penalty are
stored separately. Conditional transformed covariance and local resolution
matrices are exported in HDF5 row-major order; fixed/active-bound entries
are NaN. These omit range/lag covariance and instrumental systematics.
Knot spacing does not establish spatial resolution. The inferred FIR phase,
TX shape and range-dependent background are still assumptions.

```sh
target/release/profile-fit lpm.h5 seed-fits.h5 noon-reference.h5 100 1 auto 120 25
target/release/plot-profile noon-reference.h5 seed-fits.h5 noon-profile
mpirun --host localhost:14 --bind-to none --mca btl self,sm -np 14 \
  target/release/profile-campaign run old-campaign new-campaign noon-reference.h5
target/release/profile-campaign collect old-campaign new-campaign noon-reference.h5
target/release/profile-validate new-campaign
target/release/plot-campaign new-campaign/campaign.h5 full-profile-time-altitude
```

Optional `profile-fit` arguments are knot step, roughness weight, `auto` or
fixed gain, maximum iterations, and physical-cell width. Knot/cell steps must
partition the 200--1700 km physical domain. Existing products are not silently
overwritten. `profile-campaign` reuses converged windows with retained FIR
metadata and retries nonconverged/incomplete ones; all scientific outputs
remain separate from the original independent fits. Runtime is local/offline.

Profile HDF5 contains intrinsic knot fields (`ne_m3`, `ti_k`, `te_k`, fractions,
velocity), observed range/lag axes, signed observations/predictions/errors,
the physical-cell edges, receiver FIR and profiled background amplitudes.
Metadata records the input products, calibration assumption, regularization,
fit history, costs and limitations. The collector displays only supported
knots; density additionally needs <50% conditional relative error and local
resolution diagonal >0.1. Bound fraction errors are withheld, so a blank
fraction cell can mean a bound-limited estimate rather than missing data.

Validation includes direct voltage/range quadrature of the filtered response,
partition-invariant inverse-square integrals, agreement with the stored
trapezoidal kernels, analytic variable-projection Jacobian checks, stratified
synthetic plasma recovery, and independent replay of saved campaign fit costs,
penalties, physical states and source pulse-mask counts.

For the focused range-lag proof audit (UiT Memo 58), run:

```sh
target/release/ambiguity-proof noon-reference.h5 range-lag-proof
```

This independently replays physical-time voltage products for four irregular
range populations with different strengths and positive/negative frequencies,
using both a prescribed complex FIR and the inferred noon FIR. The diagnostic
also exports exact unfiltered geometric gains/centroids in HDF5 and an
explanatory rustplotlib figure. Its JSON report is an audit summary, duplicated
in the HDF5 metadata. The test suite includes the stratified-voltage
replay and the survival of lag-dependent geometric weighting after zero-lag
normalization. No estimator or campaign fit changes are made by this audit.

`ambiguity-impulse noon-reference.h5 output-prefix` exports two-dimensional
responses of range-Dirac / lag-Kronecker inputs at 600 km and intrinsic lags
170 and 500 us. It also shows one fixed measured product's sensitivity in
input coordinates. Both figures include trapezoidal summation and inferred
voltage FIR cross-terms. The input spike is a formal operator basis, not a
standalone physical ACF; its Hermitian partner must be added when appropriate.
Coefficient densities in km^-1 and input/output axes are saved in HDF5.
All real plot panels share an explicit normalizer and use an asinh stretch;
imaginary panels use the same unit with expanded limits. The full 23-test
suite checks the impulse map against direct physical-time tap products,
single-lag preservation for a delta receiver, Hermitian zero lag, and the
unit-area finite-cell limit. Production profile estimates are unchanged.

### Initial independent broad-volume fits

From this directory, with a selected raw window already extracted:

```sh
target/release/process selected.dat lpm.h5 --screen-bright
target/release/geometry lpm.h5 geometry.h5 350 1450 50
target/release/fit lpm.h5 geometry.h5 fits.h5 1
target/release/plot lpm.h5 normalized-lpm
target/release/plot-filter lpm.h5 top-gate-response
```

`plot-fits` is a four-range pilot layout; use it with four geometry targets.
`plot-campaign` provides the multi-range time–altitude view. `diagnostics`
and `background-audit` expose the measured filter/noise and range kernels.
Time–altitude plasma-parameter panels use Turbo; signed normalized
lag-profile matrices retain the original blue–white–red diverging scale.
Only scientific content appears in figures; implementation provenance is
kept in source comments and scientific-data metadata.

For the complete directory of supported raw files, using a previously
processed noon reference to establish the provisional density scale:

```sh
target/release/campaign plan /path/to/SYISR_data/20251218 output noon-lpm.h5
mpirun --bind-to none -np 14 target/release/campaign run output &
target/release/campaign watch output
```

`mpirun` supplies Open MPI rank/size environment variables. Each independent
Rust rank handles every Nth window with one Rayon thread. No MPI crate or
network service is needed; workers operate on local files and run the Rust
`process`/`fit` executables. Use the desired Mac core count for `-np`.
`watch` collects and renders new completed windows every 30 seconds; it can
be restarted independently. Window HDF5 files are isolated per rank and
completion JSON markers are atomic. Restarting `run` skips completed windows
and retries failed ones, reusing complete temporary raw selections when
available. Results and restart state remain in `output/windows/NNN`.

The campaign planner assigns each frame exactly once to consecutive
30-minute windows, including partial first/last windows. It writes
`inventory.json`, `plan.json`, `fit-geometry.h5`, and later `status.json`.
JSON holds configuration/provenance/restart state; numerical products remain
HDF5. The current layout assumes one extra IQ pair and a constant frame
stride across this campaign; new experiment layouts require explicit support.

## Validation

The tests cover signed I/Q decoding and malformed/truncated records,
range-kernel area/centre/support, independent nonuniform-range convolution,
colored-noise subtraction, satellite rejection of bright and weak persistent
tones while preserving a thermal population, finite-pulse/filter response,
IGRF against an independent reference, and recovery of known H/He mixtures
with free background and with zero lag omitted. Runtime checks validate the
electron cache and ion limit before fitting. The executed full-recording
campaign additionally checked all 48 intervals, masks, missing-block NaNs,
weighted ACFs, normalization, fit costs and fraction sums. Raw data are not
part of the repository.

### Sigmoid H and Gaussian He trial

The optional global composition model adapts the alternating-code formulation
from `syisr/src/h_transition.rs` at commit `0dbc990`. It evaluates the analytic
fractions at each physical response-cell altitude, inside the same exact
long-pulse lag/range operator. The model has five bounded parameters: H crossing
height 550--2000 km and local scale width 20--300 km, plus He non-H amplitude
0--0.5, centre 550--1600 km and Gaussian sigma 50--500 km. H is a logistic
(tanh-equivalent) function in the fixed IRI-temperature/gravity height
coordinate. He is `(1-H)*A*exp[-(z-centre)^2/(2*sigma^2)]`. Both retain the
alternating-code C2 onset: zero below 500 km, full weight above 530 km. Thus
H+He+O=1 without clipping physical fractions.

IRI supplies a shape prescription, not a measured temperature or composition
prior. Evaluate IRI2020 at each window midpoint and the verified zenith latitude
and longitude; monotonicize its `k*Ti/(15*amu*g)` scale height and integrate the
piecewise-linear table exactly. Above the 2000-km reference table, continue its
last slope linearly, as in the alternating-code implementation. This is an
extrapolation assumption, especially in the 5000-km extension. External model
sources, indices, binaries and numerical products remain outside Git; their
hashes are saved in each reference HDF5. The external driver uses the interface
of the included `operations/iri_shape_driver.f90`, adapted from that commit.

Global coordinates occupy columns 4, 10, 17, 23 and 29 of the saved transformed
parameter vector; all other composition slots are unused. They are not local
fractions. The HDF5 physical-fraction datasets and `h_transition`, `helium_layer`
and `composition_shape` metadata explicitly describe the new model. The four
thermal/density/velocity fields retain their original knots and roughness;
composition has no knot roughness penalty. Stored covariance is propagated
through the analytic composition Jacobian, including global cross terms;
active-bound and unidentifiable directions remain unavailable. A near-zero He
amplitude cannot determine its centre or width. These are conditional
curvature errors under the imposed shape, not total physical uncertainties.

Use a separate output directory. With the previous full-covariance campaign as
`BASELINE` and its matching screened products as `SOURCE`:

```sh
cargo build --release --features plot --bins
conda run -n base python prepare_composition_shapes.py \
  /path/to/iri_shape_driver /path/to/IRI-zip SOURCE TRIAL/shapes
conda run -n base python run_composition_trial.py SOURCE BASELINE TRIAL \
  --windows 0 16 24 40 --workers 4 --threads 3
```

The trial runner checks that the compared observations, SEMs, lags, ranges,
UTC times and covariance products match exactly, independently replays each
likelihood, writes `comparison.h5`, and renders matched profiles through
2500 km. The underlying inversion still reaches 5000 km. For a campaign run,
`SYISR_COMPOSITION_SHAPE_ROOT` names per-window `000.h5` etc. references;
`profile-fit` accepts one via `SYISR_COMPOSITION_SHAPE`. Unset these to retain
the original knot-composition model. Resume checks distinguish the two modes.

Validation includes analytic composition derivatives, physical simplex and
onset checks, full-response Jacobians after correlated nuisance projection,
and joint synthetic recovery of composition and thermal fields. The trial is
a comparison of imposed model families, not evidence that smoother curves
are intrinsically more accurate. Total penalized objectives are not directly
comparable when the composition roughness prior changes; report data costs
and penalties separately.

#### Executed four-window trial (17--18 December 2025)

All four nominal fits converged at the unchanged 1e-3 normalized projected
score threshold (maximum 0.000735). They use exactly the baseline observations,
SEMs, lag/range selection, full covariance products and gain. Independent
weighted-likelihood replay agrees to 1.75e-14 relative error.

| UTC window | H=0.5 altitude (km) | Local H width (km) | Data chi-square change | He bound reached |
| --- | ---: | ---: | ---: | --- |
| 17 Dec 16:02--16:30 | 713 | 73 | +5.2% | Non-H amplitude 0.5 |
| 18 Dec 00:00--00:30 | 894 | 159 | +5.2% | Sigma 500 km |
| 18 Dec 04:00--04:30 | 1014 | 167 | +4.0% | None |
| 18 Dec 12:00--12:30 | 891 | 149 | +25.3% | Sigma 50 km |

The noon Gaussian envelope has non-H amplitude 0.084, centre 836 km and sigma
61 km; its actual He fraction also includes the `(1-H)` factor. The constrained
composition changes topside temperatures, not just the appearance of H/He.
The model is retained as an optional trial: smoother composition alone does
not establish improved inference, three He solutions reach bounds, and these
are local nominal optima rather than a multi-start or coverage study. The
48-window knot-composition baseline remains preserved. Local comparisons and
validation are in `local/syisr-composition-profile-20251217-18`, with numerical
metrics in `comparison.h5` and source records in `execution-provenance.h5`.

### Whole-disc August 2023 and December 2025 temporal campaign

`disc_campaign.py` inventories supported 2 ms zenith recordings, finds complete
frame payloads, retains header-only and short-clock-fragment exclusions, and
checks byte-identical duplicate spans before counting them once. A sampled
voltage frame is 10,576 bytes in August2023 (1194 IQ samples; no extra IQ pair)
and 31,288 bytes in December2025 (3782 samples plus the extra pair). The
1194-sample layout uses 50msIPP, 2068us gate front and430MHz; the newer layout
uses40msIPP,2001us and430.2MHz. Unsupported layouts fail explicitly. Both
retain the original signed I/Q convention and UTC+8 header-clock convention.
Zenith azimuth is physically undefined; isolated azimuth-header anomalies do
not change the validated90degree line of sight. Zero TX-pointing fields in
2023 are unavailable metadata, not a measured TX pointing.

The indexed receive gate front enters every physical range calculation;
frequency enters the scattering wave number and electron-response cache.
The short receive window supports observation centres through1650km, with
an upper nuisance guard at1900km. The December inversion still extends to
5000km with its5500km guard. No high-altitude estimates are fabricated for
2023. Native pair selection, the exact pulse/FIR operator, signed data and
full joint real/imaginary range/lag covariance are retained.

Some2023 headers contain brief date-rollover errors. Indexed spans are anchored
to their valid first header and endpoint-verified nominal50ms cadence. For
segmented2023 processing, `time_utc_s` contains that pulse timeline, while
`original_header_time_utc_s` preserves the raw clocks and metadata records the
number and maximum magnitude of corrections. Short resynchronization fragments
that cannot support a fit are counted in the exclusion inventory; no synthetic
voltages or missing pulses are added. Short acquisition gaps remain explicit.

August voltages contain a strong stationary complex mean that violates the
zero-mean Gaussian product-covariance model. Processing removes each native
sample's complex mean over retained pulses within each nominal10s block and
divides the centred product sum by `N-1`. Removed means are saved. This suppresses
coherent stationary returns and is part of the measurement estimator, not a
residual-based noise inflation. December's previously validated products are
reused unchanged. Two-fold covariance checks start with the established10%
diagonal target component; higher components0.2,0.35,0.5,0.65,0.8 are tried only if the
prediction check fails. Every accepted covariance must improve prediction over
diagonal errors and have normalized held-out energy below2; failed windows
remain explicit. These development checks do not establish total uncertainty.

The new H model replaces the IRI-based skew with the explicitly prescribed
exponential scale-height coordinate

```
Hlocal(z) = w * exp((z-h50)/L)
u(z) = integral[h50,z] dz/Hlocal(z)
     = (L/w)*(1-exp(-(z-h50)/L))
fH(z) = sigmoid(u(z))*onset(z)
```

`L=1500km` is a fixed, configurable skew assumption. `h50` and `w` retain their
fitted crossing-height/local-width meanings. The model is monotone, has a
sharper lower shoulder and slower upper approach, and asymptotes to
`sigmoid(L/w)` rather than being forced to exactly1. The retained C2 onset
sets H/He to0 below500km and reaches full weight at530km. Helium remains a
bounded Gaussian envelope of the non-H population. Analytic derivatives,
monotonicity, local slope, simplex and covariance propagation are tested.

The temporal method follows the forward prediction/update and reverse
Rauch--Tung--Striebel equations3--7 of Virtanen et al.(2021),
*Bayesian Filtering in Incoherent Scatter Plasma Parameter Fits*,
https://doi.org/10.1029/2020JA028700. This is a BAFIM-inspired nonlinear
full-profile filter in transformed coordinates, rather than the original
GUISDAP gated implementation. The previous fitted state initializes the next
fit and supplies a Gaussian prior with covariance `Pprevious+Q(dt)`. All
likelihood terms enter once. Saved forward predictions and full conditional
curvatures are used in the reverse RTS recursion; it does not average two
independent fits or refit the same data as an independent measurement.
Continuous recording spans are independent temporal chains.

The weak nominal30min process standard deviations are logNe0.7, logTi0.4,
logTe0.5, velocity200m/s, H crossing150km, H width factor2, He non-H amplitude
0.1, He centre200km and He width factor2. Variances scale with elapsed time;
`--process-sigma-multiplier` controls their common scale. Larger values weaken
continuity. Spatial regularization is retained and its cost is saved separately
from data and temporal-prior costs. Calibration is fixed within each recording
session, with an explicit provisional local-noon F-peak normalization where
absolute gain is unavailable. Prior-covariance uncertainty, gain/receiver/model
systematics and nonlinear/boundary uncertainty are not covered by the local
Gaussian RTS curvature. Smoothed out-of-bound states and poor data fits retain
quality flags; smoothing does not validate the physical model.

Reproduction, from this package directory:

```sh
cargo test --release --all-targets --features plot
cargo build --release --features plot --bins
conda run -n base python disc_campaign.py plan RAW_DIRECTORY EXTERNAL_OUTPUT
mpirun --bind-to none -np 6 conda run --no-capture-output -n base python \
  disc_campaign.py process RAW_DIRECTORY EXTERNAL_OUTPUT --threads 2
mpirun --bind-to none -np 6 conda run --no-capture-output -n base python \
  temporal_campaign.py prepare EXTERNAL_OUTPUT --threads 1
mpirun --bind-to none -np 6 conda run --no-capture-output -n base python \
  temporal_campaign.py fit EXTERNAL_OUTPUT --threads 1
```

`--wait` lets covariance and fitting consume completed upstream products.
`--retry-failed` on the raw driver retries only recorded failures. Immutable
per-window measured products and atomic covariance files support restart;
unsuccessful fit attempts are preserved. Covariance file assembly uses one
bounded temporary SSD file per rank followed by a contiguous copy to the
external output; the temporary directory is removed. Raw selections are
streamed directly from indexed spans, without storing another raw copy.
All retained numerical products are HDF5. Filtered and smoothed campaign
products and plots are separate, and the display cap remains2500km with
Ne1e13m^-3,Te8000K,Ti4000K maxima.

#### Receiver-response diagnosis: campaign halted, one step checked

The first2023 campaign parameter maps are superseded diagnostic failures.
They must not be interpreted as plasma retrievals. The bulk fit was stopped
and the21Aug2023,14:30--15:00UTC integration was checked independently.
The measured reference voltage spectrum contains a stationary DC component
and an asymmetric excess near-3.5kHz. These noise statistics do not establish
the signal transfer function. Treating that background ACF as a calibrated
signal FIR introduced short-lag ripples into the model and drove temperatures
and composition to bounds.

With the same measured products, nominal flat signal response gives local
reduced chi-squares0.69,1.78,1.03 at453,603,753km. Temperatures there are about
940--1020K. The full joint profile, using the same full measured covariance
and no temporal prior, converges with data chi-square3001.94 and reduced
weighted data chi-square0.756. Its likelihood independently replays to2.2e-14
relative accuracy. Weak upper-profile temperatures remain conditional and
can still hit bounds; this single-step check does not validate a whole-day
product or an independently measured receiver response.

`profile-fit` and local `fit` now default to the explicitly nominal flat
signal response (`SYISR_RECEIVER_MODEL=delta`). Colored measured noise remains
in its own covariance model. The legacy background-derived signal response is
available only for labeled diagnostics with both
`SYISR_RECEIVER_MODEL=inferred` and
`SYISR_ALLOW_BACKGROUND_AS_SIGNAL_FIR=1`. The covariance code may still factor
measured voltage correlations for noise prediction; that is a distinct use.
Automatic density normalization anchors one chosen knot; it no longer caps
all other density knots at1e12m^-3. The free density bound is1e13m^-3.

New temporal trials use a fresh `temporal-flat` directory by default, and
`--sessions 0` restricts execution to the first recording span. No corrected
whole-disc temporal run has been completed. The initial one-step products
are stored separately in `single-step-diagnosis`, including full complex ACF
comparisons and the measured background spectrum. The robust O/H one-day
rerun described below supersedes the corrected-response Gaussian day trial.


#### Robust temporal rerun

The corrected-response day still exposed a temperature/composition local
minimum near sunrise. Competing initializations with identical likelihood
and priors are compared by total objective before a state is propagated.
The robust rerun uses pseudo-Huber penalties on adjacent standardized profile
changes (transition 1.5 process standard deviations, process scales twice the
nominal values above). Small changes are quadratic; large changes have bounded
influence. The measured full covariance and screened lag products are retained.
Forward and reverse coordinate minimization optimize one joint nonlinear MAP
objective, with each time's likelihood and each adjacent temporal edge counted
once. Gaussian RTS is not applied to this non-Gaussian model. Saved per-time
curvature is conditional on neighbouring states, not the joint temporal
posterior uncertainty. Convergence and physical validity are checked separately.

The current controlled rerun fixes helium to zero and removes its three shape
coordinates from optimization and covariance. It tests an O/H model; it does
not establish that helium is physically absent. A second pseudo-Huber norm
(transition 3 standardized residual units) limits extreme data residuals after
fixed full-covariance whitening and linear nuisance projection. The raw Gaussian
chi-square remains a diagnostic, separate from the minimized robust data cost.

The 34-window August day was rerun with helium fixed zero and both robust
norms. Twelve forward/reverse coordinate passes reduced the joint objective
monotonically; the last pass changed predicted ACFs by at most0.0061 marginal
SEM. All windows passed independent data/prior replay (maximum relative data
cost discrepancy2.5e-14); median raw reduced chi-square0.976. One interval still
has raw reduced chi-square above3 and remains flagged. Upper guard coordinates
did not meet the parameter-change stopping criterion; predictions stabilized,
but upper temperatures, active bounds and conditional uncertainties remain
provisional. This is not a validated whole-disc result or satellite identification.
