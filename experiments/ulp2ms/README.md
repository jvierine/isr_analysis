# SYISR 2 ms uncoded long-pulse experiment

Rust processing of the SYISR uncoded 2 ms raw-voltage recording, including
range–Doppler IQR/coherence satellite rejection, lag-profile estimation,
H+/He+/O+ mixture fits, and rustplotlib figures. All scientific products use
HDF5; raw recordings and generated outputs are excluded from Git. The
implementation intentionally supports the layout below and fails on other
sampling, pulse timing, beam geometry, or frame structure.

## Build

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
