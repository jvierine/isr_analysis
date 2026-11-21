# Zenith LFM EPB RTIs

This package processes the 480 us, 4 MHz zenith LFM experiment from the SYISR
L1 complex-voltage archive. Unsupported waveform, pointing, sample count and
clock layouts fail explicitly. The March 1, 2026 archive is available beneath
`/gpfs/pub/data/L1/matdata/20260301_SY/LFM/beam_358.0_90.0/LFM`.
The raw file date labels are not reliable observation times; use pulse metadata.
The MATLAB `syspara` clock is Beijing time and is converted to UTC by subtracting
8 hours. An individual 20:00:48.351350 Beijing pulse was checked sample-for-sample
against L0: both I and Q arrays matched exactly, with no coherent averaging.
The archived LFM pulse cadence is 0.768 s, although the transmitter IPP is 12 ms.

The current v2 grid uses 30 s integration and approximately 1.5 km range cells.
The batch script and MPI launcher enforce a 64-core/rank ceiling, with 16 ranks
by default and one compute thread per rank.

The compute executable is Rust, with MPI file distribution and collective
failure reporting. Slurm requests one available node without a node constraint.
The installed `py311` conda environment supplies MPI orchestration; the default
base environment handles Python preparation and final HDF5 reduction.

```sh
sbatch --output=/gpfs/pub/data_other/juha/lfm_epb/20260301/v1/logs/slurm-%j.log \
  syisr/lfm_epb/run.slurm "$HOME/src/isr_analysis" \
  /gpfs/pub/data_other/juha/lfm_epb/20260301/v1 \
  /gpfs/pub/data/L1/matdata/20260301_SY/LFM/beam_358.0_90.0/LFM \
  2026-03-01T10:00:00Z 2026-03-01T16:00:00Z
```

Create the output log directory before submission. Products include individual
pulse powers in `parts`, a 30 s zenith RTI, source paths, processing version,
exact pulse counts, and range and altitude coordinates. Negative signal
estimates are retained. Reprocessing writes each selected part atomically;
combination rejects duplicate pulse timestamps and incompatible grids. Use a
new versioned output root if processing settings or source selection change.

The matched filter is a unit-energy, rectangular 479 us chirp (actual duration
in `para[10]`, versus commanded 480 us in `para[9]`). Both chirp polarities are
retained for diagnostics. Linear correlation uses the echo leading edge;
only fully supported gates are included. Consecutive samples are averaged
in approximately 1.5 km cells, retaining full support to roughly 1680 km.
Nominal timing comes from the 533 us gate-front delay. The old meteor satellite
range offset is not transferred to this experiment. Altitude uses the shared
spherical-Earth ray conversion and assumed 50 m radar elevation. Absolute
hardware-delay calibration is not established. Doppler displacement is not
corrected, and no line-of-sight velocity is inferred.

The noise diagnostic is the median raw voltage power above 1500 km divided by
ln(2); topside ionospheric scatter or interference can bias it. Consequently,
these products show relative echo structure, not calibrated electron density.
Both total power and signed noise-subtracted power must be inspected. The
range-squared signal is an uncalibrated density proxy. Temporal contrast uses
the median signal at each range over the requested interval, not an external
quiet-day reference. Log-display invalid values are masked only in the derived
view. Pulse maxima are retained to identify brief bright contamination;
no automatic bright-pulse rejection is applied. The display uses the median
of individual 3 km pulse powers within each minute to reduce brief target
contamination; all mean and individual-pulse products remain available. This
median is not an unbiased calibrated density estimate. Relative-contrast display
is restricted to signal and reference SNR greater than 0.1, with all signed
scientific arrays retained. The downchirp reference produces the compact peak
of a bright moving-target-like return near 20:00:51 Beijing time; the opposite
reference spreads its energy over approximately 144 km.

Validation: run `cargo test --release --bin syisr_lfm_epb` from `syisr`.
The tests check injected delay recovery, normalization, and UTC conversion.

## Initial March 1 result

The six-hour run spans 18:00–24:00 Beijing time (10:00–16:00 UTC). Slurm job
6057 completed on scheduler-selected node02 with 16 MPI ranks. It retained
28,127 pulses from 428 overlapping files, with 77–79 pulses in every one-minute
bin. The full-supported range axis ends at 1677.8 km; altitude includes the
assumed 50 m site elevation. Reduction and summary jobs were also Slurm batched.

The RTI contains a branched F-region depletion complex around 20:30–21:00
Beijing time. Several tilted branches extend into the topside to approximately
600 km. The depth varies strongly with height: using the 19:00–19:30 local
interval as the reference, the three-minute smoothed band diagnostics have
minimum relative signals of 0.034 (250–300 km, 20:33:30), 0.002 (300–350 km,
20:34:30), 0.030 (350–400 km, 20:36:30), 0.139 (400–450 km, 20:43:30), and
0.330 (450–500 km, 20:44:30). The deepest cores approach the estimated noise
floor: the very small ratios must not be interpreted as precise residual
electron densities. These quantities describe relative echo power, not a
calibrated electron-density retrieval. All heights are nominal.

Above approximately 700 km the recovered signal is weak relative to the noise
estimate, and no reliable upper bubble boundary is established. The recording
coverage to 1678 km does not imply a detected bubble at that altitude.
The morphology is consistent with equatorial plasma-bubble structure; zenith
measurements alone do not establish horizontal motion, three-dimensional
geometry, or a plasma drift speed. The relation of bubble density depletion
to radar irregularity signatures is discussed in
[the Sanya/EAR observational study](https://doi.org/10.1002/jgra.50581).

The supplied LP L2 product has approximately 17-minute zenith sampling and
contains very large isolated density peaks even among successful fit states.
It was inspected but not used to calibrate these LFM powers. A bright return
at 20:00:51–20:00:52 local time dominates ordinary pulse means and spreads over
many ranges when compressed with the opposite chirp; median pulse products
reduce its influence. The sampled L1/L0 voltage identity and both chirp
polarities are retained as explicit validation checks.

The renderer produces full-height, F-region and event-detail views. Its final
contrast panel uses a signed linear ratio to the temporal median, masking only
weak *reference* ranges (reference SNR <= 0.1), so depletion cores remain
visible. The range-squared log panel shows only signal SNR > 0.05, avoiding
amplification of positive noise fluctuations at large ranges. These are display
rules only. The earlier logarithmic contrast fields also remain in HDF5.

## Current requested display

The current renderer saves only the range-squared signal panel using
`ColorMap::storm()`, at 360 dpi. Nonpositive signed power estimates map to the
lowest color rather than white/NaN; missing observations alone remain missing.
The HDF5 signed power is unchanged by this display floor. Full-height, F-region,
and event-detail views use the same panel. The initial findings above refer to
the v1 grid; v2 doubles both time and range resolution.

## v3 resolution correction

v3 reprocesses the voltages into 15 s time bins and approximately 0.75 km
range cells (20 raw samples), retaining the full 18:00–24:00 Beijing interval
and full supported altitude coverage. This doubles time and range resolution
relative to v2; it is not a zoom or interpolated display. Default execution
remains 16 cores. Each bin contains approximately 19–20 sampled pulses, so
noise fluctuations increase relative to the coarser products. The renderer
retains storm and maps nonpositive estimates to its minimum color.

## v4 range resolution

v4 averages 10 raw range samples per cell (0.3747405725 km), doubling
range resolution relative to v3. Time integration remains 15 s, with the
same plot limits, storm palette, signed power and 16-core processing default.

## Eleven-beam field-aligned morphology study

The main display selected by the user is 5 s by 0.3747405725 km. All eleven
available beams are reduced over 11:00–14:00 UTC, with quiet references from
11:00–11:30 UTC, on 1 March 2026. The expanded event view is 11:45–13:45 UTC,
150–900 km geodetic altitude. Its inversion uses 30 s, approximately 3 km
range products sampled every second range gate, over 200–900 km. Reference
SNR below 0.05 is excluded, while weak usable references receive low weights.
Raw signed measurements and all display resolutions remain available.

WGS84 rays use recorded beam pointing and the assumed 50 m site height.
IGRF-14 at the observation epoch maps each scattering point onto a plane
normal to the field at the 400 km zenith point. A fractional contrast is
reconstructed on a 10 km perpendicular lattice while optimizing constant
cross-field pattern advection against withheld whole-beam groups. This is
an imposed field-aligned contrast model, not an unconstrained 3D inversion.
The beam model adopts the expanded-SYISR 0.9 degree normal one-way FWHM at
450 MHz, scaled to the actual 432 MHz frequency and broadened in elevation
by the planar-array scan factor. Nine angular Gaussian quadrature points
represent the product of transmit/receive beams. No measured sidelobes are
claimed. The primary inference excludes centre rays within 2 degrees of
perpendicular to the local field; 5 degree exclusion, including all aspects,
centre-ray geometry, straight-field geometry and drift-shape sensitivities
are retained. Absolute density and coherent/ISR separation remain uncalibrated.

Run `multibeam.slurm` first for the raw reductions. `morphology.slurm` then
performs magnetic mapping, MPI native inversion, full-cell-support rendering
and the expanded 16-resolution comparison. All Sanya compute must be submitted
with `sbatch`; the main jobs use 16 cores and do not pin a node. After the main
job, submit `cavity_checks.slurm` with an `afterok` dependency for noise-free
synthetic recovery and iteration-refinement diagnostics. HDF5 outputs remain
under `/gpfs/pub/data_other/juha/lfm_epb/20260301/`.

The lifted volume contains supported nodes with relative-density proxy <=0.7,
continued along IGRF for a prescribed +/-400 km, clipped at 150–1400 km height.
That field-parallel extent is a modelling choice, not an inferred length.
Projected-gradient density bounds act on model parameters; observations are
never clipped to enforce a positive power or density.

The beam atlas and overhead geometry can be composed without resampling their embedded images:
```sh
conda run -n base python syisr/lfm_epb/assemble_beam_reference.py ROOT/figures
```

`syisr/lfm_epb/symmetry_validation.slurm` runs an exploratory comparison of
frozen, along-field-varying, time-evolving, and combined contrast models.
Three outer whole-beam folds hold event observations out; smoothing is selected
with two inner whole-beam folds. Drift is fixed to the earlier data-selected
value, so this is conditional validation, not a blind end-to-end test.
The linear diagnostic models are unconstrained; they are distinct from the
bounded physical proxy reconstruction. Additional coefficients use 500 km and
1 hour scales and a stated ridge penalty. HDF5 retains predictions, scores,
selected penalties and local time/parallel-coordinate conditioning.
`symmetry_injections.py` checks noise-free discrimination on the actual finite
beam sampling. Neither check establishes calibrated errors, coherent-scatter
cleanliness, or a measured field-parallel depletion length.
