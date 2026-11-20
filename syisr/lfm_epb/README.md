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
pulse powers in `parts`, a 60 s zenith RTI, source paths, processing version,
exact pulse counts, and range and altitude coordinates. Negative signal
estimates are retained. Reprocessing writes each selected part atomically;
combination rejects duplicate pulse timestamps and incompatible grids. Use a
new versioned output root if processing settings or source selection change.

The matched filter is a unit-energy, rectangular 479 us chirp (actual duration
in `para[10]`, versus commanded 480 us in `para[9]`). Both chirp polarities are
retained for diagnostics. Linear correlation uses the echo leading edge;
only fully supported gates are included. Consecutive samples are averaged
in approximately 3 km cells, retaining full support to roughly 1680 km.
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
